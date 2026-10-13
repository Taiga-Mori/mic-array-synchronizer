#!/usr/bin/env python3
"""Browser GUI for sync_media.py, compose_media.py and trim_media.py.

A small HTTP server (standard library only) serves the page in ``web/`` and a
JSON API.  Jobs run detached through jobs.py, so they survive a closed browser
or a dropped SSH connection.  The server listens on localhost only; use
``ssh -L`` to reach it on another machine.  Start it with ``python app.py``.

API (all JSON unless noted):

* ``GET  /api/info``                 home folder, encoders
* ``GET  /api/browse?path=``         folder listing
* ``GET  /api/probe?path=``          duration, video size, audio channels
* ``GET  /api/jobs?kind=``           job list with progress
* ``GET  /api/jobs/<id>``            one job: progress, log, report, outputs
* ``POST /api/jobs``                 ``{"kind", "arguments"}`` -> start a job
* ``POST /api/jobs/<id>/stop``       stop a job
* ``POST /api/preview``              ``{"videos", "layout", "cell", "at"}`` -> PNG
* ``GET  /api/file?path=``           any file, with HTTP ranges (video playback)
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import tempfile
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import compose_media
import jobs
from sync_media import (
    REPORT_NAME,
    json_safe,
    VIDEO_ENCODERS,
    SyncError,
    check_binary,
    default_ffprobe,
    find_tool,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".mts", ".m2ts", ".avi", ".mxf",
    ".wav", ".mp3", ".m4a", ".flac", ".aac", ".aif", ".aiff",
}  # fmt: skip
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("text/javascript", ".js")

_tools: dict[str, str] = {}
_tools_lock = threading.Lock()


def tools() -> tuple[str, str]:
    with _tools_lock:
        if not _tools:
            ffmpeg = check_binary(find_tool("ffmpeg"), "ffmpeg")
            _tools["ffmpeg"] = ffmpeg
            _tools["ffprobe"] = check_binary(default_ffprobe(ffmpeg), "ffprobe")
    return _tools["ffmpeg"], _tools["ffprobe"]


def as_path(text: str | None) -> Path:
    if not text:
        raise SyncError("パスを指定してください。")
    return Path(text).expanduser()


# ---------------------------------------------------------------------------
# API


def api_info() -> dict:
    return {
        "home": str(Path.home()),
        "cwd": os.getcwd(),
        "video_encoders": ["auto", *VIDEO_ENCODERS],
        "media_extensions": sorted(MEDIA_EXTENSIONS),
    }


def api_browse(path_text: str | None) -> dict:
    folder = as_path(path_text or str(Path.home()))
    if folder.is_file():
        folder = folder.parent
    if not folder.is_dir():
        raise SyncError(f"フォルダが見つかりません: {folder}")
    entries = []
    try:
        children = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as error:
        raise SyncError(f"フォルダを開けません: {error}") from error
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
            stat = child.stat()
        except OSError:
            continue
        entries.append({
            "name": child.name,
            "path": str(child),
            "dir": is_dir,
            "size": None if is_dir else stat.st_size,
            "mtime": stat.st_mtime,
            "media": not is_dir and child.suffix.lower() in MEDIA_EXTENSIONS,
        })  # fmt: skip
    parent = folder.parent if folder.parent != folder else None
    return {"path": str(folder), "parent": str(parent) if parent else None, "entries": entries}


def api_probe(path_text: str | None) -> dict:
    path = as_path(path_text)
    if not path.is_file():
        raise SyncError(f"ファイルが見つかりません: {path}")
    _, ffprobe = tools()
    source = compose_media.probe(path, ffprobe)
    return {
        "path": str(path),
        "duration": source.duration,
        "video_size": source.video_size,
        "fps": source.fps,
        "video_codec": source.video_codec,
        "audio_channels": source.audio_channels,
    }


def job_summary(job: jobs.Job) -> dict:
    fraction, status = jobs.progress(job)
    info = job.info
    return {
        "id": job.id,
        "kind": job.kind,
        "state": job.state,
        "started": info.get("started"),
        "progress": fraction,
        "status": status,
        "output": jobs.option(job.arguments, "-o", "--output", "--output-dir"),
    }


def job_outputs(job: jobs.Job, report: dict | None) -> list[str]:
    """Files the job wrote, for the result list (existing files only)."""
    target = jobs.option(job.arguments, "-o", "--output", "--output-dir")
    if not target:
        return []
    if job.kind == "sync":
        if not report:
            return []
        files = [item["output"] for item in report.get("inputs", []) if item.get("output")]
        if report.get("check_wav"):
            files.append(report["check_wav"]["path"])
        return [f for f in files if Path(f).is_file()]
    if job.kind == "compose":
        return [target] if Path(target).is_file() else []
    folder = Path(target)
    started = job.info.get("started", 0) - 5
    if not folder.is_dir():
        return []
    return sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".wav"} and p.stat().st_mtime >= started
    )  # fmt: skip


def api_job(job_id: str) -> dict:
    job = jobs.Job(jobs.JOBS_DIR / job_id)
    if not (job.directory / "job.json").exists():
        raise SyncError(f"実行が見つかりません: {job_id}")
    summary = job_summary(job)
    report = None
    if job.kind == "sync" and summary["output"]:
        try:
            report = json.loads((Path(summary["output"]) / REPORT_NAME).read_text(encoding="utf-8"))
            if report.get("created") and Path(summary["output"], REPORT_NAME).stat().st_mtime < job.info.get("started", 0):
                report = None  # left over from an earlier run
        except (OSError, json.JSONDecodeError):
            report = None
    lines = [line for line in job.log_lines if not line.startswith("PROGRESS ")]
    return {
        **summary,
        "arguments": job.arguments,
        "log": "\n".join(lines[-400:]),
        "report": report,
        "outputs": job_outputs(job, report),
    }


def api_start(body: dict) -> dict:
    kind = body.get("kind")
    arguments = body.get("arguments")
    if kind not in jobs.SCRIPTS or not isinstance(arguments, list):
        raise SyncError("kind と arguments を指定してください。")
    job = jobs.start_job(kind, [str(a) for a in arguments])
    return {"id": job.id}


def api_preview(body: dict) -> bytes:
    ffmpeg, ffprobe = tools()
    paths = [as_path(p) for p in body.get("videos", [])]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SyncError("見つからないファイルがあります: " + ", ".join(missing))
    sources = [s for s in (compose_media.probe(p, ffprobe) for p in paths) if s.video_size]
    if not sources:
        raise SyncError("映像のあるファイルを指定してください。")
    cell = compose_media.parse_size(body["cell"]) if body.get("cell") else None
    at = body.get("at")
    at = max(s.duration for s in sources) / 2 if at in (None, "") else float(at)
    handle, name = tempfile.mkstemp(prefix="compose_preview_", suffix=".png")
    os.close(handle)
    image = Path(name)
    try:
        command = compose_media.build_preview_command(
            sources, image, ffmpeg=ffmpeg, layout=body.get("layout", "grid"),
            cell=cell, at=at, max_width=1600,
        )  # fmt: skip
        compose_media.run_checked(command, echo=False)
        return image.read_bytes()
    finally:
        image.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# HTTP


class Handler(BaseHTTPRequestHandler):
    server_version = "mic-array-synchronizer"

    def log_message(self, format: str, *args: object) -> None:  # quieter console
        if not self.path.startswith("/api/jobs"):
            super().log_message(format, *args)

    def send_json(self, data: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        # json_safe: reports written before NaN became null would break JSON.parse.
        body = json.dumps(json_safe(data), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def handle_errors(self, action) -> None:
        try:
            action()
        except SyncError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:  # report, keep serving
            traceback.print_exc()
            self.send_json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        route = url.path

        def action() -> None:
            if route == "/api/info":
                self.send_json(api_info())
            elif route == "/api/browse":
                self.send_json(api_browse(query.get("path")))
            elif route == "/api/probe":
                self.send_json(api_probe(query.get("path")))
            elif route == "/api/jobs":
                kind = query.get("kind")
                self.send_json([job_summary(j) for j in jobs.list_jobs(kind)[:50]])
            elif match := re.fullmatch(r"/api/jobs/([\w-]+)", route):
                self.send_json(api_job(match.group(1)))
            elif route == "/api/file":
                self.send_file(as_path(query.get("path")))
            else:
                self.send_static(route)

        self.handle_errors(action)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path

        def action() -> None:
            if route == "/api/jobs":
                self.send_json(api_start(self.read_json()))
            elif match := re.fullmatch(r"/api/jobs/([\w-]+)/stop", route):
                jobs.stop_job(jobs.Job(jobs.JOBS_DIR / match.group(1)))
                self.send_json({"ok": True})
            elif route == "/api/preview":
                self.send_bytes(api_preview(self.read_json()), "image/png")
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        self.handle_errors(action)

    def send_static(self, route: str) -> None:
        name = "index.html" if route in ("", "/") else unquote(route.lstrip("/"))
        path = (WEB_DIR / name).resolve()
        if WEB_DIR not in path.parents or not path.is_file():
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        self.send_bytes(path.read_bytes(), content_type)

    def send_file(self, path: Path) -> None:
        """Serve a file with single-range support, so that <video> can seek."""
        if not path.is_file():
            raise SyncError(f"ファイルが見つかりません: {path}")
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start = int(match.group(1))
                end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
            else:
                start = max(0, size - int(match.group(2)))
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def serve(address: str, port: int | None, open_browser: bool) -> None:
    """Listen on ``port`` (or the first free one from 8501) and serve forever."""
    candidates = [port] if port else list(range(8501, 8521))
    server = None
    for candidate in candidates:
        try:
            server = ThreadingHTTPServer((address, candidate), Handler)
            break
        except OSError:
            if port:
                raise SystemExit(f"ポート {port} は使用中です。")
    if server is None:
        raise SystemExit("8501〜8520 に空いているポートがありません。--port で指定してください。")
    server.daemon_threads = True
    url = f"http://{'localhost' if address in ('127.0.0.1', 'localhost', '::1') else address}:{server.server_port}/"
    print(f"GUI: {url}  （終了は Ctrl+C）", flush=True)
    if open_browser:
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n終了しました。実行中の処理は止まりません。")
    finally:
        server.server_close()


if __name__ == "__main__":
    serve("localhost", None, open_browser=False)
