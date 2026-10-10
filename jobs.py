#!/usr/bin/env python3
"""Background jobs for the web GUI.

A job runs sync_media.py or compose_media.py in a detached process, so it keeps
going when the browser tab is closed or the SSH connection drops.  Everything
about a job lives in its own folder under ``~/.mic-array-synchronizer/jobs``:

* ``job.json``  - kind, arguments and start time
* ``log.txt``   - the script's output (including ``PROGRESS`` lines)
* ``pid``       - the runner's process id (also its process group)
* ``exit_code`` - written when the script ends
* ``stopped``   - written when the user stops the job

The GUI only reads these files.  ``python jobs.py run <job folder>`` is the
detached runner itself.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sync_media import PROGRESS_ENV

HERE = Path(__file__).resolve().parent
JOBS_DIR = Path.home() / ".mic-array-synchronizer" / "jobs"
SCRIPTS = {"sync": "sync_media.py", "compose": "compose_media.py"}


@dataclass
class Job:
    directory: Path

    @property
    def id(self) -> str:
        return self.directory.name

    @property
    def info(self) -> dict:
        return json.loads((self.directory / "job.json").read_text(encoding="utf-8"))

    @property
    def kind(self) -> str:
        return self.info["kind"]

    @property
    def arguments(self) -> list[str]:
        return self.info["arguments"]

    @property
    def log_lines(self) -> list[str]:
        path = self.directory / "log.txt"
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines()

    @property
    def exit_code(self) -> int | None:
        path = self.directory / "exit_code"
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            return None

    @property
    def stopped(self) -> bool:
        return (self.directory / "stopped").exists()

    @property
    def pid(self) -> int | None:
        try:
            return int((self.directory / "pid").read_text().strip())
        except (OSError, ValueError):
            return None

    @property
    def running(self) -> bool:
        if self.exit_code is not None:
            return False
        pid = self.pid
        if pid is None:  # the runner has not written its pid yet
            started = self.info.get("started", 0)
            return time.time() - started < 30
        return _process_alive(pid)

    @property
    def state(self) -> str:
        """running / done / failed / stopped / lost"""
        if self.running:
            return "running"
        if self.stopped:
            return "stopped"
        code = self.exit_code
        if code is None:
            return "lost"  # the runner died without writing an exit code
        return "done" if code == 0 else "failed"


def _process_alive(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def start_job(kind: str, arguments: list[str]) -> Job:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    directory = JOBS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}"
    counter = 2
    while directory.exists():
        directory = directory.with_name(f"{directory.name}-{counter}")
        counter += 1
    directory.mkdir()
    (directory / "job.json").write_text(
        json.dumps({"kind": kind, "arguments": arguments, "started": time.time()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )  # fmt: skip
    detach = (
        {"creationflags": 0x00000200 | 0x00000008}  # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
        if os.name == "nt"
        else {"start_new_session": True}
    )
    subprocess.Popen(
        [sys.executable, str(HERE / "jobs.py"), "run", str(directory)],
        cwd=HERE,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **detach,
    )
    return Job(directory)


def stop_job(job: Job) -> None:
    if not job.running:
        return
    (job.directory / "stopped").touch()
    pid = job.pid
    if pid is None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)  # the runner leads its own group
        except ProcessLookupError:
            pass


def list_jobs(kind: str | None = None) -> list[Job]:
    """Jobs, newest first."""
    if not JOBS_DIR.exists():
        return []
    jobs = [Job(d) for d in JOBS_DIR.iterdir() if (d / "job.json").exists()]
    if kind:
        jobs = [j for j in jobs if j.kind == kind]
    return sorted(jobs, key=lambda j: j.id, reverse=True)


def run(directory: Path) -> int:
    """The detached runner: run the script, log its output, record the exit code."""
    job = Job(directory)
    (directory / "pid").write_text(str(os.getpid()))
    environment = dict(
        os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", **{PROGRESS_ENV: "1"}
    )
    with open(directory / "log.txt", "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(HERE / SCRIPTS[job.kind]), *job.arguments],
            cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=environment,
        )  # fmt: skip
        try:
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            code = process.wait()
    (directory / "exit_code").write_text(str(code))
    return code


# ---------------------------------------------------------------------------
# Progress from the log


def _seconds(line: str) -> float | None:
    match = re.search(r"([0-9.]+)\s*秒", line)
    return float(match.group(1)) if match else None


def sync_progress(lines: list[str], file_count: int, analyze_only: bool) -> tuple[float, str]:
    """(fraction, status) for a sync job.

    Phases: beep search 0-10 %, correlation 10-30 %, cutting and the check WAV
    30-100 % (analysis alone is scaled to the whole bar).
    """
    fraction, status = 0.0, "開始しています…"
    searched = cut = 0
    duration: float | None = None
    for line in lines:
        if line.startswith("PROGRESS ") and duration and cut:
            seconds = float(line.split()[1])
            done = (cut - 1 + min(1.0, seconds / duration)) / (file_count + 1)
            fraction = 0.30 + 0.70 * done
        elif line.startswith("同期音を探索中"):
            searched += 1
            fraction, status = 0.10 * searched / file_count, "同期音を探しています…"
        elif line.strip().startswith("相関区間"):
            match = re.search(r"(\d+)/(\d+)", line)
            if match:
                step, total = (int(v) for v in match.groups())
                fraction = 0.10 + 0.20 * step / total
                status = f"相互相関を計算中（{step}/{total}）"
        elif line.startswith("共通区間"):
            duration = _seconds(line)
        elif line.startswith("切り出し中"):
            cut += 1
            name = line.split(":", 1)[1].split("->")[0].strip()
            fraction = 0.30 + 0.70 * (cut - 1) / (file_count + 1)
            status = f"切り出し中（{cut}/{file_count}）: {name}"
        elif line.startswith("確認用WAV"):
            cut = file_count + 1
            fraction = 0.30 + 0.70 * file_count / (file_count + 1)
            status = "確認用WAVを作成中…"
    if analyze_only:
        fraction = min(1.0, fraction / 0.30)
    return fraction, status


def compose_progress(lines: list[str]) -> tuple[float, str]:
    fraction, status = 0.0, "開始しています…"
    duration: float | None = None
    for line in lines:
        if line.startswith("長さ"):
            duration = _seconds(line)
        elif line.startswith("PROGRESS ") and duration:
            seconds = float(line.split()[1])
            fraction = min(1.0, seconds / duration)
            status = f"合成中… {seconds / 60:.1f} / {duration / 60:.1f} 分"
    return fraction, status


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        raise SystemExit(run(Path(sys.argv[2])))
    raise SystemExit("usage: python jobs.py run <job folder>")
