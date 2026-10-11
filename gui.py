"""Streamlit GUI for sync_media.py and compose_media.py.

Start it with ``python app.py`` (or start.sh / start.bat).  Jobs run detached
through jobs.py, so closing the browser or losing the SSH connection does not
stop them; reopening the page shows the latest job again.
"""

from __future__ import annotations

import html
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import compose_media
import jobs
from sync_media import (
    REPORT_NAME,
    VIDEO_ENCODERS,
    SyncError,
    check_binary,
    default_ffprobe,
    find_tool,
    open_in_default_app,
)

MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".mts", ".m2ts", ".avi",
    ".wav", ".mp3", ".m4a", ".flac", ".aac",
}  # fmt: skip
TILE_COLORS = ["#3B6FB6", "#E08A2E", "#3E9B5B", "#8A5BB0", "#C9483F", "#2E9BA6"]
CODECS = {"入力のまま": "source", "HEVC": "hevc", "H.264": "h264"}
ENCODER_MODES = {"自動": "auto", "ハードウェア": "hardware", "CPU": "software"}
LAYOUTS = {"横一列": "row", "縦一列": "column", "なるべく正方形": "grid"}
AUDIO_MODES = {"1chにミックス": "mix", "全チャンネル": "multi"}
STATE_LABELS = {
    "running": "実行中",
    "done": "完了",
    "failed": "エラーで終了",
    "stopped": "中止",
    "lost": "中断（実行プロセスが見つかりません）",
}


# ---------------------------------------------------------------------------
# Paths


def parse_paths(text: str) -> list[Path]:
    """One path per line; surrounding quotes (from "copy as path") are dropped."""
    paths = []
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'")
        if line:
            paths.append(Path(line).expanduser())
    return paths


def running_remotely() -> bool:
    """True when a dialog opened by this server would not be on the user's screen."""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
        return True
    return sys.platform.startswith("linux") and not os.environ.get("DISPLAY")


def native_dialog(kind: str) -> list[str]:
    """OS file dialog on the server's screen: kind is files, folder or save."""
    if sys.platform == "darwin":
        scripts = {
            "files": 'set picked to choose file with prompt "メディアを選択" with multiple selections allowed\n'
                     'set out to ""\nrepeat with f in picked\nset out to out & POSIX path of f & linefeed\n'
                     "end repeat\nreturn out",
            "folder": 'return POSIX path of (choose folder with prompt "フォルダを選択")',
            "save": 'return POSIX path of (choose file name with prompt "出力ファイル" default name "composite.mp4")',
        }  # fmt: skip
        result = subprocess.run(["osascript", "-e", scripts[kind]], capture_output=True, text=True, check=False)
        return [line for line in result.stdout.splitlines() if line.strip()]
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "files":
            return list(filedialog.askopenfilenames(title="メディアを選択"))
        if kind == "folder":
            chosen = filedialog.askdirectory(title="フォルダを選択")
        else:
            chosen = filedialog.asksaveasfilename(title="出力ファイル", defaultextension=".mp4")
        return [chosen] if chosen else []
    finally:
        root.destroy()


def _append_lines(key: str, lines: list[str]) -> None:
    current = parse_paths(st.session_state.get(key, ""))
    known = {str(p) for p in current}
    added = [line for line in lines if line not in known]
    st.session_state[key] = "\n".join([*map(str, current), *added])


def _pick_native(key: str, kind: str) -> None:
    chosen = native_dialog(kind)
    if not chosen:
        return
    if kind == "files":
        _append_lines(key, chosen)
    else:
        st.session_state[key] = chosen[0]


def path_buttons(key: str, kind: str) -> None:
    """「参照」 for local use plus an in-page browser of the server's folders."""
    if not running_remotely() and (sys.platform == "darwin" or sys.platform.startswith("win")):
        st.button("参照…", key=f"{key}_native", on_click=_pick_native, args=(key, kind))
    with st.expander("サーバー上のフォルダから選ぶ"):
        server_browser(key, kind)


def server_browser(key: str, kind: str) -> None:
    folder_key = f"{key}_folder"
    if folder_key not in st.session_state:
        existing = parse_paths(st.session_state.get(key, ""))
        start = existing[-1] if existing else Path.home()
        st.session_state[folder_key] = str(start if start.is_dir() else start.parent)
    folder = Path(st.session_state[folder_key]).expanduser()
    if not folder.is_dir():
        st.warning(f"フォルダが見つかりません: {folder}")
        folder = Path.home()

    def go(target: Path) -> None:
        st.session_state[folder_key] = str(target)

    st.text_input("フォルダ", key=folder_key)
    try:
        entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
    except OSError as error:
        st.warning(f"フォルダを開けません: {error}")
        return
    subfolders = [p for p in entries if p.is_dir() and not p.name.startswith(".")]
    columns = st.columns([1, 3, 1])
    columns[0].button("↑ 上へ", key=f"{key}_up", on_click=go, args=(folder.parent,))
    choice = columns[1].selectbox(
        "サブフォルダ", subfolders, format_func=lambda p: p.name, key=f"{key}_sub",
        label_visibility="collapsed", index=None, placeholder="サブフォルダを選ぶ",
    )  # fmt: skip
    columns[2].button("開く", key=f"{key}_open", on_click=go, args=(choice,), disabled=choice is None)

    if kind == "files":
        media = [p for p in entries if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
                 and not p.name.startswith("._")]  # fmt: skip
        picked = st.multiselect("このフォルダのメディア", media, format_func=lambda p: p.name, key=f"{key}_pick")
        st.button("リストに追加", key=f"{key}_add", disabled=not picked,
                  on_click=lambda: (_append_lines(key, [str(p) for p in picked]),
                                    st.session_state.update({f"{key}_pick": []})))  # fmt: skip
    elif kind == "folder":
        st.button("このフォルダを出力先にする", key=f"{key}_use",
                  on_click=lambda: st.session_state.update({key: str(folder)}))  # fmt: skip
    else:
        st.button("このフォルダに保存する", key=f"{key}_use", on_click=lambda: st.session_state.update(
            {key: str(folder / Path(st.session_state.get(key) or "composite.mp4").name)}))  # fmt: skip


def path_list_table(paths: list[Path], numbered_label: str) -> None:
    if not paths:
        return
    rows = [{numbered_label: i, "ファイル": p.name, "フォルダ": str(p.parent),
             "状態": "OK" if p.is_file() else "見つかりません"} for i, p in enumerate(paths, start=1)]  # fmt: skip
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Jobs


def job_selector(kind: str) -> jobs.Job | None:
    history = jobs.list_jobs(kind)
    if not history:
        st.caption("まだ実行していません。")
        return None
    labels = {j.id: f"{j.id}（{STATE_LABELS[j.state]}）" for j in history[:20]}
    selected = st.selectbox("実行履歴", list(labels), format_func=labels.get, key=f"{kind}_job")
    return jobs.Job(jobs.JOBS_DIR / selected)


def show_log(job: jobs.Job) -> None:
    lines = [line for line in job.log_lines if not line.startswith("PROGRESS ")]
    with st.expander("ログ", expanded=job.state == "failed"):
        st.code("\n".join(lines[-300:]) or "（まだ出力はありません）", language=None)


def job_controls(job: jobs.Job) -> None:
    state = job.state
    columns = st.columns([1, 4])
    if state == "running":
        columns[0].button("中止", key=f"stop_{job.id}", on_click=jobs.stop_job, args=(job,))
    label = STATE_LABELS[state]
    if state == "failed":
        columns[1].error(label)
    elif state in {"stopped", "lost"}:
        columns[1].warning(label)
    elif state == "done":
        columns[1].success(label)


def output_flag(arguments: list[str], name: str) -> str | None:
    return arguments[arguments.index(name) + 1] if name in arguments else None


# ---------------------------------------------------------------------------
# 同期 tab


def sync_tab() -> None:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("入力ファイル")
        st.caption("同じ会話のカメラ映像とマイク音声のパスを1行に1つずつ。順番は sync_check.wav のチャンネル順になります。")
        st.text_area("入力ファイル", key="sync_inputs", height=160, label_visibility="collapsed",
                     placeholder="/Volumes/TEIDAN/A/GX010310.MP4\n/Volumes/TEIDAN/micarray/0_120_240.wav")  # fmt: skip
        path_buttons("sync_inputs", "files")
        path_list_table(parse_paths(st.session_state.get("sync_inputs", "")), "番号")

        st.subheader("出力フォルダ")
        st.text_input("出力フォルダ", key="sync_output", label_visibility="collapsed")
        path_buttons("sync_output", "folder")

        st.subheader("オプション")
        st.checkbox("解析だけ行う（report.json のみ作成）", key="sync_analyze_only")
        st.checkbox("映像を再エンコードしない（速いが開始位置はキーフレーム単位）", key="sync_fast_copy")
        columns = st.columns(2)
        columns[0].selectbox("映像エンコーダ", ["auto", *VIDEO_ENCODERS], key="sync_encoder",
                             help="auto: 使えるハードウェアのHEVC（Mac/NVIDIA/Intel/AMD）、なければlibx264")  # fmt: skip
        columns[1].text_input("ビットレート（空欄=標準）", key="sync_bitrate", placeholder="20M")
        columns = st.columns(2)
        columns[0].number_input("同期音の何秒前から残すか", key="sync_pre_roll", value=0.05, min_value=0.0,
                                step=0.01, format="%.2f")  # fmt: skip
        columns[1].text_input("その他の引数", key="sync_extra", placeholder="--manual-beep MIC3.WAV=24.4")

        running = any(j.running for j in jobs.list_jobs("sync"))
        if st.button("同期を開始", type="primary", disabled=running):
            start_sync()
        if running:
            st.caption("同期の処理を実行中です。終わってから次を開始できます。")

    with right:
        st.subheader("実行状況")
        sync_status()


def start_sync() -> None:
    inputs = parse_paths(st.session_state.get("sync_inputs", ""))
    output = st.session_state.get("sync_output", "").strip()
    missing = [str(p) for p in inputs if not p.is_file()]
    if not inputs:
        st.error("入力ファイルを指定してください。")
        return
    if missing:
        st.error("見つからないファイルがあります:\n\n" + "\n".join(f"- `{m}`" for m in missing))
        return
    if not output:
        st.error("出力フォルダを指定してください。")
        return
    arguments = ["-o", output, *map(str, inputs),
                 "--video-encoder", st.session_state["sync_encoder"],
                 "--pre-roll", str(st.session_state["sync_pre_roll"])]  # fmt: skip
    if st.session_state.get("sync_analyze_only"):
        arguments.append("--analyze-only")
    if st.session_state.get("sync_fast_copy"):
        arguments.append("--fast-copy")
    if st.session_state.get("sync_bitrate", "").strip():
        arguments += ["--video-bitrate", st.session_state["sync_bitrate"].strip()]
    try:
        import shlex

        arguments += shlex.split(st.session_state.get("sync_extra", ""))
    except ValueError as error:
        st.error(f"その他の引数を解釈できません: {error}")
        return
    job = jobs.start_job("sync", arguments)
    st.session_state["sync_job"] = job.id
    st.rerun()


@st.fragment(run_every=3)
def sync_status() -> None:
    job = job_selector("sync")
    if job is None:
        return
    arguments = job.arguments
    output = Path(output_flag(arguments, "-o") or ".")
    analyze_only = "--analyze-only" in arguments
    fraction, status = jobs.sync_progress(job.log_lines, max(1, _count_inputs(arguments)), analyze_only)
    if job.state == "done":
        fraction, status = 1.0, "完了しました"
    st.progress(fraction, text=status)
    job_controls(job)
    show_log(job)
    report = _load_report(output)
    if report:
        sync_results(report, job)


def _count_inputs(arguments: list[str]) -> int:
    """Inputs are the positional arguments right after ``-o OUTPUT``."""
    count = 0
    for argument in arguments[2:]:
        if argument.startswith("--"):
            break
        count += 1
    return count


def _load_report(output: Path) -> dict | None:
    try:
        return json.loads((output / REPORT_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def sync_results(report: dict, job: jobs.Job) -> None:
    st.markdown("**結果（report.json）**")
    rows = []
    for item in report["inputs"]:
        warn = item["correlation_pairs_used"] == 0 or item["confidence"] in {"low", "timecode-only"}
        rows.append({
            "要確認": "⚠️" if warn else "",
            "ファイル": Path(item["input"]).name,
            "検出": item["confidence"],
            "相関補正 ms": item["correlation_correction_ms"],
            "ドリフト ppm": item["drift_ppm"],
            "ペア": item["correlation_pairs_used"],
            "タイムコード": item["timecode_status"],
        })  # fmt: skip
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    drifts = [item["drift_ppm"] for item in report["inputs"] if item["drift_ppm"] is not None] or [0.0]
    duration = report.get("common_duration_seconds") or 0
    spread = (max(drifts) - min(drifts)) * 1e-6 * duration * 1000
    st.caption(f"共通区間 {duration / 60:.1f} 分。ドリフトは補正しないため、終端では最大 約{spread:.0f} ms ずれます。"
               + ("　⚠️ のファイルは sync_check.wav で同期を確認してください。" if any(r["要確認"] for r in rows) else ""))  # fmt: skip
    if job.state == "done" and report.get("check_wav"):
        columns = st.columns(2)
        if columns[0].button("合成タブに入力する →", key=f"to_compose_{job.id}"):
            send_to_compose(report)
            st.rerun(scope="app")  # the button sits in a fragment; refresh the 合成 tab too
        if not running_remotely():
            columns[1].button("出力フォルダを開く", key=f"open_{job.id}", on_click=open_in_default_app,
                              args=(Path(report["check_wav"]["path"]).parent,))  # fmt: skip


def send_to_compose(report: dict) -> None:
    videos = [i["output"] for i in report["inputs"] if i["has_video"]]
    audios = [i["output"] for i in report["inputs"] if not i["has_video"]]
    st.session_state["compose_videos"] = "\n".join(videos)
    st.session_state["compose_audios"] = "\n".join(audios)
    st.session_state["compose_output"] = str(Path(report["check_wav"]["path"]).parent / "composite.mp4")
    st.session_state["toast"] = "合成タブに入力しました。「合成」タブを開いてください。"


# ---------------------------------------------------------------------------
# 合成 tab


def compose_tab() -> None:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("映像（左上から順に配置）")
        st.text_area("映像", key="compose_videos", height=110, label_visibility="collapsed")
        path_buttons("compose_videos", "files")
        st.radio("配置", list(LAYOUTS), key="compose_layout", horizontal=True)

        st.subheader("音声")
        st.text_area("音声", key="compose_audios", height=90, label_visibility="collapsed")
        path_buttons("compose_audios", "files")
        st.radio("音声の形", list(AUDIO_MODES), key="compose_audio_mode", horizontal=True)

        st.subheader("映像の形式・大きさ・ビットレート")
        columns = st.columns(2)
        columns[0].selectbox("形式", list(CODECS), key="compose_codec")
        columns[1].selectbox("エンコーダ", list(ENCODER_MODES), key="compose_encoder")
        columns = st.columns(2)
        columns[0].text_input("1画面の大きさ（WxH、空欄=入力のまま）", key="compose_cell", placeholder="640x360")
        columns[1].text_input("ビットレート（空欄=入力のまま）", key="compose_bitrate", placeholder="8M")

        st.subheader("出力ファイル")
        st.text_input("出力ファイル", key="compose_output", label_visibility="collapsed",
                      placeholder="/Volumes/TEIDAN/sync/composite.mp4")  # fmt: skip
        path_buttons("compose_output", "save")

    with right:
        st.subheader("配置イメージ")
        videos = parse_paths(st.session_state.get("compose_videos", ""))
        st.markdown(layout_svg(videos), unsafe_allow_html=True)
        preview_area(videos)

        st.subheader("実行")
        output = st.session_state.get("compose_output", "").strip()
        overwrite = True
        if output and Path(output).exists():
            overwrite = st.checkbox("出力ファイルはすでにあります。上書きする", key="compose_overwrite")
        running = any(j.running for j in jobs.list_jobs("compose"))
        if st.button("合成を開始", type="primary", disabled=running or not overwrite):
            start_compose()
        compose_status()


def _cell() -> tuple[int, int] | None:
    text = st.session_state.get("compose_cell", "").strip()
    return compose_media.parse_size(text) if text else None


def layout_svg(videos: list[Path]) -> str:
    if not videos:
        return "<p style='color:gray'>映像を指定すると配置が表示されます。</p>"
    try:
        cell = _cell()
    except SyncError:
        cell = None
    aspect = cell[0] / cell[1] if cell else 16 / 9
    columns, rows = compose_media.grid_shape(len(videos), LAYOUTS[st.session_state.get("compose_layout", "横一列")])
    cell_w = 160.0
    cell_h = cell_w / aspect
    width, height = columns * cell_w, rows * cell_h
    parts = [f'<svg viewBox="0 0 {width:.0f} {height:.0f}" style="width:100%;max-height:320px" '
             f'xmlns="http://www.w3.org/2000/svg">']  # fmt: skip
    for slot in range(columns * rows):
        x, y = (slot % columns) * cell_w, (slot // columns) * cell_h
        if slot >= len(videos):
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="#000"/>')
            continue
        color = TILE_COLORS[slot % len(TILE_COLORS)]
        name = html.escape(videos[slot].name)
        parts.append(
            f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="#fff" stroke-width="2"/>'
            f'<text x="{x + cell_w / 2}" y="{y + cell_h * 0.5}" font-size="{cell_h * 0.38:.0f}" fill="#fff" '
            f'text-anchor="middle" font-weight="bold" font-family="sans-serif">{slot + 1}</text>'
            f'<text x="{x + cell_w / 2}" y="{y + cell_h * 0.82}" font-size="{min(11, cell_h * 0.11):.0f}" '
            f'fill="#fff" text-anchor="middle" font-family="sans-serif">{name}</text>'
        )  # fmt: skip
    parts.append("</svg>")
    return "".join(parts) + f"<p style='color:gray;font-size:0.85em'>{columns}×{rows} の配置（番号は映像の順番）</p>"


def preview_area(videos: list[Path]) -> None:
    if st.button("プレビュー（実際の1フレーム）", disabled=not videos):
        try:
            st.session_state["compose_preview"] = render_preview(videos)
        except SyncError as error:
            st.error(str(error))
    preview = st.session_state.get("compose_preview")
    if preview:
        image, at = preview
        st.image(image, caption=f"{at / 60:.1f} 分の時点", width="stretch")
        st.button("プレビューを閉じる", on_click=lambda: st.session_state.pop("compose_preview", None))


def render_preview(videos: list[Path]) -> tuple[bytes, float]:
    missing = [str(p) for p in videos if not p.is_file()]
    if missing:
        raise SyncError("見つからないファイルがあります: " + ", ".join(missing))
    ffmpeg = check_binary(find_tool("ffmpeg"), "ffmpeg")
    ffprobe = check_binary(default_ffprobe(ffmpeg), "ffprobe")
    with st.spinner("プレビューを作成中…"):
        sources = [s for s in (compose_media.probe(p, ffprobe) for p in videos) if s.video_size]
        if not sources:
            raise SyncError("映像トラックのあるファイルがありません。")
        at = max(s.duration for s in sources) / 2
        handle, name = tempfile.mkstemp(prefix="compose_preview_", suffix=".png")
        os.close(handle)
        image = Path(name)
        try:
            command = compose_media.build_preview_command(
                sources, image, ffmpeg=ffmpeg, layout=LAYOUTS[st.session_state["compose_layout"]],
                cell=_cell(), at=at, max_width=1600,
            )  # fmt: skip
            compose_media.run_checked(command, echo=False)
            return image.read_bytes(), at
        finally:
            image.unlink(missing_ok=True)


def start_compose() -> None:
    videos = parse_paths(st.session_state.get("compose_videos", ""))
    audios = parse_paths(st.session_state.get("compose_audios", ""))
    output = st.session_state.get("compose_output", "").strip()
    missing = [str(p) for p in videos + audios if not p.is_file()]
    if not videos and not audios:
        st.error("映像か音声を指定してください。")
        return
    if missing:
        st.error("見つからないファイルがあります:\n\n" + "\n".join(f"- `{m}`" for m in missing))
        return
    if not output:
        st.error("出力ファイルを指定してください。")
        return
    try:
        _cell()
    except SyncError as error:
        st.error(str(error))
        return
    arguments = [
        "--layout", LAYOUTS[st.session_state["compose_layout"]],
        "--audio-mode", AUDIO_MODES[st.session_state["compose_audio_mode"]],
        "--codec", CODECS[st.session_state["compose_codec"]],
        "--encoder", ENCODER_MODES[st.session_state["compose_encoder"]],
        "--cell-size", st.session_state.get("compose_cell", "").strip() or "source",
        "--video-bitrate", st.session_state.get("compose_bitrate", "").strip() or "source",
        "--no-preview", "-o", output,
    ]  # fmt: skip
    if videos:
        arguments += ["-v", *map(str, videos)]
    if audios:
        arguments += ["-a", *map(str, audios)]
    job = jobs.start_job("compose", arguments)
    st.session_state["compose_job"] = job.id
    st.rerun()


@st.fragment(run_every=3)
def compose_status() -> None:
    job = job_selector("compose")
    if job is None:
        return
    fraction, status = jobs.compose_progress(job.log_lines)
    if job.state == "done":
        fraction, status = 1.0, "完了しました"
    st.progress(fraction, text=status)
    job_controls(job)
    show_log(job)
    output = output_flag(job.arguments, "-o")
    if job.state == "done" and output and not running_remotely():
        st.button("出力を開く", key=f"open_{job.id}", on_click=open_in_default_app, args=(output,))


# ---------------------------------------------------------------------------


def initialize_state() -> None:
    defaults = {
        "sync_inputs": "", "sync_output": "", "sync_encoder": "auto", "sync_bitrate": "",
        "sync_extra": "", "compose_videos": "", "compose_audios": "", "compose_output": "",
        "compose_layout": "横一列", "compose_audio_mode": "1chにミックス", "compose_codec": "入力のまま",
        "compose_encoder": "自動", "compose_cell": "", "compose_bitrate": "",
    }  # fmt: skip
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def main() -> None:
    st.set_page_config(page_title="mic-array-synchronizer", layout="wide")
    initialize_state()
    if "toast" in st.session_state:
        st.toast(st.session_state.pop("toast"))
    header, quit_column = st.columns([6, 1])
    with header:
        st.title("会話収録メディアの同期・合成")
        st.caption("mic-array-recorder の同期音と会話音声で、カメラ映像とマイク音声を揃えて切り出し、並べて1本にまとめます。")
    with quit_column:
        if st.button("アプリを終了", help="サーバーを止めます。実行中の処理は止まりません。"):
            st.info("終了しました。このタブを閉じてください。")
            os.kill(os.getpid(), signal.SIGTERM)
    sync, compose = st.tabs(["同期", "合成"])
    with sync:
        sync_tab()
    with compose:
        compose_tab()


main()
