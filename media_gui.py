#!/usr/bin/env python3
"""GUI for sync_media.py and compose_media.py.

Two tabs: 「同期」 runs sync_media.py and shows report.json as a table;
「合成」 builds the compose_media.py command, draws the layout as files are
added, and shows a one-frame preview before encoding.  Both scripts run as
child processes, so the command line and the GUI behave identically.

    python3 media_gui.py
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import compose_media
import sync_media
from sync_media import (
    PROGRESS_ENV,
    REPORT_NAME,
    SyncError,
    check_binary,
    default_ffprobe,
    open_in_default_app,
)

HERE = Path(__file__).resolve().parent
WINDOW_TITLE = "メディア同期・合成"

# Same palette as mic-array-recorder.
BG_COLOR = "#12161c"
CARD_COLOR = "#1b212b"
BORDER_COLOR = "#2a3341"
TEXT_COLOR = "#f2f4f7"
MUTED_COLOR = "#8a94a3"
ERROR_COLOR = "#f87171"
WARNING_COLOR = "#fbbf24"
ACCENT_COLOR = "#3b82f6"
ACCENT_HOVER = "#2563eb"
TILE_COLORS = ["#3B6FB6", "#E08A2E", "#3E9B5B", "#8A5BB0", "#C9483F", "#2E9BA6"]

# Font families per OS; Tk falls back to its default for missing ones.
if sys.platform == "darwin":
    UI_FAMILY, MONO_FAMILY = "Helvetica", "Menlo"
elif sys.platform.startswith("win"):
    UI_FAMILY, MONO_FAMILY = "Yu Gothic UI", "Consolas"
else:
    UI_FAMILY, MONO_FAMILY = "Noto Sans CJK JP", "DejaVu Sans Mono"


def ui_font(size: int, bold: bool = False) -> tuple:
    return (UI_FAMILY, size, "bold") if bold else (UI_FAMILY, size)


def mono_font(size: int) -> tuple:
    return (MONO_FAMILY, size)

MEDIA_TYPES = [
    ("メディア", "*.mp4 *.MP4 *.mov *.MOV *.m4v *.mkv *.mts *.MTS *.wav *.WAV *.mp3 *.m4a *.flac"),
    ("すべて", "*"),
]
VIDEO_ENCODERS = ["auto", *sync_media.VIDEO_ENCODERS]
CODECS = {"入力のまま": "source", "HEVC": "hevc", "H.264": "h264"}
ENCODER_MODES = {"自動": "auto", "ハードウェア": "hardware", "CPU": "software"}
LAYOUTS = {"横一列": "row", "縦一列": "column", "なるべく正方形": "grid"}
AUDIO_MODES = {"1chにミックス": "mix", "全チャンネル": "multi"}


# ---------------------------------------------------------------------------
# Running the scripts


class ScriptRunner:
    """Run a script in a child process and hand its output lines to the UI thread."""

    def __init__(self, widget: tk.Misc, on_line, on_done) -> None:
        self.widget = widget
        self.on_line = on_line
        self.on_done = on_done
        self.process: subprocess.Popen | None = None
        self.stopped = False
        self.lines: queue.Queue[str | None] = queue.Queue()

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, arguments: list[str]) -> None:
        environment = dict(
            os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", **{PROGRESS_ENV: "1"}
        )
        # A separate process group, so that stop() also ends ffmpeg.
        group = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
            if os.name == "nt"
            else {"start_new_session": True}
        )
        self.stopped = False
        self.process = subprocess.Popen(
            [sys.executable, *arguments],
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            **group,
        )
        threading.Thread(target=self._read, daemon=True).start()
        self.widget.after(100, self._drain)

    def stop(self) -> None:
        if not self.running:
            return
        assert self.process is not None
        self.stopped = True
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                           capture_output=True, check=False)  # fmt: skip
        else:
            os.killpg(self.process.pid, signal.SIGTERM)

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def _drain(self) -> None:
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    assert self.process is not None
                    self.on_done(self.process.wait())
                    return
                try:
                    self.on_line(line)
                except Exception as error:  # a parsing slip must not stop the drain
                    print(f"GUI: {error!r} while handling {line!r}", file=sys.stderr)
        except queue.Empty:
            self.widget.after(100, self._drain)


# ---------------------------------------------------------------------------
# Widgets


def card(parent: tk.Misc, title: str) -> tk.Frame:
    outer = tk.Frame(parent, bg=CARD_COLOR, highlightthickness=1, highlightbackground=BORDER_COLOR)
    tk.Label(outer, text=title, bg=CARD_COLOR, fg=TEXT_COLOR, font=ui_font(14, True),
             anchor="w").pack(fill="x", padx=12, pady=(10, 4))  # fmt: skip
    body = tk.Frame(outer, bg=CARD_COLOR)
    body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    outer.body = body  # type: ignore[attr-defined]
    return outer


def label(parent: tk.Misc, text: str, muted: bool = False, **options) -> tk.Label:
    return tk.Label(parent, text=text, bg=CARD_COLOR, fg=MUTED_COLOR if muted else TEXT_COLOR,
                    anchor="w", justify="left", **options)  # fmt: skip


class FileList(tk.Frame):
    """Ordered list of files with add / remove / move buttons."""

    def __init__(self, parent: tk.Misc, on_change=None, height: int = 6) -> None:
        super().__init__(parent, bg=CARD_COLOR)
        self.on_change = on_change
        self.paths: list[Path] = []
        self.listbox = tk.Listbox(
            self, height=height, selectmode="extended", activestyle="none",
            bg=BG_COLOR, fg=TEXT_COLOR, selectbackground=ACCENT_COLOR,
            highlightthickness=1, highlightbackground=BORDER_COLOR, relief="flat",
            font=mono_font(12),
        )  # fmt: skip
        self.listbox.pack(side="left", fill="both", expand=True)
        buttons = tk.Frame(self, bg=CARD_COLOR)
        buttons.pack(side="left", fill="y", padx=(8, 0))
        for text, command in [("追加…", self.add), ("削除", self.remove),
                              ("↑", lambda: self.move(-1)), ("↓", lambda: self.move(1))]:  # fmt: skip
            ttk.Button(buttons, text=text, command=command, width=6).pack(pady=2)

    def set(self, paths: list[Path]) -> None:
        self.paths = list(paths)
        self._refresh()

    def add(self) -> None:
        chosen = filedialog.askopenfilenames(parent=self, filetypes=MEDIA_TYPES)
        self.paths += [Path(p) for p in chosen if Path(p) not in self.paths]
        self._refresh()

    def remove(self) -> None:
        for index in sorted(self.listbox.curselection(), reverse=True):
            del self.paths[index]
        self._refresh()

    def move(self, step: int) -> None:
        selection = list(self.listbox.curselection())
        if len(selection) != 1:
            return
        index = selection[0]
        target = index + step
        if 0 <= target < len(self.paths):
            self.paths[index], self.paths[target] = self.paths[target], self.paths[index]
            self._refresh()
            self.listbox.selection_set(target)

    def _refresh(self) -> None:
        self.listbox.delete(0, "end")
        for number, path in enumerate(self.paths, start=1):
            self.listbox.insert("end", f"{number:>2}. {path.name}    ({path.parent})")
        if self.on_change:
            self.on_change()


class PathField(tk.Frame):
    def __init__(self, parent: tk.Misc, *, directory: bool, save_types=None) -> None:
        super().__init__(parent, bg=CARD_COLOR)
        self.directory = directory
        self.save_types = save_types
        self.variable = tk.StringVar()
        ttk.Entry(self, textvariable=self.variable).pack(side="left", fill="x", expand=True)
        ttk.Button(self, text="参照…", command=self.browse, width=6).pack(side="left", padx=(8, 0))

    def browse(self) -> None:
        if self.directory:
            chosen = filedialog.askdirectory(parent=self)
        else:
            chosen = filedialog.asksaveasfilename(
                parent=self, defaultextension=".mp4", filetypes=self.save_types
            )
        if chosen:
            self.variable.set(chosen)

    def get(self) -> str:
        return self.variable.get().strip()


class ProgressPanel(tk.Frame):
    """Progress bar, status line and a collapsible log."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, bg=CARD_COLOR)
        self.status = label(self, "待機中", muted=True)
        self.status.pack(fill="x")
        self.bar = ttk.Progressbar(self, maximum=1000, style="Accent.Horizontal.TProgressbar")
        self.bar.pack(fill="x", pady=6)
        self.log = tk.Text(self, height=8, bg=BG_COLOR, fg=MUTED_COLOR, relief="flat",
                           font=mono_font(11), highlightthickness=0, wrap="none")  # fmt: skip
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("error", foreground=ERROR_COLOR)
        self.log.tag_configure("warning", foreground=WARNING_COLOR)

    def reset(self) -> None:
        self.log.delete("1.0", "end")
        self.set(0.0, "開始しています…")

    def set(self, fraction: float, status: str | None = None) -> None:
        self.bar["value"] = max(0.0, min(1.0, fraction)) * 1000
        if status is not None:
            self.status.configure(text=status, fg=TEXT_COLOR)

    def append(self, line: str) -> None:
        tag = ""
        if line.startswith(("エラー", "Traceback")):
            tag = "error"
        elif line.startswith(("警告", "注意")):
            tag = "warning"
        self.log.insert("end", line + "\n", tag)
        self.log.see("end")


# ---------------------------------------------------------------------------
# 同期 tab


class SyncTab(tk.Frame):
    def __init__(self, parent: tk.Misc, app: "App") -> None:
        super().__init__(parent, bg=BG_COLOR)
        self.app = app
        self.runner = ScriptRunner(self, self._on_line, self._on_done)
        self.report: dict | None = None

        left = tk.Frame(self, bg=BG_COLOR)
        left.pack(side="left", fill="both", expand=True, padx=(16, 8), pady=16)
        right = tk.Frame(self, bg=BG_COLOR)
        right.pack(side="left", fill="both", expand=True, padx=(8, 16), pady=16)

        inputs = card(left, "入力ファイル")
        inputs.pack(fill="both", expand=True)
        label(inputs.body, "同じ会話のカメラ映像とマイク音声をすべて追加します（順番は sync_check.wav のチャンネル順になります）。",
              muted=True, wraplength=520).pack(fill="x", pady=(0, 6))  # fmt: skip
        self.files = FileList(inputs.body, height=9)
        self.files.pack(fill="both", expand=True)

        output = card(left, "出力フォルダ")
        output.pack(fill="x", pady=(12, 0))
        self.output = PathField(output.body, directory=True)
        self.output.pack(fill="x")

        options = card(left, "オプション")
        options.pack(fill="x", pady=(12, 0))
        grid = options.body
        self.analyze_only = tk.BooleanVar(value=False)
        self.fast_copy = tk.BooleanVar(value=False)
        self.encoder = tk.StringVar(value=VIDEO_ENCODERS[0])
        self.bitrate = tk.StringVar()
        self.pre_roll = tk.StringVar(value="0.05")
        self.extra = tk.StringVar()
        ttk.Checkbutton(grid, text="解析だけ行う（report.jsonのみ作成）",
                        variable=self.analyze_only).grid(row=0, column=0, columnspan=2, sticky="w")  # fmt: skip
        ttk.Checkbutton(grid, text="映像を再エンコードしない（速いがキーフレーム単位）",
                        variable=self.fast_copy).grid(row=1, column=0, columnspan=2, sticky="w")  # fmt: skip
        rows = [
            ("映像エンコーダ", ttk.Combobox(grid, textvariable=self.encoder, values=VIDEO_ENCODERS,
                                         state="readonly", width=20)),
            ("ビットレート（空欄=標準）", ttk.Entry(grid, textvariable=self.bitrate, width=10)),
            ("同期音の何秒前から残すか", ttk.Entry(grid, textvariable=self.pre_roll, width=10)),
            ("その他の引数", ttk.Entry(grid, textvariable=self.extra, width=30)),
        ]  # fmt: skip
        for row, (text, widget) in enumerate(rows, start=2):
            label(grid, text).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 12))
            widget.grid(row=row, column=1, sticky="w", pady=3)

        run = card(right, "実行")
        run.pack(fill="both", expand=True)
        buttons = tk.Frame(run.body, bg=CARD_COLOR)
        buttons.pack(fill="x")
        self.start_button = ttk.Button(buttons, text="同期を開始", style="Accent.TButton",
                                       command=self.start)  # fmt: skip
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="中止", command=self.runner.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.open_button = ttk.Button(buttons, text="出力フォルダを開く", command=self.open_output,
                                      state="disabled")  # fmt: skip
        self.open_button.pack(side="right")
        self.compose_button = ttk.Button(buttons, text="合成タブへ送る →", command=self.send_to_compose,
                                         state="disabled")  # fmt: skip
        self.compose_button.pack(side="right", padx=8)
        self.progress = ProgressPanel(run.body)
        self.progress.pack(fill="both", expand=True, pady=(10, 0))

        result = card(right, "結果（report.json）")
        result.pack(fill="both", expand=True, pady=(12, 0))
        columns = ("file", "confidence", "correction", "drift", "pairs", "timecode")
        self.table = ttk.Treeview(result.body, columns=columns, show="headings", height=8)
        for column, text, width in [("file", "ファイル", 220), ("confidence", "検出", 110),
                                    ("correction", "相関補正 ms", 90), ("drift", "ドリフト ppm", 90),
                                    ("pairs", "ペア", 50), ("timecode", "タイムコード", 100)]:  # fmt: skip
            self.table.heading(column, text=text)
            self.table.column(column, width=width, anchor="w" if column == "file" else "e")
        self.table.tag_configure("warn", foreground=WARNING_COLOR)
        self.table.pack(fill="both", expand=True)
        self.summary = label(result.body, "", muted=True, wraplength=560)
        self.summary.pack(fill="x", pady=(6, 0))

    # --- running ---------------------------------------------------------

    def start(self) -> None:
        if not self.files.paths:
            messagebox.showwarning(WINDOW_TITLE, "入力ファイルを追加してください。", parent=self)
            return
        if not self.output.get():
            messagebox.showwarning(WINDOW_TITLE, "出力フォルダを指定してください。", parent=self)
            return
        arguments = ["sync_media.py", "-o", self.output.get(), *map(str, self.files.paths)]
        if self.analyze_only.get():
            arguments.append("--analyze-only")
        if self.fast_copy.get():
            arguments.append("--fast-copy")
        arguments += ["--video-encoder", self.encoder.get(), "--pre-roll", self.pre_roll.get() or "0.05"]
        if self.bitrate.get().strip():
            arguments += ["--video-bitrate", self.bitrate.get().strip()]
        try:
            arguments += shlex.split(self.extra.get())
        except ValueError as error:
            messagebox.showerror(WINDOW_TITLE, f"その他の引数を解釈できません: {error}", parent=self)
            return

        self.report = None
        self.table.delete(*self.table.get_children())
        self.summary.configure(text="")
        self.progress.reset()
        self.state = {"files": len(self.files.paths), "searched": 0, "cut": 0, "duration": None}
        self._set_running(True)
        self.runner.start(arguments)

    def _set_running(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.app.set_busy(running)

    def _on_line(self, line: str) -> None:
        # Phases: search 0-10 %, correlation 10-30 %, cutting + check WAV 30-100 %.
        state = self.state
        files = state["files"]
        if line.startswith("PROGRESS ") and state["duration"]:
            seconds = float(line.split()[1])
            done = (state["cut"] - 1 + min(1.0, seconds / state["duration"])) / (files + 1)
            self.progress.set(0.30 + 0.70 * done)
            return
        self.progress.append(line)
        if line.startswith("同期音を探索中"):
            state["searched"] += 1
            self.progress.set(0.10 * state["searched"] / files, "同期音を探しています…")
        elif line.strip().startswith("相関区間"):
            match = re.search(r"(\d+)/(\d+)", line)
            if match:
                step, total = (int(v) for v in match.groups())
                self.progress.set(0.10 + 0.20 * step / total, f"相互相関を計算中（{step}/{total}）")
        elif line.startswith("共通区間"):
            state["duration"] = float(line.split(":")[1].strip().rstrip("秒"))
            self._load_report()
        elif line.startswith("切り出し中"):
            state["cut"] += 1
            name = line.split(":", 1)[1].split("->")[0].strip()
            self.progress.set(0.30 + 0.70 * (state["cut"] - 1) / (files + 1),
                              f"切り出し中（{state['cut']}/{files}）: {name}")  # fmt: skip
        elif line.startswith("確認用WAV"):
            state["cut"] = files + 1
            self.progress.set(0.30 + 0.70 * files / (files + 1), "確認用WAVを作成中…")

    def _on_done(self, code: int) -> None:
        self._set_running(False)
        self._load_report()
        self.open_button.configure(state="normal")
        if code == 0:
            self.progress.set(1.0, "完了しました")
            if self.report and not self.analyze_only.get():
                self.compose_button.configure(state="normal")
        elif self.runner.stopped:
            self.progress.set(0.0, "中止しました")
        else:
            self.progress.set(0.0, "エラーで終了しました（ログを確認してください）")
            self.progress.status.configure(fg=ERROR_COLOR)

    def _load_report(self) -> None:
        path = Path(self.output.get()) / REPORT_NAME
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.report = report
        self.table.delete(*self.table.get_children())
        warnings = 0
        for item in report["inputs"]:
            warn = item["correlation_pairs_used"] == 0 or item["confidence"] in {"low", "timecode-only"}
            warnings += warn
            self.table.insert("", "end", tags=("warn",) if warn else (), values=(
                Path(item["input"]).name, item["confidence"],
                f"{item['correlation_correction_ms']:+.2f}", f"{item['drift_ppm']:+.1f}",
                item["correlation_pairs_used"], item["timecode_status"],
            ))  # fmt: skip
        drifts = [item["drift_ppm"] for item in report["inputs"]]
        duration = report.get("common_duration_seconds") or 0
        spread = (max(drifts) - min(drifts)) * 1e-6 * duration * 1000
        text = f"共通区間 {duration / 60:.1f} 分。ドリフトは補正しないため、終端では最大 約{spread:.0f} ms ずれます。"
        if warnings:
            text += f"\n黄色の {warnings} ファイルは同期結果を sync_check.wav で確認してください。"
        self.summary.configure(text=text)

    # --- after the run ---------------------------------------------------

    def open_output(self) -> None:
        open_in_default_app(self.output.get())

    def send_to_compose(self) -> None:
        if not self.report:
            return
        videos, audios = [], []
        for item in self.report["inputs"]:
            output = Path(item["output"])
            (videos if item["has_video"] else audios).append(output)
        self.app.compose.load(videos, audios, Path(self.output.get()) / "composite.mp4")
        self.app.show_compose()


# ---------------------------------------------------------------------------
# 合成 tab


class ComposeTab(tk.Frame):
    def __init__(self, parent: tk.Misc, app: "App") -> None:
        super().__init__(parent, bg=BG_COLOR)
        self.app = app
        self.runner = ScriptRunner(self, self._on_line, self._on_done)
        self.duration: float | None = None

        left = tk.Frame(self, bg=BG_COLOR)
        left.pack(side="left", fill="both", expand=True, padx=(16, 8), pady=16)
        right = tk.Frame(self, bg=BG_COLOR)
        right.pack(side="left", fill="both", expand=True, padx=(8, 16), pady=16)

        video = card(left, "映像（左上から順に配置）")
        video.pack(fill="both", expand=True)
        self.videos = FileList(video.body, on_change=self.draw_layout, height=5)
        self.videos.pack(fill="both", expand=True)
        self.layout = tk.StringVar(value="横一列")
        layout_row = tk.Frame(video.body, bg=CARD_COLOR)
        layout_row.pack(fill="x", pady=(8, 0))
        label(layout_row, "配置").pack(side="left", padx=(0, 12))
        for text in LAYOUTS:
            ttk.Radiobutton(layout_row, text=text, value=text, variable=self.layout,
                            command=self.draw_layout).pack(side="left", padx=4)  # fmt: skip

        audio = card(left, "音声")
        audio.pack(fill="both", expand=True, pady=(12, 0))
        self.audios = FileList(audio.body, height=4)
        self.audios.pack(fill="both", expand=True)
        self.audio_mode = tk.StringVar(value="1chにミックス")
        mode_row = tk.Frame(audio.body, bg=CARD_COLOR)
        mode_row.pack(fill="x", pady=(8, 0))
        label(mode_row, "音声の形").pack(side="left", padx=(0, 12))
        for text in AUDIO_MODES:
            ttk.Radiobutton(mode_row, text=text, value=text, variable=self.audio_mode).pack(side="left", padx=4)

        encode = card(left, "映像の形式・大きさ・ビットレート")
        encode.pack(fill="x", pady=(12, 0))
        grid = encode.body
        self.codec = tk.StringVar(value="入力のまま")
        self.cell_size = tk.StringVar(value="入力のまま")
        self.bitrate = tk.StringVar(value="入力のまま")
        self.encoder = tk.StringVar(value="自動")
        rows = [
            ("形式", ttk.Combobox(grid, textvariable=self.codec, values=list(CODECS), state="readonly", width=14)),
            ("1画面の大きさ（例 640x360）", ttk.Combobox(grid, textvariable=self.cell_size,
                                                     values=["入力のまま", "1280x720", "640x360"], width=14)),
            ("ビットレート（例 8M）", ttk.Combobox(grid, textvariable=self.bitrate,
                                               values=["入力のまま", "8M", "4M", "2M"], width=14)),
            ("エンコーダ", ttk.Combobox(grid, textvariable=self.encoder, values=list(ENCODER_MODES),
                                      state="readonly", width=14)),
        ]  # fmt: skip
        for row, (text, widget) in enumerate(rows):
            label(grid, text).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 12))
            widget.grid(row=row, column=1, sticky="w", pady=3)
        self.cell_size_box = rows[1][1]
        self.cell_size_box.bind("<<ComboboxSelected>>", lambda _: self.draw_layout())
        self.cell_size_box.bind("<KeyRelease>", lambda _: self.draw_layout())

        output = card(left, "出力ファイル")
        output.pack(fill="x", pady=(12, 0))
        self.output = PathField(output.body, directory=False,
                                save_types=[("MP4", "*.mp4"), ("MOV", "*.mov"), ("MKV", "*.mkv"), ("WAV", "*.wav")])  # fmt: skip
        self.output.pack(fill="x")

        layout_card = card(right, "配置イメージ")
        layout_card.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(layout_card.body, bg=BG_COLOR, highlightthickness=0, height=220)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self.draw_layout())
        self.size_label = label(layout_card.body, "", muted=True)
        self.size_label.pack(fill="x", pady=(6, 0))

        run = card(right, "実行")
        run.pack(fill="both", expand=True, pady=(12, 0))
        buttons = tk.Frame(run.body, bg=CARD_COLOR)
        buttons.pack(fill="x")
        self.preview_button = ttk.Button(buttons, text="プレビュー", command=self.preview)
        self.preview_button.pack(side="left")
        self.start_button = ttk.Button(buttons, text="合成を開始", style="Accent.TButton", command=self.start)
        self.start_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(buttons, text="中止", command=self.runner.stop, state="disabled")
        self.stop_button.pack(side="left")
        self.open_button = ttk.Button(buttons, text="出力を開く", command=self.open_output, state="disabled")
        self.open_button.pack(side="right")
        self.progress = ProgressPanel(run.body)
        self.progress.pack(fill="both", expand=True, pady=(10, 0))
        self.draw_layout()

    def load(self, videos: list[Path], audios: list[Path], output: Path) -> None:
        self.videos.set(videos)
        self.audios.set(audios)
        self.output.variable.set(str(output))

    # --- layout drawing --------------------------------------------------

    def _cell_aspect(self) -> float:
        text = self.cell_size.get().lower()
        if "x" in text:
            try:
                width, height = (int(v) for v in text.split("x"))
                return width / height
            except ValueError:
                pass
        return 16 / 9

    def draw_layout(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        count = len(self.videos.paths)
        width, height = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
        if count == 0:
            canvas.create_text(width / 2, height / 2, text="映像を追加すると配置が表示されます",
                               fill=MUTED_COLOR, font=ui_font(13))  # fmt: skip
            self.size_label.configure(text="")
            return
        columns, rows = compose_media.grid_shape(count, LAYOUTS[self.layout.get()])
        aspect = self._cell_aspect()
        cell_w = min((width - 20) / columns, (height - 20) / rows * aspect)
        cell_h = cell_w / aspect
        left = (width - cell_w * columns) / 2
        top = (height - cell_h * rows) / 2
        for slot in range(columns * rows):
            x = left + (slot % columns) * cell_w
            y = top + (slot // columns) * cell_h
            if slot >= count:
                canvas.create_rectangle(x, y, x + cell_w, y + cell_h, fill="#000000", outline=BORDER_COLOR)
                continue
            color = TILE_COLORS[slot % len(TILE_COLORS)]
            canvas.create_rectangle(x, y, x + cell_w, y + cell_h, fill=color, outline=BG_COLOR, width=2)
            canvas.create_text(x + cell_w / 2, y + cell_h * 0.42, text=str(slot + 1), fill="white",
                               font=ui_font(max(10, int(cell_h * 0.35)), True))  # fmt: skip
            name = self.videos.paths[slot].name
            canvas.create_text(x + cell_w / 2, y + cell_h * 0.78, text=name, fill="white",
                               width=cell_w - 8, font=ui_font(max(8, int(cell_h * 0.09))))  # fmt: skip
        self.size_label.configure(text=f"{columns}×{rows} の配置（番号は映像リストの順番）")

    # --- preview ---------------------------------------------------------

    def _cell(self) -> tuple[int, int] | None:
        text = self.cell_size.get().strip()
        return None if text in {"", "入力のまま"} else compose_media.parse_size(text)

    def preview(self) -> None:
        if not self.videos.paths:
            messagebox.showwarning(WINDOW_TITLE, "映像を追加してください。", parent=self)
            return
        self.preview_button.configure(state="disabled")
        self.progress.set(0.0, "プレビューを作成中…")
        layout, cell = LAYOUTS[self.layout.get()], self._cell()
        paths = list(self.videos.paths)

        def work() -> None:
            try:
                ffmpeg = check_binary("ffmpeg", "ffmpeg")
                ffprobe = check_binary(default_ffprobe(ffmpeg), "ffprobe")
                videos = [compose_media.probe(path, ffprobe) for path in paths]
                videos = [v for v in videos if v.video_size is not None]
                if not videos:
                    raise SyncError("映像トラックのあるファイルがありません。")
                at = max(v.duration for v in videos) / 2
                handle, name = tempfile.mkstemp(prefix="compose_preview_", suffix=".png")
                os.close(handle)
                command = compose_media.build_preview_command(
                    videos, Path(name), ffmpeg=ffmpeg, layout=layout, cell=cell, at=at, max_width=1400
                )
                compose_media.run_checked(command, echo=False)
                self.after(0, lambda: self._show_preview(Path(name), at))
            except (SyncError, OSError) as error:
                message = str(error)
                self.after(0, lambda: self._preview_failed(message))

        threading.Thread(target=work, daemon=True).start()

    def _preview_failed(self, message: str) -> None:
        self.preview_button.configure(state="normal")
        self.progress.set(0.0, "プレビューを作成できませんでした")
        messagebox.showerror(WINDOW_TITLE, message, parent=self)

    def _show_preview(self, image_path: Path, at: float) -> None:
        self.preview_button.configure(state="normal")
        self.progress.set(0.0, "待機中")
        try:
            image = tk.PhotoImage(file=str(image_path))
        finally:
            image_path.unlink(missing_ok=True)
        window = tk.Toplevel(self, bg=BG_COLOR)
        window.title(f"プレビュー（{at / 60:.1f} 分の時点）")
        screen = self.winfo_screenwidth() - 80
        factor = max(1, math.ceil(image.width() / screen))
        shown = image.subsample(factor) if factor > 1 else image
        holder = tk.Label(window, image=shown, bg=BG_COLOR)
        holder.image = shown  # type: ignore[attr-defined]  # keep a reference
        holder.pack(padx=12, pady=12)
        bar = tk.Frame(window, bg=BG_COLOR)
        bar.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(bar, text="閉じる", command=window.destroy).pack(side="right")
        ttk.Button(bar, text="この配置で合成を開始", style="Accent.TButton",
                   command=lambda: (window.destroy(), self.start())).pack(side="right", padx=8)  # fmt: skip

    # --- running ---------------------------------------------------------

    def start(self) -> None:
        if not self.videos.paths and not self.audios.paths:
            messagebox.showwarning(WINDOW_TITLE, "映像か音声を追加してください。", parent=self)
            return
        if not self.output.get():
            messagebox.showwarning(WINDOW_TITLE, "出力ファイルを指定してください。", parent=self)
            return
        if self.runner.running:
            return
        arguments = ["compose_media.py", "--layout", LAYOUTS[self.layout.get()],
                     "--audio-mode", AUDIO_MODES[self.audio_mode.get()],
                     "--codec", CODECS[self.codec.get()],
                     "--encoder", ENCODER_MODES[self.encoder.get()], "--no-preview",
                     "-o", self.output.get()]  # fmt: skip
        cell = self.cell_size.get().strip()
        arguments += ["--cell-size", "source" if cell in {"", "入力のまま"} else cell]
        bitrate = self.bitrate.get().strip()
        arguments += ["--video-bitrate", "source" if bitrate in {"", "入力のまま"} else bitrate]
        if self.videos.paths:
            arguments += ["-v", *map(str, self.videos.paths)]
        if self.audios.paths:
            arguments += ["-a", *map(str, self.audios.paths)]
        if Path(self.output.get()).exists() and not messagebox.askyesno(
            WINDOW_TITLE, "出力ファイルはすでにあります。上書きしますか？", parent=self
        ):
            return
        self.duration = None
        self.progress.reset()
        self._set_running(True)
        self.runner.start(arguments)

    def _set_running(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.preview_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.app.set_busy(running)

    def _on_line(self, line: str) -> None:
        if line.startswith("PROGRESS "):
            if self.duration:
                seconds = float(line.split()[1])
                self.progress.set(seconds / self.duration,
                                  f"合成中… {seconds / 60:.1f} / {self.duration / 60:.1f} 分")  # fmt: skip
            return
        self.progress.append(line)
        if line.startswith("長さ"):
            self.duration = float(line.split(":")[1].strip().rstrip("秒"))

    def _on_done(self, code: int) -> None:
        self._set_running(False)
        if code == 0:
            self.progress.set(1.0, "完了しました")
            self.open_button.configure(state="normal")
        elif self.runner.stopped:
            self.progress.set(0.0, "中止しました")
        else:
            self.progress.set(0.0, "エラーで終了しました（ログを確認してください）")
            self.progress.status.configure(fg=ERROR_COLOR)

    def open_output(self) -> None:
        open_in_default_app(self.output.get())


# ---------------------------------------------------------------------------
# Application


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(WINDOW_TITLE)
        self.configure(bg=BG_COLOR)
        self.geometry("1280x860")
        self.minsize(1080, 720)
        self._setup_style()
        self.busy = False
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.sync = SyncTab(self.notebook, self)
        self.compose = ComposeTab(self.notebook, self)
        self.notebook.add(self.sync, text="  同期  ")
        self.notebook.add(self.compose, text="  合成  ")
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=CARD_COLOR, foreground=TEXT_COLOR, fieldbackground=BG_COLOR,
                        bordercolor=BORDER_COLOR, lightcolor=CARD_COLOR, darkcolor=CARD_COLOR,
                        insertcolor=TEXT_COLOR, font=ui_font(13))  # fmt: skip
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=CARD_COLOR, foreground=MUTED_COLOR, padding=(16, 8))
        style.map("TNotebook.Tab", background=[("selected", BG_COLOR)], foreground=[("selected", TEXT_COLOR)])
        style.configure("TButton", background=BORDER_COLOR, padding=(10, 5))
        style.map("TButton", background=[("active", "#364152"), ("disabled", CARD_COLOR)],
                  foreground=[("disabled", MUTED_COLOR)])  # fmt: skip
        style.configure("Accent.TButton", background=ACCENT_COLOR, foreground="white")
        style.map("Accent.TButton", background=[("active", ACCENT_HOVER), ("disabled", CARD_COLOR)])
        style.configure("TCheckbutton", background=CARD_COLOR)
        style.configure("TRadiobutton", background=CARD_COLOR)
        style.map("TCheckbutton", background=[("active", CARD_COLOR)])
        style.map("TRadiobutton", background=[("active", CARD_COLOR)])
        style.configure("TEntry", fieldbackground=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TCombobox", fieldbackground=BG_COLOR, background=BORDER_COLOR,
                        foreground=TEXT_COLOR, arrowcolor=TEXT_COLOR)  # fmt: skip
        style.map("TCombobox", fieldbackground=[("readonly", BG_COLOR)],
                  foreground=[("readonly", TEXT_COLOR)], selectbackground=[("readonly", BG_COLOR)],
                  selectforeground=[("readonly", TEXT_COLOR)])  # fmt: skip
        self.option_add("*TCombobox*Listbox.background", BG_COLOR)
        self.option_add("*TCombobox*Listbox.foreground", TEXT_COLOR)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT_COLOR)
        style.configure("Treeview", background=BG_COLOR, fieldbackground=BG_COLOR, foreground=TEXT_COLOR,
                        rowheight=24, font=ui_font(12))  # fmt: skip
        style.configure("Treeview.Heading", background=BORDER_COLOR, foreground=TEXT_COLOR)
        style.configure("Accent.Horizontal.TProgressbar", background=ACCENT_COLOR, troughcolor=BG_COLOR)

    def set_busy(self, busy: bool) -> None:
        self.busy = self.sync.runner.running or self.compose.runner.running or busy

    def show_compose(self) -> None:
        self.notebook.select(self.compose)

    def close(self) -> None:
        running = [tab for tab in (self.sync, self.compose) if tab.runner.running]
        if running and not messagebox.askyesno(
            WINDOW_TITLE, "処理を実行中です。中止して終了しますか？", parent=self
        ):
            return
        for tab in running:
            tab.runner.stop()
        self.destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
