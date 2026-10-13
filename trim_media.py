#!/usr/bin/env python3
"""Cut the same time ranges out of several media files.

Meant for files already synchronized by ``sync_media.py``: every range is
taken at the same time in every input.  Outputs are MP4 for video and WAV
for audio-only inputs, named ``<stem>_<start>-<end>`` (or ``<stem>_<label>``).
Video is re-encoded so that cuts are frame-accurate (``--fast-copy`` cuts at
keyframes instead); audio is cut to the sample.

Run ``python trim_media.py --help`` for usage.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import compose_media
from sync_media import (
    SyncError,
    check_binary,
    default_ffprobe,
    eprint,
    find_tool,
    output_suffix,
    probe_media,
    run_checked,
    unique_stems,
    wav_codec,
)


@dataclass
class TimeRange:
    start: float
    end: float
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or f"{format_time(self.start)}-{format_time(self.end)}"


def parse_time(text: str) -> float:
    """``600.5``, ``10:00`` or ``1:02:03.5`` -> seconds."""
    text = text.strip()
    if not re.fullmatch(r"\d+(\.\d+)?|(\d+:){1,2}\d+(\.\d+)?", text):
        raise SyncError(f"時刻は秒（600.5）か [時:]分:秒（10:00、1:02:03.5）で指定してください: {text}")
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def parse_range(text: str) -> TimeRange:
    """``START-END``, ``START+DURATION``, optionally followed by ``=LABEL``."""
    body, _, label = text.partition("=")
    if label and not re.fullmatch(r"[\w.-]+", label):
        raise SyncError(f"名前には英数字・_・-・. だけを使ってください: {label}")
    match = re.fullmatch(r"\s*([\d:.]+)\s*([-+])\s*([\d:.]+)\s*", body)
    if not match:
        raise SyncError(f"範囲は 開始-終了 か 開始+長さ で指定してください: {text}")
    start = parse_time(match.group(1))
    other = parse_time(match.group(3))
    end = start + other if match.group(2) == "+" else other
    if end <= start:
        raise SyncError(f"終了が開始より後になるように指定してください: {text}")
    return TimeRange(start, end, label or None)


def format_time(seconds: float) -> str:
    """Seconds -> ``HHMMSS`` (plus ``.mmm`` when not a whole second), for file names."""
    millis = round(seconds * 1000)
    whole, fraction = divmod(millis, 1000)
    text = f"{whole // 3600:02d}{whole % 3600 // 60:02d}{whole % 60:02d}"
    return f"{text}.{fraction:03d}" if fraction else text


def build_command(
    source: compose_media.Source,
    media_has_video: bool,
    wav: str,
    span: TimeRange,
    output: Path,
    *,
    ffmpeg: str,
    fast_copy: bool,
    encoder: tuple[str, list[str]] | None,
    bitrate: float | None,
) -> list[str]:
    duration = span.end - span.start
    command = [ffmpeg, "-y", "-v", "error", "-stats"]
    if media_has_video and not fast_copy and sys.platform == "darwin":
        command += ["-hwaccel", "videotoolbox"]
    # Input seeking: exact when re-encoding (and for PCM), keyframe-based with stream copy.
    command += ["-ss", f"{span.start:.6f}", "-t", f"{duration:.6f}", "-i", str(source.path)]
    if media_has_video:
        command += ["-map", "0:v:0", "-map", "0:a?", "-map_metadata", "-1"]
        if fast_copy:
            command += ["-c:v", "copy", "-avoid_negative_ts", "make_zero"]
        else:
            assert encoder is not None
            name, options = encoder
            command += ["-c:v", name, *options, "-pix_fmt", "yuv420p"]
            if bitrate:
                command += ["-b:v", f"{max(100, round(bitrate / 1000))}k"]
        command += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
    else:
        command += ["-map", "0:a", "-c:a", wav, "-rf64", "auto"]
    command.append(str(output))
    return command


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="複数のメディアから、同じ時間の範囲を切り抜きます（同期済みのファイル向け）。"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="切り抜くメディア")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="出力先ディレクトリ")
    parser.add_argument(
        "-r",
        "--range",
        dest="ranges",
        action="append",
        required=True,
        metavar="START-END[=NAME]",
        help="切り抜く範囲。開始-終了 か 開始+長さ。時刻は秒か [時:]分:秒。=NAME で出力名を指定。複数回指定できる",
    )
    parser.add_argument(
        "--fast-copy", action="store_true", help="映像を再エンコードせず切り抜く（速いが開始はキーフレーム単位）"
    )
    parser.add_argument(
        "--codec",
        choices=["source", "hevc", "h264"],
        default="source",
        help="映像の形式。source=入力と同じ（標準）",
    )
    parser.add_argument(
        "--video-bitrate",
        default="source",
        help="映像のビットレート（例: 8M）。source=入力と同じ（標準）",
    )
    parser.add_argument(
        "--encoder",
        choices=["auto", "hardware", "software"],
        default="auto",
        help="auto=使えればハードウェア（Mac/NVIDIA/Intel/AMD）、だめならCPU（標準）",
    )
    parser.add_argument("--ffmpeg", default=None, help="ffmpegのパス（標準は同梱のffmpeg、なければPATH）")
    parser.add_argument("--ffprobe", default=None, help="ffprobeのパス")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    ffmpeg = check_binary(args.ffmpeg or find_tool("ffmpeg"), "ffmpeg")
    ffprobe = check_binary(args.ffprobe or default_ffprobe(ffmpeg), "ffprobe")
    ranges = [parse_range(text) for text in args.ranges]
    names = [r.name for r in ranges]
    if len(set(names)) != len(names):
        raise SyncError("同じ名前の範囲があります。=NAME で別の名前を付けてください。")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [p.expanduser().resolve() for p in args.inputs]
    for path in paths:
        if not path.is_file():
            raise SyncError(f"入力ファイルが見つかりません: {path}")
    stems = unique_stems(paths)
    jobs: list[list[str]] = []
    available = compose_media.available_encoders(ffmpeg)
    for index, path in enumerate(paths):
        media = probe_media(path, index, ffprobe)
        source = compose_media.probe(path, ffprobe)
        stem = stems[index]
        encoder = None
        bitrate = None
        if media.has_video and not args.fast_copy:
            assert media.video_size is not None
            codec = source.video_codec if args.codec == "source" else args.codec
            if codec not in {"hevc", "h264"}:
                codec = "h264"
            encoder = compose_media.choose_encoder(
                codec, args.encoder, *media.video_size, ffmpeg=ffmpeg, available=available
            )
            if args.video_bitrate == "source":
                bitrate = source.video_bitrate
            else:
                bitrate = compose_media.parse_bitrate(args.video_bitrate)
        for span in ranges:
            if span.start >= media.duration:
                raise SyncError(
                    f"範囲 {span.name} の開始がファイルの長さ（{media.duration:.3f}秒）を超えています: {path.name}"
                )
            if span.end > media.duration + 1e-3:
                eprint(
                    f"警告: 範囲 {span.name} がファイルの終わり（{media.duration:.3f}秒）を超えるので、"
                    f"終わりまでを切り抜きます: {path.name}"
                )
            output = output_dir / f"{stem}_{span.name}{output_suffix(media)}"
            if output == path:
                raise SyncError(f"出力が入力を上書きします: {output}")
            jobs.append(build_command(
                source, media.has_video, wav_codec(media), span, output, ffmpeg=ffmpeg,
                fast_copy=args.fast_copy, encoder=encoder, bitrate=bitrate,
            ))  # fmt: skip

    for number, command in enumerate(jobs, start=1):
        eprint(f"切り抜き中 ({number}/{len(jobs)}): {Path(command[-1]).name}")
        run_checked(command, echo=False)
    print(f"完了: {output_dir}（{len(jobs)}ファイル）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SyncError as exc:
        eprint(f"エラー: {exc}")
        raise SystemExit(2)
