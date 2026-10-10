#!/usr/bin/env python3
"""Compose synchronized recordings into one video (or one WAV).

Video inputs are tiled in the given order, starting at the top left, as one
row, one column, or the grid closest to a square.  Audio inputs (which may be
video files) become either one mixed mono channel or all of their channels in
input order.  With no video inputs the output is a WAV file.

Inputs are expected to be already synchronized (for example by
``sync_media.py``); they are not shifted.  The output lasts as long as the
longest input: a shorter video holds its last frame and shorter audio is padded
with silence.

Run ``python compose_media.py --help`` for usage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sync_media import (
    VIDEO_ENCODERS,
    SyncError,
    _parse_rate,
    check_binary,
    default_ffprobe,
    eprint,
    open_in_default_app,
    pick_hardware_encoder,
    run_checked,
)

# CPU encoders per codec.  Hardware ones (VideoToolbox / NVENC / QSV / AMF)
# come from sync_media and are tried first; some stop at 4096 pixels wide.
SOFTWARE_ENCODERS: dict[str, tuple[str, list[str]]] = {
    "hevc": ("libx265", ["-tag:v", "hvc1", "-preset", "medium"]),
    "h264": ("libx264", ["-preset", "medium"]),
}


@dataclass
class Source:
    path: Path
    duration: float
    video_size: tuple[int, int] | None
    fps: float | None
    video_codec: str | None
    video_bitrate: float | None  # bit/s
    audio_channels: list[int]  # channels of each audio stream
    audio_rates: list[int]


def probe(path: Path, ffprobe: str) -> Source:
    result = run_checked(
        [
            ffprobe, "-v", "error",
            "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height,"
            "r_frame_rate,bit_rate,sample_rate,channels:stream_disposition=attached_pic",
            "-of", "json", str(path),
        ],
        capture=True,
        echo=False,
    )  # fmt: skip
    data = json.loads(result.stdout.decode("utf-8", errors="replace"))
    streams = data.get("streams", [])
    video = [
        s
        for s in streams
        if s.get("codec_type") == "video"
        and not s.get("disposition", {}).get("attached_pic", 0)
    ]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or 0.0)
    video_bitrate = None
    if video:
        if video[0].get("bit_rate"):
            video_bitrate = float(video[0]["bit_rate"])
        elif duration > 0 and fmt.get("size"):
            # Container total minus audio, when the stream does not say.
            audio_bits = sum(float(s.get("bit_rate") or 0) for s in audio)
            video_bitrate = float(fmt["size"]) * 8 / duration - audio_bits
    return Source(
        path=path,
        duration=duration,
        video_size=(int(video[0]["width"]), int(video[0]["height"])) if video else None,
        fps=_parse_rate(video[0].get("r_frame_rate")) if video else None,
        video_codec=video[0].get("codec_name") if video else None,
        video_bitrate=video_bitrate,
        audio_channels=[int(s.get("channels") or 0) for s in audio],
        audio_rates=[int(s.get("sample_rate") or 0) for s in audio],
    )


def grid_shape(count: int, layout: str) -> tuple[int, int]:
    """Return ``(columns, rows)``."""
    if layout == "row":
        return count, 1
    if layout == "column":
        return 1, count
    columns = math.ceil(math.sqrt(count))
    return columns, math.ceil(count / columns)


def parse_bitrate(text: str) -> float:
    match = re.fullmatch(r"([0-9.]+)\s*([kKmM]?)", text.strip())
    if not match:
        raise SyncError(f"ビットレートは 8M や 800k の形式で指定してください: {text}")
    value, unit = float(match.group(1)), match.group(2).lower()
    return value * {"": 1, "k": 1e3, "m": 1e6}[unit]


def available_encoders(ffmpeg: str) -> set[str]:
    result = run_checked([ffmpeg, "-hide_banner", "-encoders"], capture=True, echo=False)
    lines = result.stdout.decode("utf-8", errors="replace").splitlines()
    return {line.split()[1] for line in lines if len(line.split()) > 1 and line.startswith(" V")}


def choose_encoder(
    codec: str, mode: str, width: int, height: int, *, ffmpeg: str, available: set[str]
) -> tuple[str, list[str]]:
    """Pick the encoder for ``codec``; ``mode`` is auto, hardware or software."""
    hardware = None
    if mode != "software":
        hardware = pick_hardware_encoder(ffmpeg, codec, width, height)
    if hardware:
        return hardware, VIDEO_ENCODERS[hardware][0]
    if mode != "software":
        too_big = pick_hardware_encoder(ffmpeg, codec, 1280, 720) is not None
        reason = f"{width}x{height}の大きさに対応していない" if too_big else "見つからない"
        if mode == "hardware":
            raise SyncError(
                f"{codec.upper()}のハードウェアエンコーダが{reason}ため、実行できません。"
                "--encoder software を使ってください。"
            )
        eprint(f"注意: {codec.upper()}のハードウェアエンコーダが{reason}ため、CPUでエンコードします（時間がかかります）。")
    name, options = SOFTWARE_ENCODERS[codec]
    if name not in available:
        raise SyncError(f"このffmpegでは {name} を使えません。")
    return name, options


def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise SyncError(f"サイズは WxH の形式で指定してください: {text}") from exc
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise SyncError(f"サイズは正の偶数で指定してください: {text}")
    return width, height


def video_filters(
    videos: list[Source],
    *,
    layout: str,
    cell: tuple[int, int] | None,
    rate: float,
    duration: float,
) -> tuple[list[str], int, int]:
    """Filters tiling inputs ``0..len(videos)-1`` into ``[vout]``."""
    assert videos[0].video_size is not None
    cell_width, cell_height = cell or videos[0].video_size
    columns, rows = grid_shape(len(videos), layout)
    filters: list[str] = []
    labels = []
    for index in range(len(videos)):
        label = f"v{index}"
        filters.append(
            f"[{index}:v:0]fps={rate:.6f},"
            f"scale={cell_width}:{cell_height}:force_original_aspect_ratio=decrease,"
            f"pad={cell_width}:{cell_height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration={duration:.6f},"
            f"trim=duration={duration:.6f},setpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")
    if len(videos) == 1:
        filters[-1] = filters[-1].replace("[v0]", "[vout]")
    else:
        positions = [
            f"{(i % columns) * cell_width}_{(i // columns) * cell_height}"
            for i in range(len(videos))
        ]
        filters.append(
            "".join(labels)
            + f"xstack=inputs={len(videos)}:layout={'|'.join(positions)}:fill=black[vout]"
        )
    eprint(
        f"映像: {columns}×{rows}配置、{columns * cell_width}x{rows * cell_height}、{rate:g} fps"
    )
    return filters, columns * cell_width, rows * cell_height


def build_preview_command(
    videos: list[Source],
    image: Path,
    *,
    ffmpeg: str,
    layout: str,
    cell: tuple[int, int] | None,
    at: float,
    max_width: int | None = None,
) -> list[str]:
    """One composed frame at ``at`` seconds; a finished video shows its last frame."""
    rate = videos[0].fps or 30.0
    command = [ffmpeg, "-y", "-v", "error"]
    for source in videos:
        command += ["-ss", f"{max(0.0, min(at, source.duration - 2.0 / rate)):.6f}"]
        command += ["-i", str(source.path)]
    filters, _, _ = video_filters(
        videos, layout=layout, cell=cell, rate=rate, duration=1.0 / rate
    )
    label = "[vout]"
    if max_width:
        filters.append(f"[vout]scale='min({max_width},iw)':-2[vsmall]")
        label = "[vsmall]"
    command += ["-filter_complex", ";".join(filters), "-map", label]
    command += ["-frames:v", "1", "-update", "1", str(image)]
    return command


def confirm_with_preview(
    videos: list[Source],
    *,
    ffmpeg: str,
    layout: str,
    cell: tuple[int, int] | None,
    at: float,
) -> bool:
    """Show one composed frame, then ask whether to go ahead.  Nothing is kept."""
    handle, name = tempfile.mkstemp(prefix="compose_preview_", suffix=".png")
    os.close(handle)
    image = Path(name)
    try:
        run_checked(
            build_preview_command(videos, image, ffmpeg=ffmpeg, layout=layout, cell=cell, at=at),
            echo=False,
        )
        open_in_default_app(image)
        eprint(f"プレビュー（{at:.1f}秒の時点）を表示しました。")
        try:
            answer = input("この配置で実行しますか？ [y/N]: ")
        except EOFError:
            answer = ""
        return answer.strip().lower() in {"y", "yes"}
    finally:
        try:
            image.unlink(missing_ok=True)
        except OSError:  # Windows: the viewer may still hold the file
            pass


def build_command(
    videos: list[Source],
    audios: list[Source],
    output: Path,
    *,
    ffmpeg: str,
    layout: str,
    audio_mode: str,
    cell: tuple[int, int] | None,
    fps: float | None,
    codec: str,
    encoder_mode: str,
    video_bitrate: str,
) -> list[str]:
    duration = max(s.duration for s in videos + audios)
    eprint(f"長さ: {duration:.3f}秒")
    command = [ffmpeg, "-y", "-v", "error", "-stats"]
    available = available_encoders(ffmpeg)
    for source in videos:
        if sys.platform == "darwin":  # decode on the media engine too
            command += ["-hwaccel", "videotoolbox"]
        command += ["-i", str(source.path)]
    for source in audios:
        command += ["-i", str(source.path)]
    filters: list[str] = []
    maps: list[str] = []

    if videos:
        rate = fps or videos[0].fps or 30.0
        filters, width, height = video_filters(
            videos, layout=layout, cell=cell, rate=rate, duration=duration
        )
        maps += ["-map", "[vout]"]

    audio_rate = 0
    channel_count = 0
    if audios:
        audio_rate = max(r for s in audios for r in s.audio_rates)
        labels = []
        for offset, source in enumerate(audios, start=len(videos)):
            for stream, channels in enumerate(source.audio_channels):
                label = f"a{offset}_{stream}"
                routing = "|".join(f"c{c}=c{c}" for c in range(channels))
                filters.append(
                    f"[{offset}:a:{stream}]aresample={audio_rate},"
                    f"pan={channels}c|{routing},"
                    f"apad=whole_dur={duration:.6f},atrim=duration={duration:.6f}[{label}]"
                )
                labels.append(f"[{label}]")
                channel_count += channels
        if len(labels) > 64:
            raise SyncError("まとめられる音声トラックは64本までです。")
        merged = "[amerged]" if len(labels) > 1 else labels[0]
        if len(labels) > 1:
            filters.append("".join(labels) + f"amerge=inputs={len(labels)}[amerged]")
        if audio_mode == "mix":
            # Mean of every channel: never clips, unlike a plain sum.
            gain = 1.0 / channel_count
            terms = "+".join(f"{gain:.8f}*c{c}" for c in range(channel_count))
            filters.append(f"{merged}pan=mono|c0={terms}[aout]")
        else:
            filters.append(f"{merged}anull[aout]")
        maps += ["-map", "[aout]"]
        eprint(
            "音声: "
            + ("全チャンネルをミックスしたモノラル" if audio_mode == "mix" else f"{channel_count}ch")
            + f"、{audio_rate} Hz"
        )

    command += ["-filter_complex", ";".join(filters), *maps]
    if videos:
        if codec == "source":
            codec = videos[0].video_codec or ""
            if codec not in {"hevc", "h264"}:
                eprint(f"注意: 入力の形式 {codec or '不明'} では出力できないため、H.264にします。")
                codec = "h264"
        encoder, options = choose_encoder(
            codec, encoder_mode, width, height, ffmpeg=ffmpeg, available=available
        )
        if video_bitrate == "source":
            # Keep each input's bitrate, scaled by how much its picture shrinks.
            assert videos[0].video_size is not None
            cell_width, cell_height = cell or videos[0].video_size
            bitrate = 0.0
            for source in videos:
                assert source.video_size is not None
                if source.video_bitrate is None:
                    raise SyncError(
                        f"ビットレートを取得できません: {source.path}\n"
                        "--video-bitrate で指定してください。"
                    )
                area = (cell_width * cell_height) / (source.video_size[0] * source.video_size[1])
                bitrate += source.video_bitrate * min(1.0, area)
        else:
            bitrate = parse_bitrate(video_bitrate)
        command += ["-c:v", encoder, *options, "-pix_fmt", "yuv420p"]
        command += ["-b:v", f"{max(100, round(bitrate / 1000))}k"]
        eprint(f"エンコード: {codec.upper()}（{encoder}）、{bitrate / 1e6:.1f} Mbps")
        if audios:
            output_channels = 1 if audio_mode == "mix" else channel_count
            if output.suffix.lower() in {".mov", ".mkv"}:
                command += ["-c:a", "pcm_s24le"]
            elif output_channels <= 8:
                command += ["-c:a", "aac", "-b:a", f"{96 * output_channels}k"]
                command += ["-ar", str(min(audio_rate, 96000))]
            else:
                raise SyncError(
                    f"{output_channels}chの音声はMP4（AAC）に入りません。"
                    "出力を .mov か .mkv にするか、--audio-mode mix を使ってください。"
                )
        if output.suffix.lower() in {".mp4", ".mov", ".m4v"}:
            command += ["-movflags", "+faststart"]
    else:
        command += ["-c:a", "pcm_s24le", "-rf64", "auto"]
    command.append(str(output))
    return command


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="同期済みの映像を並べ、音声をまとめて1本の動画（またはWAV）にします。"
    )
    parser.add_argument("-v", "--video", nargs="+", type=Path, default=[], help="映像にするメディア（左上から順に配置）")
    parser.add_argument("-a", "--audio", nargs="+", type=Path, default=[], help="音声にするメディア（動画でも可）")
    parser.add_argument("-o", "--output", type=Path, required=True, help="出力ファイル")
    parser.add_argument(
        "--layout",
        choices=["row", "column", "grid"],
        default="grid",
        help="row=横一列、column=縦一列、grid=なるべく正方形（標準）",
    )
    parser.add_argument(
        "--audio-mode",
        choices=["mix", "multi"],
        default="mix",
        help="mix=全チャンネルを平均した1ch（標準）、multi=全チャンネルを入力順に並べる",
    )
    parser.add_argument(
        "--codec",
        choices=["source", "hevc", "h264"],
        default="source",
        help="映像の形式。source=最初の映像と同じ（標準）",
    )
    parser.add_argument(
        "--cell-size",
        default="source",
        help="1画面の大きさ WxH。source=最初の映像の大きさのまま（標準）",
    )
    parser.add_argument(
        "--video-bitrate",
        default="source",
        help="出力全体の映像ビットレート（例: 8M）。source=入力の合計のまま（標準）",
    )
    parser.add_argument(
        "--encoder",
        choices=["auto", "hardware", "software"],
        default="auto",
        help="auto=使えればハードウェア（Mac/NVIDIA/Intel/AMD）、だめならCPU（標準）",
    )
    parser.add_argument("--fps", type=float, help="出力のフレームレート（標準は最初の映像）")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="実行前の静止画プレビューと確認を省く（端末以外から実行したときは常に省略）",
    )
    parser.add_argument(
        "--preview-at", type=float, metavar="SECONDS", help="プレビューする時刻（標準は中央）"
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpegのパス")
    parser.add_argument("--ffprobe", default=None, help="ffprobeのパス")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    ffmpeg = check_binary(args.ffmpeg, "ffmpeg")
    ffprobe = check_binary(args.ffprobe or default_ffprobe(ffmpeg), "ffprobe")

    def load(paths: list[Path]) -> list[Source]:
        sources = []
        for path in paths:
            path = path.expanduser().resolve()
            if not path.is_file():
                raise SyncError(f"入力ファイルが見つかりません: {path}")
            sources.append(probe(path, ffprobe))
        return sources

    videos = []
    for source in load(args.video):
        if source.video_size is None:
            eprint(f"警告: 映像トラックがないため映像には使いません: {source.path}")
        else:
            videos.append(source)
    audios = []
    for source in load(args.audio):
        if not source.audio_channels:
            eprint(f"警告: 音声トラックがないため音声には使いません: {source.path}")
        else:
            audios.append(source)
    if not videos and not audios:
        raise SyncError("使える映像も音声もありません。")

    output = args.output.expanduser().resolve()
    if not videos and output.suffix.lower() != ".wav":
        output = output.with_suffix(".wav")
        eprint(f"警告: 映像がないためWAVで出力します: {output}")
    if any(output == s.path for s in videos + audios):
        raise SyncError(f"出力が入力を上書きします: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    cell = None if args.cell_size == "source" else parse_size(args.cell_size)
    if videos and not args.no_preview and sys.stdin.isatty():
        at = args.preview_at
        if at is None:
            at = max(s.duration for s in videos + audios) / 2
        if not confirm_with_preview(videos, ffmpeg=ffmpeg, layout=args.layout, cell=cell, at=at):
            print("中止しました。")
            return 1

    command = build_command(
        videos,
        audios,
        output,
        ffmpeg=ffmpeg,
        layout=args.layout,
        audio_mode=args.audio_mode,
        cell=cell,
        fps=args.fps,
        codec=args.codec,
        encoder_mode=args.encoder,
        video_bitrate=args.video_bitrate,
    )
    run_checked(command)
    print(f"完了: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SyncError as exc:
        eprint(f"エラー: {exc}")
        raise SystemExit(2)
