#!/usr/bin/env python3
"""Synchronize recordings of one conversation using a known acoustic beep.

Every input is treated alike.  Each file is aligned in three steps:

1. the beep is located roughly (harmonic energy, then an onset threshold);
2. video timecode, when present, cross-checks the beep positions and guides
   files whose beep could not be found reliably; and
3. the sound around the beep is cross-correlated (GCC-PHAT) between every pair
   of files, and the pairwise lags are solved jointly by least squares.

Outputs, in the given directory:

* ``<name>_sync.<ext>`` per input, all starting ``--pre-roll`` before the beep
  and all ending at the earliest common end;
* ``sync_check.wav``: every audio channel of every input, in input order, for
  checking the alignment; and
* ``report.json``.

Run ``python sync_media.py --help`` for usage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only on missing dependency
    raise SystemExit(
        "NumPy が必要です。`python3 -m pip install numpy` を実行してください。"
    ) from exc


DETECTION_RATE = 16000
CORRELATION_RATE = 16000
HARMONIC_WEIGHTS = (1.0, 0.45, 0.20)
CHECK_WAV_NAME = "sync_check.wav"
PROGRESS_ENV = "MEDIA_SYNC_PROGRESS"
REPORT_NAME = "report.json"
# Encoder -> (extra output options, default bitrate).  The hardware encoders
# are Apple VideoToolbox, NVIDIA NVENC, Intel QSV and AMD AMF.  On an M3,
# VideoToolbox ran ~5x real time for 1080p and at these bitrates measured at
# least as close to the source as libx264 CRF 18 (VMAF ~97 vs ~96).
VIDEO_ENCODERS: dict[str, tuple[list[str], str | None]] = {
    "hevc_videotoolbox": (["-tag:v", "hvc1"], "20M"),
    "hevc_nvenc": (["-tag:v", "hvc1", "-preset", "p5"], "20M"),
    "hevc_qsv": (["-tag:v", "hvc1"], "20M"),
    "hevc_amf": (["-tag:v", "hvc1"], "20M"),
    "h264_videotoolbox": ([], "30M"),
    "h264_nvenc": (["-preset", "p5"], "30M"),
    "h264_qsv": ([], "30M"),
    "h264_amf": ([], "30M"),
    "libx264": (["-preset", "medium", "-crf", "18"], None),
}
HARDWARE_ENCODERS = {
    codec: [f"{codec}_{suffix}" for suffix in ("videotoolbox", "nvenc", "qsv", "amf")]
    for codec in ("hevc", "h264")
}


class SyncError(RuntimeError):
    """A user-facing synchronization error."""


@dataclass
class AudioStream:
    index: int
    sample_rate: int
    channels: int
    codec: str
    offset: float  # stream start relative to the file start, in seconds


@dataclass
class Media:
    position: int  # 0-based input order
    path: Path
    duration: float
    has_video: bool
    fps: float | None
    audio: list[AudioStream]
    timecode: str | None
    timecode_seconds: float | None
    label: str = ""
    output_path: Path | None = None

    @property
    def main_audio(self) -> AudioStream:
        return self.audio[0]


@dataclass
class Alignment:
    """Per-file synchronization state, filled in step by step."""

    media: Media
    coarse_seconds: float | None = None
    coarse_details: dict[str, float] = field(default_factory=dict)
    confidence: str = "none"
    source: str = "none"  # beep / manual / timecode
    channel: int = 0  # 0-based channel of the main audio stream used for analysis
    channel_scores: list[float] = field(default_factory=list)
    channel_onsets: dict[int, float] = field(default_factory=dict)  # 1-based channel
    onset_seconds: float | None = None  # beep onset before correlation
    timecode_beep_seconds: float | None = None
    timecode_status: str = "unavailable"
    correction_seconds: float = 0.0
    drift_ppm: float = 0.0  # clock rate relative to the mean of all files
    beep_seconds: float | None = None
    trim_start_seconds: float | None = None
    rms_residual_seconds: float | None = None
    pairs_used: int = 0
    locked_with: list[int] = field(default_factory=list)  # same-clock files


@dataclass
class PairResult:
    first: int
    second: int
    offset_seconds: float  # correction(second) - correction(first) at the beep
    drift_ppm: float  # growth rate of that difference
    windows: int
    inliers: int
    rms_residual_seconds: float  # scatter of the windows around the fitted line
    accepted: bool
    residual_seconds: float | None = None  # disagreement with the joint solution
    note: str = ""


# ---------------------------------------------------------------------------
# External tools


def eprint(*values: object) -> None:
    print(*values, file=sys.stderr)


def shell_quote(value: str) -> str:
    if value and all(c.isalnum() or c in "_./:=+,-" for c in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def run_checked(
    command: list[str], *, capture: bool = False, echo: bool = True
) -> subprocess.CompletedProcess:
    pretty = " ".join(shell_quote(part) for part in command)
    if echo:
        eprint(f"+ {pretty}")
    if os.environ.get(PROGRESS_ENV) and not capture and Path(command[0]).stem.lower() == "ffmpeg":
        return _run_with_progress(command, pretty)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        raise SyncError(f"コマンドが失敗しました ({result.returncode}):\n{pretty}\n{stderr}")
    return result


def _run_with_progress(command: list[str], pretty: str) -> subprocess.CompletedProcess:
    """Run ffmpeg, turning its ``-progress`` output into ``PROGRESS <seconds>`` lines.

    Used by the GUI (``MEDIA_SYNC_PROGRESS=1``) to draw a progress bar.
    """
    command = [command[0], "-progress", "pipe:2", "-nostats"] + [
        part for part in command[1:] if part != "-stats"
    ]
    process = subprocess.Popen(
        command, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"
    )
    assert process.stderr is not None
    other: list[str] = []
    for line in process.stderr:
        key, separator, value = line.strip().partition("=")
        if separator and key == "out_time_us" and value.lstrip("-").isdigit():
            eprint(f"PROGRESS {max(0, int(value)) / 1e6:.3f}")
        elif not (separator and re.fullmatch(r"[a-z_0-9]+", key)):
            other.append(line)
    if process.wait() != 0:
        raise SyncError(
            f"コマンドが失敗しました ({process.returncode}):\n{pretty}\n{''.join(other[-50:])}"
        )
    return subprocess.CompletedProcess(command, process.returncode)


def check_binary(path_or_name: str, label: str) -> str:
    resolved = shutil.which(path_or_name) if "/" not in path_or_name else path_or_name
    if not resolved or not Path(resolved).exists():
        raise SyncError(f"{label} が見つかりません: {path_or_name}")
    result = subprocess.run(
        [resolved, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SyncError(f"{label} を起動できません: {resolved}\n{detail}")
    return str(resolved)


_encoder_checks: dict[tuple[str, str, int, int], bool] = {}


def encoder_works(ffmpeg: str, encoder: str, width: int = 1280, height: int = 720) -> bool:
    """Whether ``encoder`` can really encode a ``width``x``height`` frame here.

    Being compiled into ffmpeg is not enough: the GPU may be missing, and some
    hardware encoders have size limits (VideoToolbox / NVENC H.264: 4096 wide).
    """
    key = (ffmpeg, encoder, width, height)
    if key not in _encoder_checks:
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-f", "lavfi", "-i",
             f"color=black:s={width}x{height}:r=30:d=0.1", "-frames:v", "1",
             "-c:v", encoder, "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )  # fmt: skip
        _encoder_checks[key] = result.returncode == 0
    return _encoder_checks[key]


def pick_hardware_encoder(ffmpeg: str, codec: str, width: int, height: int) -> str | None:
    for encoder in HARDWARE_ENCODERS[codec]:
        if encoder_works(ffmpeg, encoder, width, height):
            return encoder
    return None


def open_in_default_app(path: Path | str) -> None:
    """Open a file or folder with the OS's default application."""
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        if shutil.which(opener):
            subprocess.run([opener, str(path)], check=False)


def find_tool(name: str) -> str:
    """A bundled ffmpeg/ffprobe (``ffmpeg/mac``, ``ffmpeg/win``, ``ffmpeg/AMD``), else PATH."""
    base = Path(__file__).resolve().parent / "ffmpeg"
    if sys.platform == "darwin":
        bundled = base / "mac" / name
    elif sys.platform.startswith("win"):
        bundled = base / "win" / f"{name}.exe"
    else:
        bundled = base / "AMD" / name
    if bundled.exists():
        return str(bundled)
    return shutil.which(name) or name


def default_ffprobe(ffmpeg: str) -> str:
    executable = Path(ffmpeg)
    sibling = executable.with_name("ffprobe" + executable.suffix)  # keeps ".exe" on Windows
    return str(sibling) if sibling.exists() else find_tool("ffprobe")


# ---------------------------------------------------------------------------
# Probing


def _parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        if float(denominator) == 0:
            return None
        return float(numerator) / float(denominator)
    return float(value)


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_timecode(text: str, fps: float | None) -> float | None:
    """Convert ``HH:MM:SS:FF`` (``;`` for drop-frame) to seconds of label time."""
    match = re.fullmatch(r"(\d+):(\d+):(\d+)[:;.,](\d+)", text.strip())
    if not match or not fps:
        return None
    hours, minutes, seconds, frames = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + frames / max(1, round(fps))


def probe_media(path: Path, position: int, ffprobe: str) -> Media:
    result = run_checked(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,start_time:format_tags=timecode:"
            "stream=index,codec_type,codec_name,sample_rate,channels,r_frame_rate,"
            "start_time,duration:stream_tags=timecode:stream_disposition=attached_pic",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
        echo=False,
    )
    data = json.loads(result.stdout.decode("utf-8", errors="replace"))
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    format_start = _float_or(fmt.get("start_time"), 0.0)

    video = [
        s
        for s in streams
        if s.get("codec_type") == "video"
        and not s.get("disposition", {}).get("attached_pic", 0)
    ]
    audio = [
        AudioStream(
            index=int(s["index"]),
            sample_rate=int(s.get("sample_rate") or 0),
            channels=int(s.get("channels") or 0),
            codec=str(s.get("codec_name", "")),
            offset=max(0.0, _float_or(s.get("start_time"), format_start) - format_start),
        )
        for s in streams
        if s.get("codec_type") == "audio"
    ]
    if not audio:
        raise SyncError(f"音声トラックがありません: {path}")

    duration = _float_or(fmt.get("duration"), 0.0)
    if duration <= 0:
        raise SyncError(f"長さを取得できません: {path}")

    fps = _parse_rate(video[0].get("r_frame_rate")) if video else None
    timecode = None
    if video:
        candidates = [video[0].get("tags", {}).get("timecode"), fmt.get("tags", {}).get("timecode")]
        candidates += [s.get("tags", {}).get("timecode") for s in streams]
        timecode = next((tc for tc in candidates if tc), None)
    return Media(
        position=position,
        path=path,
        duration=duration,
        has_video=bool(video),
        fps=fps,
        audio=audio,
        timecode=timecode,
        timecode_seconds=parse_timecode(timecode, fps) if timecode else None,
    )


def assign_labels_and_outputs(media: list[Media], output_dir: Path) -> None:
    """Name outputs ``<stem>_sync<ext>``; prefix the folder name on clashes."""
    stems = [m.path.stem for m in media]
    used: set[str] = set()
    for item in media:
        stem = item.path.stem
        if stems.count(stem) > 1:
            stem = f"{item.path.parent.name}_{stem}"
        candidate = stem
        counter = 2
        while candidate.lower() in used:
            candidate = f"{stem}_{counter}"
            counter += 1
        used.add(candidate.lower())
        item.label = candidate
        item.output_path = output_dir / f"{candidate}_sync{item.path.suffix}"


# ---------------------------------------------------------------------------
# Decoding


def decode_audio(
    media: Media,
    ffmpeg: str,
    *,
    start: float,
    duration: float,
    rate: int | None = None,
    mix: bool = False,
) -> tuple[np.ndarray, int, float]:
    """Decode the main audio stream.

    Returns ``(samples[n, channels], rate, start_of_first_sample)``.  With
    ``mix`` the channels are averaged into one.
    """
    stream = media.main_audio
    actual_start = max(start, stream.offset)
    command = [ffmpeg, "-v", "error"]
    if actual_start > 0:
        command += ["-ss", f"{actual_start:.9f}"]
    command += ["-i", str(media.path), "-t", f"{max(0.0, duration):.9f}", "-map", "0:a:0"]
    channels = stream.channels
    if mix and channels > 1:
        gain = 1.0 / channels
        terms = "+".join(f"{gain:.8f}*c{c}" for c in range(channels))
        command += ["-af", f"pan=mono|c0={terms}"]
        channels = 1
    output_rate = rate or stream.sample_rate
    command += ["-ar", str(output_rate), "-f", "f32le", "pipe:1"]
    result = run_checked(command, capture=True, echo=False)
    samples = np.frombuffer(result.stdout, dtype="<f4")
    usable = samples.size - samples.size % channels
    samples = samples[:usable].reshape(-1, channels).copy()
    if samples.shape[0] == 0:
        raise SyncError(f"音声をデコードできませんでした: {media.path}")
    return samples, output_rate, actual_start


# ---------------------------------------------------------------------------
# Beep detection


def harmonic_amplitudes(
    blocks: np.ndarray, sample_rate: int, frequencies: tuple[float, ...]
) -> np.ndarray:
    block_size = blocks.shape[1]
    window = np.hanning(block_size).astype(np.float32)
    scale = 2.0 / float(np.sum(window))
    positions = np.arange(block_size)
    output: list[np.ndarray] = []
    for frequency in frequencies:
        kernel = (
            window * np.exp(-2j * np.pi * float(frequency) * positions / sample_rate)
        ).astype(np.complex64)
        output.append(scale * np.abs(blocks @ kernel))
    return np.stack(output, axis=1)


def beep_frequencies(frequency_hz: float, sample_rate: int) -> tuple[float, float, float]:
    frequencies = (frequency_hz, frequency_hz * 2.0, frequency_hz * 3.0)
    if frequencies[-1] >= sample_rate / 2:
        raise SyncError(f"サンプルレート{sample_rate}Hzでは第3倍音を解析できません。")
    return frequencies


def detect_beep(
    samples: np.ndarray,
    sample_rate: int,
    *,
    search_start_seconds: float,
    frequency_hz: float,
) -> tuple[float, dict[str, float]]:
    if samples.size < sample_rate:
        raise SyncError("解析対象が短すぎます（1秒未満）。")

    frequencies = beep_frequencies(frequency_hz, sample_rate)
    weights = np.asarray(HARMONIC_WEIGHTS, dtype=np.float64)

    coarse_block_seconds = 0.050
    coarse_block = max(32, round(coarse_block_seconds * sample_rate))
    block_count = samples.size // coarse_block
    blocks = samples[: block_count * coarse_block].reshape(block_count, coarse_block)
    score = harmonic_amplitudes(blocks, sample_rate, frequencies) @ weights

    smooth_blocks = max(1, round(0.4 / coarse_block_seconds))
    smooth = np.convolve(score, np.ones(smooth_blocks) / smooth_blocks, mode="same")
    peak_index = int(np.argmax(smooth))
    peak_time_local = (peak_index + 0.5) * coarse_block_seconds
    peak_value = float(smooth[peak_index])

    exclusion = max(1, round(1.5 / coarse_block_seconds))
    comparison = smooth.copy()
    comparison[max(0, peak_index - exclusion) : peak_index + exclusion + 1] = np.nan
    second_value = float(np.nanmax(comparison)) if np.any(np.isfinite(comparison)) else 0.0
    background = float(np.median(smooth))
    peak_to_second = peak_value / max(second_value, 1e-12)
    peak_to_background_db = 20.0 * math.log10(
        max(peak_value, 1e-12) / max(background, 1e-12)
    )

    fine_start_local = max(0.0, peak_time_local - 1.0)
    fine_end_local = min(samples.size / sample_rate, peak_time_local + 0.6)
    fine = samples[round(fine_start_local * sample_rate) : round(fine_end_local * sample_rate)]

    fine_window = max(32, round(0.008 * sample_rate))
    fine_hop = max(1, round(0.001 * sample_rate))
    starts = np.arange(0, fine.size - fine_window + 1, fine_hop)
    if starts.size < 20:
        raise SyncError("同期音候補の精密解析範囲が短すぎます。")

    window = np.hanning(fine_window)
    positions = np.arange(fine_window)
    frames = np.lib.stride_tricks.sliding_window_view(fine, fine_window)[starts]
    fine_score = np.zeros(starts.size, dtype=np.float64)
    for frequency, weight in zip(frequencies, weights):
        kernel = window * np.exp(-2j * np.pi * frequency * positions / sample_rate)
        fine_score += weight * 2.0 * np.abs(frames @ kernel) / np.sum(window)

    centers_local = fine_start_local + (starts + fine_window / 2) / sample_rate
    baseline_mask = centers_local < peak_time_local - 0.30
    if np.count_nonzero(baseline_mask) < 20:
        baseline_mask = np.arange(starts.size) < max(20, starts.size // 4)
    baseline = fine_score[baseline_mask]
    median = float(np.median(baseline))
    robust_sigma = 1.4826 * float(np.median(np.abs(baseline - median)))
    dynamic_range = max(0.0, float(np.max(fine_score)) - median)
    threshold = median + max(12.0 * robust_sigma, 0.002 * dynamic_range)
    threshold = min(threshold, median + 0.25 * dynamic_range)

    run_length, required = 10, 8
    onset_frame: int | None = None
    search_limit = peak_time_local + 0.10
    for index in range(0, fine_score.size - run_length + 1):
        if centers_local[index] > search_limit:
            break
        if np.count_nonzero(fine_score[index : index + run_length] > threshold) >= required:
            onset_frame = index
            break
    if onset_frame is None:
        raise SyncError("同期音候補は見つかりましたが、立ち上がりを確定できません。")

    # The threshold is low, so the first window above it only just reaches the
    # beep: its end, not its start, marks the onset.
    onset_local = centers_local[onset_frame] + fine_window / (2 * sample_rate)
    details = {
        "coarse_peak_seconds": search_start_seconds + peak_time_local,
        "peak_to_second_ratio": peak_to_second,
        "peak_to_background_db": peak_to_background_db,
    }
    return search_start_seconds + float(onset_local), details


def confidence_label(details: dict[str, float]) -> str:
    ratio = details["peak_to_second_ratio"]
    background_db = details["peak_to_background_db"]
    if ratio >= 3.0 and background_db >= 20.0:
        return "high"
    if ratio >= 1.5 and background_db >= 12.0:
        return "medium"
    return "low"


def locate_beep(
    media: Media,
    ffmpeg: str,
    *,
    start: float,
    length: float,
    frequency_hz: float,
) -> tuple[float, dict[str, float]]:
    """Find the beep in a mono mixdown of ``[start, start + length)``."""
    length = min(length, media.duration - start)
    if length <= 1.0:
        raise SyncError("探索範囲がファイルの外です。")
    samples, rate, actual_start = decode_audio(
        media, ffmpeg, start=start, duration=length, rate=DETECTION_RATE, mix=True
    )
    return detect_beep(
        samples[:, 0], rate, search_start_seconds=actual_start, frequency_hz=frequency_hz
    )


def refine_on_best_channel(
    alignment: Alignment, ffmpeg: str, *, frequency_hz: float
) -> None:
    """Choose the analysis channel and re-detect the beep onset there.

    Some devices add a processed channel with extra latency (for example the
    ReSpeaker array's channel 1 is ~70 ms late), so the onset is measured on
    every channel that hears the beep and channels that disagree with the
    majority are skipped.  Among the rest, the clearest one is used.
    """
    media = alignment.media
    assert alignment.coarse_seconds is not None
    start = max(0.0, alignment.coarse_seconds - 3.0)
    samples, rate, actual_start = decode_audio(media, ffmpeg, start=start, duration=6.0)
    if 3 * frequency_hz >= rate / 2:
        samples, rate, actual_start = decode_audio(
            media, ffmpeg, start=start, duration=6.0, rate=DETECTION_RATE
        )
    frequencies = beep_frequencies(frequency_hz, rate)
    weights = np.asarray(HARMONIC_WEIGHTS)
    block = round(0.05 * rate)
    count = samples.shape[0] // block
    scores: list[float] = []
    for channel in range(samples.shape[1]):
        blocks = samples[: count * block, channel].reshape(count, block)
        score = harmonic_amplitudes(blocks, rate, frequencies) @ weights
        scores.append(float(np.max(score) / max(float(np.median(score)), 1e-12)))
    alignment.channel = int(np.argmax(scores))
    alignment.channel_scores = [round(s, 2) for s in scores]
    alignment.onset_seconds = alignment.coarse_seconds
    if alignment.source == "timecode":
        return

    onsets: dict[int, float] = {}
    for channel, score in enumerate(scores):
        if score < 0.25 * max(scores):
            continue
        try:
            onset, _ = detect_beep(
                samples[:, channel],
                rate,
                search_start_seconds=actual_start,
                frequency_hz=frequency_hz,
            )
        except SyncError:
            continue
        if abs(onset - alignment.coarse_seconds) < 0.5:
            onsets[channel] = onset
    if not onsets:
        return
    median = float(np.median(list(onsets.values())))
    agreeing = [c for c, onset in onsets.items() if abs(onset - median) <= 0.003]
    if not agreeing:
        agreeing = list(onsets)
    alignment.channel = max(agreeing, key=lambda c: scores[c])
    alignment.channel_onsets = {c + 1: round(o, 6) for c, o in onsets.items()}
    alignment.onset_seconds = onsets[alignment.channel]


# ---------------------------------------------------------------------------
# Timecode


def apply_timecode(
    alignments: list[Alignment],
    ffmpeg: str,
    *,
    tolerance: float,
    frequency_hz: float,
) -> None:
    """Cross-check beep positions against timecode and help ambiguous files.

    The beep happens at one wall-clock instant, so ``timecode + beep position``
    should agree across cameras.  Files whose timecode disagrees are marked and
    their timecode is not used.
    """
    with_tc = [a for a in alignments if a.media.timecode_seconds is not None]
    if not with_tc:
        return
    trusted = [
        a.media.timecode_seconds + a.coarse_seconds
        for a in with_tc
        if a.coarse_seconds is not None and a.confidence in {"high", "medium", "manual"}
    ]
    if not trusted:
        for item in with_tc:
            item.timecode_status = "no-reference"
        return
    wall_beep = float(np.median(trusted))

    for item in with_tc:
        expected = wall_beep - item.media.timecode_seconds
        item.timecode_beep_seconds = expected
        if item.coarse_seconds is not None and abs(item.coarse_seconds - expected) <= tolerance:
            item.timecode_status = "consistent"
            if item.confidence == "low":
                item.confidence = "timecode-confirmed"
            continue
        if item.confidence in {"high", "medium", "manual"}:
            item.timecode_status = "inconsistent"
            continue
        if not 0.0 <= expected < item.media.duration:
            item.timecode_status = "outside-file"
            continue
        # The beep was ambiguous: search again only where timecode says it is.
        eprint(f"  タイムコードを使って再探索: {item.media.label}")
        try:
            onset, details = locate_beep(
                item.media,
                ffmpeg,
                start=max(0.0, expected - 2.0),
                length=4.0,
                frequency_hz=frequency_hz,
            )
        except SyncError:
            onset = None
        if onset is not None and abs(onset - expected) <= tolerance:
            item.coarse_seconds = onset
            item.coarse_details = details
            item.confidence = "timecode-guided"
            item.source = "beep"
        else:
            item.coarse_seconds = expected
            item.confidence = "timecode-only"
            item.source = "timecode"
        item.timecode_status = "used"


# ---------------------------------------------------------------------------
# Cross-correlation


def tukey(length: int, alpha: float) -> np.ndarray:
    window = np.ones(length)
    taper = int(alpha * length / 2)
    if taper > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(taper) / taper))
        window[:taper] = ramp
        window[-taper:] = ramp[::-1]
    return window


def gcc_phat(
    spectrum_x: np.ndarray,
    spectrum_y: np.ndarray,
    nfft: int,
    rate: int,
    max_lag: float,
    *,
    upsample: int = 4,
    min_frequency: float = 80.0,
) -> float:
    """GCC-PHAT lag in seconds between two ``rfft(·, nfft)`` spectra.

    A positive lag means the same sound appears later in ``y``.
    """
    cross = np.conj(spectrum_x) * spectrum_y
    magnitude = np.abs(cross)
    cross = cross / np.maximum(magnitude, 1e-12 * max(float(magnitude.max()), 1e-30))
    cross[np.fft.rfftfreq(nfft, 1.0 / rate) < min_frequency] = 0.0
    correlation = np.fft.irfft(cross, nfft * upsample)

    max_index = min(int(math.ceil(max_lag * rate * upsample)), nfft * upsample // 2 - 1)
    values = np.concatenate([correlation[-max_index:], correlation[: max_index + 1]])
    peak = int(np.argmax(values))
    offset = 0.0
    if 0 < peak < values.size - 1:
        left, centre, right = values[peak - 1 : peak + 2]
        denominator = left - 2 * centre + right
        if denominator < 0:
            offset = 0.5 * (left - right) / denominator
    return float((peak - max_index + offset) / (rate * upsample))


def fit_line(
    times: np.ndarray, lags: np.ndarray, tolerance: float
) -> tuple[float, float, np.ndarray]:
    """Robust ``lag = intercept + slope * time``; returns (intercept, slope, inliers)."""
    if times.size == 1:
        return float(lags[0]), 0.0, np.ones(1, dtype=bool)
    # Theil-Sen start, so that a few wrong windows cannot drag the first fit.
    i, j = np.triu_indices(times.size, 1)
    slope = float(np.median((lags[j] - lags[i]) / (times[j] - times[i])))
    intercept = float(np.median(lags - slope * times))
    inliers = np.zeros(times.size, dtype=bool)
    for _ in range(10):
        residuals = lags - (intercept + slope * times)
        reference = residuals[inliers] if inliers.any() else residuals
        spread = 1.4826 * float(np.median(np.abs(reference)))
        updated = np.abs(residuals) <= max(tolerance, 4.0 * spread)
        if np.count_nonzero(updated) < 2 or np.array_equal(updated, inliers):
            break
        inliers = updated
        slope, intercept = (float(v) for v in np.polyfit(times[inliers], lags[inliers], 1))
    if not inliers.any():
        inliers = np.abs(lags - (intercept + slope * times)) <= tolerance
    return intercept, slope, inliers


def correlation_times(
    alignments: list[Alignment], *, window: float, interval: float
) -> list[float]:
    """Window start times, in seconds after each file's beep onset."""
    span = min(
        a.media.duration - (a.onset_seconds or 0.0) for a in alignments
    ) - window
    first = 5.0  # skip the beep itself: a pure tone correlates ambiguously
    if span <= first:
        return [max(0.0, span)]
    return [float(t) for t in np.arange(first, span, interval)]


def solve_corrections(
    count: int,
    pairs: list[PairResult],
    anchors: list[bool],
    value: str,
    locks: list[tuple[int, int, float]] = (),
) -> np.ndarray:
    """Least-squares per-file values from accepted pairwise differences.

    ``locks`` are exact ``value[j] - value[i] = d`` constraints.  Each
    connected group is pinned so that the mean of its anchor members is zero
    (for offsets, the beep onsets stay the absolute reference).
    """
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    accepted = [p for p in pairs if p.accepted]
    for pair in accepted:
        parent[find(pair.first)] = find(pair.second)
    for first, second, _ in locks:
        parent[find(first)] = find(second)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    rows: list[np.ndarray] = []
    values: list[float] = []
    for pair in accepted:
        row = np.zeros(count)
        row[pair.second], row[pair.first] = 1.0, -1.0
        rows.append(row)
        values.append(getattr(pair, value))
    for first, second, difference in locks:
        row = np.zeros(count)
        row[second], row[first] = 1e4, -1e4
        rows.append(row)
        values.append(1e4 * difference)
    for members in groups.values():
        pinned = [m for m in members if anchors[m]] or members
        row = np.zeros(count)
        row[pinned] = 1.0
        rows.append(row)
        values.append(0.0)
    solution, *_ = np.linalg.lstsq(np.array(rows), np.array(values), rcond=None)
    return solution


def correlate_all(
    alignments: list[Alignment],
    ffmpeg: str,
    *,
    window: float,
    interval: float,
    max_lag: float,
    timecode_max_lag: float,
    max_drift_ppm: float,
    fit_tolerance: float,
    outlier_tolerance: float,
) -> tuple[list[PairResult], list[float]]:
    """Measure pairwise lags over the whole recording and solve the offsets.

    Lags are measured in windows spread over the shared span.  Recorder clocks
    drift linearly, so each pair's lags are fitted with a line and its value at
    the beep (``t = 0``) is the pairwise offset.
    """
    rate = CORRELATION_RATE
    length = round(window * rate)
    nfft = 1 << (2 * length - 1).bit_length()
    taper = tukey(length, 0.1)
    times = correlation_times(alignments, window=window, interval=interval)
    count = len(alignments)
    lags = np.full((len(times), count, count), np.nan)

    for step, t in enumerate(times):
        eprint(f"  相関区間 {step + 1}/{len(times)}（同期音から{t:.0f}秒後）")
        spectra = []
        for alignment in alignments:
            assert alignment.onset_seconds is not None
            samples, _, _ = decode_audio(
                alignment.media,
                ffmpeg,
                start=alignment.onset_seconds + t,
                duration=window,
                rate=rate,
            )
            mono = samples[:length, alignment.channel].astype(np.float64)
            mono = np.pad(mono, (0, length - mono.size))
            spectra.append(np.fft.rfft(mono * taper, nfft))
        for i in range(count):
            for j in range(i + 1, count):
                limit = max_lag
                if "timecode" in (alignments[i].source, alignments[j].source):
                    limit = max(limit, timecode_max_lag)
                limit += max_drift_ppm * 1e-6 * t
                lags[step, i, j] = gcc_phat(spectra[i], spectra[j], nfft, rate, limit)

    time_array = np.asarray(times)
    pairs: list[PairResult] = []
    for i in range(count):
        for j in range(i + 1, count):
            intercept, slope, inliers = fit_line(time_array, lags[:, i, j], fit_tolerance)
            residuals = lags[inliers, i, j] - (intercept + slope * time_array[inliers])
            rms = float(np.sqrt(np.mean(np.square(residuals))))
            used = int(np.count_nonzero(inliers))
            enough = used >= min(3, len(times)) and used >= 0.5 * len(times)
            accepted = enough and rms <= fit_tolerance
            note = "" if accepted else ("few-consistent-windows" if not enough else "scattered")
            pairs.append(
                PairResult(
                    first=i,
                    second=j,
                    offset_seconds=intercept,
                    drift_ppm=slope * 1e6,
                    windows=len(times),
                    inliers=used,
                    rms_residual_seconds=rms,
                    accepted=accepted,
                    note=note,
                )
            )

    # Files from one recorder (same rate, same sample count, no drift between
    # them) already share a timeline: keep their trims identical instead of
    # letting speaker-dependent acoustic lags pull them apart.
    offset_locks: list[tuple[int, int, float]] = []
    drift_locks: list[tuple[int, int, float]] = []
    for pair in pairs:
        a, b = alignments[pair.first], alignments[pair.second]
        rate = a.media.main_audio.sample_rate
        if (
            rate == b.media.main_audio.sample_rate
            and abs(a.media.duration - b.media.duration) < 1.0 / rate
            and pair.inliers >= 3
            and abs(pair.drift_ppm) < 1.0
        ):
            assert a.onset_seconds is not None and b.onset_seconds is not None
            offset_locks.append((pair.first, pair.second, a.onset_seconds - b.onset_seconds))
            drift_locks.append((pair.first, pair.second, 0.0))
            a.locked_with.append(pair.second)
            b.locked_with.append(pair.first)

    anchors = [a.source in {"beep", "manual"} for a in alignments]
    while True:
        corrections = solve_corrections(
            count, pairs, anchors, "offset_seconds", offset_locks
        )
        worst: PairResult | None = None
        for pair in pairs:
            pair.residual_seconds = float(
                pair.offset_seconds - (corrections[pair.second] - corrections[pair.first])
            )
            if pair.accepted and (
                worst is None or abs(pair.residual_seconds) > abs(worst.residual_seconds or 0)
            ):
                worst = pair
        if worst is None or abs(worst.residual_seconds or 0) <= outlier_tolerance:
            break
        worst.accepted = False
        worst.note = "inconsistent"
    drifts = solve_corrections(count, pairs, [True] * count, "drift_ppm", drift_locks)

    for alignment in alignments:
        position = alignment.media.position
        alignment.correction_seconds = float(corrections[position])
        alignment.drift_ppm = float(drifts[position])
        residuals = [
            p.residual_seconds or 0.0
            for p in pairs
            if p.accepted and position in (p.first, p.second)
        ]
        alignment.pairs_used = len(residuals)
        if residuals:
            alignment.rms_residual_seconds = float(np.sqrt(np.mean(np.square(residuals))))
    return pairs, times


# ---------------------------------------------------------------------------
# Output


def write_synced_file(
    media: Media,
    start: float,
    duration: float,
    *,
    ffmpeg: str,
    fast_copy: bool,
    video_encoder: str,
    video_bitrate: str | None,
) -> None:
    assert media.output_path is not None
    command = [ffmpeg, "-y", "-v", "error", "-stats"]
    if media.has_video:
        hardware = video_encoder.endswith("_videotoolbox") and not fast_copy
        if hardware:
            command += ["-hwaccel", "videotoolbox"]
        command += ["-ss", f"{start:.9f}", "-i", str(media.path), "-t", f"{duration:.9f}"]
        command += ["-map", "0:v:0", "-map", "0:a?", "-map_metadata", "-1"]
        if fast_copy:
            command += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        else:
            options, default_bitrate = VIDEO_ENCODERS[video_encoder]
            command += ["-c:v", video_encoder, *options]
            bitrate = video_bitrate or default_bitrate
            if bitrate:
                command += ["-b:v", bitrate]
            command += [
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            ]  # fmt: skip
    else:
        command += ["-i", str(media.path), "-map", "0:a", "-map_metadata", "0"]
        command += [
            "-af",
            f"atrim=start={start:.9f}:duration={duration:.9f},asetpts=PTS-STARTPTS",
        ]
        codec = media.main_audio.codec
        if codec.startswith("pcm_"):
            command += ["-c:a", codec]
        if media.path.suffix.lower() == ".wav":
            command += ["-rf64", "auto"]
    command.append(str(media.output_path))
    run_checked(command)


def write_check_wav(
    alignments: list[Alignment], duration: float, destination: Path, *, ffmpeg: str
) -> tuple[int, list[dict[str, Any]]]:
    rate = max(s.sample_rate for a in alignments for s in a.media.audio)
    command = [ffmpeg, "-y", "-v", "error", "-stats"]
    filters: list[str] = []
    labels: list[str] = []
    channel_map: list[dict[str, Any]] = []
    next_channel = 1
    for input_index, alignment in enumerate(alignments):
        media = alignment.media
        assert alignment.trim_start_seconds is not None
        command += ["-i", str(media.path)]
        for stream_order, stream in enumerate(media.audio):
            label = f"a{input_index}_{stream_order}"
            # Strip named layouts (2.0, 5.1, ...) so amerge just stacks channels.
            routing = "|".join(f"c{c}=c{c}" for c in range(stream.channels))
            filters.append(
                f"[{input_index}:a:{stream_order}]"
                f"atrim=start={alignment.trim_start_seconds:.9f}:duration={duration:.9f},"
                f"asetpts=PTS-STARTPTS,aresample={rate},"
                f"pan={stream.channels}c|{routing}[{label}]"
            )
            labels.append(f"[{label}]")
            channel_map.append(
                {
                    "input": str(media.path),
                    "audio_stream": stream_order,
                    "source_channels": stream.channels,
                    "first_channel": next_channel,
                    "last_channel": next_channel + stream.channels - 1,
                }
            )
            next_channel += stream.channels
    if len(labels) > 64:
        raise SyncError("確認用WAVにまとめられる音声トラックは64本までです。")
    if len(labels) == 1:
        filters[-1] = filters[-1].replace(labels[0], "[out]")
    else:
        filters.append("".join(labels) + f"amerge=inputs={len(labels)}[out]")
    command += ["-filter_complex", ";".join(filters), "-map", "[out]"]
    command += ["-c:a", "pcm_s24le", "-rf64", "auto", str(destination)]
    run_checked(command)
    return rate, channel_map


def write_report(
    path: Path,
    alignments: list[Alignment],
    pairs: list[PairResult],
    *,
    settings: dict[str, Any],
    common_duration: float | None,
    correlation_times: list[float],
    check_wav: dict[str, Any] | None,
) -> None:
    def rounded(value: float | None, digits: int = 6) -> float | None:
        return None if value is None else round(value, digits)

    inputs = []
    for a in alignments:
        m = a.media
        inputs.append(
            {
                "input": str(m.path),
                "output": str(m.output_path) if m.output_path else None,
                "has_video": m.has_video,
                "duration_seconds": rounded(m.duration),
                "audio_streams": [
                    {"sample_rate": s.sample_rate, "channels": s.channels, "codec": s.codec}
                    for s in m.audio
                ],
                "analysis_channel": a.channel + 1,
                "channel_beep_scores": a.channel_scores,
                "channel_onsets_seconds": a.channel_onsets,
                "timecode": m.timecode,
                "timecode_status": a.timecode_status,
                "timecode_predicted_beep_seconds": rounded(a.timecode_beep_seconds),
                "coarse_source": a.source,
                "confidence": a.confidence,
                "coarse_peak_to_second_ratio": rounded(
                    a.coarse_details.get("peak_to_second_ratio"), 3
                ),
                "coarse_peak_to_background_db": rounded(
                    a.coarse_details.get("peak_to_background_db"), 2
                ),
                "onset_seconds": rounded(a.onset_seconds),
                "correlation_correction_ms": rounded(a.correction_seconds * 1000, 4),
                "drift_ppm": rounded(a.drift_ppm, 3),
                "drift_at_end_ms": rounded(
                    None if common_duration is None
                    else a.drift_ppm * 1e-6 * common_duration * 1000, 3
                ),
                "correlation_pairs_used": a.pairs_used,
                "same_clock_as": [str(alignments[k].media.path) for k in a.locked_with],
                "correlation_rms_residual_ms": rounded(
                    None if a.rms_residual_seconds is None else a.rms_residual_seconds * 1000, 4
                ),
                "beep_seconds": rounded(a.beep_seconds),
                "trim_start_seconds": rounded(a.trim_start_seconds),
            }
        )
    report = {
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": settings,
        "common_duration_seconds": rounded(common_duration),
        "correlation_window_starts_after_beep_seconds": [round(t, 3) for t in correlation_times],
        "inputs": inputs,
        "pairs": [
            {
                "first": str(alignments[p.first].media.path),
                "second": str(alignments[p.second].media.path),
                "offset_ms": round(p.offset_seconds * 1000, 4),
                "drift_ppm": round(p.drift_ppm, 3),
                "windows": p.windows,
                "consistent_windows": p.inliers,
                "window_scatter_ms": round(p.rms_residual_seconds * 1000, 4),
                "accepted": p.accepted,
                "residual_ms": rounded(
                    None if p.residual_seconds is None else p.residual_seconds * 1000, 4
                ),
                "note": p.note,
            }
            for p in pairs
        ],
        "check_wav": check_wav,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main


def parse_manual(values: list[str], media: list[Media]) -> dict[int, float]:
    result: dict[int, float] = {}
    for value in values:
        key, separator, seconds = value.rpartition("=")
        if not separator:
            raise SyncError(f"--manual-beep は FILE=SECONDS の形式です: {value}")
        matches = [
            m.position
            for m in media
            if key in {str(m.path), m.path.name, m.label, str(m.position + 1)}
        ]
        if len(matches) != 1:
            raise SyncError(f"--manual-beep の対象を一意に特定できません: {key}")
        result[matches[0]] = float(seconds)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="同期用の機械音と会話音声を手がかりに、複数の映像・音声を同期して切り出します。"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="同じ会話のメディアファイル")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="出力先ディレクトリ")
    parser.add_argument("--analyze-only", action="store_true", help="解析とreport.jsonだけを行う")
    parser.add_argument(
        "--fast-copy", action="store_true", help="映像を再エンコードせず切り出す（キーフレーム精度）"
    )
    parser.add_argument(
        "--video-encoder",
        choices=["auto", *VIDEO_ENCODERS],
        default="auto",
        help="映像のエンコーダ。auto=使えるハードウェアのHEVC（Mac/NVIDIA/Intel/AMD）、なければlibx264",
    )
    parser.add_argument(
        "--video-bitrate",
        default=None,
        help="映像のビットレート（例: 20M）。標準はHEVC 20M / H.264 30M、libx264はCRF 18",
    )
    parser.add_argument("--beep-frequency", type=float, default=1318.51, help="同期音の基本周波数Hz")
    parser.add_argument("--search-start", type=float, default=0.0, help="探索開始秒")
    parser.add_argument("--search-seconds", type=float, default=600.0, help="探索する長さ（秒）")
    parser.add_argument("--pre-roll", type=float, default=0.05, help="同期音の何秒前から残すか")
    parser.add_argument("--corr-window", type=float, default=20.0, help="相関区間の長さ（秒）")
    parser.add_argument("--corr-interval", type=float, default=60.0, help="相関区間の間隔（秒）")
    parser.add_argument(
        "--max-lag-ms", type=float, default=20.0, help="同期音位置での相関探索範囲（±ms）"
    )
    parser.add_argument(
        "--max-drift-ppm", type=float, default=200.0, help="想定するクロックずれの上限（ppm）"
    )
    parser.add_argument(
        "--timecode-tolerance", type=float, default=0.5, help="タイムコードと同期音の許容差（秒）"
    )
    parser.add_argument(
        "--manual-beep",
        action="append",
        default=[],
        metavar="FILE=SECONDS",
        help="人が確認した同期音の秒数（ファイル名・パス・入力番号で指定、複数可）",
    )
    parser.add_argument("--ffmpeg", default=None, help="ffmpegのパス（標準は同梱のffmpeg、なければPATH）")
    parser.add_argument("--ffprobe", default=None, help="ffprobeのパス")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.pre_roll < 0:
        raise SyncError("--pre-roll は0以上にしてください。")

    ffmpeg = check_binary(args.ffmpeg or find_tool("ffmpeg"), "ffmpeg")
    ffprobe = check_binary(args.ffprobe or default_ffprobe(ffmpeg), "ffprobe")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [p.expanduser().resolve() for p in args.inputs]
    for path in paths:
        if not path.is_file():
            raise SyncError(f"入力ファイルが見つかりません: {path}")
    if len(set(paths)) != len(paths):
        raise SyncError("同じファイルが複数回指定されています。")

    media = [probe_media(path, index, ffprobe) for index, path in enumerate(paths)]
    assign_labels_and_outputs(media, output_dir)
    for item in media:
        if item.output_path == item.path:
            raise SyncError(f"出力が入力を上書きします: {item.path}")
    manual = parse_manual(args.manual_beep, media)
    alignments = [Alignment(m) for m in media]

    # 1. Rough beep position per file.
    for alignment in alignments:
        m = alignment.media
        if m.position in manual:
            alignment.coarse_seconds = manual[m.position]
            alignment.confidence = "manual"
            alignment.source = "manual"
            continue
        eprint(f"同期音を探索中: {m.path.name}")
        try:
            onset, details = locate_beep(
                m,
                ffmpeg,
                start=args.search_start,
                length=args.search_seconds,
                frequency_hz=args.beep_frequency,
            )
        except SyncError as error:
            eprint(f"  検出失敗: {error}")
            continue
        alignment.coarse_seconds = onset
        alignment.coarse_details = details
        alignment.confidence = confidence_label(details)
        alignment.source = "beep"

    # 2. Timecode cross-check / guidance.
    apply_timecode(
        alignments,
        ffmpeg,
        tolerance=args.timecode_tolerance,
        frequency_hz=args.beep_frequency,
    )
    missing = [a.media.path.name for a in alignments if a.coarse_seconds is None]
    if missing:
        raise SyncError(
            "同期音を見つけられないファイルがあります: "
            + ", ".join(missing)
            + "\n--manual-beep FILE=SECONDS で位置を指定してください。"
        )
    for alignment in alignments:
        refine_on_best_channel(alignment, ffmpeg, frequency_hz=args.beep_frequency)

    # 3. Pairwise correlation and joint solution.
    eprint("相互相関を計算中")
    pairs, times = correlate_all(
        alignments,
        ffmpeg,
        window=args.corr_window,
        interval=args.corr_interval,
        max_lag=args.max_lag_ms / 1000.0,
        timecode_max_lag=0.25,
        max_drift_ppm=args.max_drift_ppm,
        fit_tolerance=0.002,
        outlier_tolerance=0.001,
    )

    for alignment in alignments:
        assert alignment.onset_seconds is not None
        alignment.beep_seconds = alignment.onset_seconds + alignment.correction_seconds
        alignment.trim_start_seconds = alignment.beep_seconds - args.pre_roll
    too_early = [a.media.path.name for a in alignments if (a.trim_start_seconds or 0) < 0]
    common_duration = min(
        a.media.duration - (a.trim_start_seconds or 0.0) for a in alignments
    )

    settings = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
        if k not in {"inputs", "ffmpeg", "ffprobe"}
    }
    report_path = output_dir / REPORT_NAME
    write_report(
        report_path, alignments, pairs,
        settings=settings, common_duration=common_duration,
        correlation_times=times, check_wav=None,
    )  # fmt: skip

    for a in alignments:
        print(
            f"{a.media.label:>24}  beep={a.beep_seconds:12.6f}s  "
            f"corr={a.correction_seconds * 1000:+8.3f}ms  "
            f"drift={a.drift_ppm:+7.2f}ppm  pairs={a.pairs_used}  "
            f"confidence={a.confidence}  tc={a.timecode_status}"
        )
    spread = (max(a.drift_ppm for a in alignments) - min(a.drift_ppm for a in alignments))
    eprint(
        f"注意: クロックのずれは補正していません。終端では最大で約"
        f"{spread * 1e-6 * common_duration * 1000:.1f} ms ずれます。"
    )
    review = [
        a.media.path.name
        for a in alignments
        if a.confidence == "timecode-only" or a.pairs_used == 0
    ]
    if review:
        eprint("警告: 次のファイルは同期結果を確認してください: " + ", ".join(review))
    if too_early:
        raise SyncError(
            "同期音がファイル先頭に近すぎて --pre-roll 分を確保できません: "
            + ", ".join(too_early)
        )
    if common_duration <= 0:
        raise SyncError("全ファイルに共通する区間がありません。")

    if args.analyze_only:
        print(f"レポート: {report_path}")
        return 0

    video_encoder = args.video_encoder
    videos = [a.media for a in alignments if a.media.has_video]
    if videos and not args.fast_copy and video_encoder == "auto":
        video_encoder = pick_hardware_encoder(ffmpeg, "hevc", 1920, 1080) or "libx264"
        eprint(f"映像エンコーダ: {video_encoder}")
    eprint(f"共通区間: {common_duration:.3f}秒")
    for alignment in alignments:
        eprint(f"切り出し中: {alignment.media.path.name} -> {alignment.media.output_path}")
        write_synced_file(
            alignment.media,
            alignment.trim_start_seconds or 0.0,
            common_duration,
            ffmpeg=ffmpeg,
            fast_copy=args.fast_copy,
            video_encoder=video_encoder,
            video_bitrate=args.video_bitrate,
        )
    eprint("確認用WAVを作成中")
    check_path = output_dir / CHECK_WAV_NAME
    rate, channel_map = write_check_wav(alignments, common_duration, check_path, ffmpeg=ffmpeg)
    write_report(
        report_path, alignments, pairs,
        settings=settings,
        common_duration=common_duration,
        correlation_times=times,
        check_wav={"path": str(check_path), "sample_rate": rate, "channels": channel_map},
    )  # fmt: skip
    print(f"完了: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SyncError as exc:
        eprint(f"エラー: {exc}")
        raise SystemExit(2)
