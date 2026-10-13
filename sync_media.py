#!/usr/bin/env python3
"""Synchronize recordings of one conversation using a known acoustic beep.

Every input is treated alike.  Each file is aligned in three steps:

1. the beep is located roughly (how far its three harmonics stand out from
   nearby frequencies, preferring candidates after a quiet moment, then an
   onset threshold) on the channels that keep time with the majority;
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
from fractions import Fraction
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
# Frequencies (relative to each harmonic) whose median level is the local
# background in tonal_scores(): outside a 50 ms block's main lobe, and inside
# a speech harmonic spacing so that voices raise them as much as the harmonic.
NEIGHBOUR_OFFSETS_HZ = (-240.0, -180.0, -120.0, 120.0, 180.0, 240.0)
# The beep follows a quiet moment.  Candidates whose preceding 0.05-0.25 s is
# this much quieter than the beep keep their full score; louder ones lose up
# to PRE_SILENCE_PENALTY of it.  A preference, not a requirement.
PRE_SILENCE_DB = -12.0
PRE_SILENCE_PENALTY = 0.4
# The beep is near the start of a recording (except later files of a split
# one, which are placed by correlation).  In the first search of a file, a
# candidate at the end of the searched span loses this much of its score,
# linearly from none at its start.  Also a preference only.
EARLY_PREFERENCE_PENALTY = 0.2
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
    duration: float | None = None


@dataclass
class Media:
    position: int  # 0-based input order
    path: Path
    duration: float
    has_video: bool
    fps: float | None
    audio: list[AudioStream]
    video_size: tuple[int, int] | None
    timecode: str | None
    timecode_seconds: float | None
    label: str = ""
    output_path: Path | None = None
    video_offset: float = 0.0
    video_duration: float | None = None

    @property
    def main_audio(self) -> AudioStream:
        return self.audio[0]


@dataclass
class Segment:
    """A later file of a split recording (``A1.MP4+A2.MP4``).

    Positions are on the output timeline in seconds, 0 at the beep.  A piece
    of a recording is drawn at ``out_start`` and, like every output, is not
    drift-corrected: an event at mean time ``t`` appears at
    ``t + drift * (t - anchor)``.
    """

    media: Media
    out_start: float | None = None
    anchor: float = 0.0
    gap_seconds: float | None = None  # silence / black inserted before this file
    method: str = "none"  # contiguous / correlation / timecode-only / manual
    reference: str | None = None
    timecode_estimate: float | None = None  # out_start predicted from timecode
    coarse_peak_ratio: float | None = None
    windows: int = 0
    inliers: int = 0
    rms_residual_seconds: float | None = None
    measured_drift_ppm: float | None = None


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
    channel_lags_ms: dict[int, float] = field(default_factory=dict)  # 1-based, vs majority
    onset_seconds: float | None = None  # beep onset before correlation
    timecode_beep_seconds: float | None = None
    timecode_status: str = "unavailable"
    correction_seconds: float = 0.0
    drift_ppm: float = 0.0  # clock rate relative to the mean of all files
    drift_known: bool = True  # False when no pair had enough windows to fit a slope
    beep_seconds: float | None = None
    trim_start_seconds: float | None = None
    rms_residual_seconds: float | None = None
    pairs_used: int = 0
    locked_with: list[int] = field(default_factory=list)  # same-clock files
    segments: list[Segment] = field(default_factory=list)  # later files of a split recording

    def pieces(self) -> list[tuple[Media, float, float]]:
        """``(media, out_start, anchor)`` of every placed file, in recording order."""
        assert self.beep_seconds is not None
        placed = [(self.media, -self.beep_seconds, 0.0)]
        for segment in self.segments:
            if segment.out_start is None:
                break
            placed.append((segment.media, segment.out_start, segment.anchor))
        return placed

    @property
    def out_end(self) -> float:
        media, start, _ = self.pieces()[-1]
        return start + media.duration


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
    drift_known: bool = True  # False with fewer than 3 windows (no slope fitted)
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
            "stream=index,codec_type,codec_name,sample_rate,channels,r_frame_rate,width,height,"
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
            duration=_float_or(s.get("duration"), 0.0) or None,
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
        video_size=(int(video[0]["width"]), int(video[0]["height"])) if video else None,
        timecode=timecode,
        timecode_seconds=parse_timecode(timecode, fps) if timecode else None,
        video_offset=(
            max(0.0, _float_or(video[0].get("start_time"), format_start) - format_start)
            if video else 0.0
        ),
        video_duration=(_float_or(video[0].get("duration"), 0.0) or None) if video else None,
    )


def output_suffix(media: Media) -> str:
    """Synced files are MP4 when they have video and WAV otherwise, whatever the source."""
    return ".mp4" if media.has_video else ".wav"


LOSSLESS_CODECS = {"flac", "alac", "wavpack", "tta", "ape", "mlp", "truehd"}


def wav_codec(media: Media) -> str:
    """PCM codec for a WAV output: keep PCM as it is (little-endian), else 24 / 16 bit."""
    codec = media.main_audio.codec
    if codec.startswith("pcm_"):
        return codec.replace("be", "le") if codec.endswith("be") else codec
    return "pcm_s24le" if codec in LOSSLESS_CODECS else "pcm_s16le"


def unique_stems(paths: list[Path]) -> list[str]:
    """Output stems that cannot collide although every output is .mp4 or .wav.

    Files that share a stem get what tells them apart: the folder name in
    front (different folders), the source extension behind (same folder), or
    both; a counter is the last resort.
    """
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.stem.lower(), []).append(path)
    names = []
    for path in paths:
        group = groups[path.stem.lower()]
        name = path.stem
        if len(group) > 1:
            folders = [p.parent.name for p in group]
            suffixes = [p.suffix.lower() for p in group]
            if len(set(folders)) == len(group):
                name = f"{path.parent.name}_{name}"
            elif len(set(suffixes)) == len(group):
                name = f"{name}_{path.suffix.lstrip('.')}"
            else:
                name = f"{path.parent.name}_{name}_{path.suffix.lstrip('.')}"
        names.append(name)
    used: set[str] = set()
    result = []
    for name in names:
        candidate, counter = name, 2
        while candidate.lower() in used:
            candidate, counter = f"{name}_{counter}", counter + 1
        used.add(candidate.lower())
        result.append(candidate)
    return result


def assign_labels_and_outputs(media: list[Media], output_dir: Path) -> None:
    """Name outputs ``<stem>_sync.mp4`` / ``.wav`` (see ``unique_stems``)."""
    for item, stem in zip(media, unique_stems([m.path for m in media])):
        item.label = stem
        item.output_path = output_dir / f"{stem}_sync{output_suffix(item)}"


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
    if frequencies[-1] + max(NEIGHBOUR_OFFSETS_HZ) >= sample_rate / 2:
        raise SyncError(f"サンプルレート{sample_rate}Hzでは第3倍音を解析できません。")
    return frequencies


def tonal_scores(blocks: np.ndarray, sample_rate: int, frequency_hz: float) -> np.ndarray:
    """How clearly each block holds the beep's three harmonics.

    Each harmonic's amplitude is divided by the median amplitude at nearby
    frequencies, and the three ratios are combined by geometric mean.  Speech,
    clatter and other broadband sounds are loud at the neighbours as well, and
    a pure tone near the fundamental lacks the harmonics, so both score low
    however loud they are.  The ratios are not weighted: speakers and rooms
    change the harmonics' balance a lot (recorded beeps often have a stronger
    second harmonic than fundamental).
    """
    frequencies = beep_frequencies(frequency_hz, sample_rate)
    probes = [
        f + offset for f in frequencies for offset in (0.0, *NEIGHBOUR_OFFSETS_HZ)
    ]
    amplitudes = harmonic_amplitudes(blocks, sample_rate, tuple(probes))
    amplitudes = amplitudes.reshape(blocks.shape[0], len(frequencies), -1)
    floor = 1e-7 * max(float(np.max(amplitudes)), 1e-30)
    peaks = amplitudes[:, :, 0] + floor
    neighbours = np.median(amplitudes[:, :, 1:], axis=2) + floor
    ratios = np.clip(peaks / neighbours, 1.0, 1000.0)
    return np.exp(np.mean(np.log(ratios), axis=1))


def detect_beep(
    samples: np.ndarray,
    sample_rate: int,
    *,
    search_start_seconds: float,
    frequency_hz: float,
    prefer_early: bool = False,
) -> tuple[float, dict[str, float]]:
    if samples.size < sample_rate:
        raise SyncError("解析対象が短すぎます（1秒未満）。")

    frequencies = beep_frequencies(frequency_hz, sample_rate)
    weights = np.asarray(HARMONIC_WEIGHTS, dtype=np.float64)

    coarse_block_seconds = 0.050
    coarse_block = max(32, round(coarse_block_seconds * sample_rate))
    block_count = samples.size // coarse_block
    blocks = samples[: block_count * coarse_block].reshape(block_count, coarse_block)
    score = tonal_scores(blocks, sample_rate, frequency_hz)
    loudness = np.sqrt(np.mean(np.square(blocks, dtype=np.float64), axis=1)) + 1e-12

    # Averaging over 0.4 s favours the ~0.8 s beep over short tonal clicks.
    smooth_blocks = max(1, round(0.4 / coarse_block_seconds))
    smooth = np.convolve(score, np.ones(smooth_blocks) / smooth_blocks, mode="same")

    def onset_block(index: int) -> int:
        """First block of the tone around a smoothed peak."""
        low = max(0, index - 3 * smooth_blocks // 2)
        high = min(score.size, index + smooth_blocks // 2 + 1)
        local = score[low:high]
        return low + int(np.argmax(local >= 0.5 * float(local.max())))

    def pre_silence_weight(index: int) -> float:
        start = onset_block(index)
        before = loudness[max(0, start - 5) : max(0, start - 1)]
        if before.size == 0:
            return 1.0
        during = float(np.median(loudness[start : start + 6]))
        level_db = 20.0 * math.log10(float(before.max()) / during)
        excess = min(1.0, max(0.0, (level_db - PRE_SILENCE_DB) / -PRE_SILENCE_DB))
        return 1.0 - PRE_SILENCE_PENALTY * excess

    # Several candidates, then a preference for those that follow a quiet
    # moment (and, in a whole-file search, for early ones).
    exclusion = max(1, round(1.5 / coarse_block_seconds))
    remaining = smooth.copy()
    candidates: list[tuple[float, int]] = []
    while len(candidates) < 8 and np.any(np.isfinite(remaining)):
        index = int(np.nanargmax(remaining))
        weight = pre_silence_weight(index)
        if prefer_early:
            weight *= 1.0 - EARLY_PREFERENCE_PENALTY * index / max(1, smooth.size - 1)
        candidates.append((float(smooth[index]) * weight, index))
        remaining[max(0, index - exclusion) : index + exclusion + 1] = np.nan
    candidates.sort(reverse=True)
    peak_value, peak_index = candidates[0]
    second_value = candidates[1][0] if len(candidates) > 1 else 0.0
    peak_time_local = (peak_index + 0.5) * coarse_block_seconds
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
    # At least -40 dB from the peak: very quiet recordings (phones with noise
    # suppression) otherwise trigger on faint noise well before the beep.
    threshold = median + max(12.0 * robust_sigma, 0.01 * dynamic_range)
    threshold = min(threshold, median + 0.25 * dynamic_range)

    # The beep lasts much longer than 50 ms; brief noises do not.
    run_length, required = 50, 45
    onset_frame: int | None = None
    search_limit = peak_time_local + 0.10
    for index in range(0, fine_score.size - run_length + 1):
        if centers_local[index] > search_limit:
            break
        if np.count_nonzero(fine_score[index : index + run_length] > threshold) >= required:
            # The run may start with a few frames still below the threshold.
            onset_frame = index + int(np.argmax(fine_score[index : index + run_length] > threshold))
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
    prefer_early: bool = False,
) -> tuple[float, dict[str, float]]:
    """Find the beep in a mono mixdown of ``[start, start + length)``."""
    length = min(length, media.duration - start)
    if length <= 1.0:
        raise SyncError("探索範囲がファイルの外です。")
    samples, rate, actual_start = decode_audio(
        media, ffmpeg, start=start, duration=length, rate=DETECTION_RATE, mix=True
    )
    return detect_beep(
        samples[:, 0], rate, search_start_seconds=actual_start, frequency_hz=frequency_hz,
        prefer_early=prefer_early,
    )


def channel_majority(
    samples: np.ndarray, rate: int, channels: list[int], tolerance: float = 0.001
) -> tuple[list[int], dict[int, float]]:
    """Channels that are in time with each other, measured by GCC-PHAT between them.

    Some devices add a processed channel with extra latency (for example the
    ReSpeaker array's channel 1 is ~60 ms late).  Each channel's lag against
    every other is measured in four sub-windows; a lag counts only when the
    sub-windows agree.  Returns the largest group of channels within
    ``tolerance`` of each other (all of ``channels`` when no group is larger
    than the rest, e.g. two disagreeing channels) and each channel's lag
    against the group's first member.
    """
    if len(channels) < 2:
        return channels, {}
    pieces = 4
    length = samples.shape[0] // pieces
    nfft = 1 << (2 * length - 1).bit_length()
    taper = tukey(length, 0.1)
    spectra = {
        c: [
            np.fft.rfft(samples[k * length : (k + 1) * length, c].astype(np.float64) * taper, nfft)
            for k in range(pieces)
        ]
        for c in channels
    }
    lag: dict[tuple[int, int], float] = {}
    for i, first in enumerate(channels):
        for second in channels[i + 1 :]:
            values = [
                gcc_phat(spectra[first][k], spectra[second][k], nfft, rate, 0.2)
                for k in range(pieces)
            ]
            if max(values) - min(values) <= tolerance:
                lag[first, second] = float(np.median(values))
                lag[second, first] = -lag[first, second]

    def together(a: int, b: int) -> bool:
        return abs(lag.get((a, b), math.inf)) <= tolerance

    groups = [[c for c in channels if c == seed or together(seed, c)] for seed in channels]
    groups.sort(key=len, reverse=True)
    best = groups[0]
    if any(len(g) == len(best) and set(g) != set(best) for g in groups[1:]):
        return channels, {}
    lags = {c: lag[best[0], c] for c in channels if (best[0], c) in lag}
    return best, {best[0]: 0.0, **lags}


def refine_on_best_channel(
    alignment: Alignment, ffmpeg: str, *, frequency_hz: float
) -> None:
    """Choose the analysis channel and re-detect the beep onset there.

    Only channels that hear the beep and keep time with the majority of the
    file's channels are used (see channel_majority()); among them the onset
    is measured on each, channels that disagree with the others' median are
    skipped, and the clearest remaining one is used.
    """
    media = alignment.media
    assert alignment.coarse_seconds is not None
    start = max(0.0, alignment.coarse_seconds - 3.0)
    samples, rate, actual_start = decode_audio(media, ffmpeg, start=start, duration=6.0)
    try:
        beep_frequencies(frequency_hz, rate)
    except SyncError:
        samples, rate, actual_start = decode_audio(
            media, ffmpeg, start=start, duration=6.0, rate=DETECTION_RATE
        )
    block = round(0.05 * rate)
    count = samples.shape[0] // block
    scores: list[float] = []
    for channel in range(samples.shape[1]):
        blocks = samples[: count * block, channel].reshape(count, block)
        score = tonal_scores(blocks, rate, frequency_hz)
        scores.append(float(np.max(score) / max(float(np.median(score)), 1e-12)))
    alignment.channel = int(np.argmax(scores))
    alignment.channel_scores = [round(s, 2) for s in scores]
    alignment.onset_seconds = alignment.coarse_seconds
    if alignment.source == "timecode":
        return

    hearing = [c for c, s in enumerate(scores) if s >= 0.25 * max(scores)]
    if len(hearing) > 1:
        # Conversation after the beep: broadband, unlike the tone itself.
        after, after_rate, _ = decode_audio(
            media, ffmpeg, start=alignment.coarse_seconds + 2.0, duration=20.0,
            rate=DETECTION_RATE,
        )  # fmt: skip
        levels = np.sqrt(np.mean(np.square(after, dtype=np.float64), axis=0))
        hearing = [c for c in hearing if levels[c] >= 1e-3 * float(levels.max())]
        majority, lags = channel_majority(after, after_rate, hearing)
        alignment.channel_lags_ms = {c + 1: round(v * 1000, 3) for c, v in lags.items()}
        hearing = majority
    alignment.channel = max(hearing, key=lambda c: scores[c])

    onsets: dict[int, float] = {}
    for channel in hearing:
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
    """Window start times, in seconds after each file's beep onset.

    They cover the longest file; a pair is only compared in windows that both
    files reach (the first file of a split recording is often short).
    """
    span = max(
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
        spectra: list[np.ndarray | None] = []
        for alignment in alignments:
            assert alignment.onset_seconds is not None
            if alignment.onset_seconds + t + window > alignment.media.duration:
                spectra.append(None)
                continue
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
                if spectra[i] is None or spectra[j] is None:
                    continue
                limit = max_lag
                if "timecode" in (alignments[i].source, alignments[j].source):
                    limit = max(limit, timecode_max_lag)
                limit += max_drift_ppm * 1e-6 * t
                lags[step, i, j] = gcc_phat(spectra[i], spectra[j], nfft, rate, limit)

    time_array = np.asarray(times)
    pairs: list[PairResult] = []
    for i in range(count):
        for j in range(i + 1, count):
            valid = np.isfinite(lags[:, i, j])
            measured = int(np.count_nonzero(valid))
            if measured == 0:
                pairs.append(PairResult(i, j, 0.0, 0.0, 0, 0, 0.0, False, note="no-overlap"))
                continue
            pair_times, pair_lags = time_array[valid], lags[valid, i, j]
            drift_known = measured >= 3
            if drift_known:
                intercept, slope, inliers = fit_line(pair_times, pair_lags, fit_tolerance)
            else:
                # A line through two points always fits: with so few windows
                # (a short file) assume no drift and require the windows to agree.
                intercept, slope = float(np.median(pair_lags)), 0.0
                inliers = np.abs(pair_lags - intercept) <= fit_tolerance
            residuals = pair_lags[inliers] - (intercept + slope * pair_times[inliers])
            rms = float(np.sqrt(np.mean(np.square(residuals))))
            used = int(np.count_nonzero(inliers))
            enough = used >= min(3, measured) and used >= 0.5 * measured
            if not drift_known:
                enough = used == measured
            accepted = enough and rms <= fit_tolerance
            note = "" if accepted else ("few-consistent-windows" if not enough else "scattered")
            pairs.append(
                PairResult(
                    first=i,
                    second=j,
                    offset_seconds=intercept,
                    drift_ppm=slope * 1e6,
                    windows=measured,
                    inliers=used,
                    rms_residual_seconds=rms,
                    accepted=accepted,
                    drift_known=drift_known,
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
    drift_pairs = [p for p in pairs if p.drift_known]
    drifts = solve_corrections(count, drift_pairs, [True] * count, "drift_ppm", drift_locks)

    for alignment in alignments:
        position = alignment.media.position
        alignment.correction_seconds = float(corrections[position])
        alignment.drift_ppm = float(drifts[position])
        alignment.drift_known = any(
            p.drift_known and p.accepted and position in (p.first, p.second) for p in pairs
        )
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
# Split recordings


def decode_timeline(
    alignment: Alignment, ffmpeg: str, start: float, duration: float, rate: int
) -> tuple[np.ndarray, float]:
    """The recording's analysis channel on the output timeline ``[start, start + duration)``.

    Gaps between placed files are zeros.  Returns ``(samples, covered fraction)``.
    """
    length = round(duration * rate)
    output = np.zeros(length)
    covered = 0
    for media, out_start, _ in alignment.pieces():
        first = max(start, out_start)
        last = min(start + duration, out_start + media.duration)
        if last - first < 1.0 / rate:
            continue
        samples, _, actual = decode_audio(
            media, ffmpeg, start=first - out_start, duration=last - first, rate=rate
        )
        mono = samples[:, alignment.channel].astype(np.float64)
        index = round((actual + out_start - start) * rate)
        mono = mono[: max(0, length - index)]
        output[index : index + mono.size] = mono
        covered += mono.size
    return output, covered / max(1, length)


def mean_time(out_time: float, drift_ppm: float, anchor: float) -> float:
    """Undo ``out = t + drift * (t - anchor)`` for one recording."""
    drift = drift_ppm * 1e-6
    return (out_time + drift * anchor) / (1.0 + drift)


def anchor_at(alignment: Alignment, out_time: float) -> float:
    """Anchor of the recording's file that is drawn at ``out_time``."""
    anchor = 0.0
    for _, start, piece_anchor in alignment.pieces():
        if start <= out_time:
            anchor = piece_anchor
    return anchor


def previous_end(alignment: Alignment, index: int) -> tuple[float, float, float]:
    """``(out end, anchor, mean-time end)`` of the file before ``segments[index]``."""
    media, start, anchor = alignment.pieces()[index]
    end = start + media.duration
    return end, anchor, mean_time(end, alignment.drift_ppm, anchor)


def set_segment_gap(
    alignment: Alignment, index: int, gap: float, method: str, tolerance: float
) -> None:
    """Place ``segments[index]`` ``gap`` seconds (mean time) after the previous file."""
    segment = alignment.segments[index]
    end, anchor, end_mean = previous_end(alignment, index)
    if abs(gap) <= tolerance:
        # Seamless: continue the previous file, as if it had never been split.
        segment.out_start, segment.anchor, segment.gap_seconds = end, anchor, 0.0
        segment.method = "manual" if method == "manual" else "contiguous"
    else:
        segment.out_start = segment.anchor = end_mean + gap
        segment.gap_seconds = gap
        segment.method = method


def coarse_search(
    query: np.ndarray,
    reference: Alignment,
    ffmpeg: str,
    lowest: float,
    highest: float,
    query_offset: float,
    rate: int,
    block_seconds: float = 600.0,
) -> tuple[float, float] | None:
    """Where ``query`` (a file's audio from ``query_offset`` s) starts on the reference's timeline.

    Searches file starts in ``[lowest, highest]`` with GCC-PHAT on a low sample
    rate, block by block.  Returns ``(out_start, peak-to-second ratio)``.
    """
    query = query - query.mean()
    peaks: list[tuple[float, float]] = []  # (value, out_start)
    block_start = lowest
    while block_start <= highest:
        block_end = min(highest, block_start + block_seconds)
        span = block_end - block_start
        ref, covered = decode_timeline(
            reference, ffmpeg, block_start + query_offset, span + query.size / rate, rate
        )
        if covered > 0.0 and np.any(ref):
            nfft = 1 << (ref.size + query.size).bit_length()
            cross = np.conj(np.fft.rfft(query, nfft)) * np.fft.rfft(ref, nfft)
            cross /= np.maximum(np.abs(cross), 1e-12 * max(float(np.abs(cross).max()), 1e-30))
            cross[np.fft.rfftfreq(nfft, 1.0 / rate) < 80.0] = 0.0
            values = np.fft.irfft(cross, nfft)[: round(span * rate) + 1]
            # Best and runner-up outside a +-0.25 s neighbourhood, per block.
            for _ in range(2):
                index = int(np.argmax(values))
                peaks.append((float(values[index]), block_start + index / rate))
                guard = round(0.25 * rate)
                values[max(0, index - guard) : index + guard + 1] = -np.inf
        block_start = block_end + 1.0 / rate
        if block_end >= highest:
            break
    if not peaks:
        return None
    peaks.sort(reverse=True)
    best_value, best_start = peaks[0]
    runner_up = next((v for v, t in peaks[1:] if abs(t - best_start) > 0.25), 0.0)
    return best_start, best_value / max(runner_up, 1e-12)


def relocate_by_conversation(
    alignment: Alignment,
    references: list[Alignment],
    ffmpeg: str,
    *,
    frequency_hz: float,
    coarse_rate: int = 2000,
) -> bool:
    """Look for the beep again where the file's conversation says it must be.

    Used for a file that no correlation pair confirmed: its beep position was
    probably wrong by more than the correlation search range.  A minute of
    its audio is searched for on the whole timeline of each confirmed
    reference; the beep is then sought within 2 s of the predicted position.
    """
    media = alignment.media
    query_offset = min(300.0, max(0.0, media.duration - 60.0))
    query_length = min(60.0, media.duration - query_offset)
    if query_length < 10.0:
        return False
    samples, _, _ = decode_audio(
        media, ffmpeg, start=query_offset, duration=query_length, rate=coarse_rate
    )
    query = samples[:, alignment.channel].astype(np.float64)
    # Audio-only recordings first: decoding them is far cheaper than a 4 GB video.
    for reference in sorted(references, key=lambda r: r.media.has_video):
        lowest = reference.pieces()[0][1] - query_offset
        highest = reference.out_end - query_offset - query_length
        if highest <= lowest:
            continue
        found = coarse_search(query, reference, ffmpeg, lowest, highest, query_offset, coarse_rate)
        if found is None or found[1] < 1.5:
            continue
        # The output timeline is 0 at the beep, so the file's beep is at -start.
        expected = -found[0]
        if not 0.0 <= expected < media.duration:
            continue
        try:
            onset, details = locate_beep(
                media, ffmpeg, start=max(0.0, expected - 2.0), length=4.0,
                frequency_hz=frequency_hz,
            )  # fmt: skip
        except SyncError:
            onset = None
        if onset is not None and abs(onset - expected) <= 1.0:
            alignment.coarse_seconds, alignment.coarse_details = onset, details
            alignment.confidence, alignment.source = "conversation-guided", "beep"
        else:
            alignment.coarse_seconds = expected
            alignment.confidence, alignment.source = "conversation-only", "conversation"
        eprint(
            f"  {media.path.name}: {reference.media.label}との会話の相関から"
            f"同期音を{alignment.coarse_seconds:.3f}秒に置き直しました"
        )
        refine_on_best_channel(alignment, ffmpeg, frequency_hz=frequency_hz)
        return True
    return False


def fine_alignment(
    segment: Segment,
    channel: int,
    reference: Alignment,
    estimate: float,
    ffmpeg: str,
    *,
    window: float,
    interval: float,
    max_lag: float,
    max_drift_ppm: float,
    fit_tolerance: float,
) -> tuple[float, float, int, int, float] | None:
    """Refine ``estimate`` with windowed GCC-PHAT over the whole file.

    Returns ``(out_start on the reference timeline, drift vs reference in ppm,
    windows, inliers, rms)`` or ``None`` when too few windows agree.
    """
    rate = CORRELATION_RATE
    media = segment.media
    window = min(window, max(5.0, media.duration - 2.0))
    length = round(window * rate)
    nfft = 1 << (2 * length - 1).bit_length()
    taper = tukey(length, 0.1)
    starts = list(np.arange(2.0, media.duration - window, interval)) or [0.0]
    times, lags = [], []
    for t in starts:
        if t + window > media.duration:
            continue
        ref, covered = decode_timeline(reference, ffmpeg, estimate + t, window, rate)
        if covered < 0.9:
            continue
        samples, _, _ = decode_audio(media, ffmpeg, start=t, duration=window, rate=rate)
        mono = samples[:length, channel].astype(np.float64)
        mono = np.pad(mono, (0, length - mono.size))
        lag = gcc_phat(
            np.fft.rfft(ref[:length] * taper, nfft),
            np.fft.rfft(mono * taper, nfft),
            nfft, rate, max_lag + max_drift_ppm * 1e-6 * t,
        )  # fmt: skip
        times.append(t)
        lags.append(lag)
    if not times:
        return None
    time_array, lag_array = np.asarray(times), np.asarray(lags)
    intercept, slope, inliers = fit_line(time_array, lag_array, fit_tolerance)
    used = int(np.count_nonzero(inliers))
    residuals = lag_array[inliers] - (intercept + slope * time_array[inliers])
    rms = float(np.sqrt(np.mean(np.square(residuals)))) if used else math.inf
    if used < min(3, len(times)) or used < 0.5 * len(times) or rms > fit_tolerance:
        return None
    # A positive lag means the sound comes later in the file than predicted; a
    # growing lag means the file's clock runs fast against the reference.
    return estimate - intercept, slope * 1e6, len(times), used, rms


def place_segment(
    alignment: Alignment,
    index: int,
    references: list[Alignment],
    ffmpeg: str,
    *,
    contiguous_tolerance: float,
    coarse_rate: int,
    timecode_window: float,
    window: float,
    interval: float,
    max_drift_ppm: float,
) -> bool:
    """Place ``alignment.segments[index]`` by correlation with ``references``.

    False when no reference covers the file's time yet.
    """
    segment = alignment.segments[index]
    media = segment.media
    # A beep-only placement can be wrong even with a strong tone detection.
    # Prefer references validated by the conversation correlation graph.
    verified = [r for r in references if r.pairs_used > 0]
    if verified:
        references = verified
    previous_media, previous_start, _ = alignment.pieces()[index]
    end, _, end_mean = previous_end(alignment, index)
    if media.timecode_seconds is not None and previous_media.timecode_seconds is not None:
        # Same camera, same clock: the timecode gap is the recording gap.
        elapsed = (media.timecode_seconds - previous_media.timecode_seconds) % 86400.0
        segment.timecode_estimate = previous_start + elapsed

    def fall_back(reason: str) -> bool:
        if segment.timecode_estimate is None:
            raise SyncError(
                f"{reason}: {media.path.name}\n"
                "--segment-gap FILE=SECONDS で前のファイルとの間隔を指定してください（連続なら0）。"
            )
        set_segment_gap(
            alignment, index, segment.timecode_estimate - end_mean, "timecode-only",
            contiguous_tolerance,
        )  # fmt: skip
        return True

    query_offset = min(2.0, max(0.0, media.duration - 1.0))
    query_length = min(30.0, media.duration - query_offset)
    if query_length < 3.0:
        return fall_back("短すぎて相関で位置を決められません")
    samples, _, _ = decode_audio(
        media, ffmpeg, start=query_offset, duration=query_length, rate=coarse_rate
    )
    query = samples[:, alignment.channel].astype(np.float64)

    if segment.timecode_estimate is not None:
        lowest = segment.timecode_estimate - timecode_window
        highest = segment.timecode_estimate + timecode_window
    else:
        lowest = end - 1.0
        highest = max((r.out_end for r in references), default=lowest)
    # Audio-only recordings first: decoding them is far cheaper than a 4 GB video.
    candidates: list[tuple[float, float, Alignment]] = []
    for reference in sorted(references, key=lambda r: r.media.has_video):
        reach_low = max(lowest, reference.pieces()[0][1] - query_offset)
        reach_high = min(highest, reference.out_end - query_offset - query_length)
        if reach_high <= reach_low:
            continue
        found = coarse_search(
            query, reference, ffmpeg, reach_low, reach_high, query_offset, coarse_rate
        )
        if found:
            candidates.append((found[0], found[1], reference))
    if not candidates:
        return False

    refined = None
    for estimate, ratio, reference in sorted(candidates, key=lambda item: item[1], reverse=True):
        if ratio < 1.5:
            continue
        refined = fine_alignment(
            segment, alignment.channel, reference, estimate, ffmpeg,
            window=window, interval=interval, max_lag=0.02,
            max_drift_ppm=max_drift_ppm, fit_tolerance=0.002,
        )  # fmt: skip
        if refined is not None:
            break
        eprint(f"  {media.path.name}: {reference.media.label}との精密相関が不成立、別の参照を試します")
    if refined is None:
        return fall_back("相関で位置を決められません")
    segment.coarse_peak_ratio = round(ratio, 2)
    start_on_reference, drift_vs_reference, windows, inliers, rms = refined
    # Express the start in mean time: the reference itself drifts.
    start = mean_time(
        start_on_reference, reference.drift_ppm, anchor_at(reference, start_on_reference)
    )
    segment.reference = reference.media.label
    segment.windows, segment.inliers, segment.rms_residual_seconds = windows, inliers, rms
    if windows >= 3:
        segment.measured_drift_ppm = round(drift_vs_reference + reference.drift_ppm, 3)
    set_segment_gap(alignment, index, start - end_mean, "correlation", contiguous_tolerance)
    return True


def place_segments(
    alignments: list[Alignment],
    manual_gaps: dict[Path, float],
    ffmpeg: str,
    *,
    contiguous_tolerance: float,
    **options: Any,
) -> None:
    """Place every later file of every split recording.

    Each file is matched against the other recordings placed so far, so files
    are taken in whatever order references become available.
    """
    pending = [(a, k) for a in alignments for k in range(len(a.segments))]
    while pending:
        progressed = False
        for alignment, index in list(pending):
            if index > 0 and alignment.segments[index - 1].out_start is None:
                continue
            segment = alignment.segments[index]
            if segment.media.path in manual_gaps:
                set_segment_gap(
                    alignment, index, manual_gaps[segment.media.path], "manual",
                    contiguous_tolerance,
                )  # fmt: skip
            else:
                eprint(f"分割ファイルの位置を推定中: {segment.media.path.name}")
                references = [a for a in alignments if a is not alignment]
                if not place_segment(
                    alignment, index, references, ffmpeg,
                    contiguous_tolerance=contiguous_tolerance, **options,
                ):  # fmt: skip
                    continue
            pending.remove((alignment, index))
            progressed = True
            eprint(
                f"  {segment.media.path.name}: {segment.method}、"
                f"前のファイルとの間隔 {segment.gap_seconds or 0.0:.3f}秒"
            )
        if not progressed:
            # Nothing else recorded that time: only timecode (or the user) can help.
            alignment, index = pending[0]
            segment = alignment.segments[index]
            previous_media, previous_start, _ = alignment.pieces()[index]
            if segment.media.timecode_seconds is None or previous_media.timecode_seconds is None:
                raise SyncError(
                    "同じ時間を録っているほかの機材がないため、位置を決められません: "
                    f"{segment.media.path.name}\n--segment-gap FILE=SECONDS で指定してください。"
                )
            elapsed = (segment.media.timecode_seconds - previous_media.timecode_seconds) % 86400.0
            segment.timecode_estimate = previous_start + elapsed
            _, _, end_mean = previous_end(alignment, index)
            set_segment_gap(
                alignment, index, segment.timecode_estimate - end_mean, "timecode-only",
                contiguous_tolerance,
            )  # fmt: skip
            pending.remove((alignment, index))
            eprint(f"  {segment.media.path.name}: timecode-only（相関の相手がいません）")


# ---------------------------------------------------------------------------
# Output


def output_layout(
    alignment: Alignment, pre_roll: float, duration: float, *, stream: str | int | None = None
) -> list[tuple[int | None, float, float, float]]:
    """Pieces of the output ``[0, duration)``: ``(file index or None for a gap, from, to, file time at from)``.

    A file is cut where the next one starts.  Times are output seconds.
    """
    pieces = alignment.pieces()
    layout: list[tuple[int | None, float, float, float]] = []
    cursor = 0.0

    def bounds(media: Media, start: float) -> tuple[float, float]:
        if stream == "video":
            offset, length = media.video_offset, media.video_duration
        elif isinstance(stream, int):
            audio = media.audio[stream]
            offset, length = audio.offset, audio.duration
        else:
            offset, length = 0.0, None
        return start + offset + pre_roll, length if length is not None else media.duration - offset

    for index, (media, out_start, _) in enumerate(pieces):
        track_start, track_duration = bounds(media, out_start)
        begin = max(cursor, track_start)
        end = min(duration, track_start + track_duration)
        if index + 1 < len(pieces):
            next_media, next_start, _ = pieces[index + 1]
            end = min(end, bounds(next_media, next_start)[0])
        if end <= begin:
            continue
        if begin > cursor:
            layout.append((None, cursor, begin, 0.0))
        layout.append((index, begin, end, begin - track_start))
        cursor = end
    if cursor < duration:
        layout.append((None, cursor, duration, 0.0))
    return layout


def channel_layout_name(channels: int) -> str:
    return {1: "mono", 2: "stereo"}.get(channels, f"{channels}c")


def stitch_filters(
    layout: list[tuple[int | None, float, float, float]],
    first_input: int,
    label: str,
    *,
    stream: int | None,
    rate: float,
    channels: int = 0,
    size: tuple[int, int] | None = None,
) -> list[str]:
    """Filters that join the files of one recording into ``[label]``.

    ``stream`` is an audio stream number (``rate`` = sample rate, gaps are
    silence) or ``None`` for video (``rate`` = frame rate, gaps are black).
    Piece boundaries are rounded on the output grid, so they never accumulate.
    """
    filters, parts = [], []
    exact = Fraction(rate).limit_denominator(1001)  # 29.97... -> 30000/1001
    for number, (index, begin, end, file_start) in enumerate(layout):
        count = round(end * rate) - round(begin * rate)
        if count <= 0:
            continue
        part = f"{label}_{number}"
        if stream is None:
            assert size is not None
            if index is None:
                source = f"color=c=black:s={size[0]}x{size[1]}:r={exact}"
            else:
                source = (
                    f"[{first_input + index}:v:0]setpts=PTS-STARTPTS,trim=start={file_start:.9f},"
                    f"setpts=PTS-STARTPTS,fps={exact},tpad=stop=-1:stop_mode=add"
                )
            filters.append(
                f"{source},trim=end_frame={count},setpts=PTS-STARTPTS,"
                f"scale={size[0]}:{size[1]},format=yuv420p,setsar=1[{part}]"
            )
        else:
            name = channel_layout_name(channels)
            if index is None:
                routing = "|".join(f"c{c}=0*c0" for c in range(channels))
                source = f"anullsrc=r={round(rate)}:cl=mono,pan={name}|{routing}"
            else:
                routing = "|".join(f"c{c}=c{c}" for c in range(channels))
                source = (
                    f"[{first_input + index}:a:{stream}]asetpts=PTS-STARTPTS,atrim=start={file_start:.9f},"
                    f"asetpts=PTS-STARTPTS,aresample={round(rate)},pan={name}|{routing},apad"
                )
            filters.append(f"{source},atrim=end_sample={count},asetpts=PTS-STARTPTS[{part}]")
        parts.append(f"[{part}]")
    if len(parts) == 1:
        filters[-1] = filters[-1].rsplit("[", 1)[0] + f"[{label}]"
    else:
        kind = "v=1:a=0" if stream is None else "v=0:a=1"
        filters.append("".join(parts) + f"concat=n={len(parts)}:{kind}[{label}]")
    return filters


def write_stitched_file(
    alignment: Alignment,
    pre_roll: float,
    duration: float,
    *,
    ffmpeg: str,
    video_encoder: str,
    video_bitrate: str | None,
) -> None:
    """Write a split recording as one file, gaps filled with black and silence."""
    media = alignment.media
    assert media.output_path is not None
    files = [m for m, _, _ in alignment.pieces()]
    command = [ffmpeg, "-y", "-v", "error", "-stats"]
    for item in files:
        if media.has_video and video_encoder.endswith("_videotoolbox"):
            command += ["-hwaccel", "videotoolbox"]
        command += ["-i", str(item.path)]
    filters: list[str] = []
    maps: list[str] = []
    if media.has_video:
        assert media.fps and media.video_size
        filters += stitch_filters(
            output_layout(alignment, pre_roll, duration, stream="video"), 0, "v",
            stream=None, rate=media.fps, size=media.video_size,
        )
        maps += ["-map", "[v]"]
    for number, stream in enumerate(media.audio):
        filters += stitch_filters(
            output_layout(alignment, pre_roll, duration, stream=number), 0, f"a{number}",
            stream=number, rate=stream.sample_rate,
            channels=stream.channels,
        )  # fmt: skip
        maps += ["-map", f"[a{number}]"]
    command += ["-filter_complex", ";".join(filters), *maps]
    if media.has_video:
        options, default_bitrate = VIDEO_ENCODERS[video_encoder]
        command += ["-c:v", video_encoder, *options]
        bitrate = video_bitrate or default_bitrate
        if bitrate:
            command += ["-b:v", bitrate]
        command += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
    else:
        command += ["-c:a", wav_codec(media), "-rf64", "auto"]
    command.append(str(media.output_path))
    run_checked(command)


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
            # The video is copied; the audio becomes AAC, since MP4 cannot hold
            # every source codec (PCM from MOV, for example).
            command += [
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
            ]  # fmt: skip
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
        command += ["-c:a", wav_codec(media), "-rf64", "auto"]
    command.append(str(media.output_path))
    run_checked(command)


def write_check_wav(
    alignments: list[Alignment],
    duration: float,
    destination: Path,
    *,
    ffmpeg: str,
    pre_roll: float,
) -> tuple[int, list[dict[str, Any]]]:
    rate = max(s.sample_rate for a in alignments for s in a.media.audio)
    command = [ffmpeg, "-y", "-v", "error", "-stats"]
    filters: list[str] = []
    labels: list[str] = []
    channel_map: list[dict[str, Any]] = []
    next_channel = 1
    next_input = 0
    for recording, alignment in enumerate(alignments):
        media = alignment.media
        files = [m for m, _, _ in alignment.pieces()]
        for item in files:
            command += ["-i", str(item.path)]
        for stream_order, stream in enumerate(media.audio):
            label = f"a{recording}_{stream_order}"
            # "Nc" layouts (not 2.0, 5.1, ...) so that amerge just stacks channels.
            filters += [
                f.replace(f"pan={channel_layout_name(stream.channels)}|", f"pan={stream.channels}c|")
                for f in stitch_filters(
                    output_layout(alignment, pre_roll, duration, stream=stream_order),
                    next_input, label, stream=stream_order, rate=rate,
                    channels=stream.channels,
                )
            ]  # fmt: skip
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
        next_input += len(files)
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


def json_safe(value: Any) -> Any:
    """NaN and infinity become null, so that report.json is standard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


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
    pre_roll = float(settings.get("pre_roll", 0.0))
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
                "channel_lags_ms": a.channel_lags_ms,
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
                "drift_ppm": rounded(a.drift_ppm, 3) if a.drift_known else None,
                "drift_at_end_ms": rounded(
                    None if common_duration is None or not a.drift_known
                    else a.drift_ppm * 1e-6 * common_duration * 1000, 3
                ),
                "correlation_pairs_used": a.pairs_used,
                "same_clock_as": [str(alignments[k].media.path) for k in a.locked_with],
                "correlation_rms_residual_ms": rounded(
                    None if a.rms_residual_seconds is None else a.rms_residual_seconds * 1000, 4
                ),
                "beep_seconds": rounded(a.beep_seconds),
                "trim_start_seconds": rounded(a.trim_start_seconds),
                "segments": [
                    {
                        "input": str(seg.media.path),
                        "duration_seconds": rounded(seg.media.duration),
                        # Where the file starts in the output; the gap before it is filled.
                        "output_start_seconds": rounded(
                            None if seg.out_start is None else seg.out_start + pre_roll
                        ),
                        "gap_before_seconds": rounded(seg.gap_seconds),
                        "method": seg.method,
                        "reference": seg.reference,
                        "timecode_output_start_seconds": rounded(
                            None if seg.timecode_estimate is None else seg.timecode_estimate + pre_roll
                        ),
                        "coarse_peak_ratio": seg.coarse_peak_ratio,
                        "windows": seg.windows,
                        "consistent_windows": seg.inliers,
                        "rms_residual_ms": rounded(
                            None if seg.rms_residual_seconds is None
                            else seg.rms_residual_seconds * 1000, 4
                        ),
                        "measured_drift_ppm": seg.measured_drift_ppm,
                    }
                    for seg in a.segments
                ],
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
                "drift_ppm": round(p.drift_ppm, 3) if p.drift_known else None,
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
    path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def split_input(text: str) -> list[Path]:
    """``A1.MP4+A2.MP4`` -> the files of one split recording, in order."""
    whole = Path(text).expanduser()
    if whole.is_file():  # a file name that itself contains "+"
        return [whole.resolve()]
    parts = [Path(part).expanduser().resolve() for part in text.split("+")]
    for part in parts:
        if not part.is_file():
            raise SyncError(f"入力ファイルが見つかりません: {part}")
    return parts


def check_segment_format(primary: Media, later: Media) -> None:
    """Files of one recording must be joinable without conversion."""
    same = (
        later.has_video == primary.has_video
        and len(later.audio) == len(primary.audio)
        and all(
            (a.channels, a.sample_rate) == (b.channels, b.sample_rate)
            for a, b in zip(later.audio, primary.audio)
        )
        and (not primary.has_video or later.video_size == primary.video_size)
    )
    if not same:
        raise SyncError(
            f"つなげるファイルの形式（映像の有無・大きさ、音声のチャンネル数・サンプルレート）が違います: "
            f"{primary.path.name} / {later.path.name}"
        )


def parse_segment_gaps(values: list[str], alignments: list[Alignment]) -> dict[Path, float]:
    later = [s.media for a in alignments for s in a.segments]
    result: dict[Path, float] = {}
    for value in values:
        key, separator, seconds = value.rpartition("=")
        if not separator:
            raise SyncError(f"--segment-gap は FILE=SECONDS の形式です: {value}")
        matches = [m.path for m in later if key in {str(m.path), m.path.name}]
        if len(matches) != 1:
            raise SyncError(
                f"--segment-gap の対象を一意に特定できません（分割された2つ目以降のファイルを指定）: {key}"
            )
        result[matches[0]] = float(seconds)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="同期用の機械音と会話音声を手がかりに、複数の映像・音声を同期して切り出します。"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="同じ会話のメディアファイル。分割されたファイルは録画順に + でつなぐ（例: A1.MP4+A2.MP4）",
    )
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="出力先ディレクトリ")
    parser.add_argument("--analyze-only", action="store_true", help="解析とreport.jsonだけを行う")
    parser.add_argument(
        "--duration-mode", choices=["shortest", "longest"], default="shortest",
        help="出力の長さ。shortest=最短の入力、longest=最長の入力（不足分は黒画面・無音）",
    )
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
    parser.add_argument(
        "--segment-gap",
        action="append",
        default=[],
        metavar="FILE=SECONDS",
        help="分割された2つ目以降のファイルについて、前のファイルとの間隔（秒）を指定する。連続なら0",
    )
    parser.add_argument(
        "--contiguous-tolerance",
        type=float,
        default=0.1,
        help="分割ファイルの間隔がこの秒数以内なら、連続（隙間なし）として扱う",
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

    groups = [split_input(text) for text in args.inputs]
    every_path = [p for group in groups for p in group]
    if len(set(every_path)) != len(every_path):
        raise SyncError("同じファイルが複数回指定されています。")
    if args.fast_copy and any(len(group) > 1 for group in groups):
        raise SyncError("--fast-copy は分割ファイル（+ でつないだ入力）には使えません。")
    if args.fast_copy and args.duration_mode == "longest":
        raise SyncError("--duration-mode longest は黒画面を生成するため --fast-copy と併用できません。")

    media = [probe_media(group[0], index, ffprobe) for index, group in enumerate(groups)]
    assign_labels_and_outputs(media, output_dir)
    for item in media:
        if item.output_path in every_path:
            raise SyncError(f"出力が入力を上書きします: {item.output_path}")
    manual = parse_manual(args.manual_beep, media)
    alignments = [Alignment(m) for m in media]
    for alignment, group in zip(alignments, groups):
        for path in group[1:]:
            later = probe_media(path, alignment.media.position, ffprobe)
            check_segment_format(alignment.media, later)
            alignment.segments.append(Segment(later))
    segment_gaps = parse_segment_gaps(args.segment_gap, alignments)

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
                prefer_early=True,
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
    def correlate() -> tuple[list[PairResult], list[float]]:
        found = correlate_all(
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
        return found

    eprint("相互相関を計算中")
    pairs, times = correlate()
    # A file no pair confirmed probably had its beep misplaced by more than the
    # correlation range: place it by the conversation and correlate again.
    unconfirmed = [a for a in alignments if a.pairs_used == 0 and a.source != "manual"]
    confirmed = [a for a in alignments if a.pairs_used > 0]
    if unconfirmed and confirmed:
        moved = False
        for alignment in unconfirmed:
            eprint(f"相関が取れないため、会話の位置から同期音を探し直します: {alignment.media.path.name}")
            moved |= relocate_by_conversation(
                alignment, confirmed, ffmpeg, frequency_hz=args.beep_frequency
            )
        if moved:
            eprint("相互相関を再計算中")
            pairs, times = correlate()
    too_early = [a.media.path.name for a in alignments if (a.trim_start_seconds or 0) < 0]

    # 4. Later files of split recordings.
    if any(a.segments for a in alignments):
        place_segments(
            alignments, segment_gaps, ffmpeg,
            contiguous_tolerance=args.contiguous_tolerance,
            coarse_rate=2000, timecode_window=3.0,
            window=args.corr_window, interval=args.corr_interval,
            max_drift_ppm=args.max_drift_ppm,
        )  # fmt: skip
    choose_duration = max if args.duration_mode == "longest" else min
    common_duration = choose_duration(a.out_end + args.pre_roll for a in alignments)

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
            + (f"drift={a.drift_ppm:+7.2f}ppm  " if a.drift_known else "drift=   不明     ")
            + f"pairs={a.pairs_used}  "
            f"confidence={a.confidence}  tc={a.timecode_status}"
        )
    spread = (max(a.drift_ppm for a in alignments) - min(a.drift_ppm for a in alignments))
    eprint(
        f"注意: クロックのずれは補正していません。終端では最大で約"
        f"{spread * 1e-6 * common_duration * 1000:.1f} ms ずれます。"
    )
    for a in alignments:
        for segment in a.segments:
            assert segment.out_start is not None
            print(
                f"{'+ ' + segment.media.path.name:>24}  start={segment.out_start + args.pre_roll:12.6f}s  "
                f"gap={segment.gap_seconds or 0.0:9.3f}s  {segment.method}"
                + (f"  ref={segment.reference}" if segment.reference else "")
            )
    review = [
        a.media.path.name
        for a in alignments
        if a.confidence in {"timecode-only", "conversation-only"} or a.pairs_used == 0
    ] + [s.media.path.name for a in alignments for s in a.segments if s.method == "timecode-only"]
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
        if alignment.segments or args.duration_mode == "longest":
            write_stitched_file(
                alignment, args.pre_roll, common_duration,
                ffmpeg=ffmpeg, video_encoder=video_encoder, video_bitrate=args.video_bitrate,
            )  # fmt: skip
            continue
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
    rate, channel_map = write_check_wav(
        alignments, common_duration, check_path, ffmpeg=ffmpeg, pre_roll=args.pre_roll
    )
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
