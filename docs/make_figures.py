#!/usr/bin/env python3
"""Regenerate the README figures in docs/images/.

The layout figures are real compose_media.py output: numbered test tiles are
turned into one-second clips and composed exactly as a user would.  Needs
Pillow (``pip install pillow``), ffmpeg and macOS's Hiragino font.

    python3 docs/make_figures.py
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"
FONT_BOLD = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_REGULAR = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
COLORS = ["#3B6FB6", "#E08A2E", "#3E9B5B", "#8A5BB0", "#C9483F"]
CELL = (320, 180)


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def tile(path: Path, number: int, size: tuple[int, int], caption: str) -> None:
    image = Image.new("RGB", size, COLORS[(number - 1) % len(COLORS)])
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.text((width / 2, height * 0.42), str(number), fill="white",
              font=font(int(height * 0.45)), anchor="mm")  # fmt: skip
    draw.text((width / 2, height * 0.8), caption, fill="white",
              font=font(int(height * 0.11), bold=False), anchor="mm")  # fmt: skip
    image.save(path)


def clip(image: Path, video: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(image), "-t", "1",
         "-r", "30", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(video)],
        check=True,
    )  # fmt: skip


def compose(videos: list[Path], layout: str, name: str, work: Path, cell: str) -> None:
    output = work / f"{name}.mp4"
    subprocess.run(
        [sys.executable, str(ROOT / "compose_media.py"), "--layout", layout,
         "-v", *map(str, videos), "--cell-size", cell, "--codec", "h264",
         "--video-bitrate", "2M", "--no-preview", "-o", str(output)],
        check=True, capture_output=True,
    )  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(output),
         "-frames:v", "1", "-update", "1", str(IMAGES / f"{name}.png")],
        check=True,
    )  # fmt: skip


def audio_figure(path: Path) -> None:
    width, height = 980, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    label, small = font(20), font(16, bold=False)
    inputs = [("MIC1.WAV", "1ch", COLORS[0]), ("MIC2.WAV", "1ch", COLORS[1]),
              ("camera.MP4", "2ch", COLORS[2])]  # fmt: skip

    def box(x: int, y: int, w: int, h: int, color: str, text: str, sub: str = "") -> None:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=color)
        draw.text((x + w / 2, y + h / 2 - (9 if sub else 0)), text, fill="white",
                  font=label, anchor="mm")  # fmt: skip
        if sub:
            draw.text((x + w / 2, y + h / 2 + 14), sub, fill="white", font=small, anchor="mm")

    def arrow(x1: int, y1: int, x2: int, y2: int) -> None:
        draw.line((x1, y1, x2, y2), fill="#555555", width=3)
        angle = math.atan2(y2 - y1, x2 - x1)
        head = [
            (x2 - 13 * math.cos(angle + side), y2 - 13 * math.sin(angle + side))
            for side in (-0.45, 0.45)
        ]
        draw.polygon([(x2, y2), *head], fill="#555555")

    for panel, (mode, title) in enumerate(
        [("mix", "--audio-mode mix（標準）"), ("multi", "--audio-mode multi")]
    ):
        left = 20 + panel * 490
        draw.text((left, 22), title, fill="#222222", font=label)
        draw.text((left, 52), "-a MIC1.WAV MIC2.WAV camera.MP4", fill="#555555", font=small)
        for row, (name, channels, color) in enumerate(inputs):
            y = 100 + row * 95
            box(left, y, 170, 70, color, name, channels)
        if mode == "mix":
            box(left + 290, 195, 160, 70, "#444444", "1ch", "全4chの平均")
            for row in range(3):
                arrow(left + 172, 135 + row * 95, left + 286, 230)
        else:
            outputs = [("ch1", COLORS[0]), ("ch2", COLORS[1]),
                       ("ch3", COLORS[2]), ("ch4", COLORS[2])]  # fmt: skip
            for index, (name, color) in enumerate(outputs):
                box(left + 290, 92 + index * 72, 160, 56, color, name)
            arrow(left + 172, 135, left + 286, 120)
            arrow(left + 172, 230, left + 286, 192)
            arrow(left + 172, 325, left + 286, 264)
            arrow(left + 172, 325, left + 286, 336)
    image.save(path)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as name:
        work = Path(name)
        numbered = []
        for number in range(1, 6):
            image = work / f"tile{number}.png"
            tile(image, number, (1280, 720), f"-v の{number}番目")
            clip(image, work / f"tile{number}.mp4")
            numbered.append(work / f"tile{number}.mp4")
        cell = f"{CELL[0]}x{CELL[1]}"
        compose(numbered[:3], "row", "layout_row", work, cell)
        compose(numbered[:3], "column", "layout_column", work, cell)
        compose(numbered, "grid", "layout_grid", work, cell)

        shapes = []
        for number, (size, caption) in enumerate(
            [((1280, 720), "16:9"), ((960, 720), "4:3"), ((720, 720), "1:1")], start=1
        ):
            image = work / f"shape{number}.png"
            tile(image, number, size, caption)
            clip(image, work / f"shape{number}.mp4")
            shapes.append(work / f"shape{number}.mp4")
        compose(shapes, "row", "cell_size", work, cell)
    audio_figure(IMAGES / "audio_mode.png")
    print(f"書き出し先: {IMAGES}")


if __name__ == "__main__":
    main()
