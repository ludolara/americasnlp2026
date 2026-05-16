#!/usr/bin/env python3
"""Plot sentence-level chrF++ histograms for the SFT(MT)+RVLR(MT) translator.

This intentionally uses Pillow rather than matplotlib so it can run in the
current project environment without adding plotting dependencies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

# SFT(MT)+RVLR(MT) runs for each direction and language.
LANGUAGES = ("Wixárika", "Nahuatl", "Guaraní", "Bribri")
DIRECTION_RUNS = {
    "xxx→spa": {
        "Wixárika": ROOT
        / "results/xxx_to_spa/grpo/"
        "checkpoint-125__americasnlp2026__test__dir-xxx-to-spa__langs-hch__gb-100__53eea896d99e/"
        "records.json",
        "Nahuatl": ROOT
        / "results/xxx_to_spa/grpo/nah/"
        "checkpoint-125__americasnlp2026__test__dir-xxx-to-spa__langs-nah__gb-100__a902553d659f/"
        "records.json",
        "Guaraní": ROOT
        / "results/xxx_to_spa/grpo/gn/"
        "checkpoint-125__americasnlp2026__test__dir-xxx-to-spa__langs-gn__gb-100__badad8dac8b4/"
        "records.json",
        "Bribri": ROOT
        / "results/xxx_to_spa/grpo/"
        "checkpoint-125__americasnlp2026__test__dir-xxx-to-spa__langs-bzd__gb-100__9ae2317ded6b/"
        "records.json",
    },
    "spa→xxx": {
        "Wixárika": ROOT
        / "results/checkpoint-125__americasnlp2026__test__langs-hch__gb-100__1438b5fc2b2c/"
        "records.json",
        "Nahuatl": ROOT
        / "results/checkpoint-125__americasnlp2026__test__langs-nah__gb-100__415035618e2c/"
        "records.json",
        "Guaraní": ROOT
        / "results/checkpoint-125__americasnlp2026__test__langs-gn__gb-100__f56fe9fb9a59/"
        "records.json",
        "Bribri": ROOT
        / "results/checkpoint-125__americasnlp2026__test__langs-bzd__gb-100__7d0ac2a08e08/"
        "records.json",
    },
}

FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def load_scores(path: Path) -> list[float]:
    with path.open() as f:
        rows = json.load(f)
    return [float(row["chrf_pp"]) for row in rows]


def histogram(scores: list[float], width: int = 5) -> list[int]:
    bins = [0] * (100 // width)
    for score in scores:
        index = min(int(score // width), len(bins) - 1)
        bins[index] += 1
    return bins


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    scores: list[float],
    max_count: int,
    small_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box

    plot_left = left + 58
    plot_right = right - 18
    plot_top = top + 18
    plot_bottom = bottom - 46
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    counts = histogram(scores)

    # Horizontal grid and y labels.
    tick_step = 50
    y_ticks = list(range(0, max_count + tick_step, tick_step))
    for tick in y_ticks:
        y = plot_bottom - int((tick / max_count) * plot_h)
        draw.line((plot_left, y, plot_right, y), fill="#d9d9d9", width=1)
        label = str(tick)
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text(
            (plot_left - 10 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2),
            label,
            fill="#555555",
            font=small_font,
        )

    # Bars.
    bar_w = plot_w / len(counts)
    for i, count in enumerate(counts):
        x0 = plot_left + int(i * bar_w) + 1
        x1 = plot_left + int((i + 1) * bar_w) - 1
        y0 = plot_bottom - int((count / max_count) * plot_h)
        draw.rectangle((x0, y0, x1, plot_bottom), fill="#4C78A8")

    # Axes.
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#222222", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#222222", width=2)

    # x labels.
    for tick in range(0, 101, 20):
        x = plot_left + int((tick / 100) * plot_w)
        draw.line((x, plot_bottom, x, plot_bottom + 6), fill="#222222", width=2)
        label = str(tick)
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text(
            (x - (bbox[2] - bbox[0]) / 2, plot_bottom + 10),
            label,
            fill="#555555",
            font=small_font,
        )

def main() -> None:
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)

    scores_by_direction = {
        direction: {
            language: load_scores(path)
            for language, path in runs.items()
        }
        for direction, runs in DIRECTION_RUNS.items()
    }
    max_count = max(
        max(histogram(scores))
        for scores_by_language in scores_by_direction.values()
        for scores in scores_by_language.values()
    )
    max_count = int(math.ceil(max_count / 50) * 50)

    width, height = 3200, 1800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = font(40)
    header_font = font(30)
    label_font = font(26)
    small_font = font(22)

    title = "Distribution of Sentence-level chrF++ Scores on the Test Set"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((width - (title_bbox[2] - title_bbox[0])) / 2, 24),
        title,
        fill="#111111",
        font=title_font,
    )

    left_margin = 150
    right_margin = 40
    top_margin = 170
    bottom_margin = 120
    row_gap = 80
    col_gap = 28
    row_count = len(scores_by_direction)
    col_count = len(LANGUAGES)
    panel_w = (width - left_margin - right_margin - col_gap * (col_count - 1)) // col_count
    panel_h = (height - top_margin - bottom_margin - row_gap * (row_count - 1)) // row_count

    for col_index, language in enumerate(LANGUAGES):
        left = left_margin + col_index * (panel_w + col_gap)
        header_bbox = draw.textbbox((0, 0), language, font=header_font)
        header_x = left + (panel_w - (header_bbox[2] - header_bbox[0])) / 2
        draw.text((header_x, 108), language, fill="#1f1f1f", font=header_font)

    for row_index, (direction, scores_by_language) in enumerate(scores_by_direction.items()):
        top = top_margin + row_index * (panel_h + row_gap)
        row_label_bbox = draw.textbbox((0, 0), direction, font=header_font)
        row_label_img = Image.new(
            "RGBA",
            (
                row_label_bbox[2] - row_label_bbox[0] + 20,
                row_label_bbox[3] - row_label_bbox[1] + 20,
            ),
            (255, 255, 255, 0),
        )
        row_label_draw = ImageDraw.Draw(row_label_img)
        row_label_draw.text((10, 10), direction, fill="#1f1f1f", font=header_font)
        row_label_img = row_label_img.rotate(90, expand=True)
        image.paste(
            row_label_img,
            (
                52,
                top + (panel_h - row_label_img.height) // 2,
            ),
            row_label_img,
        )

        for col_index, language in enumerate(LANGUAGES):
            left = left_margin + col_index * (panel_w + col_gap)
            box = (left, top, left + panel_w, top + panel_h)
            draw_panel(
                draw,
                box,
                scores_by_language[language],
                max_count,
                small_font,
            )

    # Shared axis labels.
    xlabel = "Sentence-level chrF++"
    xlabel_bbox = draw.textbbox((0, 0), xlabel, font=label_font)
    draw.text(
        ((width - (xlabel_bbox[2] - xlabel_bbox[0])) / 2, height - 52),
        xlabel,
        fill="#111111",
        font=label_font,
    )

    ylabel = "Number of examples"
    ylabel_img = Image.new("RGBA", (420, 60), (255, 255, 255, 0))
    ylabel_draw = ImageDraw.Draw(ylabel_img)
    ylabel_draw.text((0, 0), ylabel, fill="#111111", font=label_font)
    ylabel_img = ylabel_img.rotate(90, expand=True)
    image.paste(ylabel_img, (18, (height - ylabel_img.height) // 2), ylabel_img)

    png_path = output_dir / "translation_chrfpp_histograms_sft_mt_rvlr_mt.png"
    pdf_path = output_dir / "translation_chrfpp_histograms_sft_mt_rvlr_mt.pdf"
    image.save(png_path, dpi=(300, 300))
    image.save(pdf_path, "PDF", resolution=300.0)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
