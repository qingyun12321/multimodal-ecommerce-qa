from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/workspace/multimodal_ecommerce_qa")
DEFAULT_INPUT_JSON = ROOT / "reports/generated/official_multilingual_t2i_20260403T125235Z/combined_metrics.json"
DEFAULT_OUTPUT_IMAGE = ROOT / "official_multilingual_results_sample_style.jpg"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MODEL_ORDER = ["clip", "chinese_clip", "siglip2", "gme_2b"]
LANG_ORDER = ["zh", "en"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render benchmark results as a white table image."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-image", type=Path, default=DEFAULT_OUTPUT_IMAGE)
    parser.add_argument("--title", default="Experiment results:")
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def mean_precision(row: dict[str, float]) -> float:
    if "avg_precision_mean" in row:
        return float(row["avg_precision_mean"])
    keys = [
        "avg_precision@1",
        "avg_precision@3",
        "avg_precision@5",
        "avg_precision@10",
        "avg_precision@20",
        "avg_precision@50",
    ]
    return sum(float(row[key]) for key in keys) / len(keys)


def sort_key(row: dict[str, object]) -> tuple[int, int]:
    input_language = str(row.get("input_language", "en"))
    model_name = str(row["model_name"])
    return (
        LANG_ORDER.index(input_language) if input_language in LANG_ORDER else 999,
        MODEL_ORDER.index(model_name) if model_name in MODEL_ORDER else 999,
    )


def build_rows(payload: dict) -> list[list[str]]:
    source_rows = sorted(payload["combined_rows"], key=sort_key)
    rows = []
    for row in source_rows:
        rows.append(
            [
                str(row["model_name"]).replace("_", "-"),
                str(row.get("input_language", "en")),
                pct(float(row["avg_precision@1"])),
                pct(float(row["avg_precision@3"])),
                pct(float(row["avg_precision@5"])),
                pct(float(row["avg_precision@10"])),
                pct(float(row["avg_precision@20"])),
                pct(float(row["avg_precision@50"])),
                pct(mean_precision(row)),
            ]
        )
    return rows


def find_best_row_index(payload: dict) -> int:
    source_rows = sorted(payload["combined_rows"], key=sort_key)
    best_index = 0
    best_score = -1.0
    for idx, row in enumerate(source_rows):
        score = mean_precision(row)
        if score > best_score:
            best_score = score
            best_index = idx
    return best_index


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    rows = build_rows(payload)
    best_row_index = find_best_row_index(payload)
    headers = ["Model", "Input\nLang", "Top1", "Top3", "Top5", "Top10", "Top20", "Top50", "Average"]

    col_widths = [240, 110, 160, 160, 160, 170, 170, 170, 190]
    left = 80
    top = 60
    title_h = 60
    header_h = 120
    row_h = 74
    bottom = 40
    width = max(1650, left * 2 + sum(col_widths))
    table_w = width - left * 2
    height = top + title_h + header_h + row_h * len(rows) + bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = ImageFont.truetype(FONT_BOLD, 34)
    head_font = ImageFont.truetype(FONT_BOLD, 24)
    body_font = ImageFont.truetype(FONT_REGULAR, 22)
    body_bold_font = ImageFont.truetype(FONT_BOLD, 22)

    draw.text((left, top), args.title, fill=(45, 45, 45), font=title_font)

    y0 = top + title_h
    x_positions = [left]
    for width_part in col_widths:
        x_positions.append(x_positions[-1] + width_part)

    line_color = (225, 225, 225)
    header_bg = (255, 255, 255)
    winner_bg = (244, 251, 244)

    draw.rectangle([left, y0, left + table_w, y0 + header_h], fill=header_bg)

    total_rows_h = header_h + row_h * len(rows)
    for idx in range(len(rows) + 2):
        y = y0 + (0 if idx == 0 else header_h + row_h * (idx - 1))
        draw.line([(left, y), (left + table_w, y)], fill=line_color, width=2)
    draw.line([(left, y0 + total_rows_h), (left + table_w, y0 + total_rows_h)], fill=line_color, width=2)

    for x in x_positions:
        draw.line([(x, y0), (x, y0 + total_rows_h)], fill=line_color, width=2)

    for i, header in enumerate(headers):
        x1, x2 = x_positions[i], x_positions[i + 1]
        bbox = draw.multiline_textbbox((0, 0), header, font=head_font, spacing=4)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x1 + (x2 - x1 - tw) / 2
        ty = y0 + (header_h - th) / 2
        if i == 0:
            tx = x1 + 18
        draw.multiline_text((tx, ty), header, fill=(60, 60, 60), font=head_font, spacing=4, align="center")

    for r, row in enumerate(rows):
        y1 = y0 + header_h + r * row_h
        y2 = y1 + row_h
        is_best = r == best_row_index
        if is_best:
            draw.rectangle([left, y1, left + table_w, y2], fill=winner_bg)
            for x in x_positions:
                draw.line([(x, y1), (x, y2)], fill=line_color, width=2)
            draw.line([(left, y1), (left + table_w, y1)], fill=line_color, width=2)
            draw.line([(left, y2), (left + table_w, y2)], fill=line_color, width=2)

        for c, value in enumerate(row):
            x1, x2 = x_positions[c], x_positions[c + 1]
            font = body_bold_font if is_best else body_font
            if c == 0:
                bbox = draw.textbbox((0, 0), value, font=font)
                tx = x1 + 18
                ty = y1 + (row_h - (bbox[3] - bbox[1])) / 2
                draw.text((tx, ty), value, fill=(65, 65, 65), font=font)
            else:
                bbox = draw.textbbox((0, 0), value, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = x1 + (x2 - x1 - tw) / 2
                ty = y1 + (row_h - th) / 2
                draw.text((tx, ty), value, fill=(65, 65, 65), font=font)

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output_image, quality=95)
    print(args.output_image)


if __name__ == "__main__":
    main()
