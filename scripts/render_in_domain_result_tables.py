from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/workspace/multimodal_ecommerce_qa")
RUN_DIR = ROOT / "reports/generated/siglip2_in_domain_pipeline_20260403T155304Z"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render in-domain RAG result tables in the sample white-table style.")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def draw_table_image(
    *,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    output_path: Path,
    highlight_row: int | None = None,
) -> None:
    left = 80
    top = 60
    title_h = 60
    header_h = 120
    row_h = 74
    bottom = 40
    if len(headers) == 4:
        col_widths = [350, 260, 260, 260]
    elif len(headers) == 5:
        col_widths = [320, 220, 220, 220, 220]
    elif len(headers) == 7:
        col_widths = [260, 150, 150, 150, 150, 150, 170]
    else:
        width = max(1280, 180 + 170 * len(headers))
        table_w = width - left * 2
        first = 320
        rest = (table_w - first) // (len(headers) - 1)
        col_widths = [first] + [rest] * (len(headers) - 1)
        col_widths[-1] += table_w - sum(col_widths)
    width = max(1280, left * 2 + sum(col_widths))
    table_w = width - left * 2
    height = top + title_h + header_h + row_h * len(rows) + bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = ImageFont.truetype(FONT_BOLD, 34)
    head_font = ImageFont.truetype(FONT_BOLD, 24)
    body_font = ImageFont.truetype(FONT_REGULAR, 22)
    body_bold_font = ImageFont.truetype(FONT_BOLD, 22)

    draw.text((left, top), title, fill=(45, 45, 45), font=title_font)

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
        is_best = highlight_row is not None and r == highlight_row
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
                bbox = draw.multiline_textbbox((0, 0), value, font=font, spacing=4)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = x1 + (x2 - x1 - tw) / 2
                ty = y1 + (row_h - th) / 2
                draw.multiline_text((tx, ty), value, fill=(65, 65, 65), font=font, spacing=4, align="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def build_retrieval_rows(summary: dict[str, float]) -> list[list[str]]:
    keys = ["avg_precision@1", "avg_precision@3", "avg_precision@5", "avg_precision@10", "avg_precision@20", "avg_precision@50"]
    avg_mean = sum(float(summary[key]) for key in keys) / len(keys)
    return [[
        "siglip2",
        pct(float(summary["avg_precision@1"])),
        pct(float(summary["avg_precision@3"])),
        pct(float(summary["avg_precision@5"])),
        pct(float(summary["avg_precision@10"])),
        pct(float(summary["avg_precision@20"])),
        pct(float(summary["avg_precision@50"])),
        pct(avg_mean),
    ]]


def render_images(run_dir: Path) -> list[Path]:
    payload = json.loads((run_dir / "in_domain_summary.json").read_text(encoding="utf-8"))
    summary_metrics = payload["summary_metrics"]
    rerank_rows = read_csv_rows(run_dir / "rerank_summary.csv")
    gpt_rows = read_csv_rows(run_dir / "gpt54_eval_summary.csv")

    outputs: list[Path] = []

    image_to_image_path = run_dir / "siglip2_image_to_image_results.jpg"
    draw_table_image(
        title="Image-to-Image Experiment Results",
        headers=["Model", "Top1", "Top3", "Top5", "Top10", "Top20", "Top50", "Average"],
        rows=build_retrieval_rows(summary_metrics["image_to_image"]),
        output_path=image_to_image_path,
        highlight_row=0,
    )
    outputs.append(image_to_image_path)

    image_text_path = run_dir / "siglip2_image_text_to_image_results.jpg"
    draw_table_image(
        title="Image+Text-to-Image Experiment Results",
        headers=["Model", "Top1", "Top3", "Top5", "Top10", "Top20", "Top50", "Average"],
        rows=build_retrieval_rows(summary_metrics["image_text_to_image"]),
        output_path=image_text_path,
        highlight_row=0,
    )
    outputs.append(image_text_path)

    multi_route_rows = [
        [
            "multi-route recall",
            pct(float(summary_metrics["multi_route_top10"]["avg_precision@10"])),
            pct(float(summary_metrics["multi_route_top10"]["avg_hit@10"])),
        ],
        [
            "image+text baseline",
            pct(float(summary_metrics["image_text_top10_baseline"]["avg_precision@10"])),
            pct(float(summary_metrics["image_text_top10_baseline"]["avg_hit@10"])),
        ],
    ]
    multi_route_path = run_dir / "siglip2_multi_route_results.jpg"
    draw_table_image(
        title="Multi-Route Recall Experiment Results",
        headers=["Method", "Top10 Precision", "Top10 Hit"],
        rows=multi_route_rows,
        output_path=multi_route_path,
        highlight_row=0,
    )
    outputs.append(multi_route_path)

    gpt_by_method = {row["method_name"]: row for row in gpt_rows}
    rerank_table_rows: list[list[str]] = []
    for row in rerank_rows:
        method = row["method_name"]
        gpt_row = gpt_by_method[method]
        rerank_table_rows.append(
            [
                method.replace("_", "-"),
                pct(float(row["avg_top5_relevance_ratio"])),
                pct(float(row["avg_filtered_irrelevant_ratio"])),
                pct(float(gpt_row["avg_top5_relevant_ratio_gpt"])),
                pct(float(gpt_row["avg_filtered_correct_ratio_gpt"])),
            ]
        )

    best_index = max(range(len(rerank_table_rows)), key=lambda idx: float(gpt_rows[idx]["avg_top5_relevant_ratio_gpt"]))
    rerank_path = run_dir / "siglip2_rerank_filter_results.jpg"
    draw_table_image(
        title="Reranking and Filtering Experiment Results",
        headers=[
            "Method",
            "Label Top5\nRelevant",
            "Label Filtered\nIrrelevant",
            "GPT Top5\nRelevant",
            "GPT Filtered\nCorrect",
        ],
        rows=rerank_table_rows,
        output_path=rerank_path,
        highlight_row=best_index,
    )
    outputs.append(rerank_path)

    return outputs


def main() -> None:
    args = parse_args()
    outputs = render_images(args.run_dir)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
