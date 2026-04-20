from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ecom_qa.data.catalog import load_products
from ecom_qa.data.q_generation import (
    ServerConfig,
    build_preview_requests,
    build_result_record,
    build_run_summary,
    generate_one_qa,
    run_llama_server,
    write_json,
    write_jsonl,
)


MULTIMODAL_COUNT = 4000
TEXT_COUNT = 1000


def render_progress(current: int, total: int, succeeded: int, failed: int) -> None:
    width = 30
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    percent = (current / total * 100) if total else 100.0
    print(
        f"\r[{bar}] {current}/{total} ({percent:5.1f}%) success={succeeded} failed={failed}",
        end="",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 5000 QA pairs with the local Qwen3.5 setup.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/qa/generated"))
    parser.add_argument("--seed", type=int, default=20260416)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()

    products = load_products(args.dataset_dir)
    requests = build_preview_requests(
        products,
        dataset_dir=args.dataset_dir,
        multimodal_count=MULTIMODAL_COUNT,
        text_count=TEXT_COUNT,
        seed=args.seed,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"qwen35_qa_5000_{timestamp}"
    run_dir = args.output_dir / run_name
    qa_path = run_dir / "qa_pairs.jsonl"
    summary_path = run_dir / "run_summary.json"
    server_log_path = run_dir / "llama_server.log"
    failed_path = run_dir / "failed_rows.jsonl"

    records: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    total = len(requests)
    render_progress(0, total, 0, 0)
    with run_llama_server(ServerConfig(), log_path=server_log_path) as startup_seconds:
        for index, request in enumerate(requests, start=1):
            try:
                qa, elapsed_seconds = generate_one_qa(request, ServerConfig())
            except Exception as exc:
                failed_rows.append(
                    {
                        "qa_type": request.qa_type,
                        "source_row_index": request.source_row_index,
                        "product_id": request.product.id,
                        "image_path": request.product.image_path,
                        "source_title": request.product.title,
                        "error": str(exc),
                    }
                )
                render_progress(index, total, len(records), len(failed_rows))
                continue
            records.append(build_result_record(request, qa, elapsed_seconds))
            render_progress(index, total, len(records), len(failed_rows))
    print()

    total_wall_seconds = time.perf_counter() - started_at
    summary = build_run_summary(
        records,
        multimodal_count=MULTIMODAL_COUNT,
        text_count=TEXT_COUNT,
        startup_seconds=startup_seconds,
        total_wall_seconds=total_wall_seconds,
    )
    summary["run_name"] = run_name
    summary["requested_count"] = MULTIMODAL_COUNT + TEXT_COUNT
    summary["succeeded_count"] = len(records)
    summary["failed_count"] = len(failed_rows)
    summary["qa_path"] = str(qa_path)
    summary["server_log_path"] = str(server_log_path)
    if failed_rows:
        summary["failed_rows_path"] = str(failed_path)

    write_jsonl(qa_path, records)
    if failed_rows:
        write_jsonl(failed_path, failed_rows)
    write_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
