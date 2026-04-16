from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ecom_qa.data.catalog import load_products
from ecom_qa.data.qa_generation import (
    ServerConfig,
    build_preview_requests,
    build_result_record,
    build_run_summary,
    generate_one_qa,
    run_llama_server,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small local QA preview.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/qa/generated"))
    parser.add_argument("--multimodal-count", type=int, default=2)
    parser.add_argument("--text-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260416)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()

    products = load_products(args.dataset_dir)
    requests = build_preview_requests(
        products,
        dataset_dir=args.dataset_dir,
        multimodal_count=args.multimodal_count,
        text_count=args.text_count,
        seed=args.seed,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"qwen35_preview_{timestamp}"
    run_dir = args.output_dir / run_name
    qa_path = run_dir / "qa_pairs.jsonl"
    summary_path = run_dir / "run_summary.json"
    server_log_path = run_dir / "llama_server.log"

    records: list[dict[str, object]] = []
    with run_llama_server(ServerConfig(), log_path=server_log_path) as startup_seconds:
        for request in requests:
            qa, elapsed_seconds = generate_one_qa(request, ServerConfig())
            records.append(build_result_record(request, qa, elapsed_seconds))

    total_wall_seconds = time.perf_counter() - started_at
    summary = build_run_summary(
        records,
        multimodal_count=args.multimodal_count,
        text_count=args.text_count,
        startup_seconds=startup_seconds,
        total_wall_seconds=total_wall_seconds,
    )
    summary["run_name"] = run_name
    summary["qa_path"] = str(qa_path)
    summary["server_log_path"] = str(server_log_path)

    write_jsonl(qa_path, records)
    write_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
