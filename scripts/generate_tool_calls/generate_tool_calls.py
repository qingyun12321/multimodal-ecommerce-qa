from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ecom_qa.data.qa_generation import ServerConfig
from ecom_qa.data.tool_call_generation import (
    SOURCES,
    load_source_rows,
    make_sampling_plan,
    run_generation,
    sample_requests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VLM tool-call annotations with local llama.cpp.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCES),
        default=sorted(SOURCES),
        help="Unified VQA sources to include.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Balanced total sample count across the selected sources.",
    )
    parser.add_argument(
        "--per-source-limit",
        type=int,
        default=None,
        help="Fixed sample count to draw from each selected source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/tool_call/generated"),
        help="Parent directory for generation runs.",
    )
    parser.add_argument("--seed", type=int, default=20260421)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = load_source_rows(args.sources)
    plan = make_sampling_plan(
        source_rows,
        count=args.count,
        per_source_limit=args.per_source_limit,
        seed=args.seed,
    )
    requests = sample_requests(source_rows, plan=plan, seed=args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"qwen35_tool_call_{plan.total}_{timestamp}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    _, failed_rows, summary = run_generation(requests, config=ServerConfig(), run_dir=run_dir)
    summary["run_name"] = run_name
    summary["seed"] = args.seed
    summary["sources"] = args.sources
    summary["requested_total"] = plan.total
    summary["failed_count"] = len(failed_rows)
    summary["run_dir"] = str(run_dir)
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
