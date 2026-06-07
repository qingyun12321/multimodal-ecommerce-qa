from __future__ import annotations

import argparse
import gc
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

from ecom_qa.common.paths import data_path
from ecom_qa.tool_calls.generation import (
    DEFAULT_SOURCE_COUNTS,
    SOURCES,
    default_tool_call_server_config,
    load_source_rows,
    make_sampling_plan,
    run_generation,
    sample_requests,
)


def parse_source_counts(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"Expected NAME=COUNT, got {value!r}.")
        name, count_text = value.split("=", 1)
        name = name.strip()
        if name not in SOURCES:
            allowed = ", ".join(sorted(SOURCES))
            raise argparse.ArgumentTypeError(f"Unknown source {name!r}. Expected one of: {allowed}.")
        try:
            count = int(count_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid count in {value!r}.") from exc
        parsed[name] = count
    return parsed


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def repair_partial_jsonl(path: Path) -> None:
    if not path.exists():
        return

    offset = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            next_offset = offset + len(raw_line)
            line = raw_line.strip()
            if not line:
                offset = next_offset
                continue
            try:
                json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                backup_path = path.with_name(
                    f"{path.name}.broken_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                )
                shutil.copy2(path, backup_path)
                with path.open("r+b") as writable:
                    writable.truncate(offset)
                print(
                    f"Repaired partial JSONL line {line_number} in {path}; "
                    f"backup saved to {backup_path}."
                )
                return
            offset = next_offset


def request_key(source_name: str, record_key: str) -> tuple[str, str]:
    return source_name, record_key


def load_completed_keys(run_dir: Path) -> set[tuple[str, str]]:
    repair_partial_jsonl(run_dir / "tool_call_records.jsonl")
    repair_partial_jsonl(run_dir / "failed_rows.jsonl")

    completed: set[tuple[str, str]] = set()
    for row in iter_jsonl(run_dir / "tool_call_records.jsonl"):
        completed.add(request_key(str(row["source_dataset"]), str(row["source_record_key"])))
    for row in iter_jsonl(run_dir / "failed_rows.jsonl"):
        completed.add(request_key(str(row["source_dataset"]), str(row["source_record_key"])))
    return completed


def make_tool_ratio_table(
    *,
    generated_count: int,
    search_tool_distribution: Counter[str],
    grounding_distribution: Counter[bool],
) -> list[dict[str, object]]:
    rows = [
        ("直接回答", search_tool_distribution["none"]),
        ("RAG_search", search_tool_distribution["RAG_search"]),
        ("Web_search", search_tool_distribution["Web_search"]),
        ("图像裁剪", grounding_distribution[True]),
    ]
    return [
        {
            "tool_class": name,
            "count": count,
            "percent": round(count / generated_count * 100, 2) if generated_count else 0.0,
        }
        for name, count in rows
    ]


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def rebuild_run_summary_from_files(
    *,
    run_dir: Path,
    base_summary: dict[str, object],
    requested_counts: dict[str, int],
    total_requested: int,
    resume_mode: bool,
) -> dict[str, object]:
    generated_count = 0
    generation_seconds_total = 0.0
    retried_record_count = 0
    retry_attempts_total = 0
    search_tool_distribution: Counter[str] = Counter()
    action_distribution: Counter[str] = Counter()
    grounding_distribution: Counter[bool] = Counter()
    warning_distribution: Counter[str] = Counter()
    per_source_counts: Counter[str] = Counter()

    for record in iter_jsonl(run_dir / "tool_call_records.jsonl"):
        generated_count += 1
        generation_seconds_total += float((record.get("metadata") or {}).get("elapsed_seconds") or 0.0)
        retry_count = int(record.get("retry_count") or 0)
        if retry_count:
            retried_record_count += 1
            retry_attempts_total += retry_count
        search_tool_distribution[str(record.get("search_tool"))] += 1
        action_distribution[str(record.get("answer"))] += 1
        grounding_distribution[bool(record.get("use_grounding"))] += 1
        warning_distribution.update(record.get("parse_warnings") or [])
        per_source_counts[str(record.get("source_dataset"))] += 1

    failed_count = count_jsonl_rows(run_dir / "failed_rows.jsonl")
    summary = dict(base_summary)
    last_invocation_wall_seconds = summary.get("total_wall_seconds")
    summary.update(
        {
            "requested_counts": requested_counts,
            "generated_count": generated_count,
            "generation_seconds_total": round(generation_seconds_total, 3),
            "average_elapsed_seconds": (
                round(generation_seconds_total / generated_count, 3)
                if generated_count
                else 0.0
            ),
            "retried_record_count": retried_record_count,
            "retry_attempts_total": retry_attempts_total,
            "tool_ratio_table": make_tool_ratio_table(
                generated_count=generated_count,
                search_tool_distribution=search_tool_distribution,
                grounding_distribution=grounding_distribution,
            ),
            "search_tool_distribution": dict(search_tool_distribution),
            "action_distribution": dict(action_distribution),
            "use_grounding_distribution": {
                str(key).lower(): value
                for key, value in grounding_distribution.items()
            },
            "parse_warning_distribution": dict(warning_distribution),
            "per_source_counts": dict(per_source_counts),
            "failed_count": failed_count,
            "requested_total": total_requested,
        }
    )
    if failed_count:
        summary["failed_rows_path"] = str(run_dir / "failed_rows.jsonl")
    if resume_mode:
        summary["last_invocation_wall_seconds"] = last_invocation_wall_seconds
        summary["total_wall_seconds"] = None
        summary["total_wall_seconds_note"] = (
            "Unavailable for resumed interrupted runs because the original invocation stopped before writing a summary."
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate first-round VLM tool-call annotations with the project prompt format "
            "reference prompt."
        )
    )
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
        help="Balanced total sample count across the selected sources. Overrides the default source counts.",
    )
    parser.add_argument(
        "--per-source-limit",
        type=int,
        default=None,
        help="Fixed sample count to draw from each selected source. Overrides the default source counts.",
    )
    parser.add_argument(
        "--source-count",
        action="append",
        default=[],
        metavar="NAME=COUNT",
        help=(
            "Per-source sample count. Can be repeated. If omitted, defaults follow the project source policy: "
            + ", ".join(f"{name}={count}" for name, count in DEFAULT_SOURCE_COUNTS.items())
            + "."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_path("tool_call", "generated"),
        help="Parent directory for generation runs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run directory name. Defaults to a timestamped qwen35_tool_call name.",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Resume an interrupted run directory by skipping rows already written to tool_call_records.jsonl or failed_rows.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for each first-round tool-call annotation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retry attempts for request errors or malformed tool-call tag output.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Override llama-server --parallel for benchmarking. Defaults to the tool-call generation setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit_source_counts = parse_source_counts(args.source_count) if args.source_count else None

    source_rows = load_source_rows(args.sources)
    plan = make_sampling_plan(
        source_rows,
        count=args.count,
        per_source_limit=args.per_source_limit,
        source_counts=explicit_source_counts,
        seed=args.seed,
    )
    requests = sample_requests(source_rows, plan=plan, seed=args.seed)
    del source_rows
    gc.collect()

    completed_before_resume = 0
    if args.resume_run_dir is not None:
        if args.run_name is not None:
            raise ValueError("Use either --resume-run-dir or --run-name, not both.")
        run_dir = args.resume_run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
        completed_keys = load_completed_keys(run_dir)
        completed_before_resume = len(completed_keys)
        requests = [
            request
            for request in requests
            if request_key(request.source_name, request.record_id) not in completed_keys
        ]
        run_name = run_dir.name
        print(
            f"Resuming {run_dir}: skip {completed_before_resume} completed rows, "
            f"generate {len(requests)} remaining rows."
        )
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = args.run_name or f"qwen35_tool_call_{plan.total}_{timestamp}"
        run_dir = args.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)

    server_config = default_tool_call_server_config()
    if args.parallel is not None:
        if args.parallel <= 0:
            raise ValueError("--parallel must be greater than 0.")
        server_config = replace(server_config, parallel=args.parallel)

    if requests:
        _, _, summary = run_generation(
            requests,
            config=server_config,
            run_dir=run_dir,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )
    else:
        summary = {
            "combined_path": str(run_dir / "tool_call_records.jsonl"),
            "server_log_path": str(run_dir / "llama_server.log"),
            "prompt_path": str(run_dir / "first_round_prompt.md"),
            "server_config": {
                "ctx_size": server_config.ctx_size,
                "parallel": server_config.parallel,
                "request_workers": max(1, int(server_config.parallel or 1)),
                "threads": server_config.threads,
                "flash_attn": server_config.flash_attn,
            },
            "max_retries": args.max_retries,
            "total_wall_seconds": 0.0,
        }

    summary = rebuild_run_summary_from_files(
        run_dir=run_dir,
        base_summary=summary,
        requested_counts=plan.per_source,
        total_requested=plan.total,
        resume_mode=args.resume_run_dir is not None,
    )
    summary["run_name"] = run_name
    summary["seed"] = args.seed
    summary["sources"] = args.sources
    summary["completed_before_resume"] = completed_before_resume
    summary["remaining_requested_at_start"] = len(requests)
    summary["run_dir"] = str(run_dir)
    summary["default_source_counts"] = {
        name: DEFAULT_SOURCE_COUNTS[name]
        for name in args.sources
        if name in DEFAULT_SOURCE_COUNTS
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
