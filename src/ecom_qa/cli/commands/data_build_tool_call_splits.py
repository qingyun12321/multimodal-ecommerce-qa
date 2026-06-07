from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ecom_qa.common.paths import data_path


DEFAULT_SOURCE_RUN_DIR = data_path("tool_call", "generated", "qwen35_tool_call_30000_20260512T215543Z")
DEFAULT_OUTPUT_DIR = data_path("tool_call")
SEED = 20260514
THINK_MAX_CHARS = 250
DIRECT_ANSWER_MAX_CHARS = 400

IN_DOMAIN_QUOTAS = {
    ("text_only", "none", False): 40,
    ("text_only", "RAG_search", False): 903,
    ("text_only", "Web_search", False): 57,
    ("multimodal", "none", False): 1210,
    ("multimodal", "RAG_search", False): 607,
    ("multimodal", "RAG_search", True): 1210,
    ("multimodal", "Web_search", False): 783,
    ("multimodal", "Web_search", True): 190,
}
OUT_OF_DOMAIN_QUOTAS = {
    ("multimodal", "none", False): 2500,
    ("multimodal", "Web_search", True): 800,
    ("multimodal", "Web_search", False): 3700,
}
TRAIN_RATIO = 0.8
NO_GROUNDING_VALUES = {"", "no", "none", "false", "不需要", "无需", "否", "不用"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_out_of_domain(record: dict[str, Any]) -> bool:
    return str(record.get("source_dataset")) == "infoseek_sample"


def quota_key(record: dict[str, Any]) -> tuple[str, bool]:
    return str(record["search_tool"]), bool(record["use_grounding"])


def qa_type(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    value = str(metadata.get("qa_type") or "").strip()
    if value in {"multimodal", "text_only"}:
        return value
    return "multimodal"


def reject_reason(record: dict[str, Any]) -> str | None:
    required_fields = {
        "record_id",
        "source_dataset",
        "source_record_key",
        "query",
        "think",
        "answer",
        "decision_type",
        "search_tool",
        "search_input",
        "use_grounding",
        "grounding_input",
        "tool_calls",
        "metadata",
    }
    missing = sorted(required_fields - set(record))
    if missing:
        return "missing_required_field"

    if record.get("parse_warnings"):
        return "parse_warning"

    source_dataset = str(record["source_dataset"])
    if source_dataset not in {
        "ecom_qa_pairs",
        "ecom_qa_pairs_open_question",
        "ecom_qa_pairs_supplement",
        "infoseek_sample",
    }:
        return "unknown_source_dataset"

    search_tool = str(record["search_tool"])
    if search_tool not in {"none", "RAG_search", "Web_search"}:
        return "invalid_search_tool"

    if is_out_of_domain(record) and search_tool == "RAG_search":
        return "out_of_domain_rag"

    think = str(record.get("think") or "")
    if not think or len(think) > THINK_MAX_CHARS:
        return "think_too_long_or_empty"

    tool_calls = list(record.get("tool_calls") or [])
    use_grounding = bool(record["use_grounding"])
    grounding_input = str(record.get("grounding_input") or "").strip().lower()

    if search_tool == "none":
        if str(record["decision_type"]) != "direct_answer":
            return "direct_bad_decision_type"
        if tool_calls:
            return "direct_has_tool_calls"
        if use_grounding:
            return "direct_has_grounding"
        if not str(record.get("direct_answer") or "").strip():
            return "direct_missing_answer"
        if len(str(record["answer"])) > DIRECT_ANSWER_MAX_CHARS:
            return "direct_answer_too_long"
    else:
        if str(record["decision_type"]) != "tool_call":
            return "tool_bad_decision_type"
        if search_tool not in tool_calls:
            return "tool_calls_missing_search_tool"
        if not str(record.get("search_input") or "").strip():
            return "tool_missing_search_input"
        if str(record.get("direct_answer") or "").strip():
            return "tool_has_direct_answer"
        if use_grounding:
            if "图像裁剪" not in tool_calls:
                return "grounding_missing_tool_call"
            if grounding_input in NO_GROUNDING_VALUES:
                return "grounding_missing_input"
        elif "图像裁剪" in tool_calls:
            return "non_grounding_has_tool_call"

    return None


def group_candidates(records: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for record in records:
        reason = reject_reason(record)
        if reason is not None:
            rejected[reason] += 1
            continue
        domain = "out_of_domain" if is_out_of_domain(record) else "in_domain"
        key = f"{domain}|{qa_type(record)}|{record['search_tool']}|{str(bool(record['use_grounding'])).lower()}"
        groups[key].append(record)
    return groups, rejected


def sample_group(
    groups: dict[str, list[dict[str, Any]]],
    *,
    domain: str,
    qa_type_value: str,
    search_tool: str,
    use_grounding: bool,
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    key = f"{domain}|{qa_type_value}|{search_tool}|{str(use_grounding).lower()}"
    candidates = list(groups.get(key) or [])
    if len(candidates) < count:
        raise ValueError(f"Need {count} rows for {key}, only found {len(candidates)}.")
    rng.shuffle(candidates)
    return candidates[:count]


def select_records(records: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups, rejected = group_candidates(records)
    rng = random.Random(seed)

    selected: list[dict[str, Any]] = []
    quota_details: dict[str, int] = {}
    for (qa_type_value, search_tool, use_grounding), count in IN_DOMAIN_QUOTAS.items():
        selected_rows = sample_group(
            groups,
            domain="in_domain",
            qa_type_value=qa_type_value,
            search_tool=search_tool,
            use_grounding=use_grounding,
            count=count,
            rng=rng,
        )
        selected.extend(selected_rows)
        quota_details[f"in_domain|{qa_type_value}|{search_tool}|{str(use_grounding).lower()}"] = len(selected_rows)

    for (qa_type_value, search_tool, use_grounding), count in OUT_OF_DOMAIN_QUOTAS.items():
        selected_rows = sample_group(
            groups,
            domain="out_of_domain",
            qa_type_value=qa_type_value,
            search_tool=search_tool,
            use_grounding=use_grounding,
            count=count,
            rng=rng,
        )
        selected.extend(selected_rows)
        quota_details[f"out_of_domain|{qa_type_value}|{search_tool}|{str(use_grounding).lower()}"] = len(selected_rows)

    rng.shuffle(selected)
    return selected, {
        "candidate_group_sizes": {key: len(value) for key, value in sorted(groups.items())},
        "rejected_counts": dict(rejected),
        "quota_details": quota_details,
    }


def stratified_train_test_split(
    records: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[
            (
                str(record["source_dataset"]),
                qa_type(record),
                str(record["search_tool"]),
                bool(record["use_grounding"]),
            )
        ].append(record)

    exact_train_counts: dict[tuple[str, str, str, bool], float] = {
        key: len(rows) * train_ratio
        for key, rows in groups.items()
    }
    train_counts: dict[tuple[str, str, str, bool], int] = {
        key: int(value)
        for key, value in exact_train_counts.items()
    }
    target_train_total = int(round(len(records) * train_ratio))
    missing = target_train_total - sum(train_counts.values())
    if missing > 0:
        remainders = sorted(
            exact_train_counts,
            key=lambda key: (exact_train_counts[key] - int(exact_train_counts[key]), str(key)),
            reverse=True,
        )
        for key in remainders[:missing]:
            train_counts[key] += 1

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for key, rows in groups.items():
        shuffled = list(rows)
        rng.shuffle(shuffled)
        split_at = train_counts[key]
        train.extend(shuffled[:split_at])
        test.extend(shuffled[split_at:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def tool_balance(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "直接回答": sum(1 for record in records if record["search_tool"] == "none"),
        "RAG_search": sum(1 for record in records if record["search_tool"] == "RAG_search"),
        "Web_search": sum(1 for record in records if record["search_tool"] == "Web_search"),
        "图像裁剪": sum(1 for record in records if record["use_grounding"]),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_source = Counter(str(record["source_dataset"]) for record in records)
    per_domain = Counter("out_of_domain" if is_out_of_domain(record) else "in_domain" for record in records)
    per_qa_type = Counter(qa_type(record) for record in records)
    per_domain_qa_type = Counter(
        f"{'out_of_domain' if is_out_of_domain(record) else 'in_domain'}|{qa_type(record)}"
        for record in records
    )
    per_search_tool = Counter(str(record["search_tool"]) for record in records)
    per_group = Counter(
        f"{'out_of_domain' if is_out_of_domain(record) else 'in_domain'}|{record['search_tool']}|{str(bool(record['use_grounding'])).lower()}"
        for record in records
    )
    return {
        "count": len(records),
        "per_source": dict(per_source),
        "per_domain": dict(per_domain),
        "per_qa_type": dict(per_qa_type),
        "per_domain_qa_type": dict(per_domain_qa_type),
        "per_search_tool": dict(per_search_tool),
        "tool_balance": tool_balance(records),
        "per_group": dict(per_group),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the final balanced tool-call dataset.")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_path = args.source_run_dir / "tool_call_records.jsonl"
    failed_path = args.source_run_dir / "failed_rows.jsonl"

    source_records = load_jsonl(records_path)
    selected, selection_details = select_records(source_records, seed=args.seed)
    train, test = stratified_train_test_split(selected, seed=args.seed + 1, train_ratio=TRAIN_RATIO)

    output_dir = args.output_dir
    write_jsonl(output_dir / "tool_call_records_balanced.jsonl", selected)
    write_jsonl(output_dir / "tool_call_records_train.jsonl", train)
    write_jsonl(output_dir / "tool_call_records_test.jsonl", test)

    failed_count = len(load_jsonl(failed_path)) if failed_path.exists() else 0
    summary = {
        "source_run_dir": str(args.source_run_dir),
        "source_records_count": len(source_records),
        "source_failed_count_ignored": failed_count,
        "selection_policy": {
            "target_total": len(selected),
            "target_in_domain": sum(IN_DOMAIN_QUOTAS.values()),
            "target_out_of_domain": sum(OUT_OF_DOMAIN_QUOTAS.values()),
            "train_ratio": TRAIN_RATIO,
            "seed": args.seed,
            "think_max_chars": THINK_MAX_CHARS,
            "direct_answer_max_chars": DIRECT_ANSWER_MAX_CHARS,
            "excluded_out_of_domain_rag": True,
            "in_domain_quotas": {
                f"{qa_type_value}|{search_tool}|grounding={str(use_grounding).lower()}": count
                for (qa_type_value, search_tool, use_grounding), count in IN_DOMAIN_QUOTAS.items()
            },
            "out_of_domain_quotas": {
                f"{qa_type_value}|{search_tool}|grounding={str(use_grounding).lower()}": count
                for (qa_type_value, search_tool, use_grounding), count in OUT_OF_DOMAIN_QUOTAS.items()
            },
        },
        "selected": summarize(selected),
        "train": summarize(train),
        "test": summarize(test),
        "selection_details": selection_details,
        "output_files": {
            "all": str(output_dir / "tool_call_records_balanced.jsonl"),
            "train": str(output_dir / "tool_call_records_train.jsonl"),
            "test": str(output_dir / "tool_call_records_test.jsonl"),
            "summary": str(output_dir / "tool_call_records_balanced_summary.json"),
        },
    }
    write_json(output_dir / "tool_call_records_balanced_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
