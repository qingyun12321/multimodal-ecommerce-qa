from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEICTIC_PATTERN = re.compile(r"\b(this|these|that|those)\b", re.IGNORECASE)
SCENE_PATTERN = re.compile(r"适合什么场合|适用于什么场合|适用场景|适合哪些场合|赠送")
NONVISUAL_ATTR_PATTERN = re.compile(r"材质|成分|纤维|鞋面|面料|包材质|规格|售后|退换|刷新率|能效|排水|功效")
VISUAL_OCR_HINT_PATTERN = re.compile(r"标签|包装上|瓶身上|面板上|logo|显示屏|文字|标注")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def replace_grounding_text(think: str, *, needed: bool) -> str:
    replacements = [
        ("不需要图像裁剪", "需要图像裁剪" if needed else "不需要图像裁剪"),
        ("无需图像裁剪", "需要图像裁剪" if needed else "不需要图像裁剪"),
        ("不需要裁剪", "需要图像裁剪" if needed else "不需要图像裁剪"),
        ("无需裁剪", "需要图像裁剪" if needed else "不需要图像裁剪"),
        ("需要图像裁剪", "需要图像裁剪" if needed else "不需要图像裁剪"),
        ("需要裁剪", "需要图像裁剪" if needed else "不需要图像裁剪"),
        ("需要 grounding", "需要图像裁剪" if needed else "不需要图像裁剪"),
        ("需要grounding", "需要图像裁剪" if needed else "不需要图像裁剪"),
    ]
    updated = think
    for old, new in replacements:
        updated = updated.replace(old, new)
    if "图像裁剪" not in updated:
        if needed:
            updated = updated.rstrip("。") + "。需要先定位图中对象或局部区域，再进行检索，因此需要图像裁剪。"
        else:
            updated = updated.rstrip("。") + "。不需要图像裁剪。"
    return updated


def switch_search_tool_text(think: str, *, from_tool: str, to_tool: str, reason: str) -> str:
    updated = think.replace(from_tool, to_tool)
    if to_tool not in updated:
        updated = updated.rstrip("。") + f"。{reason}"
    elif reason not in updated:
        updated = updated.rstrip("。") + f"。{reason}"
    return updated


def qa_type_of(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata") or {}
    value = metadata.get("qa_type")
    return str(value) if value is not None else None


def source_name_of(record: dict[str, Any]) -> str:
    return Path(str(record["source_file"])).name


def rebuild_tool_fields(record: dict[str, Any]) -> None:
    search_tool = str(record["search_tool"])
    use_grounding = bool(record["use_grounding"])
    tool_calls: list[str] = []
    if search_tool != "none":
        tool_calls.append(search_tool)
        if use_grounding:
            tool_calls.append("图像裁剪")
    if tool_calls:
        record["answer"] = f"调用工具【{'，'.join(tool_calls)}】"
        record["decision_type"] = "tool_call"
    else:
        record["answer"] = "直接回答，无需调用工具"
        record["decision_type"] = "direct_answer"
    record["tool_calls"] = tool_calls
    metadata = record.get("metadata") or {}
    tool_plan = dict(metadata.get("tool_plan") or {})
    tool_plan.update(
        {
            "decision_type": record["decision_type"],
            "search_tool": search_tool,
            "use_grounding": use_grounding,
            "tool_calls": tool_calls,
        }
    )
    metadata["tool_plan"] = tool_plan
    record["metadata"] = metadata


def clean_record(record: dict[str, Any], stats: Counter[str]) -> dict[str, Any]:
    cleaned = deepcopy(record)
    query = str(cleaned["query"])
    query_lower = query.lower()
    source_name = source_name_of(cleaned)
    qa_type = qa_type_of(cleaned)
    search_tool = str(cleaned["search_tool"])
    use_grounding = bool(cleaned["use_grounding"])

    if qa_type == "text_only" and use_grounding:
        cleaned["use_grounding"] = False
        cleaned["think"] = replace_grounding_text(str(cleaned["think"]), needed=False)
        stats["text_only_remove_grounding"] += 1
        use_grounding = False

    if (
        source_name == "infoseek_sample.jsonl"
        and search_tool == "Web_search"
        and not use_grounding
        and DEICTIC_PATTERN.search(query_lower)
    ):
        cleaned["use_grounding"] = True
        cleaned["think"] = replace_grounding_text(str(cleaned["think"]), needed=True)
        stats["ood_deictic_add_grounding"] += 1
        use_grounding = True

    if (
        source_name != "infoseek_sample.jsonl"
        and search_tool == "Web_search"
        and SCENE_PATTERN.search(query)
    ):
        cleaned["search_tool"] = "RAG_search"
        cleaned["use_grounding"] = False
        cleaned["think"] = switch_search_tool_text(
            replace_grounding_text(str(cleaned["think"]), needed=False),
            from_tool="Web_search",
            to_tool="RAG_search",
            reason="该问题更接近具体商品页中的适用场景或赠送场景信息，应优先使用 RAG_search。",
        )
        stats["in_domain_scene_web_to_rag"] += 1

    if (
        source_name != "infoseek_sample.jsonl"
        and str(cleaned["search_tool"]) == "RAG_search"
        and bool(cleaned["use_grounding"])
        and qa_type == "multimodal"
        and NONVISUAL_ATTR_PATTERN.search(query)
        and not VISUAL_OCR_HINT_PATTERN.search(query)
    ):
        cleaned["use_grounding"] = False
        cleaned["think"] = replace_grounding_text(str(cleaned["think"]), needed=False)
        stats["in_domain_nonvisual_rag_remove_grounding"] += 1

    rebuild_tool_fields(cleaned)
    return cleaned


def build_summary(
    records: list[dict[str, Any]],
    *,
    source_run_dir: Path,
    cleaning_stats: Counter[str],
    ignored_failed_count: int,
) -> dict[str, Any]:
    tool_distribution = Counter(record["answer"] for record in records)
    grounding_distribution = Counter(bool(record["use_grounding"]) for record in records)
    per_source = Counter(str(record["source_dataset"]) for record in records)
    return {
        "generated_count": len(records),
        "tool_answer_distribution": dict(tool_distribution),
        "use_grounding_distribution": {str(key).lower(): value for key, value in grounding_distribution.items()},
        "per_source_counts": dict(per_source),
        "cleaned_from_run_dir": str(source_run_dir),
        "ignored_failed_count": ignored_failed_count,
        "cleaning_stats": dict(cleaning_stats),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Programmatically clean tool-call labels.")
    parser.add_argument("--input-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("dataset/tool_call/generated"))
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_run_dir = args.input_run_dir.resolve()
    records_path = input_run_dir / "tool_call_records.jsonl"
    failed_rows_path = input_run_dir / "failed_rows.jsonl"
    original_records = load_jsonl(records_path)
    ignored_failed_count = len(load_jsonl(failed_rows_path)) if failed_rows_path.exists() else 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{input_run_dir.name}_cleaned_{timestamp}"
    output_run_dir = args.output_root / run_name
    output_run_dir.mkdir(parents=True, exist_ok=True)

    cleaning_stats: Counter[str] = Counter()
    cleaned_records = [clean_record(record, cleaning_stats) for record in original_records]

    combined_path = output_run_dir / "tool_call_records.jsonl"
    per_source_dir = output_run_dir / "per_source"
    summary_path = output_run_dir / "run_summary.json"
    cleaning_summary_path = output_run_dir / "cleaning_summary.json"

    write_jsonl(combined_path, cleaned_records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in cleaned_records:
        grouped[str(record["source_dataset"])].append(record)
    for source_name, rows in grouped.items():
        write_jsonl(per_source_dir / f"{source_name}.jsonl", rows)

    summary = build_summary(
        cleaned_records,
        source_run_dir=input_run_dir,
        cleaning_stats=cleaning_stats,
        ignored_failed_count=ignored_failed_count,
    )
    summary["combined_path"] = str(combined_path)
    summary["run_dir"] = str(output_run_dir)
    summary["run_name"] = run_name
    write_json(summary_path, summary)
    write_json(
        cleaning_summary_path,
        {
            "input_run_dir": str(input_run_dir),
            "output_run_dir": str(output_run_dir),
            "input_record_count": len(original_records),
            "output_record_count": len(cleaned_records),
            "ignored_failed_count": ignored_failed_count,
            "cleaning_stats": dict(cleaning_stats),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
