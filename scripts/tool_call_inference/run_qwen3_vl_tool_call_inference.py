from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ecom_qa.data.model_client import SERVER_BINARY, run_llama_server, write_json
from ecom_qa.data.tool_call_inference_model import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CTX_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PARALLEL,
    DEFAULT_PORT,
    DEFAULT_THREADS,
    DEFAULT_UBATCH_SIZE,
    MODEL_SPECS,
    infer_tool_call,
    make_server_config,
)


DEFAULT_INPUT = Path("dataset/tool_call/tool_call_records_test.jsonl")
DEFAULT_OUTPUT_ROOT = Path("dataset/tool_call_inference/generated")
DEFAULT_RUN_NAME = "qwen3_vl_tool_call_full"
LABELS = ("直接回答", "RAG_search", "Web_search", "图像裁剪")
PRIMARY_LABELS = ("直接回答", "RAG_search", "Web_search")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL-4B/8B first-round tool-call inference and compute precision/recall/F1."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), default=sorted(MODEL_SPECS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--server-binary", type=Path, default=SERVER_BINARY)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ubatch-size", type=int, default=DEFAULT_UBATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args()


def load_records(path: Path, *, offset: int, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index < offset:
                continue
            if limit is not None and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_input_line_index"] = line_index
            records.append(record)
    return records


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def primary_label_from_record(record: dict[str, Any]) -> str:
    search_tool = str(record["search_tool"])
    if search_tool == "RAG_search":
        return "RAG_search"
    if search_tool == "Web_search":
        return "Web_search"
    return "直接回答"


def expected_labels(record: dict[str, Any]) -> set[str]:
    labels = {primary_label_from_record(record)}
    if bool(record["use_grounding"]):
        labels.add("图像裁剪")
    return labels


def primary_label_from_prediction(parsed: dict[str, Any]) -> str:
    search_tool = str(parsed["search_tool"])
    if search_tool == "RAG_search":
        return "RAG_search"
    if search_tool == "Web_search":
        return "Web_search"
    if str(parsed.get("direct_answer") or "").strip():
        return "直接回答"
    return "无有效动作"


def predicted_labels(parsed: dict[str, Any]) -> set[str]:
    primary = primary_label_from_prediction(parsed)
    labels: set[str] = set()
    if primary in PRIMARY_LABELS:
        labels.add(primary)
    if bool(parsed["use_grounding"]):
        labels.add("图像裁剪")
    return labels


def expected_plan(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": record["answer"],
        "decision_type": record["decision_type"],
        "search_tool": record["search_tool"],
        "use_grounding": bool(record["use_grounding"]),
        "tool_calls": record["tool_calls"],
        "labels": sorted(expected_labels(record)),
    }


def build_prediction_record(
    record: dict[str, Any],
    *,
    model_key: str,
    parsed: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    expected = expected_plan(record)
    predicted = {
        "raw_output": parsed["raw_output"],
        "think": parsed["think"],
        "direct_answer": parsed["direct_answer"],
        "answer": parsed["answer"],
        "decision_type": parsed["decision_type"],
        "search_tool": parsed["search_tool"],
        "search_input": parsed["search_input"],
        "use_grounding": parsed["use_grounding"],
        "grounding_input": parsed["grounding_input"],
        "tool_calls": parsed["tool_calls"],
        "labels": sorted(predicted_labels(parsed)),
        "parsed_tags": parsed["parsed_tags"],
        "parse_warnings": parsed["parse_warnings"],
    }
    return {
        "record_id": record["record_id"],
        "input_line_index": record["_input_line_index"],
        "model_key": model_key,
        "domain": record["domain"],
        "source_dataset": record["source_dataset"],
        "source_record_key": record["source_record_key"],
        "query": record["query"],
        "image_path": record["image_path"],
        "expected": expected,
        "prediction": predicted,
        "metrics": {
            "primary_exact": primary_label_from_prediction(parsed) == primary_label_from_record(record),
            "grounding_exact": bool(parsed["use_grounding"]) == bool(record["use_grounding"]),
            "labels_exact": set(predicted["labels"]) == set(expected["labels"]),
            "search_tool_exact": parsed["search_tool"] == record["search_tool"],
            "tool_calls_exact": parsed["tool_calls"] == record["tool_calls"],
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "support": tp + fn,
        "predicted": tp + fp,
    }


@dataclass(slots=True)
class MetricsAccumulator:
    total: int = 0
    elapsed_seconds_total: float = 0.0
    exact_counts: Counter[str] = field(default_factory=Counter)
    label_tp: Counter[str] = field(default_factory=Counter)
    label_fp: Counter[str] = field(default_factory=Counter)
    label_fn: Counter[str] = field(default_factory=Counter)
    primary_confusion: Counter[str] = field(default_factory=Counter)
    parse_warning_distribution: Counter[str] = field(default_factory=Counter)
    by_domain_total: Counter[str] = field(default_factory=Counter)
    by_domain_labels_exact: Counter[str] = field(default_factory=Counter)

    def add(self, record: dict[str, Any]) -> None:
        self.total += 1
        self.elapsed_seconds_total += float(record["elapsed_seconds"])
        for name, value in record["metrics"].items():
            if value:
                self.exact_counts[name] += 1

        expected = set(record["expected"]["labels"])
        predicted = set(record["prediction"]["labels"])
        for label in LABELS:
            if label in expected and label in predicted:
                self.label_tp[label] += 1
            elif label not in expected and label in predicted:
                self.label_fp[label] += 1
            elif label in expected and label not in predicted:
                self.label_fn[label] += 1

        expected_primary = next(label for label in PRIMARY_LABELS if label in expected)
        predicted_primary = primary_label_from_prediction(record["prediction"])
        self.primary_confusion[f"{expected_primary} -> {predicted_primary}"] += 1
        self.parse_warning_distribution.update(record["prediction"].get("parse_warnings") or [])

        domain = str(record["domain"])
        self.by_domain_total[domain] += 1
        if record["metrics"]["labels_exact"]:
            self.by_domain_labels_exact[domain] += 1

    def to_summary(self, *, startup_seconds: float, total_wall_seconds: float) -> dict[str, Any]:
        exact_metrics = {
            name: {
                "correct": self.exact_counts[name],
                "total": self.total,
                "accuracy": round(self.exact_counts[name] / self.total, 6) if self.total else 0.0,
            }
            for name in ("primary_exact", "grounding_exact", "labels_exact", "search_tool_exact", "tool_calls_exact")
        }
        per_label = {
            label: precision_recall_f1(self.label_tp[label], self.label_fp[label], self.label_fn[label])
            for label in LABELS
        }
        micro_tp = sum(self.label_tp.values())
        micro_fp = sum(self.label_fp.values())
        micro_fn = sum(self.label_fn.values())
        macro_precision = sum(item["precision"] for item in per_label.values()) / len(LABELS)
        macro_recall = sum(item["recall"] for item in per_label.values()) / len(LABELS)
        macro_f1 = sum(item["f1"] for item in per_label.values()) / len(LABELS)
        return {
            "generated_count": self.total,
            "server_startup_seconds": round(startup_seconds, 3),
            "total_wall_seconds": round(total_wall_seconds, 3),
            "average_elapsed_seconds": round(self.elapsed_seconds_total / self.total, 3) if self.total else 0.0,
            "exact_metrics": exact_metrics,
            "per_label_precision_recall_f1": per_label,
            "micro_precision_recall_f1": precision_recall_f1(micro_tp, micro_fp, micro_fn),
            "macro_precision_recall_f1": {
                "precision": round(macro_precision, 6),
                "recall": round(macro_recall, 6),
                "f1": round(macro_f1, 6),
            },
            "by_domain_labels_exact": {
                domain: {
                    "correct": self.by_domain_labels_exact[domain],
                    "total": total,
                    "accuracy": round(self.by_domain_labels_exact[domain] / total, 6) if total else 0.0,
                }
                for domain, total in self.by_domain_total.items()
            },
            "primary_confusion": dict(self.primary_confusion),
            "parse_warning_distribution": dict(self.parse_warning_distribution),
        }


def render_progress(model_key: str, completed: int, total: int, failed: int) -> None:
    percent = completed / total * 100 if total else 100.0
    print(f"\r{model_key}: {completed}/{total} ({percent:5.1f}%) failed={failed}", end="", flush=True)


def infer_one_indexed(
    *,
    index: int,
    record: dict[str, Any],
    model_key: str,
    config: Any,
    max_tokens: int,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
    try:
        parsed, elapsed_seconds = infer_tool_call(record, config, max_tokens=max_tokens)
        return index, build_prediction_record(
            record,
            model_key=model_key,
            parsed=parsed,
            elapsed_seconds=elapsed_seconds,
        ), None
    except Exception as exc:
        return index, None, {
            "record_id": record.get("record_id"),
            "input_line_index": record.get("_input_line_index"),
            "model_key": model_key,
            "error": str(exc),
        }


def run_model(
    *,
    model_key: str,
    records: list[dict[str, Any]],
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = make_server_config(
        model_key=model_key,
        server_binary=args.server_binary,
        host=args.host,
        port=args.port,
        threads=args.threads,
        ctx_size=args.ctx_size,
        parallel=args.parallel,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
    )
    model_dir = run_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = model_dir / "predictions.jsonl"
    failed_path = model_dir / "failed_rows.jsonl"
    summary_path = model_dir / "run_summary.json"
    server_log_path = model_dir / "llama_server.log"

    started_at = time.perf_counter()
    accumulator = MetricsAccumulator()
    failed_count = 0
    completed_count = 0
    max_workers = max(1, int(args.parallel))

    render_progress(model_key, 0, len(records), 0)
    with run_llama_server(config, log_path=server_log_path) as startup_seconds:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            in_flight: dict[Future[tuple[int, dict[str, Any] | None, dict[str, Any] | None]], int] = {}
            next_index = 0

            def submit_next() -> None:
                nonlocal next_index
                if next_index >= len(records):
                    return
                future = executor.submit(
                    infer_one_indexed,
                    index=next_index,
                    record=records[next_index],
                    model_key=model_key,
                    config=config,
                    max_tokens=args.max_tokens,
                )
                in_flight[future] = next_index
                next_index += 1

            for _ in range(min(max_workers, len(records))):
                submit_next()

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future)
                    completed_count += 1
                    _, prediction_record, failed_record = future.result()
                    if failed_record is not None:
                        failed_count += 1
                        append_jsonl(failed_path, failed_record)
                    else:
                        assert prediction_record is not None
                        append_jsonl(predictions_path, prediction_record)
                        accumulator.add(prediction_record)
                    submit_next()
                    render_progress(model_key, completed_count, len(records), failed_count)
    print()

    total_wall_seconds = time.perf_counter() - started_at
    summary = accumulator.to_summary(startup_seconds=startup_seconds, total_wall_seconds=total_wall_seconds)
    summary.update(
        {
            "model_key": model_key,
            "model_alias": config.alias,
            "model_path": str(config.model_path),
            "mmproj_path": str(config.mmproj_path),
            "input_path": str(args.input),
            "input_offset": args.offset,
            "input_limit": args.limit,
            "selected_count": len(records),
            "failed_count": failed_count,
            "predictions_path": str(predictions_path),
            "failed_path": str(failed_path) if failed_count else None,
            "server_log_path": str(server_log_path),
            "server_config": {
                "ctx_size": args.ctx_size,
                "parallel": args.parallel,
                "request_workers": max_workers,
                "threads": args.threads,
                "batch_size": args.batch_size,
                "ubatch_size": args.ubatch_size,
            },
            "prompt_source": "src/ecom_qa/data/tool_call_generation.py::build_first_round_prompt",
        }
    )
    write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.parallel <= 0:
        raise ValueError("--parallel must be greater than 0.")
    records = load_records(args.input, offset=args.offset, limit=args.limit)
    if not records:
        raise ValueError("No input records selected.")

    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(run_dir / "input_records.jsonl", records)

    summaries = []
    for model_key in args.models:
        summaries.append(run_model(model_key=model_key, records=records, run_dir=run_dir, args=args))

    combined_summary = {
        "run_name": args.run_name,
        "input_path": str(args.input),
        "selected_count": len(records),
        "models": args.models,
        "parallel": args.parallel,
        "model_summaries": summaries,
    }
    write_json(run_dir / "run_summary.json", combined_summary)
    print(json.dumps(combined_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
