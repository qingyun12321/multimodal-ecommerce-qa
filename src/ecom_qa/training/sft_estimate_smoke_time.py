#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from ecom_qa.common.paths import resolve_data_root


WORKSPACE = Path(os.environ.get("WORKSPACE_HOME", "/workspace"))
REPO_DIR = Path(os.environ.get("REPO_DIR", WORKSPACE / "repos" / "multimodal-ecommerce-qa"))
DEFAULT_SFT_DATA_DIR = REPO_DIR / "data" / "training" / "sft" / "generated" / "mimo_multiturn"
DATA_DIR = Path(os.environ.get("SFT_DATA_DIR", DEFAULT_SFT_DATA_DIR))
PROJECT_DATA_DIR = resolve_data_root(REPO_DIR)
CHECKPOINT_ROOT = Path(
    os.environ.get("SFT_CHECKPOINT_ROOT", WORKSPACE / "checkpoints" / "qwen3-vl-8b-tool-call-sft-v2")
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def parse_step(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        head = value.split("/", 1)[0]
        if head.isdigit():
            return int(head)
    return default


def trainer_runtime(trainer_state: dict[str, Any], logging_records: list[dict[str, Any]]) -> tuple[float | None, int | None]:
    history = trainer_state.get("log_history") or []
    for item in reversed(history):
        if "train_runtime" in item:
            return float(item["train_runtime"]), int(item.get("step") or trainer_state.get("global_step") or 0)
    for item in reversed(logging_records):
        if "train_runtime" in item:
            return float(item["train_runtime"]), parse_step(item.get("global_step/max_steps"), trainer_state.get("global_step"))
    return None, int(trainer_state.get("global_step") or 0) or None


def main() -> int:
    validation = read_json(DATA_DIR / "validation_report.json")
    trainer_state = read_json(CHECKPOINT_ROOT / "smoke" / "checkpoint-10" / "trainer_state.json")
    logging_records = read_jsonl(CHECKPOINT_ROOT / "smoke" / "logging.jsonl")
    runtime, runtime_steps = trainer_runtime(trainer_state, logging_records)
    smoke_steps = int(trainer_state.get("global_step") or runtime_steps or 10)
    seconds_per_step = (runtime / runtime_steps) if runtime and runtime_steps else None

    source_train_rows = count_jsonl(PROJECT_DATA_DIR / "tool_call" / "tool_call_records_train.jsonl")
    source_val_rows = count_jsonl(PROJECT_DATA_DIR / "tool_call" / "tool_call_records_test.jsonl")
    smoke_train_rows = count_jsonl(DATA_DIR / "smoke_train.ms_swift.jsonl")
    data_elapsed = float(validation.get("elapsed_seconds") or 0)
    data_seconds_per_row = data_elapsed / max(1, smoke_train_rows + count_jsonl(DATA_DIR / "smoke_val.ms_swift.jsonl"))

    grad_accum = 4
    batch_size = 1
    steps_per_epoch = math.ceil(source_train_rows / max(1, grad_accum * batch_size))
    estimated_train_seconds = seconds_per_step * steps_per_epoch if seconds_per_step else None
    estimated_data_seconds = data_seconds_per_row * (source_train_rows + source_val_rows)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke_steps": smoke_steps,
        "smoke_train_runtime_seconds": runtime,
        "seconds_per_train_step": seconds_per_step,
        "source_train_rows": source_train_rows,
        "source_val_rows": source_val_rows,
        "smoke_train_rows": smoke_train_rows,
        "data_generation_seconds": data_elapsed,
        "data_generation_seconds_per_row": data_seconds_per_row,
        "assumed_effective_batch_size": grad_accum * batch_size,
        "estimated_steps_per_epoch": steps_per_epoch,
        "estimated_train_seconds_one_epoch": estimated_train_seconds,
        "estimated_train_hours_one_epoch": (estimated_train_seconds / 3600) if estimated_train_seconds else None,
        "estimated_data_generation_seconds_full": estimated_data_seconds,
        "estimated_data_generation_hours_full": estimated_data_seconds / 3600,
        "notes": [
            "This estimate is derived from the smoke test and should be treated as an order-of-magnitude forecast.",
            "Full data generation time is dominated by OpenRouter API filling plus SearXNG Web_search retrieval and rerank.",
            "Main training is intentionally not started by this workflow.",
        ],
    }
    path = DATA_DIR / "training_time_estimate_smoke.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
