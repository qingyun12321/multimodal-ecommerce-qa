#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


WORKSPACE = Path(os.environ.get("WORKSPACE_HOME", "/workspace"))
REPO_DIR = Path(os.environ.get("REPO_DIR", WORKSPACE / "repos" / "multimodal-ecommerce-qa"))
DEFAULT_SFT_DATA_DIR = REPO_DIR / "data" / "training" / "sft" / "generated" / "mimo_multiturn"
DATA_DIR = Path(os.environ.get("SFT_DATA_DIR", DEFAULT_SFT_DATA_DIR))
CHECKPOINT_ROOT = Path(
    os.environ.get("SFT_CHECKPOINT_ROOT", WORKSPACE / "checkpoints" / "qwen3-vl-8b-tool-call-sft-v2")
)
CONFIG_DIR = Path(os.environ.get("SFT_CONFIG_DIR", REPO_DIR / "configs" / "training" / "sft"))
REPORT_DIR = Path(os.environ.get("SFT_REPORT_DIR", REPO_DIR / "reports" / "sft"))
LOG_DIR = WORKSPACE / "logs"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", WORKSPACE / "models" / "Qwen" / "Qwen3-VL-8B-Instruct"))

METRIC_NAME_MAP = {
    "loss": "train/loss",
    "learning_rate": "train/learning_rate",
    "grad_norm": "train/grad_norm",
    "token_acc": "train/token_acc",
    "memory": "train/gpu_memory_gib",
    "memory(GiB)": "train/gpu_memory_gib",
    "eval_loss": "eval/loss",
    "eval_token_acc": "eval/token_acc",
    "eval_samples_per_second": "eval/samples_per_second",
}


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_./ -]+", "_", value).strip("_")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def parse_args_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    tokens = shlex.split(path.read_text(encoding="utf-8"))
    parsed: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            parsed[key] = tokens[index + 1]
            index += 2
        else:
            parsed[key] = "true"
            index += 1
    return parsed


def read_logging_records(path: Path) -> list[dict[str, Any]]:
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


def find_latest_log(run_kind: str) -> Path | None:
    logs = list(LOG_DIR.glob(f"{run_kind}-sft-*.log"))
    return max(logs, key=lambda path: path.stat().st_mtime) if logs else None


def to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_step(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        head = value.split("/", 1)[0]
        if head.isdigit():
            return int(head)
    return None


def iter_history(records: list[dict[str, Any]], trainer_state: dict[str, Any]) -> list[dict[str, Any]]:
    if records:
        last_record = records[-1]
        history = last_record.get("log_history")
        if isinstance(history, list):
            return [item for item in history if isinstance(item, dict)]
    history = trainer_state.get("log_history")
    if isinstance(history, list):
        return [item for item in history if isinstance(item, dict)]
    return []


def collect_metric_history(records: list[dict[str, Any]], trainer_state: dict[str, Any]) -> list[tuple[str, float, int]]:
    points: list[tuple[str, float, int]] = []
    seen: set[tuple[str, int]] = set()
    for item in [*records, *iter_history(records, trainer_state)]:
        step = parse_step(item.get("step") or item.get("global_step") or item.get("global_step/max_steps"))
        if step is None:
            continue
        for raw_key, metric_name in METRIC_NAME_MAP.items():
            if raw_key not in item:
                continue
            value = to_float(item[raw_key])
            key = (metric_name, step)
            if value is not None and key not in seen:
                points.append((metric_name, value, step))
                seen.add(key)
    return points


def collect_data_params(validation_report: dict[str, Any], run_kind: str) -> dict[str, str]:
    params: dict[str, str] = {}
    prefix = "smoke_" if run_kind == "smoke" else ""
    params[f"{run_kind}_train_rows"] = str(count_jsonl(DATA_DIR / f"{prefix}train.ms_swift.jsonl"))
    params[f"{run_kind}_val_rows"] = str(count_jsonl(DATA_DIR / f"{prefix}val.ms_swift.jsonl"))
    for split, payload in (validation_report.get("splits") or {}).items():
        for key in (
            "rows",
            "missing_images",
            "invalid_format",
            "api_failures",
            "information_rows",
            "grounding_rows",
        ):
            if key in payload:
                params[f"data_{split}_{key}"] = str(payload[key])
        for group_key in ("qa_type_counts", "decision_type_counts", "search_tool_counts", "turn_count_counts"):
            if group_key in payload:
                params[f"data_{split}_{group_key}"] = json.dumps(payload[group_key], ensure_ascii=False, sort_keys=True)
    return params


def log_artifact_if_exists(path: Path, artifact_path: str, logged: list[tuple[str, str]]) -> None:
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
        logged.append((str(path), artifact_path))


def sync_artifacts_to_mlflow_host(artifact_uri: str, logged: list[tuple[str, str]]) -> list[str]:
    host = os.environ.get("MLFLOW_ARTIFACT_SSH_HOST")
    if not host or not artifact_uri.startswith("/"):
        return []
    ssh_config = os.environ.get("MLFLOW_ARTIFACT_SSH_CONFIG")
    ssh_cmd = ["ssh"]
    if ssh_config:
        ssh_cmd.extend(["-F", ssh_config])
    ssh_cmd.append(host)
    rsync_ssh = f"ssh -F {ssh_config}" if ssh_config else "ssh"
    copied: list[str] = []
    for source, artifact_path in logged:
        source_path = Path(source)
        if not source_path.exists():
            continue
        dest_dir = str(Path(artifact_uri) / artifact_path)
        subprocess.run([*ssh_cmd, "mkdir", "-p", dest_dir], check=True)
        subprocess.run(["rsync", "-az", "--partial", "-e", rsync_ssh, str(source_path), f"{host}:{dest_dir}/"], check=True)
        copied.append(f"{source} -> {host}:{dest_dir}/")
    return copied


def latest_checkpoint(output_dir: Path, run_kind: str) -> Path | None:
    if run_kind == "smoke":
        candidate = output_dir / "checkpoint-10"
        if candidate.exists():
            return candidate
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: path.stat().st_mtime) if output_dir.exists() else []
    return checkpoints[-1] if checkpoints else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-kind", choices=["smoke", "main"], default="smoke")
    parser.add_argument("--experiment", default="ecom-qa-tool-call-sft-v2")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    output_dir = CHECKPOINT_ROOT / args.run_kind
    checkpoint_dir = latest_checkpoint(output_dir, args.run_kind)
    config_path = CONFIG_DIR / f"qwen3_vl_8b_lora_sft_{args.run_kind}.args"
    logging_path = output_dir / "logging.jsonl"
    validation_path = DATA_DIR / "validation_report.json"
    trainer_state_path = checkpoint_dir / "trainer_state.json" if checkpoint_dir else Path("")
    latest_log = find_latest_log(args.run_kind)

    config_args = parse_args_file(config_path)
    logging_records = read_logging_records(logging_path)
    validation_report = read_json(validation_path)
    trainer_state = read_json(trainer_state_path)
    metric_history = collect_metric_history(logging_records, trainer_state)

    params = {
        "run_kind": args.run_kind,
        "model": "Qwen3-VL-8B-Instruct",
        "model_dir": str(MODEL_DIR),
        "data_dir": str(DATA_DIR),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else "",
        "main_training_started": "false" if args.run_kind == "smoke" else "true",
        "ms_swift_version": package_version("ms-swift"),
        "mlflow_version": package_version("mlflow"),
        "torch_version": package_version("torch"),
        "transformers_version": package_version("transformers"),
    }
    params.update(collect_data_params(validation_report, args.run_kind))
    for key in (
        "model_type",
        "template",
        "tuner_type",
        "target_modules",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "max_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_steps",
        "eval_strategy",
        "save_strategy",
        "attn_impl",
        "torch_dtype",
    ):
        if key in config_args:
            params[key] = str(config_args[key])

    run_name = args.run_name or (
        f"qwen3-vl-8b-tool-call-sft-v2-{args.run_kind}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORT_DIR / f"{run_name}-mlflow-summary.json"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "stage": args.run_kind,
                "base_model": "Qwen3-VL-8B-Instruct",
                "training_framework": "ms-swift",
                "dataset": "ecom-qa-tool-call-sft-v2",
                "metric_policy": "time_series_only",
            }
        )
        mlflow.log_params(params)
        for key, value, step in metric_history:
            mlflow.log_metric(sanitize_name(key), value, step=step)

        logged_artifacts: list[tuple[str, str]] = []
        log_artifact_if_exists(config_path, "config", logged_artifacts)
        log_artifact_if_exists(validation_path, "data", logged_artifacts)
        for filename in (
            f"{'smoke_' if args.run_kind == 'smoke' else ''}review_samples.jsonl",
            f"{'smoke_' if args.run_kind == 'smoke' else ''}metadata.jsonl",
            f"{'smoke_' if args.run_kind == 'smoke' else ''}api_failures.jsonl",
            "resource_report_smoke.json",
            "training_time_estimate_smoke.json",
        ):
            log_artifact_if_exists(DATA_DIR / filename, "data", logged_artifacts)
        log_artifact_if_exists(logging_path, "training", logged_artifacts)
        if latest_log is not None:
            log_artifact_if_exists(latest_log, "training", logged_artifacts)
        if checkpoint_dir is not None:
            for filename in ("trainer_state.json", "adapter_config.json", "README.md"):
                log_artifact_if_exists(checkpoint_dir / filename, checkpoint_dir.name, logged_artifacts)
        host_copied_artifacts = sync_artifacts_to_mlflow_host(run.info.artifact_uri, logged_artifacts)

        summary = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": run_name,
            "tracking_uri": tracking_uri,
            "run_kind": args.run_kind,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
            "metric_history_points": len(metric_history),
            "metric_names": sorted({name for name, _, _ in metric_history}),
            "params": params,
            "artifacts_logged": [source for source, _ in logged_artifacts],
            "artifacts_copied_to_mlflow_host": host_copied_artifacts,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        mlflow.log_artifact(str(summary_path), artifact_path="reports")

    print(json.dumps({"run_id": summary["run_id"], "run_name": run_name, "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
