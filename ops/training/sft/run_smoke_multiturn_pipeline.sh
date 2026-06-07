#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SFT_CONFIG_DIR="${SFT_CONFIG_DIR:-${REPO_DIR}/configs/training/sft}"
source "${SFT_ENV_FILE:-${SFT_CONFIG_DIR}/sft-env.sh}"

mkdir -p "${SFT_DATA_DIR}" /workspace/logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="/workspace/logs/smoke-sft-${STAMP}.log"

"${SCRIPT_DIR}/monitor_smoke_resources.sh" "${SFT_DATA_DIR}/resource_snapshot_before_smoke.json"

(cd "${REPO_DIR}" && uv run ecom-qa train sft-build-data \
  --repo-dir "${REPO_DIR}" \
  --output-dir "${SFT_DATA_DIR}" \
  --smoke \
  --smoke-train-size "${SMOKE_TRAIN_SIZE:-32}" \
  --smoke-val-size "${SMOKE_VAL_SIZE:-16}" \
  --web-mode searxng \
  --searxng-url "${SEARXNG_URL:-http://127.0.0.1:8080}" \
  --summary-url "${SUMMARY_URL:-http://127.0.0.1:8088}" \
  --live-web-limit "${SMOKE_LIVE_WEB_LIMIT:-32}" \
  --require-live-web)

"${SCRIPT_DIR}/monitor_smoke_resources.sh" "${SFT_DATA_DIR}/resource_snapshot_after_data.json"

set -o pipefail
"${SCRIPT_DIR}/run_smoke_sft.sh" 2>&1 | tee "${LOG_FILE}"

"${SCRIPT_DIR}/monitor_smoke_resources.sh" "${SFT_DATA_DIR}/resource_snapshot_after_train.json"

(cd "${REPO_DIR}" && uv run ecom-qa train sft-estimate-smoke-time)

python - <<PY
import json
from pathlib import Path
root = Path("${SFT_DATA_DIR}")
snapshots = []
for name in ["before_smoke", "after_data", "after_train"]:
    path = root / f"resource_snapshot_{name}.json"
    if path.exists():
        snapshots.append(json.loads(path.read_text(encoding="utf-8")))
(root / "resource_report_smoke.json").write_text(json.dumps({"snapshots": snapshots}, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY

(cd "${REPO_DIR}" && uv run ecom-qa train sft-sync-mlflow \
  --run-kind smoke \
  --experiment "${MLFLOW_EXPERIMENT_NAME:-ecom-qa-tool-call-sft-v2}" \
  --run-name "qwen3-vl-8b-tool-call-sft-v2-smoke-${STAMP}")

echo "Smoke pipeline completed. Main training was not started."
