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

BUILD_ARGS=(
  --output-dir "${SFT_DATA_DIR}"
  --smoke
  --smoke-train-size "${SMOKE_TRAIN_SIZE:-24}"
  --smoke-val-size "${SMOKE_VAL_SIZE:-8}"
  --api-model "${SFT_API_MODEL}"
  --api-reasoning-effort "${SFT_API_REASONING_EFFORT}"
  --api-workers "${SFT_API_WORKERS}"
  --worker-fallbacks "${SFT_API_WORKER_FALLBACKS}"
  --web-max-items "${SFT_WEB_MAX_ITEMS}"
  --api-timeout-seconds "${SFT_API_TIMEOUT_SECONDS:-900}"
  --max-retries "${SFT_API_MAX_RETRIES:-2}"
  --searxng-url "${SFT_SEARXNG_URL}"
  --searxng-engines "${SFT_SEARXNG_ENGINES}"
  --clear-output
  --clear-cache
)

if [[ "${SFT_NO_API_CACHE:-0}" == "1" || "${SFT_NO_API_CACHE:-false}" == "true" ]]; then
  BUILD_ARGS+=(--no-api-cache)
fi
if [[ "${SFT_NO_WEB_CACHE:-0}" == "1" || "${SFT_NO_WEB_CACHE:-false}" == "true" ]]; then
  BUILD_ARGS+=(--no-web-cache)
fi
if [[ "${SFT_KEEP_API_DEBUG:-0}" == "1" || "${SFT_KEEP_API_DEBUG:-false}" == "true" ]]; then
  BUILD_ARGS+=(--keep-api-debug)
fi

(cd "${REPO_DIR}" && uv run ecom-qa train sft-build-data "${BUILD_ARGS[@]}")

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
