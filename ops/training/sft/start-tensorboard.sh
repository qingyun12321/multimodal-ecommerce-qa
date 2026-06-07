#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SFT_CONFIG_DIR="${SFT_CONFIG_DIR:-${REPO_DIR}/configs/training/sft}"
source "${SFT_ENV_FILE:-${SFT_CONFIG_DIR}/sft-env.sh}"
exec tensorboard --logdir /workspace/runs --host 0.0.0.0 --port 6006
