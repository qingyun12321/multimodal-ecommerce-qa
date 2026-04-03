#!/usr/bin/env bash
set -euo pipefail
source /workspace/codex-env.sh
cd /workspace/multimodal_ecommerce_qa
exec codex "$@"
