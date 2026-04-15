#!/usr/bin/env bash
set -euo pipefail
source /workspace/codex-env.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec codex "$@"
