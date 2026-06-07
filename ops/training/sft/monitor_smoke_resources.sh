#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_HOME="${WORKSPACE_HOME:-/workspace}"
OUT="${1:-${WORKSPACE_HOME}/data/ecom-qa-tool-call-sft-v2/resource_snapshot.json}"
mkdir -p "$(dirname "${OUT}")"

python3 - <<PY
import json
import subprocess
import time
from pathlib import Path

def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"

payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "df_workspace": run(["df", "-h", "/workspace"]),
    "free": run(["free", "-h"]),
    "gpu": run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw", "--format=csv,noheader"]),
    "processes": run(["bash", "-lc", "ps -eo pid,ppid,pcpu,pmem,rss,cmd | grep -E 'llama-server|swift|searxng|dockerd|python' | grep -v grep | head -40"]),
    "docker": run(["bash", "-lc", "docker ps --format '{{.Names}} {{.Status}} {{.Ports}}' 2>/dev/null || true"]),
    "docker_stats": run(["bash", "-lc", "docker stats --no-stream --format '{{json .}}' 2>/dev/null || true"]),
}
Path("${OUT}").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print("${OUT}")
PY
