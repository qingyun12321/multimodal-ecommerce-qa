#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_HOME="${WORKSPACE_HOME:-/workspace}"
APP_DIR="${LLAMA_CPP_DIR:-${WORKSPACE_HOME}/apps/llama.cpp}"
MODEL_REPO="${SUMMARY_MODEL_REPO:-jc-builds/Qwen3.5-9B-Q4_K_M-GGUF}"
MODEL_DIR="${SUMMARY_MODEL_DIR:-${WORKSPACE_HOME}/models/gguf/Qwen3.5-9B-Q4_K_M-GGUF}"
LOG_DIR="${WORKSPACE_HOME}/logs"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_HOME}/.cache/uv}"
export HF_HOME="${HF_HOME:-${WORKSPACE_HOME}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

mkdir -p "$(dirname "${APP_DIR}")" "${MODEL_DIR}" "${LOG_DIR}" "${UV_CACHE_DIR}" "${HF_HOME}"

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git
fi

if [ ! -d "${APP_DIR}/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${APP_DIR}"
else
  git -C "${APP_DIR}" pull --ff-only
fi

cmake -S "${APP_DIR}" -B "${APP_DIR}/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "${APP_DIR}/build" --config Release -j"$(nproc)"

uv run --no-project --with huggingface_hub python - <<PY
from huggingface_hub import snapshot_download
from pathlib import Path
repo = "${MODEL_REPO}"
target = Path("${MODEL_DIR}")
snapshot_download(repo_id=repo, local_dir=target, allow_patterns=["*.gguf"])
matches = sorted(target.rglob("*Q4_K_M*.gguf")) or sorted(target.rglob("*.gguf"))
if not matches:
    raise SystemExit(f"no GGUF files found in {target}")
(target / "ACTIVE_MODEL").write_text(str(matches[0]) + "\\n", encoding="utf-8")
print(matches[0])
PY

echo "llama.cpp build and Qwen3.5-9B GGUF model are ready."
