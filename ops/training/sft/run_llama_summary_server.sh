#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_HOME="${WORKSPACE_HOME:-/workspace}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${WORKSPACE_HOME}/apps/llama.cpp}"
MODEL_DIR="${SUMMARY_MODEL_DIR:-${WORKSPACE_HOME}/models/gguf/Qwen3.5-9B-Q4_K_M-GGUF}"
MODEL_PATH="${SUMMARY_MODEL_PATH:-$(cat "${MODEL_DIR}/ACTIVE_MODEL")}"
PORT="${SUMMARY_PORT:-8088}"
LOG_DIR="${WORKSPACE_HOME}/logs"
PID_FILE="${WORKSPACE_HOME}/run/llama-summary-server.pid"
mkdir -p "${LOG_DIR}" "$(dirname "${PID_FILE}")"

if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "llama summary server already running at http://127.0.0.1:${PORT}"
  exit 0
fi

if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
  echo "Removing stale health check state for pid $(cat "${PID_FILE}")"
fi

nohup "${LLAMA_CPP_DIR}/build/bin/llama-server" \
  -m "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --ctx-size "${SUMMARY_CTX_SIZE:-32768}" \
  --parallel "${SUMMARY_PARALLEL:-2}" \
  --n-gpu-layers "${SUMMARY_N_GPU_LAYERS:-99}" \
  --reasoning off \
  --reasoning-budget 0 \
  --alias qwen3.5-9b-summary \
  > "${LOG_DIR}/llama-summary-server.log" 2>&1 &
echo $! > "${PID_FILE}"

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "llama summary server is ready at http://127.0.0.1:${PORT}"
    exit 0
  fi
  sleep 2
done

tail -80 "${LOG_DIR}/llama-summary-server.log" || true
exit 1
