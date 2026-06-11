#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SEARXNG_RUN="${SEARXNG_RUN:-/workspace/venvs/searxng/bin/searxng-run}"
SEARXNG_SETTINGS_PATH="${SEARXNG_SETTINGS_PATH:-${REPO_DIR}/configs/retrieval/searxng-sft.yml}"
SEARXNG_PORT="${SEARXNG_PORT:-8888}"
SEARXNG_URL="${SEARXNG_URL:-http://127.0.0.1:${SEARXNG_PORT}}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "${LOG_DIR}"

export SEARXNG_SETTINGS_PATH
export SEARXNG_PORT

if [[ "${SEARXNG_FOREGROUND:-0}" == "1" ]]; then
  exec "${SEARXNG_RUN}"
fi

"${SEARXNG_RUN}" >"${LOG_DIR}/searxng-sft.log" 2>&1 &
pid="$!"
trap 'kill "${pid}" >/dev/null 2>&1 || true' INT TERM EXIT

for _ in $(seq 1 60); do
  if curl -fsS "${SEARXNG_URL}/config" >/dev/null 2>&1; then
    curl -fsS --get "${SEARXNG_URL}/search" \
      --data-urlencode "q=test" \
      --data-urlencode "format=json" \
      --data-urlencode "engines=wikipedia" >/dev/null
    echo "SearXNG SFT service is ready at ${SEARXNG_URL}"
    wait "${pid}"
    exit $?
  fi
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "SearXNG exited before readiness. Log: ${LOG_DIR}/searxng-sft.log" >&2
    exit 1
  fi
  sleep 1
done

echo "Timed out waiting for SearXNG. Log: ${LOG_DIR}/searxng-sft.log" >&2
exit 1
