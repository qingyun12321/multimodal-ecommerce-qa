#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_HOME="${WORKSPACE_HOME:-/workspace}"
APP_DIR="${SEARXNG_APP_DIR:-${WORKSPACE_HOME}/apps/searxng}"
VENV_DIR="${SEARXNG_VENV_DIR:-${WORKSPACE_HOME}/venvs/searxng}"
SERVICE_DIR="${SEARXNG_UV_SERVICE_DIR:-${WORKSPACE_HOME}/services/searxng-uv}"
LOG_DIR="${WORKSPACE_HOME}/logs"
RUN_DIR="${WORKSPACE_HOME}/run"
PORT="${SEARXNG_PORT:-8080}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_HOME}/.cache/uv}"

mkdir -p "$(dirname "${APP_DIR}")" "$(dirname "${VENV_DIR}")" "${SERVICE_DIR}" "${LOG_DIR}" "${RUN_DIR}" "${UV_CACHE_DIR}"

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git
fi

if [ ! -d "${APP_DIR}/.git" ]; then
  git clone --depth 1 https://github.com/searxng/searxng.git "${APP_DIR}"
else
  git -C "${APP_DIR}" pull --ff-only
fi

uv venv --python 3.12 "${VENV_DIR}"
uv pip install --python "${VENV_DIR}/bin/python" -r "${APP_DIR}/requirements.txt" -r "${APP_DIR}/requirements-server.txt"
uv pip install --python "${VENV_DIR}/bin/python" setuptools wheel
uv pip install --python "${VENV_DIR}/bin/python" --no-build-isolation -e "${APP_DIR}"

if [ ! -f "${SERVICE_DIR}/settings.yml" ]; then
  SECRET_KEY="$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
)"
  cat >"${SERVICE_DIR}/settings.yml" <<YAML
use_default_settings: true

general:
  debug: false
  instance_name: "workspace-searxng-uv"

search:
  safe_search: 1
  autocomplete: ""
  formats:
    - html
    - json

server:
  bind_address: "127.0.0.1"
  port: ${PORT}
  secret_key: "${SECRET_KEY}"
  limiter: false
  image_proxy: false
YAML
fi

PID_FILE="${RUN_DIR}/searxng-uv.pid"
if [ -f "${PID_FILE}" ]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if ps -p "${OLD_PID}" -o cmd= 2>/dev/null | grep -q "searx/webapp.py"; then
    kill "${OLD_PID}" || true
    sleep 2
  fi
fi

export SEARXNG_SETTINGS_PATH="${SERVICE_DIR}/settings.yml"
nohup "${VENV_DIR}/bin/python" "${APP_DIR}/searx/webapp.py" > "${LOG_DIR}/searxng-uv.log" 2>&1 < /dev/null &
echo $! > "${PID_FILE}"

for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:${PORT}/config" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -fsS "http://127.0.0.1:${PORT}/search?q=ping&format=json" >/dev/null
echo "SearXNG uv service is ready at http://127.0.0.1:${PORT}"
