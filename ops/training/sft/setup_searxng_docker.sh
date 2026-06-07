#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_HOME="${WORKSPACE_HOME:-/workspace}"
SERVICE_DIR="${SEARXNG_SERVICE_DIR:-${WORKSPACE_HOME}/services/searxng}"
DOCKER_ROOT="${DOCKER_ROOT:-${WORKSPACE_HOME}/docker}"
DOCKER_EXEC_ROOT="${DOCKER_EXEC_ROOT:-/var/run/docker}"
DOCKER_STORAGE_DRIVER="${DOCKER_STORAGE_DRIVER:-vfs}"
LOG_DIR="${WORKSPACE_HOME}/logs"
SEARXNG_PORT="${SEARXNG_PORT:-8080}"

mkdir -p "${SERVICE_DIR}/core-config" "${SERVICE_DIR}/data" "${SERVICE_DIR}/valkey-data" \
  "${DOCKER_ROOT}" "${DOCKER_EXEC_ROOT}" "${LOG_DIR}" /etc/docker

if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ca-certificates curl gnupg lsb-release
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

cat >/etc/docker/daemon.json <<JSON
{
  "data-root": "${DOCKER_ROOT}",
  "exec-root": "${DOCKER_EXEC_ROOT}",
  "storage-driver": "${DOCKER_STORAGE_DRIVER}",
  "features": {
    "containerd-snapshotter": false
  },
  "iptables": false,
  "ip6tables": false,
  "ip-forward": false,
  "ip-masq": false,
  "ipv6": false,
  "bridge": "none",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  }
}
JSON

if ! docker info >/dev/null 2>&1; then
  nohup dockerd --config-file /etc/docker/daemon.json --host unix:///var/run/docker.sock \
    > "${LOG_DIR}/dockerd.log" 2>&1 &
  for _ in $(seq 1 60); do
    if docker info >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi
docker info >/dev/null

SECRET_KEY="$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
)"

cat >"${SERVICE_DIR}/core-config/settings.yml" <<YAML
use_default_settings: true

general:
  debug: false
  instance_name: "workspace-searxng"

search:
  safe_search: 1
  autocomplete: ""
  formats:
    - html
    - json

server:
  bind_address: "0.0.0.0"
  port: 8080
  secret_key: "${SECRET_KEY}"
  limiter: false
  image_proxy: false

redis:
  url: redis://127.0.0.1:6379/0
YAML

cat >"${SERVICE_DIR}/docker-compose.yml" <<YAML
services:
  valkey:
    image: docker.io/valkey/valkey:8-alpine
    container_name: workspace-searxng-valkey
    restart: unless-stopped
    network_mode: host
    command: valkey-server --bind 127.0.0.1 --port 6379 --save 30 1 --loglevel warning
    volumes:
      - ${SERVICE_DIR}/valkey-data:/data

  searxng:
    image: docker.io/searxng/searxng:latest
    container_name: workspace-searxng
    restart: unless-stopped
    network_mode: host
    volumes:
      - ${SERVICE_DIR}/core-config:/etc/searxng:rw
      - ${SERVICE_DIR}/data:/var/cache/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://127.0.0.1:${SEARXNG_PORT}/
    depends_on:
      - valkey
YAML

docker compose -f "${SERVICE_DIR}/docker-compose.yml" up -d

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${SEARXNG_PORT}/config" >/dev/null; then
    break
  fi
  sleep 1
done

curl -fsS "http://127.0.0.1:${SEARXNG_PORT}/search?q=ping&format=json" >/dev/null
echo "SearXNG is ready at http://127.0.0.1:${SEARXNG_PORT}; DockerRootDir=$(docker info --format '{{.DockerRootDir}}')"
