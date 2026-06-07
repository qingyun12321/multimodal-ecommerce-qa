#!/usr/bin/env bash
set -euo pipefail
LOG=/workspace/logs/mlflow-tunnel.log
mkdir -p /workspace/logs
while true; do
  date '+%F %T starting mlflow ssh tunnel' >> "$LOG"
  ssh -F /root/.ssh/config -N -L 127.0.0.1:5000:127.0.0.1:5000 mlflow-tracking >> "$LOG" 2>&1 || true
  date '+%F %T tunnel exited; restarting in 5s' >> "$LOG"
  sleep 5
done
