#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def ssh(host: str, script: str, *args: str, ssh_config: str | None = None) -> str:
    cmd = ["ssh"]
    if ssh_config:
        cmd.extend(["-F", ssh_config])
    remote_cmd = " ".join(shlex.quote(part) for part in ["python3", "-", *args])
    cmd.extend([host, remote_cmd])
    return subprocess.check_output(cmd, input=script, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean old project-only MLflow runs and artifacts.")
    parser.add_argument("--experiment", action="append", default=["ecom-qa-tool-call-sft"])
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--ssh-host", default=os.environ.get("MLFLOW_ARTIFACT_SSH_HOST", "mlflow-tracking"))
    parser.add_argument("--ssh-config", default=os.environ.get("MLFLOW_ARTIFACT_SSH_CONFIG", "/root/.ssh/config"))
    parser.add_argument("--db-path", default="/root/mlflow/mlflow.db")
    parser.add_argument("--artifact-root", default="/root/mlflow/artifacts")
    args = parser.parse_args()

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=args.tracking_uri)
    deleted = {"tracking_uri": args.tracking_uri, "experiments": [], "runs": []}
    for name in args.experiment:
        exp = client.get_experiment_by_name(name)
        if exp is None:
            continue
        runs = client.search_runs([exp.experiment_id], run_view_type=3, max_results=50000)
        for run in runs:
            deleted["runs"].append(run.info.run_id)
            if run.info.lifecycle_stage != "deleted":
                client.delete_run(run.info.run_id)
        if exp.lifecycle_stage != "deleted":
            client.delete_experiment(exp.experiment_id)
        deleted["experiments"].append({"name": name, "experiment_id": exp.experiment_id})

    if args.hard and deleted["experiments"]:
        hard_script = r'''
import json
import shutil
import sqlite3
import sys
from pathlib import Path

names = json.loads(sys.argv[1])
db_path = Path(sys.argv[2])
artifact_root = Path(sys.argv[3])
conn = sqlite3.connect(db_path)
conn.execute("PRAGMA foreign_keys=OFF")
exp_ids = [
    str(row[0])
    for row in conn.execute(
        "select experiment_id from experiments where name in (%s)" % ",".join("?" for _ in names),
        names,
    )
]
run_ids = []
if exp_ids:
    run_ids = [
        row[0]
        for row in conn.execute(
            "select run_uuid from runs where experiment_id in (%s)" % ",".join("?" for _ in exp_ids),
            exp_ids,
        )
    ]
tables = [row[0] for row in conn.execute("select name from sqlite_master where type='table'")]
for table in tables:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    if run_ids and "run_uuid" in cols:
        conn.execute(
            f"delete from {table} where run_uuid in (%s)" % ",".join("?" for _ in run_ids),
            run_ids,
        )
    elif run_ids and "run_id" in cols:
        conn.execute(
            f"delete from {table} where run_id in (%s)" % ",".join("?" for _ in run_ids),
            run_ids,
        )
for table in tables:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    if exp_ids and "experiment_id" in cols:
        conn.execute(
            f"delete from {table} where cast(experiment_id as text) in (%s)" % ",".join("?" for _ in exp_ids),
            exp_ids,
        )
conn.commit()
conn.execute("VACUUM")
conn.close()
removed = []
for exp_id in exp_ids:
    path = artifact_root / exp_id
    if path.exists():
        shutil.rmtree(path)
        removed.append(str(path))
print(json.dumps({"experiment_ids": exp_ids, "run_ids": run_ids, "removed_artifact_dirs": removed}, ensure_ascii=False))
'''
        result = ssh(
            args.ssh_host,
            hard_script,
            json.dumps(args.experiment),
            args.db_path,
            args.artifact_root,
            ssh_config=args.ssh_config if args.ssh_config else None,
        )
        deleted["hard_delete"] = json.loads(result)

    print(json.dumps(deleted, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
