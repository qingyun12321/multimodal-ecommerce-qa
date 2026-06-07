from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the current repository root when commands are run from this checkout."""
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "ecom_qa").exists():
            return candidate
    return current


def data_root() -> Path:
    return Path(os.environ.get("ECOM_QA_DATA_ROOT", "data"))


def resolve_data_root(base_dir: Path | None = None) -> Path:
    root = data_root()
    if root.is_absolute():
        return root
    return (base_dir or repo_root()) / root


def data_path(*parts: str) -> Path:
    return data_root().joinpath(*parts)


def artifacts_root() -> Path:
    return Path(os.environ.get("ECOM_QA_ARTIFACTS_ROOT", "artifacts"))


def generated_reports_root() -> Path:
    return Path(os.environ.get("ECOM_QA_REPORTS_ROOT", "reports/generated"))


def experiments_root() -> Path:
    return Path("reports/experiments")
