from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_new_package_layout_imports() -> None:
    package = importlib.import_module("ecom_qa")

    assert package.__doc__
    assert importlib.import_module("ecom_qa.data.catalog")
    assert importlib.import_module("ecom_qa.retrieval.models")
    assert importlib.import_module("ecom_qa.evaluation.subcategory")


def test_dataset_and_script_layout() -> None:
    expected_paths = [
        ROOT / "dataset" / "catalog" / "products.jsonl",
        ROOT / "dataset" / "manifests" / "test_manifest.json",
        ROOT / "dataset" / "manifests" / "report.json",
        ROOT / "scripts" / "run_retrieval_benchmark.py",
        ROOT / "scripts" / "run_retrieval_pipeline.py",
        ROOT / "scripts" / "run_reranking.py",
        ROOT / "scripts" / "run_web_search.py",
        ROOT / "scripts" / "render_benchmark_report.py",
        ROOT / "scripts" / "render_pipeline_report.py",
    ]

    missing = [path for path in expected_paths if not path.exists()]
    assert missing == []
