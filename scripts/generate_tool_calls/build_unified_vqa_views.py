#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path("/home/qingyun/projects/multimodal-ecommerce-qa")
DATASET_DIR = WORKSPACE_ROOT / "dataset"
OUTPUT_DIR = DATASET_DIR / "unified_vqa"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    domain: str
    source_file: Path
    image_root: Path
    language: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unified VQA JSONL views without modifying original datasets."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing unified VQA JSONL and manifest files.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_absolute_image_path(image_root: Path, relative_path: str) -> Path:
    if not relative_path:
        raise ValueError(f"Missing image path under {image_root}")
    path = Path(relative_path)
    absolute = path if path.is_absolute() else image_root / path
    absolute = absolute.resolve()
    if not absolute.exists():
        raise FileNotFoundError(f"Image not found: {absolute}")
    return absolute


def detect_image_extension(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return suffix[1:] if suffix.startswith(".") else suffix


def normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def build_in_domain_record(
    row: dict[str, Any],
    source: SourceSpec,
    index: int,
) -> dict[str, Any]:
    image_abs = ensure_absolute_image_path(source.image_root, row["image_path"])
    answers = normalize_answers(row.get("answer"))
    source_key = row.get("product_id") or f"row_{index:06d}"
    return {
        "record_id": f"{source.name}:{index:06d}",
        "task_type": "single_image_vqa",
        "domain": source.domain,
        "source_dataset": source.name,
        "source_file": str(source.source_file),
        "source_record_key": source_key,
        "language": source.language,
        "question": row["query"],
        "answers": answers,
        "primary_answer": answers[0] if answers else "",
        "image_path": str(image_abs),
        "image_rel_path": row["image_path"],
        "image_id": row.get("product_id"),
        "image_format": detect_image_extension(image_abs),
        "metadata": {
            "qa_type": row.get("qa_type"),
            "source_row_index": row.get("source_row_index"),
            "product_id": row.get("product_id"),
            "source_title": row.get("source_title"),
            "elapsed_seconds": row.get("elapsed_seconds"),
        },
    }


def build_infoseek_record(
    row: dict[str, Any],
    source: SourceSpec,
    index: int,
) -> dict[str, Any]:
    image_abs = ensure_absolute_image_path(source.image_root, row["image_path"])
    answers = normalize_answers(row.get("answer"))
    return {
        "record_id": f"{source.name}:{index:06d}",
        "task_type": "single_image_vqa",
        "domain": source.domain,
        "source_dataset": source.name,
        "source_file": str(source.source_file),
        "source_record_key": row["data_id"],
        "language": source.language,
        "question": row["question"],
        "answers": answers,
        "primary_answer": answers[0] if answers else "",
        "image_path": str(image_abs),
        "image_rel_path": row["image_path"],
        "image_id": row.get("image_id"),
        "image_format": detect_image_extension(image_abs),
        "metadata": {
            "data_id": row.get("data_id"),
            "data_split": row.get("data_split"),
            "entity_id": row.get("entity_id"),
            "entity_text": row.get("entity_text"),
            "answer_eval": row.get("answer_eval"),
            "source_repo": row.get("source_repo"),
            "source_parquet": row.get("source_parquet"),
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    image_counts = Counter(record["image_path"] for record in records)
    return {
        "rows": len(records),
        "unique_images": len(image_counts),
        "reused_images": sum(1 for count in image_counts.values() if count > 1),
        "languages": dict(Counter(record["language"] for record in records)),
        "domains": dict(Counter(record["domain"] for record in records)),
        "source_datasets": dict(Counter(record["source_dataset"] for record in records)),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    sources = [
        SourceSpec(
            name="ecom_qa_pairs",
            domain="in_domain",
            source_file=DATASET_DIR / "qa" / "qa_pairs.jsonl",
            image_root=DATASET_DIR,
            language="zh",
        ),
        SourceSpec(
            name="ecom_qa_pairs_open_question",
            domain="in_domain",
            source_file=DATASET_DIR / "qa" / "qa_pairs_open_question.jsonl",
            image_root=DATASET_DIR,
            language="zh",
        ),
        SourceSpec(
            name="ecom_qa_pairs_supplement",
            domain="in_domain",
            source_file=DATASET_DIR / "qa" / "qa_pairs_supplement.jsonl",
            image_root=DATASET_DIR,
            language="zh",
        ),
        SourceSpec(
            name="infoseek_sample",
            domain="out_of_domain",
            source_file=DATASET_DIR / "infoseek_sample" / "metadata.jsonl",
            image_root=DATASET_DIR / "infoseek_sample",
            language="en",
        ),
    ]

    all_records: list[dict[str, Any]] = []
    per_source_outputs: dict[str, str] = {}
    per_source_summary: dict[str, dict[str, Any]] = {}

    output_dir.mkdir(parents=True, exist_ok=args.overwrite)

    for source in sources:
        rows = load_jsonl(source.source_file)
        if source.name == "infoseek_sample":
            records = [build_infoseek_record(row, source, index) for index, row in enumerate(rows)]
        else:
            records = [build_in_domain_record(row, source, index) for index, row in enumerate(rows)]

        output_path = output_dir / f"{source.name}.jsonl"
        write_jsonl(output_path, records)
        per_source_outputs[source.name] = str(output_path)
        per_source_summary[source.name] = summarize(records)
        all_records.extend(records)

    combined_path = output_dir / "combined.jsonl"
    write_jsonl(combined_path, all_records)

    manifest = {
        "description": "Unified VQA views built from in-domain ecommerce QA and out-of-domain InfoSeek samples.",
        "schema": {
            "record_id": "Stable view-specific row id.",
            "task_type": "Currently always single_image_vqa.",
            "domain": "in_domain or out_of_domain.",
            "source_dataset": "Original dataset name.",
            "source_file": "Original JSONL file path.",
            "source_record_key": "Stable source-side identifier.",
            "language": "Question language.",
            "question": "User-facing question text.",
            "answers": "Normalized list of acceptable answers.",
            "primary_answer": "First answer in answers.",
            "image_path": "Absolute path to the original image file. No image copies are made.",
            "image_rel_path": "Original relative image path from the source dataset.",
            "image_id": "Source-side image identifier when available.",
            "image_format": "File extension without dot.",
            "metadata": "Source-specific extra fields retained for traceability.",
        },
        "outputs": {
            "combined": str(combined_path),
            "per_source": per_source_outputs,
        },
        "summary": {
            "combined": summarize(all_records),
            "per_source": per_source_summary,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
