from __future__ import annotations

import argparse
import io
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from ecom_qa.common.paths import artifacts_root, data_path

DEFAULT_REPO_ID = "reonokiy/vsp-infoseek"
DEFAULT_OUTPUT_DIR = data_path("infoseek_sample")
DEFAULT_CACHE_DIR = artifacts_root() / "hf_cache"
DEFAULT_SAMPLE_SIZE = 15000
DEFAULT_SEED = 20260421
DEFAULT_ROWS_PER_SOURCE_SHARD = 2500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a reproducible random InfoSeek sample with local images and metadata."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--rows-per-source-shard",
        type=int,
        default=DEFAULT_ROWS_PER_SOURCE_SHARD,
        help="Approximate number of sampled rows to draw from each selected parquet shard.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to sample from.",
    )
    return parser.parse_args()


def proportional_targets(total: int, counts: dict[str, int]) -> dict[str, int]:
    exact = {name: total * count / sum(counts.values()) for name, count in counts.items()}
    targets = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(targets.values())
    if remainder:
        ranked = sorted(exact.items(), key=lambda item: item[1] - targets[item[0]], reverse=True)
        for name, _ in ranked[:remainder]:
            targets[name] += 1
    return targets


def random_bucket_counts(total: int, bucket_count: int, rng: random.Random) -> list[int]:
    counts = [0] * bucket_count
    for _ in range(total):
        counts[rng.randrange(bucket_count)] += 1
    return counts


def group_parquet_files(repo_files: list[str], splits: set[str]) -> dict[str, list[str]]:
    grouped = {split: [] for split in splits}
    for path in repo_files:
        if not path.startswith("data/") or not path.endswith(".parquet"):
            continue
        split = path.split("/")[-1].split("-")[0]
        if split in grouped:
            grouped[split].append(path)
    for split, files in grouped.items():
        files.sort()
        if not files:
            raise ValueError(f"No parquet files found for split {split!r}.")
    return grouped


def guess_image_extension(image_bytes: bytes) -> str:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = (image.format or "JPEG").lower()
    except Exception:
        image_format = "bin"
    return {
        "jpeg": "jpg",
        "png": "png",
        "webp": "webp",
        "gif": "gif",
        "bmp": "bmp",
        "tiff": "tiff",
    }.get(image_format, image_format)


def save_image(
    image_id: str,
    image_bytes: bytes,
    image_dir: Path,
    saved_images: dict[str, str],
) -> str:
    existing = saved_images.get(image_id)
    if existing:
        return existing

    extension = guess_image_extension(image_bytes)
    relative_path = Path("images") / f"{image_id}.{extension}"
    output_path = image_dir.parent / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    saved_images[image_id] = relative_path.as_posix()
    return saved_images[image_id]


def main() -> None:
    args = parse_args()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import load_dataset_builder
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing download dependency {exc.name!r}. Run `uv sync` before using this command.") from exc

    rng = random.Random(args.seed)

    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    image_dir = output_dir / "images"
    metadata_path = output_dir / "metadata.jsonl"
    manifest_path = output_dir / "manifest.json"

    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=False)
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    builder = load_dataset_builder(args.repo_id)
    split_rows = {
        split: builder.info.splits[split].num_examples
        for split in args.splits
        if split in builder.info.splits
    }
    missing_splits = sorted(set(args.splits) - set(split_rows))
    if missing_splits:
        raise ValueError(f"Unknown split(s): {', '.join(missing_splits)}")

    split_targets = proportional_targets(args.sample_size, split_rows)

    api = HfApi()
    repo_files = api.list_repo_files(args.repo_id, repo_type="dataset")
    parquet_files = group_parquet_files(repo_files, set(args.splits))

    saved_images: dict[str, str] = {}
    records: list[dict[str, object]] = []
    selected_files_manifest: list[dict[str, object]] = []

    for split in args.splits:
        split_target = split_targets[split]
        if split_target == 0:
            continue

        files = parquet_files[split]
        source_shard_count = min(
            len(files),
            max(1, math.ceil(split_target / args.rows_per_source_shard)),
        )
        selected_files = rng.sample(files, k=source_shard_count)
        per_file_targets = random_bucket_counts(split_target, source_shard_count, rng)

        for file_path, rows_to_take in zip(selected_files, per_file_targets, strict=True):
            if rows_to_take == 0:
                continue

            local_path = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    filename=file_path,
                    cache_dir=str(cache_dir),
                )
            )
            table = pq.read_table(local_path)
            if rows_to_take > table.num_rows:
                raise ValueError(
                    f"Requested {rows_to_take} rows from {file_path}, but shard only has {table.num_rows}."
                )

            row_indices = sorted(rng.sample(range(table.num_rows), rows_to_take))
            sampled_rows = table.take(pa.array(row_indices)).to_pylist()
            unique_images_before = len(saved_images)

            for row in sampled_rows:
                image_info = row["image"] or {}
                image_bytes = image_info.get("bytes")
                if not image_bytes:
                    continue

                image_path = save_image(
                    image_id=row["image_id"],
                    image_bytes=image_bytes,
                    image_dir=image_dir,
                    saved_images=saved_images,
                )
                records.append(
                    {
                        "data_id": row["data_id"],
                        "image_id": row["image_id"],
                        "image_path": image_path,
                        "question": row["question"],
                        "answer": row["answer"],
                        "answer_eval": row["answer_eval"],
                        "data_split": row["data_split"],
                        "entity_id": row["entity_id"],
                        "entity_text": row["entity_text"],
                        "source_repo": args.repo_id,
                        "source_parquet": file_path,
                    }
                )

            selected_files_manifest.append(
                {
                    "split": split,
                    "source_parquet": file_path,
                    "downloaded_path": str(local_path),
                    "shard_rows": table.num_rows,
                    "sampled_rows": rows_to_take,
                    "new_images_saved": len(saved_images) - unique_images_before,
                }
            )
            print(
                json.dumps(
                    {
                        "split": split,
                        "source_parquet": file_path,
                        "sampled_rows": rows_to_take,
                        "records_so_far": len(records),
                        "unique_images_so_far": len(saved_images),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    rng.shuffle(records)

    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "InfoSeek sample",
        "source_repo": args.repo_id,
        "official_repo": "https://github.com/open-vision-language/infoseek",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed,
        "requested_sample_size": args.sample_size,
        "actual_sample_size": len(records),
        "unique_images": len(saved_images),
        "splits": args.splits,
        "split_rows": split_rows,
        "split_targets": split_targets,
        "rows_per_source_shard": args.rows_per_source_shard,
        "sampling_method": (
            "Randomly preselect parquet shards per split, then uniformly sample rows within each "
            "selected shard. This keeps the download bounded while preserving cross-split randomness."
        ),
        "metadata_path": str(metadata_path),
        "image_root": str(image_dir),
        "selected_files": selected_files_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
