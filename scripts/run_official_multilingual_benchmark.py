from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from ecom_rag.data import build_subcategory_queries, load_products
from ecom_rag.eval import compute_ranked_lists
from ecom_rag.model_retrievers import ModelRunConfig, build_retriever
from ecom_rag.subcategory_eval import evaluate_subcategory_queries


MODEL_ORDER = ["clip", "chinese_clip", "siglip2", "gme_2b"]
LANGUAGE_ORDER = ["en", "zh"]


@dataclass(frozen=True, slots=True)
class LanguageVariant:
    input_language: str
    query_mode: str
    prompt_template: str | None
    prompt_label: str


OFFICIAL_VARIANTS: dict[str, tuple[LanguageVariant, LanguageVariant]] = {
    "clip": (
        LanguageVariant(
            input_language="en",
            query_mode="slug_en",
            prompt_template="a photo of a {label}",
            prompt_label="OpenAI CLIP template",
        ),
        LanguageVariant(
            input_language="zh",
            query_mode="canonical_zh",
            prompt_template="a photo of a {label}",
            prompt_label="OpenAI CLIP template with Chinese label",
        ),
    ),
    "chinese_clip": (
        LanguageVariant(
            input_language="en",
            query_mode="slug_en",
            prompt_template=None,
            prompt_label="raw label",
        ),
        LanguageVariant(
            input_language="zh",
            query_mode="canonical_zh",
            prompt_template=None,
            prompt_label="raw label",
        ),
    ),
    "siglip2": (
        LanguageVariant(
            input_language="en",
            query_mode="slug_en",
            prompt_template="This is a photo of {label}.",
            prompt_label="SigLIP2 official template",
        ),
        LanguageVariant(
            input_language="zh",
            query_mode="canonical_zh",
            prompt_template="This is a photo of {label}.",
            prompt_label="SigLIP2 official template with Chinese label",
        ),
    ),
    "gme_2b": (
        LanguageVariant(
            input_language="en",
            query_mode="slug_en",
            prompt_template=None,
            prompt_label="raw label + official retrieval instruction",
        ),
        LanguageVariant(
            input_language="zh",
            query_mode="canonical_zh",
            prompt_template=None,
            prompt_label="raw label + official retrieval instruction",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 8-row official-aligned multilingual text-to-image benchmark."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/generated"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--models",
        default="clip,chinese_clip,siglip2,gme_2b",
        help="Comma-separated serial benchmark order.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else None,
    )
    parser.add_argument("--hf-image-batch-size", type=int, default=64)
    parser.add_argument("--hf-text-batch-size", type=int, default=64)
    parser.add_argument("--hf-torch-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--gme-image-batch-size", type=int, default=48)
    parser.add_argument("--gme-text-batch-size", type=int, default=32)
    parser.add_argument("--gme-torch-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--gme-max-image-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def ensure_workspace_env() -> dict[str, str]:
    root = Path.cwd()
    hf_home = root / "artifacts" / "hf_home"
    env_updates = {
        "HF_HOME": str(hf_home),
        "TRANSFORMERS_CACHE": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    for key, value in env_updates.items():
        os.environ[key] = value
    return env_updates


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_model_configs(args: argparse.Namespace) -> list[ModelRunConfig]:
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    configs: list[ModelRunConfig] = []
    for name in requested:
        if name == "gme_2b":
            configs.append(
                ModelRunConfig(
                    name=name,
                    image_batch_size=args.gme_image_batch_size,
                    text_batch_size=args.gme_text_batch_size,
                    torch_dtype=args.gme_torch_dtype,
                    max_image_tokens=args.gme_max_image_tokens,
                )
            )
        else:
            configs.append(
                ModelRunConfig(
                    name=name,
                    image_batch_size=args.hf_image_batch_size,
                    text_batch_size=args.hf_text_batch_size,
                    torch_dtype=args.hf_torch_dtype,
                )
            )
    return configs


def prepare_queries(products, variant: LanguageVariant):
    queries = build_subcategory_queries(products, mode=variant.query_mode)
    if variant.prompt_template:
        query_texts = [
            variant.prompt_template.format(label=query.query_text) for query in queries
        ]
    else:
        query_texts = [query.query_text for query in queries]
    relevance_sets = [set(query.relevant_indices) for query in queries]
    positive_counts = [len(query.relevant_indices) for query in queries]
    return queries, query_texts, relevance_sets, positive_counts


def mean_precision(summary_metrics: dict[str, float]) -> float:
    keys = (
        "avg_precision@1",
        "avg_precision@3",
        "avg_precision@5",
        "avg_precision@10",
        "avg_precision@20",
        "avg_precision@50",
    )
    return float(sum(summary_metrics[key] for key in keys) / len(keys))


def write_variant_outputs(
    *,
    report_dir: Path,
    model_name: str,
    variant: LanguageVariant,
    queries,
    products,
    ranked_indices,
    scores,
    per_query_metrics,
    top_k: int,
) -> dict[str, str]:
    variant_dir = report_dir / model_name / variant.input_language
    variant_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = variant_dir / "subcategory_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "subcategory_slug",
                "canonical_zh",
                "query_text",
                "positive_count",
                "topk_image_paths",
                "topk_scores",
                "topk_correct_flags",
            ]
        )
        for idx, query in enumerate(queries):
            row = ranked_indices[idx]
            topk_indices = row[:top_k]
            positives = set(query.relevant_indices)
            writer.writerow(
                [
                    query.slug,
                    query.canonical_zh,
                    query.query_text,
                    len(query.relevant_indices),
                    [products[j].image_path for j in topk_indices],
                    [float(scores[idx, j]) for j in topk_indices],
                    [int(j in positives) for j in topk_indices],
                ]
            )

    per_query_path = variant_dir / "per_query_metrics.json"
    write_json(
        per_query_path,
        {
            "rows": [
                {
                    "subcategory_slug": query.slug,
                    "category_slug": query.category_slug,
                    "canonical_zh": query.canonical_zh,
                    "query_text": query.query_text,
                    **metrics,
                }
                for query, metrics in zip(queries, per_query_metrics, strict=True)
            ]
        },
    )
    return {
        "predictions_csv": str(predictions_path),
        "per_query_metrics": str(per_query_path),
    }


def build_combined_row(
    *,
    model_name: str,
    variant: LanguageVariant,
    summary_metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "input_language": variant.input_language,
        "query_mode": variant.query_mode,
        "prompt_template": variant.prompt_template,
        "prompt_label": variant.prompt_label,
        "avg_precision@1": summary_metrics["avg_precision@1"],
        "avg_precision@3": summary_metrics["avg_precision@3"],
        "avg_precision@5": summary_metrics["avg_precision@5"],
        "avg_precision@10": summary_metrics["avg_precision@10"],
        "avg_precision@20": summary_metrics["avg_precision@20"],
        "avg_precision@50": summary_metrics["avg_precision@50"],
        "avg_hit@1": summary_metrics["avg_hit@1"],
        "avg_hit@5": summary_metrics["avg_hit@5"],
        "avg_hit@10": summary_metrics["avg_hit@10"],
        "avg_precision_mean": mean_precision(summary_metrics),
    }


def main() -> None:
    args = parse_args()
    env_updates = ensure_workspace_env()
    device = resolve_device(args.device)

    products = load_products(args.dataset_dir)
    image_paths = [args.dataset_dir / product.image_path for product in products]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"official_multilingual_t2i_{timestamp}"
    report_dir = args.report_dir / run_name
    artifact_dir = args.artifact_dir / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, object]] = []
    model_summaries: dict[str, dict[str, object]] = {}

    for config in build_model_configs(args):
        print(f"=== Running {config.name} once for both languages ===", flush=True)
        retriever = build_retriever(config, device=device)
        image_embeddings = retriever.encode_images(
            image_paths,
            batch_size=config.image_batch_size,
        )

        model_summary: dict[str, object] = {
            "model_name": config.name,
            "device": device,
            "image_batch_size": config.image_batch_size,
            "text_batch_size": config.text_batch_size,
            "torch_dtype": config.torch_dtype,
            "max_image_tokens": config.max_image_tokens,
            "variants": {},
        }

        for variant in OFFICIAL_VARIANTS[config.name]:
            print(
                f"--- {config.name} / {variant.input_language} / {variant.prompt_label} ---",
                flush=True,
            )
            queries, query_texts, relevance_sets, positive_counts = prepare_queries(
                products,
                variant,
            )
            query_embeddings = retriever.encode_texts(
                query_texts,
                batch_size=config.text_batch_size,
            )
            scores, ranked_indices = compute_ranked_lists(query_embeddings, image_embeddings)
            summary_metrics, per_query_metrics = evaluate_subcategory_queries(
                ranked_indices,
                relevance_sets,
                positive_counts,
            )

            artifacts = write_variant_outputs(
                report_dir=report_dir,
                model_name=config.name,
                variant=variant,
                queries=queries,
                products=products,
                ranked_indices=ranked_indices,
                scores=scores,
                per_query_metrics=per_query_metrics,
                top_k=args.top_k,
            )
            combined_rows.append(
                build_combined_row(
                    model_name=config.name,
                    variant=variant,
                    summary_metrics=summary_metrics,
                )
            )
            model_summary["variants"][variant.input_language] = {
                "input_language": variant.input_language,
                "query_mode": variant.query_mode,
                "prompt_template": variant.prompt_template,
                "prompt_label": variant.prompt_label,
                "metrics": summary_metrics,
                "artifacts": artifacts,
            }

            del query_embeddings
            del scores
            del ranked_indices
            gc.collect()

        model_summaries[config.name] = model_summary
        write_json(report_dir / f"{config.name}_metrics.json", model_summary)
        retriever.cleanup()
        del image_embeddings
        gc.collect()

    combined_rows.sort(
        key=lambda row: (
            LANGUAGE_ORDER.index(str(row["input_language"])),
            MODEL_ORDER.index(str(row["model_name"])),
        )
    )

    write_json(
        report_dir / "combined_metrics.json",
        {
            "assignment_target": "文搜图（四模型中英文输入官方对齐对比）",
            "dataset_dir": str(args.dataset_dir),
            "image_count": len(image_paths),
            "workspace_env": env_updates,
            "row_count": len(combined_rows),
            "models": model_summaries,
            "combined_rows": combined_rows,
        },
    )
    with (report_dir / "combined_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(combined_rows[0].keys()))
        writer.writeheader()
        writer.writerows(combined_rows)

    write_json(
        artifact_dir / "run_manifest.json",
        {
            "run_name": run_name,
            "report_dir": str(report_dir),
            "artifact_dir": str(artifact_dir),
            "row_count": len(combined_rows),
        },
    )

    print(
        json.dumps(
            {
                "run_name": run_name,
                "report_dir": str(report_dir),
                "artifact_dir": str(artifact_dir),
                "combined_rows": combined_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
