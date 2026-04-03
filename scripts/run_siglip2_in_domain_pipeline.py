from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ecom_rag.data import build_query_text, build_subcategory_queries, load_products
from ecom_rag.in_domain_pipeline import (
    TOP_KS,
    build_aggregated_candidates,
    build_case_metrics,
    build_ranked_cases,
    build_siglip2_prompted_queries,
    sample_repeat_cases,
    write_json,
)
from ecom_rag.model_retrievers import ModelRunConfig, build_retriever
from ecom_rag.rerankers import BGEReranker, QwenFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the remaining in-domain RAG pipeline with SigLIP2.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/generated"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--skip-qwen", action="store_true")
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


def make_custom_query(category: str, canonical_zh: str) -> str:
    return (
        f"我想找一款属于“{category}”类目的“{canonical_zh}”。"
        "请优先保留真正属于这个子类别、商品信息匹配、明显不是其他近似子类别的结果。"
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_rerank_payload(
    *,
    products,
    aggregated_candidates: list[dict[str, object]],
    method_name: str,
    ranked_rows: list[dict[str, object]],
) -> dict[str, object]:
    relevance_ratios = [row["top5_relevance_ratio"] for row in ranked_rows]
    filtered_ratios = [row["filtered_irrelevant_ratio"] for row in ranked_rows]
    return {
        "method_name": method_name,
        "case_count": len(ranked_rows),
        "avg_top5_relevance_ratio": sum(relevance_ratios) / len(relevance_ratios),
        "avg_filtered_irrelevant_ratio": sum(filtered_ratios) / len(filtered_ratios),
        "rows": ranked_rows,
    }


def main() -> None:
    args = parse_args()
    env_updates = ensure_workspace_env()

    products = load_products(args.dataset_dir)
    queries = build_subcategory_queries(products, mode="slug_en")
    query_texts = build_siglip2_prompted_queries(queries)
    image_paths = [args.dataset_dir / product.image_path for product in products]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"siglip2_in_domain_pipeline_{timestamp}"
    report_dir = args.report_dir / run_name
    artifact_dir = args.artifact_dir / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    retriever = build_retriever(
        ModelRunConfig(
            name="siglip2",
            image_batch_size=args.image_batch_size,
            text_batch_size=args.text_batch_size,
            torch_dtype=args.torch_dtype,
        ),
        device=args.device,
    )
    image_embeddings = retriever.encode_images(image_paths, batch_size=args.image_batch_size)
    text_embeddings = retriever.encode_texts(query_texts, batch_size=args.text_batch_size)
    repeat_cases = sample_repeat_cases(queries, repeats=args.repeat_count, seed=args.seed)
    ranked_cases = build_ranked_cases(
        products=products,
        queries=queries,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        repeat_cases=repeat_cases,
    )

    summary_metrics, per_case_metrics = build_case_metrics(ranked_cases, queries)
    aggregated_candidates = build_aggregated_candidates(ranked_cases, products)

    write_json(
        report_dir / "in_domain_summary.json",
        {
            "run_name": run_name,
            "model_name": "siglip2",
            "checkpoint": "google/siglip2-base-patch16-224",
            "dataset_dir": str(args.dataset_dir),
            "repeat_count": args.repeat_count,
            "seed": args.seed,
            "ks": list(TOP_KS),
            "workspace_env": env_updates,
            "summary_metrics": summary_metrics,
        },
    )
    write_json(report_dir / "per_case_metrics.json", {"rows": per_case_metrics})
    write_json(report_dir / "aggregated_candidates.json", {"rows": aggregated_candidates})
    write_json(
        artifact_dir / "run_manifest.json",
        {
            "run_name": run_name,
            "report_dir": str(report_dir),
            "artifact_dir": str(artifact_dir),
        },
    )

    with (report_dir / "ranked_cases.jsonl").open("w", encoding="utf-8") as fh:
        for ranked_case in ranked_cases:
            fh.write(json.dumps(asdict(ranked_case), ensure_ascii=False) + "\n")

    bge = BGEReranker(device=args.device, torch_dtype=args.torch_dtype)
    rerank_rows_bge: list[dict[str, object]] = []
    rerank_rows_qwen4b: list[dict[str, object]] = []
    rerank_rows_qwen9b: list[dict[str, object]] = []

    qwen4b = None if args.skip_qwen else QwenFilter(model_id="Qwen/Qwen3.5-4B", device=args.device)
    qwen9b = None if args.skip_qwen else QwenFilter(model_id="Qwen/Qwen3.5-9B", device=args.device)

    query_lookup = {query.slug: query for query in queries}

    for row in aggregated_candidates:
        candidate_pairs = [
            (item["index"], products[item["index"]])
            for item in row["multi_route_top10"]
        ]
        query = make_custom_query(row["category"], row["canonical_zh"])
        query_subcategory = query_lookup[row["subcategory_slug"]].canonical_zh

        def compute_ratios(top5_indices: list[int], filtered_indices: list[int]) -> tuple[float, float]:
            top5_matches = [products[index].subcategory == query_subcategory for index in top5_indices]
            filtered_mismatches = [products[index].subcategory != query_subcategory for index in filtered_indices]
            relevance_ratio = float(sum(top5_matches) / len(top5_matches))
            filtered_ratio = float(sum(filtered_mismatches) / len(filtered_mismatches)) if filtered_indices else 0.0
            return relevance_ratio, filtered_ratio

        bge_ranked = bge.rank(query=query, documents=candidate_pairs, batch_size=args.rerank_batch_size)
        bge_top5 = bge_ranked[:5]
        bge_filtered = bge_ranked[5:]
        bge_rel_ratio, bge_filtered_ratio = compute_ratios(
            [item.index for item in bge_top5],
            [item.index for item in bge_filtered],
        )
        rerank_rows_bge.append(
            {
                "subcategory_slug": row["subcategory_slug"],
                "category": row["category"],
                "canonical_zh": row["canonical_zh"],
                "custom_query": query,
                "top5_product_ids": [item.product_id for item in bge_top5],
                "top5_indices": [item.index for item in bge_top5],
                "top5_scores": [item.score for item in bge_top5],
                "filtered_product_ids": [item.product_id for item in bge_filtered],
                "filtered_indices": [item.index for item in bge_filtered],
                "top5_relevance_ratio": bge_rel_ratio,
                "filtered_irrelevant_ratio": bge_filtered_ratio,
            }
        )

        if qwen4b is not None:
            qwen4b_ranked, qwen4b_payload = qwen4b.rank(query=query, candidates=candidate_pairs)
            qwen4b_top5 = qwen4b_ranked[:5]
            qwen4b_filtered = [item for item in candidate_pairs if item[0] not in {doc.index for doc in qwen4b_top5}]
            qwen4b_rel_ratio, qwen4b_filtered_ratio = compute_ratios(
                [item.index for item in qwen4b_top5],
                [index for index, _ in qwen4b_filtered],
            )
            rerank_rows_qwen4b.append(
                {
                    "subcategory_slug": row["subcategory_slug"],
                    "category": row["category"],
                    "canonical_zh": row["canonical_zh"],
                    "custom_query": query,
                    "top5_product_ids": [item.product_id for item in qwen4b_top5],
                    "top5_indices": [item.index for item in qwen4b_top5],
                    "filtered_product_ids": [products[index].id for index, _ in qwen4b_filtered],
                    "filtered_indices": [index for index, _ in qwen4b_filtered],
                    "top5_relevance_ratio": qwen4b_rel_ratio,
                    "filtered_irrelevant_ratio": qwen4b_filtered_ratio,
                    "raw_response": qwen4b_payload,
                }
            )

        if qwen9b is not None:
            qwen9b_ranked, qwen9b_payload = qwen9b.rank(query=query, candidates=candidate_pairs)
            qwen9b_top5 = qwen9b_ranked[:5]
            qwen9b_filtered = [item for item in candidate_pairs if item[0] not in {doc.index for doc in qwen9b_top5}]
            qwen9b_rel_ratio, qwen9b_filtered_ratio = compute_ratios(
                [item.index for item in qwen9b_top5],
                [index for index, _ in qwen9b_filtered],
            )
            rerank_rows_qwen9b.append(
                {
                    "subcategory_slug": row["subcategory_slug"],
                    "category": row["category"],
                    "canonical_zh": row["canonical_zh"],
                    "custom_query": query,
                    "top5_product_ids": [item.product_id for item in qwen9b_top5],
                    "top5_indices": [item.index for item in qwen9b_top5],
                    "filtered_product_ids": [products[index].id for index, _ in qwen9b_filtered],
                    "filtered_indices": [index for index, _ in qwen9b_filtered],
                    "top5_relevance_ratio": qwen9b_rel_ratio,
                    "filtered_irrelevant_ratio": qwen9b_filtered_ratio,
                    "raw_response": qwen9b_payload,
                }
            )

    bge_payload = build_rerank_payload(
        products=products,
        aggregated_candidates=aggregated_candidates,
        method_name="bge_reranker_v2_m3",
        ranked_rows=rerank_rows_bge,
    )
    write_json(report_dir / "bge_reranker_results.json", bge_payload)

    if rerank_rows_qwen4b:
        write_json(
            report_dir / "qwen3_5_4b_filter_results.json",
            build_rerank_payload(
                products=products,
                aggregated_candidates=aggregated_candidates,
                method_name="qwen3_5_4b",
                ranked_rows=rerank_rows_qwen4b,
            ),
        )
    if rerank_rows_qwen9b:
        write_json(
            report_dir / "qwen3_5_9b_filter_results.json",
            build_rerank_payload(
                products=products,
                aggregated_candidates=aggregated_candidates,
                method_name="qwen3_5_9b",
                ranked_rows=rerank_rows_qwen9b,
            ),
        )

    write_csv(
        report_dir / "rerank_summary.csv",
        [
            {
                "method_name": "bge_reranker_v2_m3",
                "avg_top5_relevance_ratio": bge_payload["avg_top5_relevance_ratio"],
                "avg_filtered_irrelevant_ratio": bge_payload["avg_filtered_irrelevant_ratio"],
            },
            *(
                [
                    {
                        "method_name": "qwen3_5_4b",
                        "avg_top5_relevance_ratio": sum(row["top5_relevance_ratio"] for row in rerank_rows_qwen4b) / len(rerank_rows_qwen4b),
                        "avg_filtered_irrelevant_ratio": sum(row["filtered_irrelevant_ratio"] for row in rerank_rows_qwen4b) / len(rerank_rows_qwen4b),
                    }
                ]
                if rerank_rows_qwen4b
                else []
            ),
            *(
                [
                    {
                        "method_name": "qwen3_5_9b",
                        "avg_top5_relevance_ratio": sum(row["top5_relevance_ratio"] for row in rerank_rows_qwen9b) / len(rerank_rows_qwen9b),
                        "avg_filtered_irrelevant_ratio": sum(row["filtered_irrelevant_ratio"] for row in rerank_rows_qwen9b) / len(rerank_rows_qwen9b),
                    }
                ]
                if rerank_rows_qwen9b
                else []
            ),
        ],
    )

    retriever.cleanup()
    bge.cleanup()
    if qwen4b is not None:
        qwen4b.cleanup()
    if qwen9b is not None:
        qwen9b.cleanup()
    del image_embeddings
    del text_embeddings
    gc.collect()

    print(
        json.dumps(
            {
                "run_name": run_name,
                "report_dir": str(report_dir),
                "summary_metrics": summary_metrics,
                "rerank_methods": [
                    "bge_reranker_v2_m3",
                    *([] if args.skip_qwen else ["qwen3_5_4b", "qwen3_5_9b"]),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
