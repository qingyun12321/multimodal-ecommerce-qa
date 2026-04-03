from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from ecom_rag.data import build_query_text, build_subcategory_queries, load_products
from ecom_rag.rerankers import BGEReranker, QwenFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BGE and Qwen3.5 reranking on saved multi-route candidates.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--aggregated-candidates",
        type=Path,
        required=True,
        help="Path to aggregated_candidates.json from the in-domain pipeline run.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="Directory where rerank outputs should be written.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument(
        "--models",
        default="bge_reranker_v2_m3,qwen3_5_4b,qwen3_5_9b",
        help="Comma-separated method order.",
    )
    return parser.parse_args()


def ensure_workspace_env() -> None:
    root = Path.cwd()
    hf_home = root / "artifacts" / "hf_home"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_aggregated_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["rows"]


def make_custom_query(category: str, canonical_zh: str) -> str:
    return (
        f"我想找一款属于“{category}”类目的“{canonical_zh}”。"
        "请优先保留真正属于这个子类别、商品信息匹配、明显不是其他近似子类别的结果。"
    )


def build_case_row(
    *,
    method_name: str,
    subcategory_slug: str,
    category: str,
    canonical_zh: str,
    custom_query: str,
    top5_products,
    filtered_products,
    top5_relevance_ratio: float,
    filtered_irrelevant_ratio: float,
    raw_response: dict | None = None,
) -> dict[str, object]:
    return {
        "method_name": method_name,
        "subcategory_slug": subcategory_slug,
        "category": category,
        "canonical_zh": canonical_zh,
        "custom_query": custom_query,
        "top5_product_ids": [product["product_id"] for product in top5_products],
        "top5_indices": [product["index"] for product in top5_products],
        "top5_image_paths": [product["image_path"] for product in top5_products],
        "top5_documents": [product["document"] for product in top5_products],
        "top5_scores": [product.get("score") for product in top5_products],
        "filtered_product_ids": [product["product_id"] for product in filtered_products],
        "filtered_indices": [product["index"] for product in filtered_products],
        "filtered_image_paths": [product["image_path"] for product in filtered_products],
        "filtered_documents": [product["document"] for product in filtered_products],
        "top5_relevance_ratio": top5_relevance_ratio,
        "filtered_irrelevant_ratio": filtered_irrelevant_ratio,
        "raw_response": raw_response,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method_name",
                "avg_top5_relevance_ratio",
                "avg_filtered_irrelevant_ratio",
                "case_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_summary(method_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "method_name": method_name,
        "case_count": len(rows),
        "avg_top5_relevance_ratio": sum(row["top5_relevance_ratio"] for row in rows) / len(rows),
        "avg_filtered_irrelevant_ratio": sum(row["filtered_irrelevant_ratio"] for row in rows) / len(rows),
        "rows": rows,
    }


def with_suffix(name: str, suffix: str) -> str:
    return f"{name}{suffix}" if suffix else name


def main() -> None:
    args = parse_args()
    ensure_workspace_env()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    products = load_products(args.dataset_dir)
    query_lookup = {
        query.slug: query
        for query in build_subcategory_queries(products, mode="slug_en")
    }
    aggregated_rows = load_aggregated_rows(args.aggregated_candidates)
    if args.row_limit > 0:
        aggregated_rows = aggregated_rows[args.row_start : args.row_start + args.row_limit]
    else:
        aggregated_rows = aggregated_rows[args.row_start :]
    requested_methods = [name.strip() for name in args.models.split(",") if name.strip()]

    summaries: list[dict[str, object]] = []

    def build_docs(candidate_pairs):
        return [
            {
                "index": index,
                "product_id": product.id,
                "image_path": product.image_path,
                "document": build_query_text(product, mode="compact_rerank_zh"),
            }
            for index, product in candidate_pairs
        ]

    if "bge_reranker_v2_m3" in requested_methods:
        reranker = BGEReranker(device=args.device, torch_dtype=args.torch_dtype)
        rows: list[dict[str, object]] = []
        for row in aggregated_rows:
            candidate_pairs = [(item["index"], products[item["index"]]) for item in row["multi_route_top10"]]
            query = make_custom_query(row["category"], row["canonical_zh"])
            canonical = query_lookup[row["subcategory_slug"]].canonical_zh
            ranked = reranker.rank(query=query, documents=candidate_pairs, batch_size=args.rerank_batch_size)
            top5 = [
                {
                    "index": item.index,
                    "product_id": item.product_id,
                    "image_path": item.image_path,
                    "document": item.document,
                    "score": item.score,
                }
                for item in ranked[:5]
            ]
            filtered = [
                {
                    "index": item.index,
                    "product_id": item.product_id,
                    "image_path": item.image_path,
                    "document": item.document,
                    "score": item.score,
                }
                for item in ranked[5:]
            ]
            top5_relevance_ratio = sum(products[item["index"]].subcategory == canonical for item in top5) / len(top5)
            filtered_irrelevant_ratio = (
                sum(products[item["index"]].subcategory != canonical for item in filtered) / len(filtered)
                if filtered
                else 0.0
            )
            rows.append(
                build_case_row(
                    method_name="bge_reranker_v2_m3",
                    subcategory_slug=row["subcategory_slug"],
                    category=row["category"],
                    canonical_zh=row["canonical_zh"],
                    custom_query=query,
                    top5_products=top5,
                    filtered_products=filtered,
                    top5_relevance_ratio=top5_relevance_ratio,
                    filtered_irrelevant_ratio=filtered_irrelevant_ratio,
                )
            )
        summary = build_summary("bge_reranker_v2_m3", rows)
        write_json(args.report_dir / with_suffix("bge_reranker_results.json", args.output_suffix), summary)
        summaries.append(summary)
        reranker.cleanup()

    qwen_model_map = {
        "qwen3_5_4b": "Qwen/Qwen3.5-4B",
        "qwen3_5_9b": "Qwen/Qwen3.5-9B",
    }
    for method_name in ("qwen3_5_4b", "qwen3_5_9b"):
        if method_name not in requested_methods:
            continue
        model = QwenFilter(
            model_id=qwen_model_map[method_name],
            device=args.device,
            torch_dtype=args.torch_dtype,
        )
        rows = []
        for row in aggregated_rows:
            candidate_pairs = [(item["index"], products[item["index"]]) for item in row["multi_route_top10"]]
            candidate_docs = build_docs(candidate_pairs)
            by_index = {doc["index"]: doc for doc in candidate_docs}
            query = make_custom_query(row["category"], row["canonical_zh"])
            canonical = query_lookup[row["subcategory_slug"]].canonical_zh
            ranked, raw_response = model.rank(query=query, candidates=candidate_pairs)
            top5 = [
                {
                    "index": item.index,
                    "product_id": item.product_id,
                    "image_path": item.image_path,
                    "document": item.document,
                    "score": item.score,
                }
                for item in ranked[:5]
            ]
            top5_index_set = {item["index"] for item in top5}
            filtered = [doc for doc in candidate_docs if doc["index"] not in top5_index_set]
            top5_relevance_ratio = sum(products[item["index"]].subcategory == canonical for item in top5) / len(top5)
            filtered_irrelevant_ratio = (
                sum(products[item["index"]].subcategory != canonical for item in filtered) / len(filtered)
                if filtered
                else 0.0
            )
            rows.append(
                build_case_row(
                    method_name=method_name,
                    subcategory_slug=row["subcategory_slug"],
                    category=row["category"],
                    canonical_zh=row["canonical_zh"],
                    custom_query=query,
                    top5_products=top5,
                    filtered_products=filtered,
                    top5_relevance_ratio=top5_relevance_ratio,
                    filtered_irrelevant_ratio=filtered_irrelevant_ratio,
                    raw_response=raw_response,
                )
            )
        summary = build_summary(method_name, rows)
        write_json(
            args.report_dir / with_suffix(f"{method_name}_filter_results.json", args.output_suffix),
            summary,
        )
        summaries.append(summary)
        model.cleanup()

    write_summary_csv(
        args.report_dir / with_suffix("rerank_summary.csv", args.output_suffix),
        [
            {
                "method_name": summary["method_name"],
                "avg_top5_relevance_ratio": summary["avg_top5_relevance_ratio"],
                "avg_filtered_irrelevant_ratio": summary["avg_filtered_irrelevant_ratio"],
                "case_count": summary["case_count"],
            }
            for summary in summaries
        ],
    )
    print(
        json.dumps(
            {
                "report_dir": str(args.report_dir),
                "methods": [
                    {
                        "method_name": summary["method_name"],
                        "avg_top5_relevance_ratio": summary["avg_top5_relevance_ratio"],
                        "avg_filtered_irrelevant_ratio": summary["avg_filtered_irrelevant_ratio"],
                    }
                    for summary in summaries
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
