from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

from ecom_qa.data.catalog import ProductRecord, load_products
from ecom_qa.data.qa_generation import (
    TARGET_TOOL_CLASSES,
    ServerConfig,
    TargetedQARequest,
    build_targeted_result_record,
    generate_one_targeted_qa,
    run_llama_server,
    validate_target_tool_class,
    write_json,
)


DEFAULT_TARGETS = ["web_search", "web_search_grounding", "rag_search_grounding"]
GROUNDING_CUES = [
    "logo",
    "标志",
    "标签",
    "包装",
    "瓶身",
    "显示屏",
    "屏幕",
    "图案",
    "文字",
    "按钮",
    "面板",
    "胸前",
    "袖口",
    "鞋底",
    "表盘",
    "拉链",
    "链条",
    "局部",
    "印花",
    "铭牌",
    "控制",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate targeted in-domain QA pairs for specific expected tool-call classes."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/qa/generated"))
    parser.add_argument("--total-count", type=int, default=5000)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, choices=sorted(TARGET_TOOL_CLASSES))
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def balanced_quotas(total: int, targets: list[str]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("total-count must be greater than 0.")
    base = total // len(targets)
    remainder = total % len(targets)
    return {target: base + (1 if index < remainder else 0) for index, target in enumerate(targets)}


def product_text(product: ProductRecord) -> str:
    return "\n".join(
        [
            product.title,
            product.brand,
            product.category,
            product.subcategory,
            json.dumps(product.parameters, ensure_ascii=False),
            product.description,
        ]
    ).lower()


def product_score(product: ProductRecord, target: str) -> int:
    text = product_text(product)
    score = sum(1 for cue in GROUNDING_CUES if cue.lower() in text)
    if target == "web_search":
        return 1 if product.brand or product.subcategory else 0
    if target == "web_search_grounding":
        return score
    if target == "rag_search_grounding":
        has_local_facts = bool(product.price or product.parameters or product.after_sales or product.shop_name)
        return score + (1 if has_local_facts else 0)
    raise AssertionError(f"Unhandled target: {target}")


def ranked_product_indices(products: list[ProductRecord], target: str, rng: Random) -> list[int]:
    scored: dict[int, list[int]] = {}
    for index, product in enumerate(products):
        scored.setdefault(product_score(product, target), []).append(index)

    ranked: list[int] = []
    for score in sorted(scored, reverse=True):
        group = scored[score]
        rng.shuffle(group)
        ranked.extend(group)
    return ranked


def render_progress(target: str, generated: int, quota: int, failed: int) -> None:
    percent = (generated / quota * 100) if quota else 100.0
    print(
        f"\r{target}: generated={generated}/{quota} ({percent:5.1f}%) failed={failed}",
        end="",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    for target in args.targets:
        validate_target_tool_class(target)

    started_at = time.perf_counter()
    products = load_products(args.dataset_dir)
    rng = Random(args.seed)
    quotas = balanced_quotas(args.total_count, args.targets)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"qwen35_targeted_qa_{args.total_count}_{timestamp}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    qa_path = run_dir / "qa_pairs.jsonl"
    failed_path = run_dir / "failed_rows.jsonl"
    summary_path = run_dir / "run_summary.json"
    server_log_path = run_dir / "llama_server.log"

    records: list[dict[str, Any]] = []
    failed_count = 0
    generated_by_target: Counter[str] = Counter()
    failed_by_target: Counter[str] = Counter()

    with run_llama_server(ServerConfig(), log_path=server_log_path) as startup_seconds:
        for target in args.targets:
            quota = quotas[target]
            product_indices = ranked_product_indices(products, target, rng)
            product_cursor = 0
            render_progress(target, 0, quota, 0)

            while generated_by_target[target] < quota:
                product_index = product_indices[product_cursor % len(product_indices)]
                product_cursor += 1

                request = TargetedQARequest(
                    target_tool_class=target,
                    product=products[product_index],
                    image_path=args.dataset_dir / products[product_index].image_path,
                    source_row_index=product_index,
                )
                try:
                    qa, qa_elapsed_seconds = generate_one_targeted_qa(request, ServerConfig())
                    record = build_targeted_result_record(request, qa, qa_elapsed_seconds)
                except Exception as exc:
                    failed_count += 1
                    failed_by_target[target] += 1
                    append_jsonl(
                        failed_path,
                        {
                            "target_tool_class": target,
                            "source_row_index": product_index,
                            "product_id": products[product_index].id,
                            "image_path": products[product_index].image_path,
                            "error": str(exc),
                        },
                    )
                    render_progress(target, generated_by_target[target], quota, failed_by_target[target])
                    continue

                records.append(record)
                generated_by_target[target] += 1
                append_jsonl(qa_path, record)
                render_progress(target, generated_by_target[target], quota, failed_by_target[target])
            print()

    total_wall_seconds = time.perf_counter() - started_at
    summary = {
        "run_name": run_name,
        "targets": args.targets,
        "target_specs": TARGET_TOOL_CLASSES,
        "requested_total": args.total_count,
        "quotas": quotas,
        "generated_count": len(records),
        "generated_by_target": dict(generated_by_target),
        "failed_count": failed_count,
        "failed_by_target": dict(failed_by_target),
        "server_startup_seconds": round(startup_seconds, 3),
        "total_wall_seconds": round(total_wall_seconds, 3),
        "qa_path": str(qa_path),
        "failed_path": str(failed_path) if failed_path.exists() else None,
        "server_log_path": str(server_log_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
