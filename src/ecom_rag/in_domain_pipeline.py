from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np

from ecom_rag.data import ProductRecord, SubcategoryQuery
from ecom_rag.subcategory_eval import evaluate_subcategory_queries, hit_at_k, precision_at_k


SIGLIP2_T2I_PROMPT = "This is a photo of {label}."
TOP_KS = (1, 3, 5, 10, 20, 50)


@dataclass(slots=True)
class RepeatCase:
    subcategory_index: int
    repeat_id: int
    reference_index: int


@dataclass(slots=True)
class RankedCase:
    subcategory_slug: str
    category: str
    canonical_zh: str
    repeat_id: int
    reference_index: int
    reference_product_id: str
    reference_image_path: str
    text_query: str
    image_to_image_top50: list[int]
    image_text_top50: list[int]
    text_to_image_top10: list[int]
    multi_route_top10: list[int]
    image_text_top10: list[int]


def build_siglip2_prompted_queries(queries: list[SubcategoryQuery]) -> list[str]:
    return [SIGLIP2_T2I_PROMPT.format(label=query.slug_text) for query in queries]


def sample_repeat_cases(
    queries: list[SubcategoryQuery],
    *,
    repeats: int,
    seed: int,
) -> list[RepeatCase]:
    rng = Random(seed)
    cases: list[RepeatCase] = []
    for subcategory_index, query in enumerate(queries):
        references = rng.sample(query.relevant_indices, k=repeats)
        for repeat_id, reference_index in enumerate(references, start=1):
            cases.append(
                RepeatCase(
                    subcategory_index=subcategory_index,
                    repeat_id=repeat_id,
                    reference_index=reference_index,
                )
            )
    return cases


def rank_corpus(
    corpus_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    *,
    exclude_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scores = corpus_embeddings @ query_embedding.astype(np.float32, copy=False)
    if exclude_index is not None:
        scores[exclude_index] = -np.inf
    ranked_indices = np.argsort(-scores, kind="stable")
    return scores, ranked_indices


def fuse_embeddings(image_embedding: np.ndarray, text_embedding: np.ndarray) -> np.ndarray:
    fused = (image_embedding + text_embedding) / 2.0
    norm = np.linalg.norm(fused)
    if norm == 0:
        return fused.astype(np.float32, copy=False)
    return (fused / norm).astype(np.float32, copy=False)


def merge_route_rankings(
    text_ranked: np.ndarray,
    image_ranked: np.ndarray,
    *,
    per_route: int = 5,
) -> list[int]:
    selected: list[int] = []
    used: set[int] = set()

    text_added = 0
    for index in text_ranked:
        if index in used:
            continue
        selected.append(int(index))
        used.add(int(index))
        text_added += 1
        if text_added == per_route:
            break

    image_added = 0
    for index in image_ranked:
        if index in used:
            continue
        selected.append(int(index))
        used.add(int(index))
        image_added += 1
        if image_added == per_route:
            break

    return selected


def aggregate_topk_lists(candidate_lists: list[list[int]], *, total: int = 10) -> list[int]:
    score_map: dict[int, float] = defaultdict(float)
    count_map: dict[int, int] = defaultdict(int)
    best_rank_map: dict[int, int] = {}

    for candidate_list in candidate_lists:
        for rank, index in enumerate(candidate_list):
            score_map[index] += 1.0 / (rank + 1)
            count_map[index] += 1
            best_rank_map[index] = min(best_rank_map.get(index, rank), rank)

    ranked = sorted(
        score_map,
        key=lambda index: (-score_map[index], -count_map[index], best_rank_map[index], index),
    )
    return ranked[:total]


def evaluate_case_rankings(
    ranked_lists: list[list[int]],
    relevance_sets: list[set[int]],
    positive_counts: list[int],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    ranked_array = np.asarray(ranked_lists, dtype=np.int32)
    return evaluate_subcategory_queries(
        ranked_array,
        relevance_sets,
        positive_counts,
        ks=TOP_KS,
    )


def summarize_top10_only(
    ranked_lists: list[list[int]],
    relevance_sets: list[set[int]],
) -> dict[str, float]:
    precisions = [
        precision_at_k(ranked, relevant, 10)
        for ranked, relevant in zip(ranked_lists, relevance_sets, strict=True)
    ]
    hits = [
        hit_at_k(ranked, relevant, 10)
        for ranked, relevant in zip(ranked_lists, relevance_sets, strict=True)
    ]
    return {
        "avg_precision@10": float(np.mean(precisions)),
        "avg_hit@10": float(np.mean(hits)),
        "query_count": len(ranked_lists),
    }


def build_ranked_cases(
    *,
    products: list[ProductRecord],
    queries: list[SubcategoryQuery],
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    repeat_cases: list[RepeatCase],
) -> list[RankedCase]:
    ranked_cases: list[RankedCase] = []
    text_rank_cache: dict[int, np.ndarray] = {}

    for repeat_case in repeat_cases:
        query = queries[repeat_case.subcategory_index]
        reference_product = products[repeat_case.reference_index]

        if repeat_case.subcategory_index not in text_rank_cache:
            _, cached_ranked = rank_corpus(
                image_embeddings,
                text_embeddings[repeat_case.subcategory_index],
            )
            text_rank_cache[repeat_case.subcategory_index] = cached_ranked

        text_ranked = text_rank_cache[repeat_case.subcategory_index]
        text_ranked = text_ranked[text_ranked != repeat_case.reference_index]

        _, image_ranked = rank_corpus(
            image_embeddings,
            image_embeddings[repeat_case.reference_index],
            exclude_index=repeat_case.reference_index,
        )

        fused_embedding = fuse_embeddings(
            image_embeddings[repeat_case.reference_index],
            text_embeddings[repeat_case.subcategory_index],
        )
        _, image_text_ranked = rank_corpus(
            image_embeddings,
            fused_embedding,
            exclude_index=repeat_case.reference_index,
        )

        multi_route_top10 = merge_route_rankings(text_ranked, image_ranked, per_route=5)
        ranked_cases.append(
            RankedCase(
                subcategory_slug=query.slug,
                category=query.category,
                canonical_zh=query.canonical_zh,
                repeat_id=repeat_case.repeat_id,
                reference_index=repeat_case.reference_index,
                reference_product_id=reference_product.id,
                reference_image_path=reference_product.image_path,
                text_query=SIGLIP2_T2I_PROMPT.format(label=query.slug_text),
                image_to_image_top50=[int(index) for index in image_ranked[:50]],
                image_text_top50=[int(index) for index in image_text_ranked[:50]],
                text_to_image_top10=[int(index) for index in text_ranked[:10]],
                multi_route_top10=multi_route_top10,
                image_text_top10=[int(index) for index in image_text_ranked[:10]],
            )
        )

    return ranked_cases


def build_case_metrics(
    ranked_cases: list[RankedCase],
    queries: list[SubcategoryQuery],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    image_ranked_lists: list[list[int]] = []
    image_text_ranked_lists: list[list[int]] = []
    multi_route_lists: list[list[int]] = []
    image_text_top10_lists: list[list[int]] = []
    relevance_sets: list[set[int]] = []
    positive_counts: list[int] = []
    per_case_rows: list[dict[str, object]] = []

    query_by_slug = {query.slug: query for query in queries}

    for ranked_case in ranked_cases:
        query = query_by_slug[ranked_case.subcategory_slug]
        relevance = set(query.relevant_indices)
        relevance.discard(ranked_case.reference_index)

        image_ranked_lists.append(ranked_case.image_to_image_top50)
        image_text_ranked_lists.append(ranked_case.image_text_top50)
        multi_route_lists.append(ranked_case.multi_route_top10)
        image_text_top10_lists.append(ranked_case.image_text_top10)
        relevance_sets.append(relevance)
        positive_counts.append(len(relevance))

        per_case_rows.append(
            {
                "subcategory_slug": ranked_case.subcategory_slug,
                "category": ranked_case.category,
                "canonical_zh": ranked_case.canonical_zh,
                "repeat_id": ranked_case.repeat_id,
                "reference_product_id": ranked_case.reference_product_id,
                "reference_image_path": ranked_case.reference_image_path,
            }
        )

    image_summary, image_per_case = evaluate_case_rankings(
        image_ranked_lists,
        relevance_sets,
        positive_counts,
    )
    image_text_summary, image_text_per_case = evaluate_case_rankings(
        image_text_ranked_lists,
        relevance_sets,
        positive_counts,
    )
    multi_route_summary = summarize_top10_only(multi_route_lists, relevance_sets)
    image_text_top10_summary = summarize_top10_only(image_text_top10_lists, relevance_sets)

    enriched_rows: list[dict[str, object]] = []
    for row, image_metrics, image_text_metrics, ranked_case, relevance in zip(
        per_case_rows,
        image_per_case,
        image_text_per_case,
        ranked_cases,
        relevance_sets,
        strict=True,
    ):
        enriched_rows.append(
            {
                **row,
                "image_to_image": image_metrics,
                "image_text_to_image": image_text_metrics,
                "multi_route_precision@10": precision_at_k(ranked_case.multi_route_top10, relevance, 10),
                "multi_route_hit@10": hit_at_k(ranked_case.multi_route_top10, relevance, 10),
                "image_text_precision@10": precision_at_k(ranked_case.image_text_top10, relevance, 10),
                "image_text_hit@10": hit_at_k(ranked_case.image_text_top10, relevance, 10),
            }
        )

    return (
        {
            "image_to_image": image_summary,
            "image_text_to_image": image_text_summary,
            "multi_route_top10": multi_route_summary,
            "image_text_top10_baseline": image_text_top10_summary,
            "query_count": len(ranked_cases),
        },
        enriched_rows,
    )


def build_aggregated_candidates(
    ranked_cases: list[RankedCase],
    products: list[ProductRecord],
) -> list[dict[str, object]]:
    grouped: dict[str, list[RankedCase]] = defaultdict(list)
    for ranked_case in ranked_cases:
        grouped[ranked_case.subcategory_slug].append(ranked_case)

    rows: list[dict[str, object]] = []
    for subcategory_slug in sorted(grouped):
        cases = grouped[subcategory_slug]
        multi_route_top10 = aggregate_topk_lists(
            [case.multi_route_top10 for case in cases],
            total=10,
        )
        image_text_top10 = aggregate_topk_lists(
            [case.image_text_top10 for case in cases],
            total=10,
        )

        rows.append(
            {
                "subcategory_slug": subcategory_slug,
                "category": cases[0].category,
                "canonical_zh": cases[0].canonical_zh,
                "text_query": cases[0].text_query,
                "reference_image_paths": [case.reference_image_path for case in cases],
                "multi_route_top10": [
                    {
                        "rank": rank + 1,
                        "index": index,
                        "product_id": products[index].id,
                        "image_path": products[index].image_path,
                    }
                    for rank, index in enumerate(multi_route_top10)
                ],
                "image_text_top10": [
                    {
                        "rank": rank + 1,
                        "index": index,
                        "product_id": products[index].id,
                        "image_path": products[index].image_path,
                    }
                    for rank, index in enumerate(image_text_top10)
                ],
            }
        )
    return rows


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
