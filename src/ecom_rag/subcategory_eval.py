from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def precision_at_k(ranked_indices: np.ndarray, positives: set[int], k: int) -> float:
    topk = ranked_indices[:k]
    return float(sum(index in positives for index in topk) / k)


def recall_at_k(
    ranked_indices: np.ndarray,
    positives: set[int],
    positive_count: int,
    k: int,
) -> float:
    topk = ranked_indices[:k]
    return float(sum(index in positives for index in topk) / positive_count)


def hit_at_k(ranked_indices: np.ndarray, positives: set[int], k: int) -> float:
    topk = ranked_indices[:k]
    return float(any(index in positives for index in topk))


def evaluate_subcategory_queries(
    ranked_indices: np.ndarray,
    relevance_sets: Sequence[set[int]],
    positive_counts: Sequence[int],
    ks: Sequence[int] = (1, 3, 5, 10, 20, 50),
) -> tuple[dict[str, float], list[dict[str, float]]]:
    per_query: list[dict[str, float]] = []
    for row, positives, positive_count in zip(
        ranked_indices,
        relevance_sets,
        positive_counts,
        strict=True,
    ):
        metrics = {"positive_count": float(positive_count)}
        for k in ks:
            metrics[f"precision@{k}"] = precision_at_k(row, positives, k)
            metrics[f"recall@{k}"] = recall_at_k(row, positives, positive_count, k)
            metrics[f"hit@{k}"] = hit_at_k(row, positives, k)
        per_query.append(metrics)

    summary: dict[str, float] = {"query_count": float(len(per_query))}
    for k in ks:
        summary[f"avg_precision@{k}"] = float(
            np.mean([row[f"precision@{k}"] for row in per_query])
        )
        summary[f"avg_recall@{k}"] = float(
            np.mean([row[f"recall@{k}"] for row in per_query])
        )
        summary[f"avg_hit@{k}"] = float(np.mean([row[f"hit@{k}"] for row in per_query]))
    return summary, per_query
