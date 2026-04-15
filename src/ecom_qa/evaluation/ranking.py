from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_ranked_lists(
    query_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = query_embeddings @ image_embeddings.T
    ranked_indices = np.argsort(-scores, axis=1)
    return scores, ranked_indices


def compute_metrics(
    ranked_indices: np.ndarray,
    positives: Sequence[int],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    ranks = []
    for row_idx, positive_index in enumerate(positives):
        row = ranked_indices[row_idx]
        positive_rank = int(np.where(row == positive_index)[0][0]) + 1
        ranks.append(positive_rank)

    ranks_np = np.asarray(ranks, dtype=np.int32)
    metrics: dict[str, float] = {
        "query_count": float(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks_np)),
        "median_rank": float(np.median(ranks_np)),
        "mean_rank": float(np.mean(ranks_np)),
    }
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean(ranks_np <= k))
    return metrics


def compute_subcategory_metrics(
    ranked_indices: np.ndarray,
    positive_indices: Sequence[Sequence[int]],
    ks: Sequence[int] = (1, 3, 5, 10, 20, 50),
) -> tuple[dict[str, float], list[dict[str, float]]]:
    per_query_rows: list[dict[str, float]] = []
    macro_metrics: dict[str, float] = {
        "query_count": float(len(positive_indices)),
    }

    for row_idx, positives in enumerate(positive_indices):
        positive_set = set(positives)
        row_metrics: dict[str, float] = {
            "query_index": float(row_idx),
            "positive_count": float(len(positive_set)),
        }
        for k in ks:
            topk = ranked_indices[row_idx, :k]
            hits = sum(1 for item in topk if int(item) in positive_set)
            row_metrics[f"precision@{k}"] = hits / k
            row_metrics[f"recall@{k}"] = hits / len(positive_set)
            row_metrics[f"hit@{k}"] = 1.0 if hits > 0 else 0.0
        per_query_rows.append(row_metrics)

    for k in ks:
        macro_metrics[f"precision@{k}"] = float(
            np.mean([row[f"precision@{k}"] for row in per_query_rows])
        )
        macro_metrics[f"recall@{k}"] = float(
            np.mean([row[f"recall@{k}"] for row in per_query_rows])
        )
        macro_metrics[f"hit@{k}"] = float(np.mean([row[f"hit@{k}"] for row in per_query_rows]))

    return macro_metrics, per_query_rows
