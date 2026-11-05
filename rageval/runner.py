"""Batch evaluation over many queries."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .metrics import hit_rate, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def evaluate_run(
    results: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Iterable[str]],
    k: int = 10,
) -> dict[str, float]:
    """Average retrieval metrics over all queries that have relevance labels.

    results: query id -> ranked list of retrieved doc ids
    qrels:   query id -> relevant doc ids
    """
    query_ids = [q for q in qrels if q in results]
    if not query_ids:
        raise ValueError("no overlapping query ids between results and qrels")

    totals = {"recall": 0.0, "precision": 0.0, "hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}
    for q in query_ids:
        retrieved, relevant = results[q], set(qrels[q])
        totals["recall"] += recall_at_k(retrieved, relevant, k)
        totals["precision"] += precision_at_k(retrieved, relevant, k)
        totals["hit_rate"] += hit_rate(retrieved, relevant, k)
        totals["mrr"] += reciprocal_rank(retrieved[:k], relevant)
        totals["ndcg"] += ndcg_at_k(retrieved, relevant, k)

    n = len(query_ids)
    return {f"{name}@{k}": value / n for name, value in totals.items()} | {"num_queries": n}
