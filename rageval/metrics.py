"""Core retrieval metrics.

Each function takes a ranked list of retrieved document ids and a set of
relevant document ids for a single query.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be a positive integer")


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant documents found in the top k results."""
    _validate_k(k)
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top k results that are relevant."""
    _validate_k(k)
    relevant = set(relevant)
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / k


def hit_rate(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant document appears in the top k, else 0.0."""
    _validate_k(k)
    relevant = set(relevant)
    return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal of the rank of the first relevant document."""
    relevant = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def mrr(runs: Iterable[tuple[Sequence[str], Iterable[str]]]) -> float:
    """Mean reciprocal rank over (retrieved, relevant) pairs."""
    scores = [reciprocal_rank(r, rel) for r, rel in runs]
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized discounted cumulative gain with binary relevance."""
    _validate_k(k)
    relevant = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
        if doc in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
