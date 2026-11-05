import math
import pytest

from rageval import recall_at_k, precision_at_k, reciprocal_rank, mrr, ndcg_at_k, hit_rate

RETRIEVED = ["d3", "d1", "d7", "d2", "d9"]
RELEVANT = {"d1", "d2"}


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, RELEVANT, 2) == 0.5
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == 1.0


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, RELEVANT, 2) == 0.5


def test_hit_rate():
    assert hit_rate(RETRIEVED, RELEVANT, 1) == 0.0
    assert hit_rate(RETRIEVED, RELEVANT, 2) == 1.0


def test_reciprocal_rank_and_mrr():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == 0.5
    assert mrr([(RETRIEVED, RELEVANT), (["d1"], {"d1"})]) == 0.75


def test_ndcg_perfect_ranking():
    assert ndcg_at_k(["d1", "d2", "d3"], RELEVANT, 3) == pytest.approx(1.0)


def test_ndcg_partial():
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(expected)


def test_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k(RETRIEVED, RELEVANT, 0)
