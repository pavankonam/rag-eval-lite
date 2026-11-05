import pytest

from rageval.runner import evaluate_run


def test_evaluate_run_averages_queries():
    results = {"q1": ["d1", "d2"], "q2": ["d9", "d3"]}
    qrels = {"q1": {"d1"}, "q2": {"d3"}}
    scores = evaluate_run(results, qrels, k=2)
    assert scores["num_queries"] == 2
    assert scores["recall@2"] == 1.0
    assert scores["mrr@2"] == pytest.approx(0.75)


def test_evaluate_run_requires_overlap():
    with pytest.raises(ValueError):
        evaluate_run({"q1": ["d1"]}, {"q2": {"d1"}})
