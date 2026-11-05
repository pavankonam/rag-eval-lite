# rag-eval-lite

Lightweight, dependency free retrieval metrics for evaluating Retrieval Augmented Generation (RAG) pipelines.

## Metrics

| Metric | Function |
|---|---|
| Recall@k | `recall_at_k` |
| Precision@k | `precision_at_k` |
| Hit Rate@k | `hit_rate` |
| MRR | `mrr` |
| nDCG@k | `ndcg_at_k` |

## Quick start

```python
from rageval import recall_at_k, ndcg_at_k

retrieved = ["d3", "d1", "d7", "d2"]
relevant = {"d1", "d2"}
print(recall_at_k(retrieved, relevant, k=3))  # 0.5
print(ndcg_at_k(retrieved, relevant, k=4))
```

## Running tests

```bash
pip install pytest
pytest
```

## License

MIT
