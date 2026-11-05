"""rageval: lightweight retrieval metrics for RAG pipelines."""
from .metrics import recall_at_k, precision_at_k, reciprocal_rank, mrr, ndcg_at_k, hit_rate

__all__ = ["recall_at_k", "precision_at_k", "reciprocal_rank", "mrr", "ndcg_at_k", "hit_rate"]
__version__ = "0.1.0"
