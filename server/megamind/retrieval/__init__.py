"""Retrieval: from a natural-language question to the best small set of memory tokens.

- `planner`: rule-based query planning (temporal expressions, query type, rewrite).
- `base`: the pluggable `Retriever` interface.
- `hybrid`: our own text-vector + BM25 + image-vector retriever fused with RRF.
- `ucc`: stub for Universal Constructor's "UCC Construction-1" indexer.
"""

from megamind.retrieval.base import RetrievalResult, RetrievedToken, Retriever
from megamind.retrieval.hybrid import HybridConfig, HybridRetriever
from megamind.retrieval.planner import QueryPlan, QueryType, Weights, plan_query
from megamind.retrieval.ucc import UCCRetriever

__all__ = [
    "HybridConfig",
    "HybridRetriever",
    "QueryPlan",
    "QueryType",
    "RetrievalResult",
    "RetrievedToken",
    "Retriever",
    "UCCRetriever",
    "Weights",
    "build_retriever",
    "plan_query",
]


def build_retriever(settings, store, text_embedder, image_embedder=None) -> Retriever:
    """Select the backend from `settings.retriever_backend` (default: our hybrid)."""
    backend = getattr(settings, "retriever_backend", "hybrid")
    if backend == "ucc":
        return UCCRetriever(settings)
    return HybridRetriever(store, text_embedder, image_embedder, tz=getattr(settings, "wearer_tz", None))
