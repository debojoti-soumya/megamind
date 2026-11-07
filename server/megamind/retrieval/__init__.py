"""Retrieval: from a natural-language question to the best small set of memory tokens.

- `planner`: rule-based query planning (temporal expressions, query type, rewrite).
- `base`: the pluggable `Retriever` interface.
- `hybrid`: our own text-vector + BM25 + image-vector retriever fused with RRF.
- `ucc`: Universal Constructor "UCC Constructor-1" backend (facet ranking over a synced corpus, hybrid fallback).
"""

import logging

from megamind.retrieval.base import RetrievalResult, RetrievedToken, Retriever
from megamind.retrieval.hybrid import HybridConfig, HybridRetriever
from megamind.retrieval.planner import QueryPlan, QueryType, Weights, plan_query
from megamind.retrieval.ucc import UCCIndexer, UCCRetriever, build_ucc

__all__ = [
    "HybridConfig",
    "HybridRetriever",
    "QueryPlan",
    "QueryType",
    "RetrievalResult",
    "RetrievedToken",
    "Retriever",
    "UCCIndexer",
    "UCCRetriever",
    "Weights",
    "build_retriever",
    "plan_query",
]


def build_retriever(settings, store, text_embedder, image_embedder=None) -> tuple[Retriever, UCCIndexer | None]:
    """Select the backend from `settings.retriever_backend` (default: our hybrid). With `ucc` (and a key) the
    second item is the store -> corpus sync job the app must start; otherwise it is None."""
    hybrid = HybridRetriever(store, text_embedder, image_embedder, tz=getattr(settings, "wearer_tz", None))
    if getattr(settings, "retriever_backend", "hybrid") == "ucc":
        if getattr(settings, "ucc_api_key", ""):
            return build_ucc(settings, store, hybrid)
        logging.getLogger(__name__).warning("RETRIEVER_BACKEND=ucc but UCC_API_KEY is empty; using hybrid")
    return hybrid, None
