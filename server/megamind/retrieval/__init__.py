"""Retrieval: from a natural-language question to the best small set of memory tokens.

- `planner`: rule-based query planning (temporal expressions, query type, rewrite).
- `base`: the pluggable `Retriever` interface.
- `hybrid`: our own text-vector + BM25 + image-vector retriever fused with RRF.
- `ucc`: Universal Constructor "UCC Constructor-1" backend (facet ranking over a synced corpus, hybrid fallback).
- `hier`: hierarchical LLM recall over day/hour/10-min notes; `combo`: per-question routing over all three.
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
    cfg = HybridConfig(speech_link_s=getattr(settings, "speech_link_s", HybridConfig.speech_link_s))
    hybrid = HybridRetriever(store, text_embedder, image_embedder, tz=getattr(settings, "wearer_tz", None), config=cfg)
    if getattr(settings, "retriever_backend", "hybrid") == "combo":
        from megamind.retrieval.combo import ComboRetriever
        from megamind.retrieval.hier import build_hier

        ucc, indexer = build_ucc(settings, store, hybrid) if getattr(settings, "ucc_api_key", "") else (None, None)
        key = getattr(settings, "hier_api_key", "") or getattr(settings, "openai_api_key", "")
        hier = build_hier(settings, store, hybrid, key) if key else None
        if ucc is None and hier is None:
            logging.getLogger(__name__).warning("RETRIEVER_BACKEND=combo but no UCC/OpenAI key; using hybrid")
            return hybrid, None
        return ComboRetriever(hybrid, ucc=ucc, hier=hier, tz=getattr(settings, "wearer_tz", None)), indexer
    if getattr(settings, "retriever_backend", "hybrid") == "ucc":
        if getattr(settings, "ucc_api_key", ""):
            return build_ucc(settings, store, hybrid)
        logging.getLogger(__name__).warning("RETRIEVER_BACKEND=ucc but UCC_API_KEY is empty; using hybrid")
    if getattr(settings, "retriever_backend", "hybrid") == "hier":
        from megamind.retrieval.hier import build_hier

        key = getattr(settings, "hier_api_key", "") or getattr(settings, "openai_api_key", "")
        if key:
            return build_hier(settings, store, hybrid, key), None
        logging.getLogger(__name__).warning("RETRIEVER_BACKEND=hier but no HIER_API_KEY/OPENAI_API_KEY; using hybrid")
    return hybrid, None
