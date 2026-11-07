"""Retrieval: from a natural-language question to the best small set of memory tokens.

- `planner`: rule-based query planning (temporal expressions, query type, rewrite).
- `base`: the pluggable `Retriever` interface.
- `hybrid`: our own text-vector + BM25 + image-vector retriever fused with RRF.
- `ucc`: stub for Universal Constructor's "UCC Construction-1" indexer.
"""

from megamind.retrieval.planner import QueryPlan, QueryType, Weights, plan_query

__all__ = ["QueryPlan", "QueryType", "Weights", "plan_query"]
