"""Reciprocal-rank fusion, shared by every MemoryStore backend."""

from __future__ import annotations

from collections.abc import Sequence

from megamind.models import MemoryHit, MemoryToken

Scored = tuple[MemoryToken, float]  # re-exported by base.py
MIN_SCORE = 1e-6


def rrf(rankings: Sequence[Sequence[str]], c: int = 60) -> dict[str, float]:
    """Fuse several ranked id lists; higher is better. `c` damps the top ranks (Cormack et al. 2009)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, token_id in enumerate(ranking):
            scores[token_id] = scores.get(token_id, 0.0) + 1.0 / (c + rank + 1)
    return scores


def fuse_hits(rankings: Sequence[Sequence[Scored]], k: int, *, min_score: float = MIN_SCORE) -> list[MemoryHit]:
    """RRF-fuse primitive results into MemoryHits.

    Entries with a non-positive raw score (no term overlap, orthogonal vector) are dropped first, so an
    unrelated query yields no hits (-> recall_memory reports not_found) instead of citing arbitrary memories.
    Ties are broken newest-first.
    """
    kept = [[(t, s) for t, s in ranking if s > min_score] for ranking in rankings]
    tokens: dict[str, MemoryToken] = {}
    for ranking in kept:
        for t, _ in ranking:
            tokens.setdefault(t.id, t)
    fused = rrf([[t.id for t, _ in ranking] for ranking in kept])
    ordered = sorted(fused.items(), key=lambda kv: (kv[1], tokens[kv[0]].ts), reverse=True)
    return [MemoryHit.from_token(tokens[i], s) for i, s in ordered[:k]]
