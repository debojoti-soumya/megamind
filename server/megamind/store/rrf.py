"""Reciprocal-rank fusion, shared by every MemoryStore backend."""

from __future__ import annotations

from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[str]], c: int = 60) -> dict[str, float]:
    """Fuse several ranked id lists; higher is better. `c` damps the top ranks (Cormack et al. 2009)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, token_id in enumerate(ranking):
            scores[token_id] = scores.get(token_id, 0.0) + 1.0 / (c + rank + 1)
    return scores
