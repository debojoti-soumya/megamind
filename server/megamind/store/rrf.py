"""Reciprocal-rank fusion, shared by every MemoryStore backend (and the hybrid retriever's image calibration)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from megamind.models import MemoryHit, MemoryToken

Scored = tuple[MemoryToken, float]  # re-exported by base.py
Calibration = tuple[float, float]  # (scale, bias): p(match) = sigmoid(scale * cos + bias)
Ranking = Sequence[str] | Sequence[tuple[str, float]]
MIN_SCORE = 1e-6


def rrf(rankings: Sequence[Sequence[str]], c: int = 60) -> dict[str, float]:
    """Fuse several ranked id lists; higher is better. `c` damps the top ranks (Cormack et al. 2009)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, token_id in enumerate(ranking):
            scores[token_id] = scores.get(token_id, 0.0) + 1.0 / (c + rank + 1)
    return scores


def weighted_rrf(rankings: Iterable[tuple[float, Ranking]], c: float = 60) -> dict[str, float]:
    """Weighted RRF: each ranking is (weight, items); an item is an id, or (id, multiplier) to scale that one
    hit's contribution: `weight * multiplier / (c + rank + 1)`. Ported from PR #54 (`store/fusion.py`)."""
    scores: dict[str, float] = {}
    for weight, ranking in rankings:
        for rank, item in enumerate(ranking):
            token_id, mult = item if isinstance(item, tuple) else (item, 1.0)
            scores[token_id] = scores.get(token_id, 0.0) + weight * mult / (c + rank + 1)
    return scores


def image_conf(cos: float, scale: float, bias: float) -> float:
    """SigLIP match probability sigmoid(scale * cos + bias) for an image-text cosine.

    Used as an RRF multiplier on image-vector hits: a nearest image that is still a poor match (a wifi-password
    question has no visual answer) contributes ~0 instead of full rank credit, so it can't outrank exact
    text/OCR hits. Monotone in `cos`, so the order *within* the image list never changes.
    """
    z = scale * cos + bias
    if z < -60:  # avoid OverflowError in exp for very negative logits
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def calibrated(cal: Calibration | None) -> Callable[[float], float] | None:
    """A score -> multiplier function for `fuse_hits(multipliers=...)`, or None when uncalibrated."""
    if cal is None:
        return None
    scale, bias = cal
    return lambda cos: image_conf(cos, scale, bias)


def fuse_hits(
    rankings: Sequence[Sequence[Scored]],
    k: int,
    *,
    min_score: float = MIN_SCORE,
    weights: Sequence[float] | None = None,
    multipliers: Sequence[Callable[[float], float] | None] | None = None,
) -> list[MemoryHit]:
    """RRF-fuse primitive results into MemoryHits.

    Entries with a non-positive raw score (no term overlap, orthogonal vector) are dropped first, so an
    unrelated query yields no hits (-> recall_memory reports not_found) instead of citing arbitrary memories.
    Ties are broken newest-first.

    Optional, per ranking (same order as `rankings`): `weights` scales the whole list, `multipliers` maps a hit's
    raw score to a factor on its contribution (e.g. `calibrated(store.image_calibration)` for the image list).
    With neither, this is plain RRF.
    """
    kept = [[(t, s) for t, s in ranking if s > min_score] for ranking in rankings]
    tokens: dict[str, MemoryToken] = {}
    for ranking in kept:
        for t, _ in ranking:
            tokens.setdefault(t.id, t)
    if weights is None and multipliers is None:
        fused = rrf([[t.id for t, _ in ranking] for ranking in kept])
    else:
        ws = list(weights) if weights is not None else [1.0] * len(kept)
        ms = list(multipliers) if multipliers is not None else [None] * len(kept)
        if len(ws) != len(kept) or len(ms) != len(kept):
            raise ValueError("weights/multipliers must have one entry per ranking")
        fused = weighted_rrf(
            (w, [(t.id, m(s)) for t, s in ranking] if m is not None else [t.id for t, _ in ranking])
            for w, m, ranking in zip(ws, ms, kept)
        )
    ordered = sorted(fused.items(), key=lambda kv: (kv[1], tokens[kv[0]].ts), reverse=True)
    return [MemoryHit.from_token(tokens[i], s) for i, s in ordered[:k]]
