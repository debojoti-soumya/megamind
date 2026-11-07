"""Shared ranking logic for every MemoryStore: weighted RRF, episode diversity, BM25 fallback."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")
RRF_C = 60
POOL = 50


@dataclass(frozen=True)
class FusionWeights:
    bm25: float = 1.0
    text: float = 1.0
    image: float = 0.6
    image_visual_mode: float = 1.5
    per_episode_cap: int = 3


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def weighted_rrf(rankings: Iterable[tuple[float, Sequence[str]]], c: int = RRF_C) -> dict[str, float]:
    scores: dict[str, float] = {}
    for weight, ranking in rankings:
        for rank, token_id in enumerate(ranking):
            scores[token_id] = scores.get(token_id, 0.0) + weight / (c + rank + 1)
    return scores


def diversify(
    ordered: Sequence[tuple[str, float]], episode_of: dict[str, str | None], k: int, cap: int
) -> list[tuple[str, float]]:
    """Keep at most `cap` hits per episode; backfill with the overflow if fewer than k remain."""
    if cap <= 0:
        return list(ordered[:k])
    seen: Counter[str] = Counter()
    picked, overflow = [], []
    for tid, s in ordered:
        ep = episode_of.get(tid)
        if ep is not None and seen[ep] >= cap:
            overflow.append((tid, s))
            continue
        if ep is not None:
            seen[ep] += 1
        picked.append((tid, s))
        if len(picked) == k:
            return picked
    return picked + overflow[: k - len(picked)]


def bm25_rank(query: str, docs: dict[str, str], k1: float = 1.2, b: float = 0.75) -> list[str]:
    """Plain Okapi BM25 for the in-memory store; ids with zero score are omitted."""
    q = set(tokenize(query))
    if not q or not docs:
        return []
    toks = {i: tokenize(t) for i, t in docs.items()}
    n = len(toks)
    avgdl = sum(len(t) for t in toks.values()) / n or 1.0
    df = Counter(w for t in toks.values() for w in set(t) & q)
    scored = []
    for i, t in toks.items():
        tf = Counter(w for w in t if w in q)
        if not tf:
            continue
        s = 0.0
        for w, f in tf.items():
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avgdl))
        scored.append((s, i))
    scored.sort(reverse=True)
    return [i for _, i in scored]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)
