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


# Same idea as LanceDB's native FTS (stop words removed, English stemming) so both backends rank alike.
STOP_WORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i i'm in is it its me my "
    "of on or our so than that the their them there these they this to was we were what when where which who why "
    "will with would you your".split()
)


@dataclass(frozen=True)
class FusionWeights:
    bm25: float = 1.0
    text: float = 1.0
    image: float = 0.6
    image_visual_mode: float = 1.5
    per_episode_cap: int = 3
    # Sigmoid calibration of image-text cosine (SigLIP is trained with p = sigmoid(scale * cos + bias)).
    # When set, each image hit's RRF contribution is scaled by p, so questions with no visual match
    # (a wifi password, something someone said) don't let unrelated photos outrank the text hits.
    image_logit_scale: float | None = None
    image_logit_bias: float = 0.0

    def image_conf(self, cos: float) -> float:
        if self.image_logit_scale is None:
            return 1.0
        return 1.0 / (1.0 + math.exp(-(self.image_logit_scale * cos + self.image_logit_bias)))


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf) and not w.endswith("ss"):
            return w[: -len(suf)]
    return w


def tokenize(text: str, stem: bool = True) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in STOP_WORDS]
    return [_stem(w) for w in words] if stem else words


Ranking = Sequence[str] | Sequence[tuple[str, float]]


def weighted_rrf(rankings: Iterable[tuple[float, Ranking]], c: int = RRF_C) -> dict[str, float]:
    """Weighted reciprocal-rank fusion. Ranking items are ids, or (id, multiplier) to scale one hit."""
    scores: dict[str, float] = {}
    for weight, ranking in rankings:
        for rank, item in enumerate(ranking):
            token_id, mult = item if isinstance(item, tuple) else (item, 1.0)
            scores[token_id] = scores.get(token_id, 0.0) + weight * mult / (c + rank + 1)
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
