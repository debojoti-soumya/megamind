"""Our own hybrid retriever: text-vector + BM25 + image-vector lists fused with weighted RRF,
a recency prior, near-duplicate suppression and temporal expansion (neighbours of the top
hits) so the reader sees context. Returns <= k tokens in chronological order.
"""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from megamind.ingest.enrich import ImageEmbedder, TextEmbedder
from megamind.ingest.filter import hamming
from megamind.models import MemoryToken
from megamind.retrieval.adapter import StoreAdapter
from megamind.retrieval.base import RetrievalResult, RetrievedToken, RetrieveMode, Retriever
from megamind.retrieval.planner import QueryPlan, Weights, plan_query
from megamind.store.base import MemoryStore
from megamind.store.rrf import image_conf


@dataclass
class HybridConfig:
    pool: int = 50  # candidates taken from each ranked list before fusion
    rrf_c: float = 60.0
    recency_tau_s: float = 2 * 3600  # e^-age/tau prior (Ego4D: "where is X" = last sighting)
    dedupe_window_s: float = 120.0  # near-identical frames closer than this collapse to one
    dedupe_hamming: int = 6  # dHash distance considered "same frame"
    episode_cap: int = 3  # max hits per episode_id (diversity; no-op until storage fills episode_id)
    expand_top: int = 3  # how many top hits get neighbours
    expand_before: int = 1
    expand_after: int = 1
    expand_max_frac: float = 0.34  # neighbours take at most this share of k
    widen_on_empty: bool = True  # drop a *parsed* (not explicit) time window if nothing matched
    anchor_pad_s: float = 15 * 60  # +/- around a resolved "when I ..." anchor
    temporal_sample_span_s: float = 3 * 3600  # windows wider than this are sampled evenly instead of by focus
    # "recent" questions: look back this far first, then each next span, then everything
    recent_lookback_s: tuple[float, ...] = (1800, 6 * 3600, 48 * 3600)
    recent_fallback_s: float = 3600  # an empty parsed window ending this close to now falls back to "recent"
    weights_by_type: dict[str, Weights] | None = None  # override planner weights (used by the eval harness)
    # cross-kind context: a top transcript hit brings the nearest frame within +-speech_link_s (the pixels of what
    # was said), a top frame hit the nearest transcript token (the words around what was seen); 0 = off
    speech_link_s: float = 15.0


class HybridRetriever(Retriever):
    def __init__(
        self,
        store: MemoryStore,
        text_embedder: TextEmbedder | None,
        image_embedder: ImageEmbedder | None = None,
        *,
        tz: str | ZoneInfo | None = None,
        config: HybridConfig | None = None,
    ) -> None:
        self.adapter = StoreAdapter(store)
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.tz = tz
        self.cfg = config or HybridConfig()
        self._cache: dict[str, MemoryToken] = {}
        # query text -> vector (LRU): the prefetch and the agent's own recall often embed the same text, and a
        # hosted embedder is a network round trip on the critical path
        self._qvecs: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self.qvec_cache_size = 256

    # ------------------------------------------------------------------ public

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 12,
        mode: RetrieveMode = "hybrid",
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        now: datetime | None = None,
    ) -> RetrievalResult:
        now = now or datetime.now(UTC)
        k = max(1, k)
        plan = plan_query(query, now=now, tz=self.tz, time_from=time_from, time_to=time_to)
        if self.cfg.weights_by_type and plan.query_type in self.cfg.weights_by_type:
            plan.weights = self.cfg.weights_by_type[plan.query_type]
        # The caller's explicit non-default mode wins; otherwise follow the plan.
        # ...except with no content words: there is nothing to match semantically/visually, only time.
        eff_mode = plan.mode if mode in ("hybrid", None) or not plan.search_text else mode
        if eff_mode == "temporal" and plan.search_text and not plan.has_time:
            eff_mode = "hybrid"

        anchor_ts = None
        if plan.anchor_event and not plan.has_time:
            anchor_ts = await self._resolve_anchor(plan, device_id)
            if anchor_ts is not None:
                plan.time_from, plan.time_to = self._anchor_window(plan.query, anchor_ts, now.timestamp())

        no_text = not (plan.search_text or plan.query.strip())
        if eff_mode in ("temporal", "recent") and not plan.search_text and not plan.has_time or no_text:
            return RetrievalResult(plan=plan, tokens=await self._recent(k, device_id, now.timestamp()))
        if eff_mode == "temporal":
            toks = await self._temporal(plan, k, device_id)
            near_now = plan.time_to is None or now.timestamp() - plan.time_to <= self.cfg.recent_fallback_s
            if not toks and anchor_ts is None and near_now and not (plan.explicit_time and plan.search_text):
                # "what happened a minute ago" between two sparse captures: the latest frames are the answer
                toks = await self._recent(k, device_id, now.timestamp())
                return RetrievalResult(plan=plan, tokens=toks, widened=bool(toks))
            return RetrievalResult(plan=plan, tokens=toks, anchor_ts=anchor_ts)

        text = plan.search_text or plan.query
        ref_ts = min(now.timestamp(), plan.time_to) if plan.time_to else now.timestamp()
        # inside an explicit/parsed window the window itself carries the time intent: no recency prior
        w = replace(plan.weights, recency=0.0) if plan.has_time else plan.weights
        fused = await self._fuse(text, eff_mode, w, plan.time_from, plan.time_to, device_id, ref_ts)
        widened = False
        if not fused and self.cfg.widen_on_empty and plan.has_time and not plan.explicit_time and anchor_ts is None:
            fused = await self._fuse(text, eff_mode, plan.weights, None, None, device_id, now.timestamp())
            widened = bool(fused)
        if not fused:
            return RetrievalResult(plan=plan, tokens=[], anchor_ts=anchor_ts)

        ranked = await self._dedupe(fused)
        lo, hi = (None, None) if widened else (plan.time_from, plan.time_to)
        picked = await self._expand(ranked, k, lo, hi)
        picked.sort(key=lambda r: r.token.ts)
        return RetrievalResult(plan=plan, tokens=picked, widened=widened, anchor_ts=anchor_ts)

    # ------------------------------------------------------------------ stages

    async def _fuse(
        self,
        text: str,
        mode: str,
        w: Weights,
        time_from: float | None,
        time_to: float | None,
        device_id: str | None,
        ref_ts: float,
    ) -> list[tuple[str, float, list[str]]]:
        """Weighted RRF over the ranked lists selected by `mode`, plus a recency prior.

        With `store.image_calibration` set (SigLIP, SEARCH_CALIBRATE_IMAGE=true) each image hit contributes
        `weight * sigmoid(scale * cos + bias) / (c + rank + 1)`, so a weak nearest image can't outrank exact
        text/OCR hits (ported from PR #54). Hybrid mode only: visual mode ranks the image list alone, unchanged."""
        flt = dict(time_from=time_from, time_to=time_to, device_id=device_id)
        n = self.cfg.pool

        async def vec(embed, field: str) -> list[tuple[str, float]]:
            return await self.adapter.vector_search(await self._embed_query(field, embed, text), n, field, **flt)

        # the legs are independent store calls (LanceDB runs each in a worker thread): run them concurrently
        legs: list[tuple[str, float, Coroutine[Any, Any, list[tuple[str, float]]]]] = []
        text_lists = mode in ("semantic", "hybrid") or (mode == "visual" and self.image_embedder is None)
        if text_lists:
            legs.append(("bm25", w.bm25, self.adapter.keyword_search(text, n, **flt)))
            if self.text_embedder is not None:
                legs.append(("text", w.text, vec(self.text_embedder.embed_query, "text_vec")))
        if mode in ("semantic", "hybrid") and w.ocr_bm25:
            legs.append(("ocr_bm25", w.ocr_bm25, self.adapter.ocr_search(text, n, **flt)))
        if mode in ("visual", "hybrid") and self.image_embedder is not None:
            weight = 1.0 if mode == "visual" else w.image
            legs.append(("image", weight, vec(self.image_embedder.embed_text, "image_vec")))
        ranked = await asyncio.gather(*(c for _, _, c in legs))

        # calibrate only when the image list is fused against text lists: in visual mode it is the only
        # list, and shrinking it would just hand the ranking to the additive recency/OCR priors
        cal = getattr(self.adapter.store, "image_calibration", None) if mode == "hybrid" else None
        # (name, weight, ranked ids, optional per-hit multipliers aligned with the ids)
        lists: list[tuple[str, float, list[str], list[float] | None]] = []
        for (name, weight, _), hits in zip(legs, ranked, strict=True):
            mults = [image_conf(cos, *cal) for _, cos in hits] if name == "image" and cal is not None else None
            lists.append((name, weight, [i for i, _ in hits], mults))

        scores: dict[str, float] = {}
        reasons: dict[str, list[str]] = {}
        c = self.cfg.rrf_c
        for name, weight, ids, mults in lists:
            for rank, i in enumerate(ids):
                m = 1.0 if mults is None else mults[rank]
                scores[i] = scores.get(i, 0.0) + weight * m / (c + rank + 1)
                reasons.setdefault(i, []).append(name)
        if not scores:
            return []

        toks = {t.id: t for t in await self.adapter.get_many(list(scores))}
        top = 1.0 / (c + 1)
        out = []
        for i, s in scores.items():
            t = toks.get(i)
            if t is None:
                continue
            age = max(0.0, ref_ts - t.ts)
            s += w.recency * top * math.exp(-age / self.cfg.recency_tau_s)
            if w.ocr and t.ocr_text.strip():
                s += w.ocr * top
                reasons[i].append("ocr")
            out.append((i, s, reasons[i]))
        out.sort(key=lambda x: (-x[1], -toks[x[0]].ts))
        self._cache = toks
        return out

    async def _dedupe(self, fused: list[tuple[str, float, list[str]]]) -> list[RetrievedToken]:
        kept: list[RetrievedToken] = []
        per_episode: dict[str, int] = {}
        for i, s, why in fused:
            t = self._cache[i]
            if any(self._same_frame(t, r.token) for r in kept):
                continue
            if t.episode_id is not None:
                if per_episode.get(t.episode_id, 0) >= self.cfg.episode_cap:
                    continue
                per_episode[t.episode_id] = per_episode.get(t.episode_id, 0) + 1
            kept.append(RetrievedToken(token=t, score=s, reasons=why))
        return kept

    def _same_frame(self, a: MemoryToken, b: MemoryToken) -> bool:
        if abs(a.ts - b.ts) > self.cfg.dedupe_window_s:
            return False
        if a.frame_hash and b.frame_hash:
            try:
                if hamming(a.frame_hash, b.frame_hash) <= self.cfg.dedupe_hamming:
                    return True
            except ValueError:
                pass
        return bool(a.caption) and a.caption == b.caption and a.ocr_text == b.ocr_text

    async def _embed_query(self, field: str, embed: Callable[[str], Awaitable[list[float]]], text: str) -> list[float]:
        key = (field, text)
        if (v := self._qvecs.get(key)) is not None:
            self._qvecs.move_to_end(key)
            return v
        v = await embed(text)
        self._qvecs[key] = v
        while len(self._qvecs) > self.qvec_cache_size:
            self._qvecs.popitem(last=False)
        return v

    async def _expand(
        self, ranked: list[RetrievedToken], k: int, lo: float | None = None, hi: float | None = None
    ) -> list[RetrievedToken]:
        """Top hits + their immediate neighbours (context for the reader), capped at k and kept inside
        the question's time window so an explicit `time_from/time_to` stays a hard filter."""
        budget = min(
            int(k * self.cfg.expand_max_frac), self.cfg.expand_top * (self.cfg.expand_before + self.cfg.expand_after)
        )
        primary = ranked[: max(1, k - budget)]
        chosen: dict[str, RetrievedToken] = {r.token.id: r for r in primary}
        anchors = primary[: self.cfg.expand_top] if len(chosen) < k else []
        # one store scan per anchor (window + linked modality): fetch them concurrently, then merge in rank order
        windows, links = await asyncio.gather(
            asyncio.gather(
                *(self.adapter.neighbors(r.token.id, self.cfg.expand_before, self.cfg.expand_after) for r in anchors)
            ),
            asyncio.gather(*(self._linked(r.token, lo, hi) for r in anchors)),
        )
        for r, nbs, link in zip(anchors, windows, links, strict=True):
            if len(chosen) >= k:
                break
            if (
                link is not None
                and link.id not in chosen
                and not any(self._same_frame(link, c.token) for c in chosen.values())
            ):
                chosen[link.id] = RetrievedToken(token=link, score=r.score * 0.5, reasons=["linked"], is_context=True)
            for nb in nbs:
                if len(chosen) >= k:
                    break
                if nb.id in chosen or not nb.caption:
                    continue
                if (lo is not None and nb.ts < lo) or (hi is not None and nb.ts > hi):
                    continue
                if any(self._same_frame(nb, c.token) for c in chosen.values()):
                    continue
                chosen[nb.id] = RetrievedToken(token=nb, score=r.score * 0.5, reasons=["neighbor"], is_context=True)
        for r in ranked[len(primary) :]:
            if len(chosen) >= k:
                break
            chosen.setdefault(r.token.id, r)
        return list(chosen.values())

    async def _linked(self, t: MemoryToken, lo: float | None, hi: float | None) -> MemoryToken | None:
        """The other modality of the same moment: nearest enriched frame for a transcript token, nearest
        transcript token for a frame, within +-speech_link_s on the same device (and inside [lo, hi])."""
        span = self.cfg.speech_link_s
        other = {"transcript": "frame", "frame": "transcript"}.get(t.kind)
        if span <= 0 or other is None:
            return None
        a, b = t.ts - span, t.ts + span
        if lo is not None:
            a = max(a, lo)
        if hi is not None:
            b = min(b, hi)
        if a > b:
            return None
        rows = await self.adapter.range(a, b, limit=200, device_id=t.device_id, kinds=(other,))
        rows = [x for x in rows if x.caption and x.id != t.id]
        return min(rows, key=lambda x: abs(x.ts - t.ts), default=None)

    async def _recent(self, k: int, device_id: str | None, now_ts: float) -> list[RetrievedToken]:
        """No content words and no time ("what did I just see?"): the device's latest k enriched frames, found by
        widening the look-back until something is there. Newest first (best-first hand-off), scored by recency."""
        toks: list[MemoryToken] = []
        for back in (*self.cfg.recent_lookback_s, None):
            lo = None if back is None else now_ts - back
            toks = [t for t in await self.adapter.range(lo, now_ts, limit=10_000, device_id=device_id) if t.caption]
            if toks:
                break
        return [
            RetrievedToken(token=t, score=1.0 / (1.0 + max(0.0, now_ts - t.ts) / 600.0), reasons=["recent"])
            for t in reversed(toks[-k:])
        ]

    async def _temporal(self, plan: QueryPlan, k: int, device_id: str | None) -> list[RetrievedToken]:
        """Pure time query ("what was I doing at 11" / "after lunch"): frames from the window, scored by
        closeness to the focus point (start for "after X", end for "before X", else the centre). Wide windows
        are sampled evenly so the reader sees the whole span; narrow ones take the k closest frames."""
        toks = [
            t
            for t in await self.adapter.range(plan.time_from, plan.time_to, limit=10_000, device_id=device_id)
            if t.caption
        ]
        if not toks:
            return []
        lo = plan.time_from if plan.time_from is not None else toks[0].ts
        hi = plan.time_to if plan.time_to is not None else toks[-1].ts
        span = max(hi - lo, 60.0)
        focus = {"start": lo, "end": hi}.get(plan.time_focus, (lo + hi) / 2)
        if plan.time_from is None and plan.time_to is None:
            focus = hi
        if len(toks) > k:
            if span <= self.cfg.temporal_sample_span_s and plan.has_time:
                toks = sorted(sorted(toks, key=lambda t: abs(t.ts - focus))[:k], key=lambda t: t.ts)
            else:
                step = len(toks) / k
                toks = [toks[int(i * step)] for i in range(k)]
        return [RetrievedToken(token=t, score=1.0 / (1.0 + abs(t.ts - focus) / span), reasons=["time"]) for t in toks]

    async def _resolve_anchor(self, plan: QueryPlan, device_id: str | None) -> float | None:
        """Find when 'I left the office' happened: best keyword/text match for the event."""
        text = plan.anchor_event or ""
        fused = await self._fuse(
            text, "semantic", Weights(recency=0.3), None, None, device_id, datetime.now(UTC).timestamp()
        )
        return self._cache[fused[0][0]].ts if fused else None

    def _anchor_window(self, query: str, anchor_ts: float, now_ts: float) -> tuple[float | None, float | None]:
        q = query.lower()
        pad = self.cfg.anchor_pad_s
        if "before" in q or "until" in q:
            return None, anchor_ts
        if "after" in q or "since" in q:
            return anchor_ts, now_ts
        return anchor_ts - pad, anchor_ts + pad
