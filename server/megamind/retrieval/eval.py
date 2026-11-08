"""Retrieval evaluation: a synthetic "day" of memory tokens + questions with gold ids -> Recall@k / MRR.

The dataset format (see `server/tools/data/retrieval_day.json`) is deliberately tiny and human-editable:

    {"date": "2026-11-08", "tz": "America/Los_Angeles", "now": "18:30",
     "stream": [{"id": "h03", "time": "07:11", "caption": "...", "ocr": "", "entities": [...], "episode": "home-am"}],
     "questions": [{"q": "where are my keys", "gold": ["e04"], "time_from": null, "time_to": null, "mode": null}]}

Metrics are computed on the *ranked* (best-first, non-context) view and, separately, on the whole hand-off set the
reader would see (`handoff_recall`). The CLI lives in `server/tools/eval_retrieval.py`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import statistics
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from megamind.ingest.enrich import ImageEmbedder, MockTextEmbedder, TextEmbedder
from megamind.models import MemoryToken
from megamind.retrieval.base import Retriever
from megamind.retrieval.hybrid import HybridConfig, HybridRetriever
from megamind.retrieval.planner import WEIGHTS_BY_TYPE, Weights
from megamind.store.memory_store import InMemoryStore

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "tools" / "data" / "retrieval_day.json"


@dataclass
class EvalCase:
    q: str
    gold: list[str]
    time_from: float | None = None
    time_to: float | None = None
    mode: str = "hybrid"
    k: int | None = None
    qtype: str = "generic"
    groups: list[list[str]] = field(default_factory=list)  # gold groups (any id of a group counts); [] = one per id

    def __post_init__(self) -> None:
        if not self.groups:
            self.groups = [[g] for g in self.gold]


@dataclass
class Dataset:
    tokens: list[MemoryToken]
    cases: list[EvalCase]
    now: datetime
    tz: ZoneInfo


@dataclass
class CaseResult:
    case: EvalCase
    ranked_ids: list[str]  # best-first, non-context
    handoff_ids: list[str]  # everything the reader would see
    first_rank: int | None  # 1-based rank of the first gold id, None if missed
    latency_ms: float = 0.0
    cost: dict[str, float] = field(default_factory=dict)  # calls / llm_calls / ucc_queries / usd for this question
    failed: bool = False  # the paid backend errored or timed out and the hybrid answered
    hybrid_only: bool = False  # routed / gated to the hybrid by design (no paid call)

    @property
    def rr(self) -> float:
        return 0.0 if self.first_rank is None else 1.0 / self.first_rank

    def _hit(self, ids: list[str]) -> float:
        got = set(ids)
        return sum(1 for g in self.case.groups if got & set(g)) / len(self.case.groups)

    def recall_at(self, n: int) -> float:
        """Share of gold groups with a member in the top n (= share of gold ids for singleton groups)."""
        return self._hit(self.ranked_ids[:n])

    @property
    def handoff_recall(self) -> float:
        return self._hit(self.handoff_ids)


@dataclass
class Report:
    results: list[CaseResult] = field(default_factory=list)
    ks: tuple[int, ...] = (1, 3, 5)

    @property
    def mrr(self) -> float:
        return sum(r.rr for r in self.results) / max(1, len(self.results))

    def recall(self, n: int) -> float:
        return sum(r.recall_at(n) for r in self.results) / max(1, len(self.results))

    @property
    def handoff_recall(self) -> float:
        return sum(r.handoff_recall for r in self.results) / max(1, len(self.results))

    def summary(self) -> dict[str, float]:
        out = {f"recall@{n}": round(self.recall(n), 3) for n in self.ks}
        out["mrr"] = round(self.mrr, 3)
        out["handoff_recall"] = round(self.handoff_recall, 3)
        return out

    def misses(self) -> list[CaseResult]:
        return [r for r in self.results if r.first_rank is None or r.first_rank > 1]

    def by_type(self) -> dict[str, Report]:
        out: dict[str, Report] = {}
        for r in self.results:
            out.setdefault(r.case.qtype, Report(ks=self.ks)).results.append(r)
        return out

    def ops(self) -> dict[str, float]:
        """Latency percentiles, mean paid calls / $ per question and failure counts."""
        lat = sorted(r.latency_ms for r in self.results) or [0.0]
        n = max(1, len(self.results))

        def pct(p: float) -> float:
            return lat[min(len(lat) - 1, round(p * (len(lat) - 1)))]

        mean = lambda key: sum(r.cost.get(key, 0.0) for r in self.results) / n  # noqa: E731
        return {
            "p50_ms": round(statistics.median(lat)),
            "p95_ms": round(pct(0.95)),
            "calls_per_q": round(mean("calls"), 2),
            "llm_calls_per_q": round(mean("llm_calls"), 2),
            "ucc_per_q": round(mean("ucc_queries"), 2),
            "usd_per_q": round(mean("usd"), 5),
            "failures": sum(r.failed for r in self.results),
            "hybrid_only": sum(r.hybrid_only for r in self.results),
        }


# --------------------------------------------------------------------------- dataset


def _local(d: date, hhmm: str, tz: ZoneInfo) -> datetime:
    h, m, *sec = hhmm.split(":")
    return datetime.combine(d, time(int(h), int(m), int(sec[0]) if sec else 0), tzinfo=tz)


def load_dataset(path: Path | str = DEFAULT_DATASET) -> Dataset:
    raw = json.loads(Path(path).read_text())
    tz = ZoneInfo(raw.get("tz", "America/Los_Angeles"))
    d = date.fromisoformat(raw["date"])
    now = _local(d, raw["now"], tz)
    tokens = []
    for e in raw["stream"]:
        tokens.append(
            MemoryToken(
                id=e["id"],
                device_id=raw.get("device_id", "eval-device"),
                ts=_local(d, e["time"], tz).timestamp(),
                caption=e["caption"],
                ocr_text=e.get("ocr", ""),
                entities=e.get("entities", []),
                kind=e.get("kind", "frame"),
                thumb_path="" if e.get("kind") == "transcript" else f"/nonexistent/{e['id']}.jpg",
                frame_hash="" if e.get("kind") == "transcript" else e.get("hash") or _id_hash(e["id"]),
                episode_id=e.get("episode"),
            )
        )
    cases = []
    for q in raw["questions"]:
        if not q.get("gold"):  # must-decline questions (agent_day.json schema) have nothing to retrieve
            continue
        groups = [list(g) if isinstance(g, list) else [g] for g in q["gold"]]
        cases.append(
            EvalCase(
                q=q["q"],
                gold=[i for g in groups for i in g],
                groups=groups,
                qtype=q.get("type", "generic"),
                time_from=_local(d, q["time_from"], tz).timestamp() if q.get("time_from") else None,
                time_to=_local(d, q["time_to"], tz).timestamp() if q.get("time_to") else None,
                mode=q.get("mode") or "hybrid",
                k=q.get("k"),
            )
        )
    return Dataset(tokens=tokens, cases=cases, now=now, tz=tz)


def _id_hash(token_id: str) -> str:
    return hashlib.sha256(token_id.encode()).hexdigest()[:16]


def blank_captions(tokens: list[MemoryToken], frac: float, *, seed: int = 0) -> list[MemoryToken]:
    """A poor enricher: `frac` of the frame captions (seeded pick) come back empty; OCR / entities stay."""
    frames = [t.id for t in tokens if t.kind == "frame" and t.caption]
    drop = set(random.Random(seed).sample(frames, round(frac * len(frames)))) if frac > 0 else set()
    return [t.model_copy(update={"caption": ""}) if t.id in drop else t for t in tokens]


def tokens_digest(tokens: list[MemoryToken]) -> str:
    """Content hash of a token list (names the UCC corpus / hier note cache of a dataset variant)."""
    raw = json.dumps([[t.id, t.ts, t.kind, t.caption, t.ocr_text, t.entities] for t in tokens])
    return hashlib.sha256(raw.encode()).hexdigest()[:10]


async def build_store(
    tokens: list[MemoryToken],
    text_embedder: TextEmbedder | None = None,
    image_embedder: ImageEmbedder | None = None,
) -> InMemoryStore:
    """In-memory store with text vectors from `text_embedder` (mock by default; real one when keys exist).
    Embedders with `embed_many` get batches of 256 (one paid call per batch instead of per token)."""
    emb = text_embedder or MockTextEmbedder()
    store = InMemoryStore()
    texts = [f"{t.caption} {t.ocr_text}" for t in tokens]
    many = getattr(emb, "embed_many", None)
    if many is not None:
        vecs = [v for i in range(0, len(texts), 256) for v in await many(texts[i : i + 256])]
    else:
        vecs = [await emb.embed(x) for x in texts]
    for t, vec in zip(tokens, vecs, strict=True):
        t = t.model_copy(update={"text_vec": vec})
        if image_embedder is not None:
            # no real frames in the synthetic set: embed the caption through the text tower as a stand-in
            t = t.model_copy(update={"image_vec": await image_embedder.embed_text(t.caption)})
        await store.add(t)
    return store


# --------------------------------------------------------------------------- evaluation


async def evaluate(
    retriever: Retriever, ds: Dataset, *, k: int = 8, ks: tuple[int, ...] = (1, 3, 5), meter=None
) -> Report:
    """One question at a time (so latency and the `meter`'s per-question call / $ deltas are attributable).
    Failures / hybrid-only answers come from the backend's `stats` ("errors"; UCC "fallback", hier "gated")."""
    rep = Report(ks=ks)
    stats = getattr(retriever, "stats", None)
    for c in ds.cases:
        before = dict(stats) if stats is not None else {}
        tally = meter.all.copy() if meter is not None else None
        t0 = _time.perf_counter()
        res = await retriever.retrieve(
            c.q, k=c.k or k, mode=c.mode, time_from=c.time_from, time_to=c.time_to, now=ds.now
        )
        latency = 1000 * (_time.perf_counter() - t0)
        ranked = [r.token.id for r in res.ranked if not r.is_context]
        handoff = [r.token.id for r in res.tokens]
        first = next((i + 1 for i, tid in enumerate(ranked) if tid in c.gold), None)
        delta = {key: stats[key] - before.get(key, 0) for key in stats} if stats is not None else {}
        cost = {}
        if tally is not None:
            d = meter.all.minus(tally)
            cost = {"calls": d.calls, "llm_calls": d.llm_calls, "ucc_queries": d.ucc_queries, "usd": d.usd}
        rep.results.append(
            CaseResult(
                case=c,
                ranked_ids=ranked,
                handoff_ids=handoff,
                first_rank=first,
                latency_ms=latency,
                cost=cost,
                failed=delta.get("errors", 0) > 0,
                hybrid_only=(delta.get("fallback", 0) > 0 and not delta.get("errors", 0)) or delta.get("gated", 0) > 0,
            )
        )
    return rep


def make_retriever(
    store: InMemoryStore,
    ds: Dataset,
    *,
    text_embedder: TextEmbedder | None = None,
    image_embedder: ImageEmbedder | None = None,
    config: HybridConfig | None = None,
) -> HybridRetriever:
    return HybridRetriever(store, text_embedder or MockTextEmbedder(), image_embedder, tz=ds.tz, config=config)


def _scaled(base: dict[str, Weights], bm25: float, image: float, recency: float) -> dict[str, Weights]:
    return {
        k: Weights(text=w.text, bm25=w.bm25 * bm25, image=w.image * image, recency=w.recency * recency)
        for k, w in base.items()
    }


async def grid_search(
    store: InMemoryStore,
    ds: Dataset,
    *,
    k: int = 8,
    bm25_scales: tuple[float, ...] = (0.5, 1.0, 1.5),
    image_scales: tuple[float, ...] = (0.5, 1.0),
    recency_scales: tuple[float, ...] = (0.0, 1.0, 2.0),
    rrf_cs: tuple[float, ...] = (10.0, 60.0),
    text_embedder: TextEmbedder | None = None,
    image_embedder: ImageEmbedder | None = None,
) -> list[tuple[dict[str, float], dict[str, float]]]:
    """Sweep weight multipliers (relative to the planner defaults) and RRF c. Returns (params, summary) best first."""
    rows = []
    for b, i, r, c in itertools.product(bm25_scales, image_scales, recency_scales, rrf_cs):
        cfg = HybridConfig(rrf_c=c, weights_by_type=_scaled(WEIGHTS_BY_TYPE, b, i, r))
        retr = make_retriever(store, ds, text_embedder=text_embedder, image_embedder=image_embedder, config=cfg)
        rep = await evaluate(retr, ds, k=k)
        rows.append(({"bm25": b, "image": i, "recency": r, "rrf_c": c}, rep.summary()))
    rows.sort(key=lambda x: (-x[1]["mrr"], -x[1]["recall@3"], -x[1]["handoff_recall"]))
    return rows
