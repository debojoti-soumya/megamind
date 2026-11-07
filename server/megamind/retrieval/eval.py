"""Retrieval evaluation: a synthetic "day" of memory tokens + questions with gold ids -> Recall@k / MRR.

The dataset format (see `server/tools/data/retrieval_day.json`) is deliberately tiny and human-editable:

    {"date": "2026-11-08", "tz": "America/Los_Angeles", "now": "18:30",
     "stream": [{"id": "h03", "time": "07:11", "caption": "...", "ocr": "", "entities": [...], "episode": "home-am"}],
     "questions": [{"q": "where are my keys", "gold": ["e04"], "time_from": null, "time_to": null, "mode": null}]}

Metrics are computed on the *ranked* (best-first, non-context) view and, separately, on the whole hand-off set the
reader would see (`handoff_recall`). The CLI lives in `server/tools/eval_retrieval.py`.
"""

from __future__ import annotations

import itertools
import json
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

    @property
    def rr(self) -> float:
        return 0.0 if self.first_rank is None else 1.0 / self.first_rank

    def recall_at(self, n: int) -> float:
        got = set(self.ranked_ids[:n]) & set(self.case.gold)
        return len(got) / len(self.case.gold)

    @property
    def handoff_recall(self) -> float:
        return len(set(self.handoff_ids) & set(self.case.gold)) / len(self.case.gold)


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


# --------------------------------------------------------------------------- dataset


def _local(d: date, hhmm: str, tz: ZoneInfo) -> datetime:
    h, m = hhmm.split(":")
    return datetime.combine(d, time(int(h), int(m)), tzinfo=tz)


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
                thumb_path=f"/nonexistent/{e['id']}.jpg",
                frame_hash=f"{abs(hash(e['id'])) % (1 << 64):016x}",
                episode_id=e.get("episode"),
            )
        )
    cases = []
    for q in raw["questions"]:
        cases.append(
            EvalCase(
                q=q["q"],
                gold=list(q["gold"]),
                time_from=_local(d, q["time_from"], tz).timestamp() if q.get("time_from") else None,
                time_to=_local(d, q["time_to"], tz).timestamp() if q.get("time_to") else None,
                mode=q.get("mode") or "hybrid",
                k=q.get("k"),
            )
        )
    return Dataset(tokens=tokens, cases=cases, now=now, tz=tz)


async def build_store(
    tokens: list[MemoryToken],
    text_embedder: TextEmbedder | None = None,
    image_embedder: ImageEmbedder | None = None,
) -> InMemoryStore:
    """In-memory store with text vectors from `text_embedder` (mock by default; real one when keys exist)."""
    emb = text_embedder or MockTextEmbedder()
    store = InMemoryStore()
    for t in tokens:
        t = t.model_copy(update={"text_vec": await emb.embed(f"{t.caption} {t.ocr_text}")})
        if image_embedder is not None:
            # no real frames in the synthetic set: embed the caption through the text tower as a stand-in
            t = t.model_copy(update={"image_vec": await image_embedder.embed_text(t.caption)})
        await store.add(t)
    return store


# --------------------------------------------------------------------------- evaluation


async def evaluate(retriever: Retriever, ds: Dataset, *, k: int = 8, ks: tuple[int, ...] = (1, 3, 5)) -> Report:
    rep = Report(ks=ks)
    for c in ds.cases:
        res = await retriever.retrieve(
            c.q, k=c.k or k, mode=c.mode, time_from=c.time_from, time_to=c.time_to, now=ds.now
        )
        ranked = [r.token.id for r in res.ranked if not r.is_context]
        handoff = [r.token.id for r in res.tokens]
        first = next((i + 1 for i, tid in enumerate(ranked) if tid in c.gold), None)
        rep.results.append(CaseResult(case=c, ranked_ids=ranked, handoff_ids=handoff, first_rank=first))
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
