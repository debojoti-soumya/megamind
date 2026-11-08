"""UCCRetriever / UCCIndexer against an in-process fake of the UCC REST API (no network, no key)."""

import asyncio
import json
import re

import httpx
import pytest

from megamind.config import Settings
from megamind.ingest.enrich import MockTextEmbedder
from megamind.models import MemoryToken
from megamind.retrieval import HybridRetriever, UCCIndexer, UCCRetriever, build_retriever
from megamind.retrieval.adapter import tokenize
from megamind.retrieval.ucc import UCCClient, UCCConfig, token_document
from megamind.store.memory_store import InMemoryStore

BASE = "http://ucc.test/v1"
TEMPLATE = set(tokenize("The wearer asks Does this snapshot show or mention what they are asking about where my I"))


class FakeUCC:
    """Facet p(yes) = share of the question's content words found in the document text."""

    def __init__(self, *, fail: int | None = None, delay: float = 0.0) -> None:
        self.docs: list[dict] = []
        self.created = False
        self.fail = fail
        self.delay = delay
        self.selects = 0

    def _ok(self, md: dict, pred: dict | None) -> bool:
        if pred is None:
            return True
        if pred["type"] == "and":
            return all(self._ok(md, c) for c in pred["children"])
        v, op, x = md.get(pred["field"]), pred["operator"], pred["value"]
        return {
            "eq": lambda: v == x,
            "ge": lambda: v is not None and v >= x,
            "le": lambda: v is not None and v <= x,
            "in": lambda: v in x,
        }[op]()

    def _p(self, question: str, text: str) -> float:
        q = set(tokenize(question)) - TEMPLATE
        return len(q & set(tokenize(text))) / len(q) if q else 0.0

    async def __call__(self, req: httpx.Request) -> httpx.Response:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            return httpx.Response(self.fail, json={"detail": "boom"})
        path = req.url.path.removeprefix("/v1")
        body = json.loads(req.content) if req.content else None
        if req.method == "GET" and re.fullmatch(r"/corpora/[\w.-]+", path):
            if not self.created:
                return httpx.Response(404, json={"detail": "corpus_not_found"})
            return httpx.Response(200, json={"id": "c-1", "status": "ready"})
        if req.method == "POST" and path == "/corpora":
            self.created = True
            return httpx.Response(201, json={"id": "c-1", "status": "ready"})
        if req.method == "POST" and path.endswith("/documents"):
            self.created = True
            self.docs.extend(body["documents"])
            return httpx.Response(202, json={"data": [{"filename": d["name"]} for d in body["documents"]]})
        if req.method == "POST" and path == "/queries/select":
            self.selects += 1
            rows = []
            for i, d in enumerate(self.docs):
                if not self._ok(d["metadata"], body["where"]):
                    continue
                row = {"document_id": f"doc_{i}", **{f: d["metadata"].get(f) for f in body["fields"]}}
                for name, col in body["columns"].items():
                    row[name] = self._p(col["question"], d["text"])
                rows.append(row)
            return httpx.Response(200, json={"status": "finished", "rows": rows, "failures": []})
        return httpx.Response(404, json={"detail": "unknown_route"})


def tok(i: str, ts: float, caption: str, ocr: str = "", device: str = "dev") -> MemoryToken:
    return MemoryToken(
        id=i,
        device_id=device,
        ts=ts,
        caption=caption,
        ocr_text=ocr,
        thumb_path=f"/x/{i}.jpg",
        frame_hash=f"{abs(hash(i)) % (1 << 64):016x}",
    )


T0 = 1_794_150_000.0
TOKENS = [
    tok("a", T0, "Kitchen counter with car keys next to a coffee mug."),
    tok("b", T0 + 600, "Office desk with a laptop showing a spreadsheet.", ocr="Q3 budget"),
    tok("c", T0 + 1200, "Living room sofa with reading glasses on a cushion."),
    tok("d", T0 + 1800, "Hallway shelf with a red umbrella.", device="other"),
    tok("e", T0 + 2400, ""),  # not enriched yet
]


async def setup(fake: FakeUCC, cfg: UCCConfig | None = None):
    store = InMemoryStore()
    emb = MockTextEmbedder()
    for t in TOKENS:
        await store.add(t.model_copy(update={"text_vec": await emb.embed(t.caption)}))
    client = UCCClient("k", "test", base_url=BASE, transport=httpx.MockTransport(fake))
    hybrid = HybridRetriever(store, emb, None, tz="UTC")
    return store, client, hybrid, UCCRetriever(client, hybrid, tz="UTC", config=cfg)


async def test_indexer_pushes_enriched_tokens_once():
    fake = FakeUCC()
    store, client, _, _ = await setup(fake)
    idx = UCCIndexer(client, store, tz="UTC")
    assert await idx.sync_once() == 4
    assert await idx.sync_once() == 0
    md = {d["metadata"]["token_id"]: d["metadata"] for d in fake.docs}
    assert set(md) == {"a", "b", "c", "d"}
    assert md["b"] == {"token_id": "b", "device_id": "dev", "ts": T0 + 600, "kind": "frame"}
    assert "Q3 budget" in next(d["text"] for d in fake.docs if d["name"] == "b.txt")
    # a fresh indexer (app restart) learns what is already in the corpus instead of re-uploading
    assert await UCCIndexer(client, store, tz="UTC").sync_once() == 0


async def test_indexer_picks_up_episode_summaries_written_behind_the_cursor():
    fake = FakeUCC()
    store, client, _, _ = await setup(fake)
    idx = UCCIndexer(client, store, tz="UTC", lookback_s=60)
    await store.add(tok("late", T0 + 7200, "Parking garage level 2."))
    assert await idx.sync_once() == 5  # cursor now at T0 + 7200
    # the consolidator closes an episode stamped with its start, far behind cursor - lookback
    ep = tok("ep1-summary", T0, "Episode: morning at home, then the office.").model_copy(
        update={"kind": "episode", "episode_id": "ep1"}
    )
    await store.add(ep)
    assert await idx.sync_once() == 1
    assert fake.docs[-1]["metadata"] == {
        "token_id": "ep1-summary",
        "device_id": "dev",
        "ts": T0,
        "kind": "episode",
        "episode_id": "ep1",
    }
    assert await idx.sync_once() == 0


def test_token_document_shape():
    d = token_document(TOKENS[1].model_copy(update={"episode_id": "ep1", "entities": ["laptop"]}))
    assert d["name"] == "b.txt" and d["metadata"]["episode_id"] == "ep1"
    assert "Scene: Office desk" in d["text"] and "Objects: laptop" in d["text"]


async def test_ranks_by_facet_probability():
    fake = FakeUCC()
    store, client, _, retr = await setup(fake)
    await UCCIndexer(client, store).sync_once()
    res = await retr.retrieve("where are my car keys", k=3)
    assert res.ranked[0].token.id == "a"
    assert "ucc" in res.ranked[0].reasons
    assert retr.stats["ucc"] == 1 and retr.stats["fallback"] == 0


async def test_time_window_and_device_are_metadata_filters():
    fake = FakeUCC()
    store, client, _, retr = await setup(fake)
    await UCCIndexer(client, store).sync_once()
    res = await retr.retrieve("where is the umbrella", k=3, device_id="dev")
    assert "d" not in [r.token.id for r in res.tokens]
    res = await retr.retrieve("car keys", k=3, time_from=T0 + 300, time_to=T0 + 1500)
    assert all(T0 + 300 <= r.token.ts <= T0 + 1500 for r in res.tokens)
    assert "a" not in [r.token.id for r in res.tokens]


async def test_candidates_mode_reranks_hybrid_pool():
    fake = FakeUCC()
    store, client, _, retr = await setup(fake, UCCConfig(candidates=2))
    await UCCIndexer(client, store).sync_once()
    res = await retr.retrieve("reading glasses", k=3)
    assert res.ranked[0].token.id == "c"


@pytest.mark.parametrize("fake", [FakeUCC(fail=503), FakeUCC(delay=0.5)], ids=["http-error", "timeout"])
async def test_falls_back_to_hybrid_and_cools_down(fake):
    store, client, hybrid, retr = await setup(fake, UCCConfig(timeout_s=0.1, cooldown_s=60))
    fake.created = True
    res = await retr.retrieve("where are my car keys", k=3)
    want = await hybrid.retrieve("where are my car keys", k=3)
    assert [r.token.id for r in res.tokens] == [r.token.id for r in want.tokens]
    assert retr.stats == {"ucc": 0, "fallback": 1, "errors": 1}
    await retr.retrieve("reading glasses", k=3)  # inside the cool-down: UCC is not called again
    assert retr.stats["errors"] == 1 and retr.stats["fallback"] == 2


async def test_empty_corpus_and_pure_time_questions_use_hybrid():
    fake = FakeUCC()
    _, _, _, retr = await setup(fake)
    fake.created = True
    res = await retr.retrieve("where are my car keys", k=3)  # nothing indexed yet
    assert res.tokens and retr.stats["fallback"] == 1
    selects = fake.selects
    await retr.retrieve("what was I doing 10 minutes ago", k=3)
    assert fake.selects == selects  # temporal: no select


def test_build_retriever_selects_backend(tmp_path):
    store, emb = InMemoryStore(), MockTextEmbedder()
    r, idx = build_retriever(Settings(_env_file=None), store, emb)
    assert isinstance(r, HybridRetriever) and idx is None
    r, idx = build_retriever(Settings(_env_file=None, retriever_backend="ucc"), store, emb)
    assert isinstance(r, HybridRetriever) and idx is None  # no key: hybrid
    r, idx = build_retriever(Settings(_env_file=None, retriever_backend="ucc", ucc_api_key="k"), store, emb)
    assert isinstance(r, UCCRetriever) and isinstance(idx, UCCIndexer)


async def test_indexer_backs_off_exponentially_while_unreachable(monkeypatch):
    fake = FakeUCC(fail=503)
    store, client, _, _ = await setup(fake)
    idx = UCCIndexer(client, store, interval_s=30, max_backoff_s=300)
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(s):
        delays.append(s)
        if len(delays) == 5:
            fake.fail = None  # UCC comes back
        if len(delays) == 7:
            raise asyncio.CancelledError
        await real_sleep(0)

    monkeypatch.setattr("megamind.retrieval.ucc.asyncio.sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await idx._loop()
    assert delays == [60, 120, 240, 300, 300, 30, 30]
    assert idx.failures == 0 and len(fake.docs) == 4
