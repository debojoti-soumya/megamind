from datetime import UTC, datetime

import pytest

from megamind.ingest.enrich import ImageEmbedder, MockTextEmbedder
from megamind.models import MemoryToken
from megamind.retrieval import HybridConfig, HybridRetriever
from megamind.retrieval.adapter import StoreAdapter, bm25_scores, tokenize
from megamind.store.memory_store import InMemoryStore

NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
T0 = NOW.timestamp() - 3600  # stream starts an hour before "now"

STREAM = [
    # (offset_s, caption, ocr, entities, frame_hash)
    (0, "A kitchen counter with a coffee machine and a mug.", "", ["coffee machine", "mug"], "ff00ff00ff00ff00"),
    (60, "Keys lying on the kitchen counter next to the coffee machine.", "", ["keys", "counter"], "ff00ff00ff00ff01"),
    (120, "Hallway with a coat rack and a closed front door.", "", ["coat rack", "door"], "0000000000000000"),
    (
        900,
        "Office desk with a laptop; a whiteboard behind.",
        "DEMO FRIDAY 5PM",
        ["laptop", "whiteboard"],
        "1234123412341234",
    ),
    (
        960,
        "Close-up of the whiteboard schedule.",
        "DEMO FRIDAY 5PM\nPROJECT FALCON",
        ["whiteboard"],
        "1234123412341235",
    ),
    (1800, "Two people sitting at a conference table talking.", "", ["people", "table"], "abcdabcdabcdabcd"),
    (2400, "Keys on a wooden desk beside a notebook.", "", ["keys", "desk", "notebook"], "5555555555555555"),
    (
        3000,
        "A grocery shelf with cartons of oat milk and a price tag.",
        "OAT MILK $4.99",
        ["oat milk"],
        "9999999999999999",
    ),
]


async def _store(with_vecs: bool = True) -> tuple[InMemoryStore, list[MemoryToken]]:
    st = InMemoryStore()
    emb = MockTextEmbedder()
    toks = []
    for i, (off, cap, ocr, ents, fh) in enumerate(STREAM):
        t = MemoryToken(
            id=f"t{i}",
            device_id="dev",
            ts=T0 + off,
            caption=cap,
            ocr_text=ocr,
            entities=ents,
            thumb_path=f"/tmp/t{i}.jpg",
            frame_hash=fh,
            text_vec=(await emb.embed(f"{cap} {ocr}")) if with_vecs else None,
        )
        await st.add(t)
        toks.append(t)
    return st, toks


def _retr(st, **cfg) -> HybridRetriever:
    return HybridRetriever(st, MockTextEmbedder(), None, tz="UTC", config=HybridConfig(**cfg))


def test_tokenize_and_bm25():
    assert tokenize("The Keys' are on the counters.") == ["the", "key", "are", "on", "the", "counter"]
    ranked = bm25_scores("keys counter", {"a": "keys on the counter", "b": "a door", "c": "keys on a desk"})
    assert [i for i, _ in ranked] == ["a", "c"]


async def test_adapter_fallback_primitives():
    st, toks = await _store()
    ad = StoreAdapter(st)
    kw = await ad.keyword_search("whiteboard", 5)
    assert {i for i, _ in kw} == {"t3", "t4"}
    rng = await ad.range(T0 + 100, T0 + 1000)
    assert [t.id for t in rng] == ["t2", "t3", "t4"]
    nb = await ad.neighbors("t3", before=1, after=2)
    assert [t.id for t in nb] == ["t2", "t4", "t5"]
    assert [t.id for t in await ad.get_many(["t1", "nope", "t0"])] == ["t1", "t0"]
    vs = await ad.vector_search(await MockTextEmbedder().embed("oat milk"), 3, "text_vec")
    assert vs[0][0] == "t7"


async def test_locate_object_prefers_most_recent_sighting_and_orders_chronologically():
    st, _ = await _store()
    r = await _retr(st).retrieve("where did I leave my keys?", k=4, now=NOW)
    ids = [x.token.id for x in r.tokens]
    assert ids == sorted(ids, key=lambda i: int(i[1:]))  # chronological
    assert r.ranked[0].token.id == "t6"  # keys on desk (newer) beats keys on counter
    assert "t1" in ids
    assert r.plan.query_type == "locate_object"


async def test_time_window_from_query_filters_results():
    st, _ = await _store()
    # keys were on the counter ~59 min ago and on the desk ~20 min ago
    r = await _retr(st).retrieve("where were my keys about an hour ago", k=3, now=NOW)
    assert r.ranked[0].token.id == "t1"
    assert all(abs((NOW.timestamp() - 3600) - x.token.ts) <= 0.35 * 3600 + 1 for x in r.tokens)


async def test_explicit_bounds_are_honoured():
    st, _ = await _store()
    r = await _retr(st).retrieve("keys", k=5, time_from=T0 + 2000, time_to=T0 + 2600, now=NOW)
    assert [x.token.id for x in r.tokens if not x.is_context] == ["t6"]


async def test_read_text_hits_ocr_and_dedupes_near_identical_frames():
    st, _ = await _store()
    r = await _retr(st, expand_top=0).retrieve("what did the whiteboard say about the demo", k=5, now=NOW)
    ids = [x.token.id for x in r.tokens]
    assert "t4" in ids or "t3" in ids
    assert not ("t3" in ids and "t4" in ids), "t3/t4 are the same frame (dHash distance 1, 60 s apart)"


async def test_temporal_expansion_adds_neighbors_as_context():
    st, _ = await _store()
    r = await _retr(st, dedupe_window_s=0).retrieve("oat milk price", k=6, now=NOW)
    best = r.ranked[0]
    assert best.token.id == "t7" and not best.is_context
    ctx = [x for x in r.tokens if x.is_context]
    assert ctx and all(x.reasons == ["neighbor"] for x in ctx)
    assert any(x.token.id == "t6" for x in ctx)  # t7's predecessor
    assert len(r.tokens) <= 6


async def test_pure_temporal_query_spreads_frames_over_window():
    st, _ = await _store()
    r = await _retr(st).retrieve("what was I doing 45 minutes ago?", k=3, now=NOW)
    assert r.plan.mode == "temporal"
    assert all(x.reasons == ["time"] for x in r.tokens)
    lo, hi = r.plan.time_from, r.plan.time_to
    assert all(lo <= x.token.ts <= hi for x in r.tokens)
    assert len(r.tokens) <= 3


async def test_widens_when_parsed_window_is_empty():
    st, _ = await _store()
    r = await _retr(st).retrieve("the oat milk price 3 days ago", k=3, now=NOW)
    assert r.widened and r.ranked[0].token.id == "t7"
    r = await _retr(st).retrieve("oat milk", k=3, time_from=T0 - 7 * 86400, time_to=T0 - 6 * 86400, now=NOW)
    assert not r.widened and r.tokens == []


async def test_anchor_event_bounds_time():
    st, _ = await _store()
    r = await _retr(st).retrieve("where were my keys before I went to the conference table", k=3, now=NOW)
    assert r.plan.anchor_event == "went to the conference table"
    assert r.anchor_ts == pytest.approx(T0 + 1800)
    assert all(x.token.ts <= T0 + 1800 for x in r.tokens)
    assert r.ranked[0].token.id == "t1"


async def test_modes_semantic_and_visual_without_image_tower_degrade():
    st, _ = await _store()
    sem = await _retr(st).retrieve("keys", k=3, mode="semantic", now=NOW)
    vis = await _retr(st).retrieve("keys", k=3, mode="visual", now=NOW)
    assert {x.token.id for x in sem.ranked if not x.is_context} >= {"t6"}
    assert {x.token.id for x in vis.ranked if not x.is_context} >= {"t6"}


class _FakeImageEmbedder(ImageEmbedder):
    dim = 4

    async def embed_image(self, jpeg: bytes) -> list[float]:
        return [1, 0, 0, 0]

    async def embed_text(self, text: str) -> list[float]:
        return [1, 0, 0, 0] if "people" in text else [0, 1, 0, 0]


async def test_visual_mode_uses_image_vectors():
    st, toks = await _store()
    for t in toks:
        t.image_vec = [1, 0, 0, 0] if t.id == "t5" else [0, 0, 1, 0]
    r = HybridRetriever(st, MockTextEmbedder(), _FakeImageEmbedder(), tz="UTC")
    out = await r.retrieve("the people I was sitting with", k=2, mode="visual", now=NOW)
    assert out.ranked[0].token.id == "t5" and "image" in out.ranked[0].reasons


async def test_empty_store_and_unknown_query():
    st = InMemoryStore()
    r = await _retr(st).retrieve("where are my keys", now=NOW)
    assert r.tokens == []
    st, _ = await _store()
    r = await _retr(st).retrieve("zebra quantum", k=3, now=NOW)
    assert r.tokens == [] or all(x.score > 0 for x in r.tokens)


async def test_device_filter():
    st, _ = await _store()
    r = await _retr(st).retrieve("keys", k=3, device_id="other", now=NOW)
    assert r.tokens == []


async def test_hits_and_memory_tokens_views():
    st, _ = await _store()
    r = await _retr(st).retrieve("keys", k=4, now=NOW)
    hits = r.hits()
    assert hits[0].id == r.ranked[0].token.id and hits[0].thumb_url.endswith(".jpg")
    assert [t.id for t in r.memory_tokens()] == [x.token.id for x in r.tokens]
