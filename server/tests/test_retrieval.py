from datetime import UTC, datetime, timedelta

import pytest

from megamind.ingest.enrich import ImageEmbedder, MockTextEmbedder
from megamind.models import MemoryToken
from megamind.retrieval import HybridConfig, HybridRetriever, Weights
from megamind.retrieval.adapter import StoreAdapter, bm25_scores, cosine, token_text, tokenize
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


# --------------------------------------------------------------------------- storage primitives (#3)


class _PrimitiveStore(InMemoryStore):
    """A store exposing the #3 primitives: token+score rankings, neighbours include the anchor."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def vector_search(
        self, vec, k, *, field="text_vec", time_from=None, time_to=None, device_id=None, kinds=None
    ):
        self.calls.append(f"vector_search:{field}")
        toks = [t for t in self._tokens.values() if t.caption and getattr(t, field)]
        toks.sort(key=lambda t: -cosine(vec, getattr(t, field)))
        return [(t, 0.9) for t in toks[:k]]

    async def keyword_search(self, query, k, *, time_from=None, time_to=None, device_id=None, kinds=None):
        self.calls.append("keyword_search")
        ranked = bm25_scores(query, {t.id: token_text(t) for t in self._tokens.values() if t.caption})
        return [(self._tokens[i], s) for i, s in ranked[:k]]

    async def get_many(self, ids):
        self.calls.append("get_many")
        return [self._tokens[i] for i in ids if i in self._tokens]

    async def range(self, time_from, time_to, limit=100, *, device_id=None, kinds=None):
        self.calls.append("range")
        toks = sorted(self._tokens.values(), key=lambda t: t.ts)
        return [t for t in toks if (time_from is None or t.ts >= time_from) and (time_to is None or t.ts <= time_to)][
            :limit
        ]

    async def neighbors(self, token_id, *, before=2, after=2):
        self.calls.append(f"neighbors:{before}:{after}")
        toks = sorted(self._tokens.values(), key=lambda t: t.ts)
        idx = next(i for i, t in enumerate(toks) if t.id == token_id)
        return toks[max(0, idx - before) : idx + 1 + after]  # includes the anchor


async def _primitive_store() -> _PrimitiveStore:
    src, toks = await _store()
    store = _PrimitiveStore()
    for t in toks:
        await store.add(t)
    return store


def _tok(id_: str, ts: float, caption: str, **kw) -> MemoryToken:
    fh = f"{abs(hash(id_)) % (1 << 64):016x}"
    return MemoryToken(
        id=id_, device_id="dev", ts=ts, caption=caption, thumb_path=f"/tmp/{id_}.jpg", frame_hash=fh, **kw
    )


async def test_adapter_delegates_to_store_primitives():
    store = await _primitive_store()
    ad = StoreAdapter(store)
    ranked = await ad.keyword_search("keys", 5)
    assert ranked and all(isinstance(i, str) for i, _ in ranked)
    assert "keyword_search" in store.calls
    # tokens returned by the store are cached, so get_many does not hit the store again
    toks = await ad.get_many([i for i, _ in ranked])
    assert [t.id for t in toks] == [i for i, _ in ranked]
    assert "get_many" not in store.calls
    nb = await ad.neighbors("t3", 1, 1)
    assert "neighbors:1:1" in store.calls
    assert "t3" not in {t.id for t in nb} and {t.id for t in nb} == {"t2", "t4"}
    rng = await ad.range(None, None, limit=3)
    assert "range" in store.calls and len(rng) == 3


async def test_retriever_works_end_to_end_on_primitive_store():
    store = await _primitive_store()
    r = HybridRetriever(store, MockTextEmbedder(), tz="UTC")
    res = await r.retrieve("where are my keys", k=4, now=NOW)
    assert res.ranked[0].token.id == "t6"
    assert any(c.startswith("vector_search") for c in store.calls)


async def test_episode_cap_limits_hits_per_episode():
    store = InMemoryStore()
    base = NOW.timestamp() - 3600
    for i in range(6):
        await store.add(
            _tok(f"d{i}", base + i * 300, f"laptop on the desk next to a coffee mug variant {i}", episode_id="ep-desk")
        )
    await store.add(_tok("k1", base + 2000, "coffee mug on the kitchen counter", episode_id="ep-kitchen"))
    r = HybridRetriever(
        store, MockTextEmbedder(), config=HybridConfig(episode_cap=2, expand_top=0, dedupe_window_s=0.0)
    )
    res = await r.retrieve("coffee mug", k=6, now=NOW)
    ids = [t.token.id for t in res.tokens if not t.is_context]
    assert sum(1 for i in ids if i.startswith("d")) == 2
    assert "k1" in ids


async def test_read_text_adds_ocr_only_bm25_list():
    st, _ = await _store()
    out = await _retr(st).retrieve("what did the oat milk label say", k=3, now=NOW)
    assert out.plan.query_type == "read_text"
    best = out.ranked[0]
    assert best.token.ocr_text.strip() and "ocr_bm25" in best.reasons
    generic = await _retr(st).retrieve("where are my keys", k=3, now=NOW)
    assert all("ocr_bm25" not in r.reasons for r in generic.ranked)


async def test_meal_question_widens_when_the_meal_window_is_empty():
    # meal eaten outside the planner's lunch hours (or "lunch" asked before lunchtime): text search still finds it
    st = InMemoryStore()
    emb = MockTextEmbedder()
    now = datetime(2026, 11, 8, 16, 0, tzinfo=UTC)
    for i, (dt, cap) in enumerate([(-7200, "A desk with a laptop."), (-3600, "A plate of pasta on a cafe table.")]):
        await st.add(
            MemoryToken(
                id=f"m{i}",
                device_id="dev",
                ts=now.timestamp() + dt,
                caption=cap,
                text_vec=await emb.embed(cap),
                thumb_path="",
                frame_hash=f"{i:016x}",
            )
        )
    r = await HybridRetriever(st, emb, None, tz="UTC").retrieve("what did I have for lunch", k=3, now=now)
    assert r.widened and r.ranked[0].token.id == "m1"


async def _recency_store(now: datetime) -> InMemoryStore:
    st = InMemoryStore()
    rows = [("r0", -7200, "dev"), ("r1", -1500, "dev"), ("r2", -1200, "dev"), ("r3", -900, "dev"), ("x", -60, "other")]
    for i, (id_, dt, dev) in enumerate(rows):
        await st.add(
            MemoryToken(
                id=id_,
                device_id=dev,
                ts=now.timestamp() + dt,
                caption=f"Frame {id_} of the day.",
                thumb_path="",
                frame_hash=f"{i:016x}",
            )
        )
    return st


@pytest.mark.parametrize("q", ["what did I just see?", "what was I looking at?", "what was that?"])
async def test_recency_question_returns_latest_frames_for_device(q):
    now = datetime(2026, 11, 8, 16, 0, tzinfo=UTC)
    r = await HybridRetriever(await _recency_store(now), MockTextEmbedder(), None).retrieve(
        q, k=2, device_id="dev", now=now
    )
    assert r.plan.mode == "recent"
    assert [x.token.id for x in r.tokens] == ["r3", "r2"]  # newest first
    assert all(x.reasons == ["recent"] for x in r.tokens)


async def test_recency_looks_back_further_when_last_half_hour_is_empty():
    now = datetime(2026, 11, 8, 18, 0, tzinfo=UTC)  # newest "dev" frame is 2h25m old
    r = await HybridRetriever(await _recency_store(now - timedelta(hours=2)), MockTextEmbedder(), None).retrieve(
        "what did I just see?", k=1, device_id="dev", now=now
    )
    assert [x.token.id for x in r.tokens] == ["r3"]


async def test_empty_recent_window_falls_back_to_latest_frames():
    now = datetime(2026, 11, 8, 16, 0, tzinfo=UTC)  # nothing captured in the last ~15 min on "dev"
    r = await HybridRetriever(await _recency_store(now), MockTextEmbedder(), None).retrieve(
        "what happened a minute ago?", k=1, device_id="dev", now=now
    )
    assert r.plan.mode == "temporal" and r.widened
    assert [x.token.id for x in r.tokens] == ["r3"]


@pytest.mark.parametrize("mode", ["visual", "semantic"])
async def test_contentless_query_is_recent_whatever_mode_the_caller_passes(mode):
    now = datetime(2026, 11, 8, 16, 0, tzinfo=UTC)
    r = await HybridRetriever(await _recency_store(now), MockTextEmbedder(), None).retrieve(
        "my surroundings", k=2, mode=mode, device_id="dev", now=now
    )
    assert [x.token.id for x in r.tokens] == ["r3", "r2"]


async def test_contentless_explicit_window_near_now_falls_back_to_latest_frames():
    now = datetime(2026, 11, 8, 16, 0, tzinfo=UTC)
    r = await HybridRetriever(await _recency_store(now), MockTextEmbedder(), None).retrieve(
        "what did I see?", k=1, time_from=now.timestamp() - 300, device_id="dev", now=now
    )
    assert r.widened and [x.token.id for x in r.tokens] == ["r3"]


async def test_speech_recency_returns_latest_transcripts_else_latest_frames():
    now = datetime(2026, 11, 8, 16, 0, tzinfo=UTC)
    st = await _recency_store(now)
    r = HybridRetriever(st, MockTextEmbedder(), None)
    res = await r.retrieve("what did I hear?", k=2, device_id="dev", now=now)
    assert res.plan.kinds == ("transcript",) and [x.token.id for x in res.tokens] == ["r3", "r2"]  # no speech yet
    await st.add(
        _tok("s1", now.timestamp() - 3000, "Speech: let's meet at noon", kind="transcript", transcript="meet at noon")
    )
    res = await r.retrieve("who did I talk to?", k=2, device_id="dev", now=now)
    assert [x.token.id for x in res.tokens] == ["s1"]


async def test_ocr_search_cache_sees_new_late_enriched_and_purged_tokens():
    store = InMemoryStore()
    base = NOW.timestamp() - 3600
    await store.add(_tok("a", base, "a wall sign", ocr_text="EXIT ROUTE B"))
    await store.add(_tok("b", base + 60, "a poster"))
    ad = StoreAdapter(store)
    assert [i for i, _ in await ad.ocr_search("exit", 5)] == ["a"]
    assert await ad.ocr_search("wifi", 5) == []
    await store.add(_tok("c", base + 120, "a router label", ocr_text="WIFI PASSWORD hunter2"))  # new token
    assert [i for i, _ in await ad.ocr_search("wifi", 5)] == ["c"]
    await store.add(_tok("b", base + 60, "a poster", ocr_text="WIFI GUEST"))  # OCR filled in late
    assert {i for i, _ in await ad.ocr_search("wifi", 5)} == {"b", "c"}
    assert [i for i, _ in await ad.ocr_search("wifi", 5, time_to=base + 90)] == ["b"]
    await store.purge("dev", since_ts=base + 100)  # count drops -> full re-read
    assert [i for i, _ in await ad.ocr_search("wifi", 5)] == ["b"]


async def test_query_embedding_is_cached_across_retrieves():
    class CountingEmbedder(MockTextEmbedder):
        calls = 0

        async def embed(self, text: str) -> list[float]:
            CountingEmbedder.calls += 1
            return await super().embed(text)

    store, _ = await _store()
    r = HybridRetriever(store, CountingEmbedder(), tz="UTC")
    first = await r.retrieve("where are my keys", k=4, now=NOW)
    again = await r.retrieve("where are my keys", k=4, now=NOW)
    assert CountingEmbedder.calls == 1
    assert [t.token.id for t in again.tokens] == [t.token.id for t in first.tokens]
    r.qvec_cache_size = 1
    await r.retrieve("oat milk price", k=4, now=NOW)
    await r.retrieve("where are my keys", k=4, now=NOW)  # evicted by the LRU bound
    assert CountingEmbedder.calls == 3


# ---------------------------------------------------------------- calibrated image fusion (port of PR #54)

SIGLIP_CAL = (117.3, -12.9)  # exp(logit_scale), logit_bias of ViT-B-16-SigLIP/webli


class _AxisImageEmbedder(ImageEmbedder):
    """Every query looks for 'visual axis 0'; image vectors pick their cosine to it directly."""

    dim = 3

    def __init__(self, calibration=None) -> None:
        self._cal = calibration

    async def embed_image(self, jpeg: bytes) -> list[float]:
        return [1.0, 0.0, 0.0]

    async def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    @property
    def calibration(self):
        return self._cal


def _img(cos: float) -> list[float]:
    """Unit vector with cosine `cos` to [1, 0, 0]."""
    return [cos, (1 - cos * cos) ** 0.5, 0.0]


_IMAGE_HEAVY = Weights(text=1.0, bm25=1.0, image=3.0, recency=0.0)
_ALL_TYPES = ("locate_object", "read_text", "activity", "people", "generic")


async def _sign_store(calibration=None) -> InMemoryStore:
    st = InMemoryStore()
    st.image_calibration = calibration
    emb = MockTextEmbedder()
    rows = [
        ("sign", "a sign on the wall", "PROJECT FALCON", _img(0.0)),  # exact text, no visual match
        ("photo", "a cafe table with cups", "", _img(0.06)),  # nearest image, but a poor match
        ("photo2", "a window onto the street", "", _img(0.04)),
    ]
    for i, (tid, cap, ocr, iv) in enumerate(rows):
        await st.add(
            MemoryToken(
                id=tid,
                device_id="dev",
                ts=T0 + i * 600,
                caption=cap,
                ocr_text=ocr,
                thumb_path=f"/tmp/{tid}.jpg",
                frame_hash=f"{i:016x}",
                text_vec=await emb.embed(f"{cap} {ocr}"),
                image_vec=iv,
            )
        )
    return st


def _image_retr(st) -> HybridRetriever:
    cfg = HybridConfig(weights_by_type={t: _IMAGE_HEAVY for t in _ALL_TYPES})
    return HybridRetriever(st, MockTextEmbedder(), _AxisImageEmbedder(), tz="UTC", config=cfg)


async def test_weak_image_match_no_longer_outranks_exact_text_hit():
    q = "where did I see the falcon"
    plain = await _image_retr(await _sign_store(None)).retrieve(q, k=3, mode="hybrid", now=NOW)
    assert plain.ranked[0].token.id == "photo"  # uncalibrated: a 0.06-cosine photo gets full rank credit
    cal = await _image_retr(await _sign_store(SIGLIP_CAL)).retrieve(q, k=3, mode="hybrid", now=NOW)
    assert cal.ranked[0].token.id == "sign"


async def test_calibration_leaves_visual_mode_order_unchanged():
    q = "where did I see the falcon"
    plain = await _image_retr(await _sign_store(None)).retrieve(q, k=3, mode="visual", now=NOW)
    cal = await _image_retr(await _sign_store(SIGLIP_CAL)).retrieve(q, k=3, mode="visual", now=NOW)
    assert [(r.token.id, r.score) for r in plain.ranked] == [(r.token.id, r.score) for r in cal.ranked]
    assert plain.ranked[0].token.id == "photo"


async def _stream_with_images(calibration) -> InMemoryStore:
    st, toks = await _store()
    for i, t in enumerate(toks):
        t.image_vec = _img(0.1 + 0.05 * i)
    st.image_calibration = calibration
    return st


@pytest.mark.parametrize("q", ["where are my keys", "the people at the table", "coffee machine"])
async def test_no_calibration_keeps_todays_hybrid_scores(q):
    """Regression guard: calibration=None is the old formula, and a multiplier of exactly 1 changes nothing."""
    r = lambda st: HybridRetriever(st, MockTextEmbedder(), _AxisImageEmbedder(), tz="UTC")  # noqa: E731
    base = await r(await _stream_with_images(None)).retrieve(q, k=6, mode="hybrid", now=NOW)
    ones = await r(await _stream_with_images((0.0, 100.0))).retrieve(q, k=6, mode="hybrid", now=NOW)
    assert base.ranked and [(x.token.id, x.score) for x in base.ranked] == [(x.token.id, x.score) for x in ones.ranked]
    assert any("image" in x.reasons for x in base.ranked)


# ------------------------------------------------------------------ cross-kind speech <-> frame linking


async def _speech_store() -> InMemoryStore:
    st = InMemoryStore()
    emb = MockTextEmbedder()
    base = NOW.timestamp() - 1800
    rows = [
        ("f1", base, "frame", "Robot arm bolted to a lab bench, cables everywhere."),
        ("f2", base + 40, "frame", "Close-up of a soldering iron on a workbench."),
        ("s1", base + 6, "transcript", 'Speech heard nearby: "hi I am Priya from the robotics team"'),
        ("f3", base + 600, "frame", "A whiteboard covered in a sprint plan."),
        ("s2", base + 590, "transcript", 'Speech heard nearby: "standup moves to nine thirty"'),
    ]
    for id_, ts, kind, cap in rows:
        await st.add(
            MemoryToken(
                id=id_,
                device_id="dev",
                ts=ts,
                caption=cap,
                kind=kind,
                thumb_path="" if kind == "transcript" else f"/tmp/{id_}.jpg",
                frame_hash="" if kind == "transcript" else f"{abs(hash(id_)) % (1 << 64):016x}",
                text_vec=await emb.embed(cap),
            )
        )
    return st


async def test_transcript_hit_brings_nearest_frame_as_context():
    r = HybridRetriever(await _speech_store(), MockTextEmbedder(), tz="UTC")
    res = await r.retrieve("who is Priya", k=6, now=NOW)
    assert res.ranked[0].token.id == "s1"
    linked = [x for x in res.tokens if x.reasons == ["linked"]]
    assert [x.token.id for x in linked] == ["f1"] and linked[0].is_context  # 6 s away; f2 (34 s) is out of range


async def test_frame_hit_brings_overlapping_transcript():
    r = HybridRetriever(await _speech_store(), MockTextEmbedder(), tz="UTC")
    res = await r.retrieve("whiteboard sprint plan", k=6, now=NOW)
    assert res.ranked[0].token.id == "f3"
    assert "s2" in {x.token.id for x in res.tokens if x.reasons == ["linked"]}


async def test_speech_link_disabled():
    r = HybridRetriever(await _speech_store(), MockTextEmbedder(), tz="UTC", config=HybridConfig(speech_link_s=0))
    res = await r.retrieve("who is Priya", k=6, now=NOW)
    assert res.ranked[0].token.id == "s1"
    assert not [x for x in res.tokens if x.reasons == ["linked"]]


async def test_speech_link_stays_inside_explicit_window():
    st = await _speech_store()
    base = NOW.timestamp() - 1800
    r = HybridRetriever(st, MockTextEmbedder(), tz="UTC")
    res = await r.retrieve("who is Priya", k=6, now=NOW, time_from=base + 3, time_to=base + 100)
    assert "f1" not in {x.token.id for x in res.tokens}  # 3 s before the window opens


def test_build_retriever_reads_speech_link_setting():
    from megamind.config import Settings
    from megamind.retrieval import build_retriever

    r, _ = build_retriever(Settings(speech_link_s=0, _env_file=None), InMemoryStore(), MockTextEmbedder())
    assert r.cfg.speech_link_s == 0


# ---------------------------------------------------------------- purge invalidates retrieval caches (#111)


async def test_purge_all_invalidates_the_ocr_cache_through_the_live_wrapper():
    from types import SimpleNamespace

    from megamind.ingest.worker import purge_all
    from megamind.live import LiveRetriever

    store = InMemoryStore()
    base = NOW.timestamp() - 3600
    await store.add(_tok("b", base, "a poster"))
    await store.add(_tok("a", base + 60, "a router label", ocr_text="WIFI PASSWORD hunter2"))
    hybrid = HybridRetriever(store, MockTextEmbedder(), tz="UTC")
    ad = hybrid.adapter
    assert [i for i, _ in await ad.ocr_search("wifi", 5)] == ["a"]

    class Worker:
        async def purge(self, device_id=None, *, since_ts=None):
            return await store.purge(device_id, since_ts=since_ts)

    state = SimpleNamespace(worker=Worker(), retriever=LiveRetriever(hybrid, live=None))
    assert await purge_all(state, "dev", since_ts=base + 30) == 1
    # a new frame inside the rebuild interval that the incremental refresh doesn't re-read (backfilled, older ts)
    # restores the store count, so only the explicit invalidation keeps the purged OCR out of search
    await store.add(_tok("c", base - 3600, "a hallway"))
    assert await ad.ocr_search("wifi", 5) == []
    assert "a" not in {t.token.id for t in (await hybrid.retrieve("wifi password", k=4, now=NOW)).tokens}


async def test_degraded_query_vectors_are_not_cached():
    class FlakyEmbedder(MockTextEmbedder):
        degraded: str | None = "last error 503"
        calls = 0

        async def embed(self, text: str) -> list[float]:
            FlakyEmbedder.calls += 1
            return await super().embed(text)

    store, _ = await _store()
    emb = FlakyEmbedder()
    r = HybridRetriever(store, emb, tz="UTC")
    await r.retrieve("where are my keys", k=4, now=NOW)
    assert FlakyEmbedder.calls == 1 and not r._qvecs  # served, but not kept
    emb.degraded = None  # the hosted embedder recovered
    await r.retrieve("where are my keys", k=4, now=NOW)
    await r.retrieve("where are my keys", k=4, now=NOW)
    assert FlakyEmbedder.calls == 2 and len(r._qvecs) == 1


async def test_anchor_window_is_one_sided_and_focused():
    st, _ = await _store()
    r = await _retr(st).retrieve("what did I see before I went to the conference table", k=3, now=NOW)
    assert r.anchor_ts == pytest.approx(T0 + 1800)
    assert {x.token.id for x in r.tokens} == {"t3", "t4", "t5"}  # the anchor moment and what led up to it
    after = await _retr(st).retrieve("what did I see after I went to the conference table", k=3, now=NOW)
    assert {x.token.id for x in after.tokens} == {"t5", "t6", "t7"}


async def test_anchor_resolves_inside_the_parsed_window():
    st, _ = await _store()
    now = datetime(2026, 11, 8, 20, 0, tzinfo=UTC)  # the stream (11:00-12:00 UTC) is "this morning" in UTC
    r = await _retr(st).retrieve("what did I see before I went to the conference table today", k=3, now=now)
    assert r.anchor_ts == pytest.approx(T0 + 1800)
    assert {x.token.id for x in r.tokens} == {"t3", "t4", "t5"}


async def test_focus_prior_pulls_towards_the_window_edge():
    st, _ = await _store()
    rt = _retr(st)
    ref = T0 + 900
    near_edge = await rt._fuse("keys", "hybrid", Weights(recency=1.0), None, None, None, ref, around=True)
    towards_now = await rt._fuse("keys", "hybrid", Weights(recency=1.0), None, None, None, ref)
    assert near_edge[0][0] == "t1" and towards_now[0][0] == "t6"
