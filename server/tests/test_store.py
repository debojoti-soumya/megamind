import random

import pyarrow as pa
import pytest

from megamind.models import MemoryToken, Pose
from megamind.store.lancedb_store import LanceDBStore
from megamind.store.memory_store import InMemoryStore
from megamind.store.rrf import calibrated, fuse_hits, image_conf, rrf, weighted_rrf

DIM = 8


def _vec(i: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _token(i: int, *, device="dev1", caption=None, ocr="", entities=(), tmp_path=None, image=True) -> MemoryToken:
    thumb = tmp_path / f"t{i}.jpg" if tmp_path else f"/nonexistent/t{i}.jpg"
    if tmp_path:
        thumb.write_bytes(b"jpeg")
    return MemoryToken(
        id=f"tok{i:04d}",
        device_id=device,
        ts=1000.0 + i,
        caption=f"frame {i} caption" if caption is None else caption,
        ocr_text=ocr,
        entities=list(entities),
        pose=Pose(x=0.0, y=1.6, z=float(i), yaw=90.0) if i % 2 == 0 else None,
        text_vec=_vec(i),
        image_vec=_vec(i + 1) if image else None,
        thumb_path=str(thumb),
        frame_hash=f"{i:016x}",
        episode_id=None,
    )


@pytest.fixture(params=["memory", "lancedb"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryStore()
    return LanceDBStore(tmp_path / "lancedb", vector_index_min_rows=10_000)


@pytest.fixture
def lance(tmp_path):
    return LanceDBStore(tmp_path / "lancedb", vector_index_min_rows=10_000)


async def test_roundtrip_and_count(store):
    t = _token(0, caption="keys on the kitchen counter", ocr="WELCOME", entities=["keys", "counter"])
    await store.add(t)
    got = await store.get("tok0000")
    assert got is not None
    assert got.model_dump() == t.model_dump()
    assert await store.count() == 1
    assert await store.count("dev1") == 1
    assert await store.count("other") == 0
    assert await store.get("missing") is None


async def test_add_is_upsert(store):
    t = _token(1)
    await store.add(t)
    t.caption = "updated caption"
    t.text_vec = _vec(3)
    await store.add(t)
    assert await store.count() == 1
    assert (await store.get(t.id)).caption == "updated caption"


async def test_persistence_across_reopen(tmp_path):
    s1 = LanceDBStore(tmp_path / "db")
    await s1.add(_token(0))
    await s1.add(_token(1, image=False))
    s2 = LanceDBStore(tmp_path / "db")
    assert await s2.count() == 2
    assert (await s2.get("tok0001")).image_vec is None


async def test_dim_mismatch_is_rejected(lance):
    await lance.add(_token(0))
    bad = _token(1)
    bad.text_vec = [1.0] * (DIM + 1)
    with pytest.raises(ValueError, match="dims"):
        await lance.add(bad)


async def test_recent_hashes_filters_device_and_time(store):
    for i in range(5):
        await store.add(_token(i, device="a" if i < 3 else "b"))
    got = await store.recent_hashes("a", since_ts=1001.0)
    assert sorted(got) == [(1001.0, f"{1:016x}"), (1002.0, f"{2:016x}")]
    assert await store.recent_hashes("c", since_ts=0) == []


async def test_vector_and_keyword_search(store):
    await store.add(_token(0, caption="keys on the kitchen counter", entities=["keys"]))
    await store.add(_token(1, caption="laptop on a wooden desk", ocr="MEETING 3PM"))
    await store.add(_token(2, caption="", image=False))  # unenriched: never searchable

    hits = await store.vector_search(_vec(1), 5)
    assert [t.id for t, _ in hits][0] == "tok0001"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)

    hits = await store.vector_search(_vec(1), 5, field="image_vec")
    assert [t.id for t, _ in hits][0] == "tok0000"

    hits = await store.keyword_search("meeting", 5)
    assert [t.id for t, _ in hits] == ["tok0001"]
    hits = await store.keyword_search("keys counter", 5)
    assert [t.id for t, _ in hits] == ["tok0000"]
    assert await store.keyword_search("!!! ???", 5) == []


async def test_hybrid_search_with_filters(store):
    for i in range(6):
        await store.add(_token(i, device="a" if i < 4 else "b", caption=f"object number{i} on the table"))
    hits = await store.search(_vec(2), None, "number2", k=3)
    assert hits[0].id == "tok0002"
    assert hits[0].thumb_url == "/v1/thumbs/tok0002.jpg"

    hits = await store.search(_vec(2), None, "table", k=10, time_from=1003.0, time_to=1004.0)
    assert sorted(h.id for h in hits) == ["tok0003", "tok0004"]

    hits = await store.search(_vec(2), None, "table", k=10, device_id="b")
    assert sorted(h.id for h in hits) == ["tok0004", "tok0005"]

    hits = await store.search(None, _vec(5), "zzz", k=10, mode="visual")
    assert hits[0].id == "tok0004"

    # query vector of a different dimension (e.g. embedder swapped) degrades to keyword-only
    hits = await store.search([1.0] * (DIM + 3), None, "number5", k=3)
    assert hits[0].id == "tok0005"


async def test_search_on_empty_store(store):
    assert await store.search(_vec(0), None, "anything") == []
    assert await store.count() == 0
    assert await store.recent_hashes("dev1", 0) == []


async def test_get_many_range_neighbors(store):
    for i in range(10):
        await store.add(_token(i, device="a" if i != 5 else "b"))
    many = await store.get_many(["tok0003", "missing", "tok0001", "tok0003"])
    assert [t.id for t in many] == ["tok0003", "tok0001"]

    rng = await store.range(1002.0, 1006.0, limit=3)
    assert [t.id for t in rng] == ["tok0002", "tok0003", "tok0004"]
    rng = await store.range(None, None, limit=100, device_id="b")
    assert [t.id for t in rng] == ["tok0005"]

    nb = await store.neighbors("tok0006", before=2, after=1)
    assert [t.id for t in nb] == ["tok0003", "tok0004", "tok0006", "tok0007"]  # tok0005 is another device
    assert await store.neighbors("missing") == []


async def test_purge(store, tmp_path):
    for i in range(4):
        await store.add(_token(i, device="a" if i < 2 else "b", tmp_path=tmp_path))
    assert await store.purge("a") == 2
    assert await store.count() == 2
    assert not (tmp_path / "t0.jpg").exists() and (tmp_path / "t2.jpg").exists()
    assert await store.purge(since_ts=1003.0) == 1
    assert await store.purge() == 1
    assert await store.count() == 0
    assert await store.purge() == 0


async def test_fts_index_sees_rows_added_after_build(lance):
    await lance.add(_token(0, caption="first row builds the index"))
    assert any(idx.columns == ["search_text"] for idx in lance._table.list_indices())
    await lance.add(_token(1, caption="banana smoothie on the counter"))
    hits = await lance.keyword_search("banana", 5)
    assert [t.id for t, _ in hits] == ["tok0001"]


async def test_add_many_and_kinds_filter(store):
    toks = [_token(i, caption=f"scene {i}") for i in range(4)]
    await store.add_many(toks)
    assert await store.count() == 4
    assert [t.id for t in await store.range(None, None, kinds=("frame",))] == [t.id for t in toks]
    assert await store.range(None, None, kinds=("episode",)) == []
    assert await store.vector_search(_vec(1), 3, kinds=("episode",)) == []
    assert sorted(t.id for t, _ in await store.keyword_search("scene", 10, kinds=("frame",))) == [t.id for t in toks]


async def test_search_unrelated_query_returns_nothing(store):
    await store.add_many([_token(i, caption=f"coffee mug on desk {i}") for i in range(3)])
    hits = await store.search(_vec(99), None, "zebra", k=5)
    assert hits == []


async def test_vector_search_drops_orthogonal_vectors(store):
    await store.add_many([_token(0), _token(1), _token(2)])
    hits = await store.vector_search(_vec(1), 10)
    assert [t.id for t, _ in hits] == ["tok0001"]
    assert await store.vector_search(_vec(5), 10) == []


# ---------------------------------------------------------------- calibrated image fusion + float16 (port of PR #54)

SIGLIP_CAL = (117.3, -12.9)


def test_image_conf_and_weighted_rrf():
    assert image_conf(0.15, *SIGLIP_CAL) > 0.9 and image_conf(0.05, *SIGLIP_CAL) < 0.01
    assert image_conf(-1.0, 1e6, 0.0) == 0.0  # no overflow on very negative logits
    assert calibrated(None) is None and calibrated(SIGLIP_CAL)(0.15) == image_conf(0.15, *SIGLIP_CAL)
    assert weighted_rrf([(1.0, [("a", 0.0), ("b", 1.0)])])["a"] == 0.0
    assert weighted_rrf([(2.0, ["a"])], c=60)["a"] == 2.0 / 61
    assert weighted_rrf([(1.0, ["a", "b"]), (1.0, ["b"])]) == rrf([["a", "b"], ["b"]])


def test_fuse_hits_defaults_are_plain_rrf():
    toks = [_token(i) for i in range(4)]
    rankings = [[(toks[0], 0.9), (toks[1], 0.5)], [(toks[2], 3.0), (toks[0], 1.0)], [(toks[3], 0.2), (toks[1], 0.1)]]
    plain = [(h.id, h.score) for h in fuse_hits(rankings, 10)]
    explicit = [(h.id, h.score) for h in fuse_hits(rankings, 10, weights=[1.0] * 3, multipliers=[None] * 3)]
    assert plain == explicit
    expected = rrf([[t.id for t, _ in r] for r in rankings])
    assert dict(plain) == expected
    with pytest.raises(ValueError):
        fuse_hits(rankings, 10, weights=[1.0])


def _unit_img(cos: float, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[0], v[1] = cos, (1 - cos * cos) ** 0.5
    return v


async def test_store_search_weak_image_does_not_outrank_text(store):
    query_img = [1.0] + [0.0] * (DIM - 1)
    sign = _token(0, caption="a sign on the wall", ocr="PROJECT FALCON")
    sign.image_vec = _unit_img(0.0)
    sign.text_vec = _vec(5)
    photo = _token(1, caption="a cafe table with cups")
    photo.image_vec = _unit_img(0.06)
    photo.text_vec = _vec(6)
    await store.add_many([sign, photo])
    # uncalibrated: the image ranking (rank 0) ties the text hit, and the newer photo wins the tie
    hits = await store.search(None, query_img, "falcon", k=2)
    assert hits[0].id == "tok0001"
    store.image_calibration = SIGLIP_CAL
    hits = await store.search(None, query_img, "falcon", k=2)
    assert hits[0].id == "tok0000"
    # visual mode: one ranking, monotone multiplier -> same order
    assert [h.id for h in await store.search(None, query_img, "", k=2, mode="visual")] == ["tok0001"]


def _rand_unit(rnd: random.Random, dim: int) -> list[float]:
    v = [rnd.gauss(0, 1) for _ in range(dim)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


_RAND_DIMS = {"text_dim": 64, "image_dim": 48}


def _rand_tokens(n: int, text_dim: int = 64, image_dim: int = 48, seed: int = 7) -> list[MemoryToken]:
    rnd = random.Random(seed)
    return [
        MemoryToken(
            id=f"r{i:04d}",
            device_id="d",
            ts=1000.0 + i,
            caption=f"random frame {i}",
            thumb_path=f"/nonexistent/r{i}.jpg",
            frame_hash=f"{i:016x}",
            text_vec=_rand_unit(rnd, text_dim),
            image_vec=_rand_unit(rnd, image_dim) if i % 4 else None,
        )
        for i in range(n)
    ]


async def test_float16_is_the_default_and_round_trips(tmp_path):
    st = LanceDBStore(tmp_path / "ldb", **_RAND_DIMS)
    toks = _rand_tokens(20)
    await st.add_many(toks)
    assert st.vector_dtype == "float16"
    assert st._table.schema.field("text_vec").type.value_type == pa.float16()
    for t in toks[::5]:
        back = await st.get(t.id)
        assert max(abs(a - b) for a, b in zip(back.text_vec, t.text_vec, strict=True)) < 1e-3
        assert (back.image_vec is None) == (t.image_vec is None)


async def test_float16_nearest_neighbours_match_float32(tmp_path):
    toks = _rand_tokens(200)
    f16 = LanceDBStore(tmp_path / "f16", vector_dtype="float16", vector_index_min_rows=10_000, **_RAND_DIMS)
    f32 = LanceDBStore(tmp_path / "f32", vector_dtype="float32", vector_index_min_rows=10_000, **_RAND_DIMS)
    await f16.add_many(toks)
    await f32.add_many(toks)
    for t in toks[::10]:  # every token is its own nearest neighbour after the float16 round trip
        assert (await f16.vector_search(t.text_vec, 1))[0][0].id == t.id
    rnd = random.Random(11)
    for _ in range(20):
        q = _rand_unit(rnd, 64)
        assert (await f16.vector_search(q, 1))[0][0].id == (await f32.vector_search(q, 1))[0][0].id
        qi = _rand_unit(rnd, 48)
        a = await f16.vector_search(qi, 1, field="image_vec")
        b = await f32.vector_search(qi, 1, field="image_vec")
        assert [t.id for t, _ in a] == [t.id for t, _ in b]


async def test_existing_float32_table_keeps_its_dtype(tmp_path):
    path = tmp_path / "ldb"
    old = LanceDBStore(path, vector_dtype="float32")
    await old.add(_token(0))
    reopened = LanceDBStore(path)  # default float16 must not migrate or break an existing table
    assert reopened.vector_dtype == "float32"
    await reopened.add(_token(1))
    assert reopened._table.schema.field("text_vec").type.value_type == pa.float32()
    assert (await reopened.get("tok0001")).text_vec == _vec(1)
    assert await reopened.count() == 2


async def test_ivf_index_builds_on_float16_nullable_columns(tmp_path):
    st = LanceDBStore(tmp_path / "ldb", vector_index_min_rows=64, **_RAND_DIMS)
    toks = _rand_tokens(120)  # every 4th token has no image vector (null)
    await st.add_many(toks)
    indexed = {idx.columns[0] for idx in st._table.list_indices()}
    assert {"text_vec", "image_vec"} <= indexed
    t = toks[5]
    assert (await st.vector_search(t.image_vec, 1, field="image_vec"))[0][0].id == t.id
    assert (await st.vector_search(t.text_vec, 1))[0][0].id == t.id


def test_unknown_vector_dtype_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        LanceDBStore(tmp_path / "ldb", vector_dtype="int8")


async def test_lancedb_decodes_tokens_exactly(lance):
    toks = [_token(i, entities=("mug", f"e{i}"), image=i % 3 != 1) for i in range(7)]
    toks[4] = toks[4].model_copy(update={"kind": "episode", "episode_id": "ep-x", "ocr_text": "EXIT"})
    await lance.add_many(toks)
    got = await lance.range(None, None, limit=100)
    assert [t.model_dump() for t in got] == [t.model_dump() for t in toks]
    assert [t.model_dump() for t in await lance.get_many([toks[5].id, toks[0].id])] == [
        toks[5].model_dump(),
        toks[0].model_dump(),
    ]
    assert (await lance.get(toks[4].id)).image_vec is None
    hits = await lance.vector_search(_vec(2), 3)
    assert hits[0][0].model_dump() == toks[2].model_dump()


async def test_range_orders_ties_by_id_and_truncates(store):
    toks = [_token(i).model_copy(update={"ts": 1000.0 + (i // 2)}) for i in range(6)]
    await store.add_many(list(reversed(toks)))
    got = await store.range(None, None, limit=4)
    assert [t.id for t in got] == [t.id for t in toks[:4]]
    assert await store.range(None, None, limit=0) == []
    assert [t.id for t in await store.neighbors(toks[2].id, before=1, after=1)] == [t.id for t in toks[1:4]]


async def test_memory_vector_search_matches_bruteforce_and_tracks_updates():
    import math
    import random

    rnd = random.Random(7)
    store = InMemoryStore()
    toks = [_token(i).model_copy(update={"text_vec": [rnd.uniform(-1, 1) for _ in range(DIM)]}) for i in range(40)]
    toks[3].text_vec = [0.0] * DIM  # zero vector scores 0 and is dropped, as before
    await store.add_many(toks)
    q = [rnd.uniform(-1, 1) for _ in range(DIM)]

    def cos(a, b):
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    want = sorted(((t, cos(q, t.text_vec)) for t in toks if cos(q, t.text_vec) > 1e-6), key=lambda x: (x[1], x[0].ts))
    got = await store.vector_search(q, 10)
    assert [t.id for t, _ in got] == [t.id for t, _ in reversed(want)][:10]
    assert [s for _, s in got] == pytest.approx([s for _, s in reversed(want)][:10])

    toks[0].text_vec = list(q)  # reassigned in place on the stored object -> cache must not go stale
    assert (await store.vector_search(q, 1))[0][0].id == toks[0].id
    await store.add(toks[5].model_copy(update={"id": toks[0].id, "text_vec": [-x for x in q]}))
    assert toks[0].id not in {t.id for t, _ in await store.vector_search(q, 40)}
    await store.purge()
    assert await store.vector_search(q, 5) == []
