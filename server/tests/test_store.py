import pytest

from megamind.models import MemoryToken, Pose
from megamind.store.lancedb_store import LanceDBStore
from megamind.store.memory_store import InMemoryStore

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
