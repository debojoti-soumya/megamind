import pytest

from megamind.models import MemoryToken
from megamind.store.fusion import diversify
from megamind.store.lancedb_store import LanceDBStore
from megamind.store.memory_store import InMemoryStore


def _vec(i: int, dim: int = 8) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _tok(i: int, ts: float, **kw) -> MemoryToken:
    base = dict(id=f"t{i}", device_id="dev1", ts=ts, thumb_path=f"/x/{i}.jpg", frame_hash=f"{i:016x}", text_vec=_vec(i))
    return MemoryToken(**{**base, **kw})


@pytest.fixture(params=["memory", "lancedb"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryStore()
    return LanceDBStore(tmp_path / "ldb", text_dim=8, image_dim=4)


async def _seed(store):
    await store.add(_tok(1, 100, caption="keys on the kitchen counter next to a mug", entities=["keys", "mug"]))
    await store.add(_tok(2, 200, ocr_text="PROJECT FALCON standup 3pm", caption="whiteboard in office"))
    await store.add(_tok(3, 300, transcript="Sam said the demo is on Friday"))
    await store.add(_tok(4, 400, image_vec=[0.0, 0.0, 1.0, 0.0], text_vec=None))
    await store.add(_tok(5, 500, caption="red bike poster on wall", device_id="dev2"))


async def test_roundtrip_and_count(store):
    await _seed(store)
    assert await store.count() == 5
    assert await store.count("dev2") == 1
    t = await store.get("t2")
    assert t.ocr_text.startswith("PROJECT FALCON") and t.text_vec == _vec(2)
    t4 = await store.get("t4")
    assert t4.text_vec is None and t4.image_vec == [0.0, 0.0, 1.0, 0.0]
    assert sorted(await store.recent_hashes("dev1", 250)) == [(300.0, f"{3:016x}"), (400.0, f"{4:016x}")]


async def test_exact_ocr_and_transcript_without_caption(store):
    await _seed(store)
    hits = await store.search(None, None, "falcon", mode="semantic")
    assert hits[0].id == "t2"
    hits = await store.search(None, None, "what did Sam say about the demo", mode="semantic")
    assert hits[0].id == "t3" and "Friday" in hits[0].transcript


async def test_text_vector_and_visual_only_token(store):
    await _seed(store)
    hits = await store.search(_vec(1), None, "", mode="semantic")
    assert hits[0].id == "t1"
    hits = await store.search(None, [0.0, 0.0, 1.0, 0.0], "", mode="visual")
    assert [h.id for h in hits] == ["t4"]


async def test_time_and_device_filters(store):
    await _seed(store)
    hits = await store.search(_vec(1), None, "kitchen keys whiteboard", time_from=150, time_to=350)
    assert {h.id for h in hits} <= {"t2", "t3"}
    hits = await store.search(None, None, "poster", device_id="dev1", mode="semantic")
    assert all(h.id != "t5" for h in hits)


async def test_episode_diversity(store):
    for i in range(6):
        await store.add(_tok(10 + i, 1000 + i, caption=f"kitchen counter view {i}", episode_id="epA"))
    await store.add(_tok(20, 2000, caption="kitchen sink", episode_id="epB"))
    hits = await store.search(None, None, "kitchen", k=4, mode="semantic")
    assert sum(h.episode_id == "epA" for h in hits) == 3
    assert any(h.episode_id == "epB" for h in hits)


async def test_purge(store):
    await _seed(store)
    assert await store.purge("dev2") == 1
    assert await store.count() == 4
    assert await store.purge() == 4
    assert await store.count() == 0


async def test_lancedb_persists(tmp_path):
    s = LanceDBStore(tmp_path / "ldb", text_dim=8, image_dim=4)
    await s.add(_tok(1, 100, caption="keys on counter"))
    s2 = LanceDBStore(tmp_path / "ldb", text_dim=8, image_dim=4)
    assert (await s2.get("t1")).caption == "keys on counter"


def test_diversify_backfills():
    ordered = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    assert diversify(ordered, {"a": "e", "b": "e", "c": "e"}, k=3, cap=1) == ordered


async def test_summary_hit_resolves_to_member_token(store):
    from megamind.models import MemorySummary

    await store.add(_tok(1, 100, caption="kitchen counter", text_vec=_vec(1)))
    await store.add(_tok(2, 130, caption="kitchen counter", text_vec=_vec(2)))
    await store.add(_tok(3, 900, caption="office desk", text_vec=_vec(3)))
    # the speech that answers the question spans the window, not any one frame's text
    await store.upsert_summary(
        MemorySummary(
            id="s1",
            device_id="dev1",
            start_ts=60,
            end_ts=180,
            text="we agreed to bake banana bread for the party",
            text_vec=_vec(2),
            token_ids=["t1", "t2"],
        )
    )
    hits = await store.search(_vec(2), None, "what did we plan to bake", mode="semantic")
    assert hits[0].id == "t2"
    hits = await store.search(None, None, "banana bread", mode="semantic")
    assert hits[0].id in ("t1", "t2")
    assert [s.id for s in await store.list_summaries(time_to=50)] == []
    # summaries never surface members outside the time filter
    hits = await store.search(None, None, "banana bread", mode="semantic", time_from=120)
    assert all(h.id != "t1" for h in hits)


async def test_description_is_searchable(store):
    await store.add(_tok(1, 100, caption="a kitchen", description="Shure is cutting strawberries on a board"))
    await store.add(_tok(2, 200, caption="a kitchen"))
    hits = await store.search(None, None, "who was cutting strawberries", mode="semantic")
    assert hits[0].id == "t1" and "strawberries" in hits[0].description
