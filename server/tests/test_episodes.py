import math

import pytest

from megamind.models import MemoryToken, Pose
from megamind.store import EpisodeTracker
from megamind.store.consolidate import Consolidator, MockSummarizer, summary_id
from megamind.store.lancedb_store import LanceDBStore
from megamind.store.memory_store import InMemoryStore


def _unit(angle_deg: float) -> list[float]:
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a), 0.0, 0.0]


def _frame(i: int, ts: float, *, device="dev1", angle=0.0, caption=None, ocr="", entities=(), tmp_path=None):
    thumb = tmp_path / f"f{i}.jpg" if tmp_path else f"/nonexistent/f{i}.jpg"
    if tmp_path:
        thumb.write_bytes(b"jpeg")
    return MemoryToken(
        id=f"f{i:04d}",
        device_id=device,
        ts=ts,
        caption=f"frame {i}" if caption is None else caption,
        ocr_text=ocr,
        entities=list(entities),
        text_vec=[1.0, 0.0, 0.0, 0.0],
        image_vec=_unit(angle),
        thumb_path=str(thumb),
        frame_hash=f"{i:016x}",
    )


@pytest.fixture(params=["memory", "lancedb"])
def make_store(request, tmp_path):
    def _make(tracker: EpisodeTracker | None):
        if request.param == "memory":
            return InMemoryStore(episodes=tracker)
        return LanceDBStore(tmp_path / "lancedb", vector_index_min_rows=10_000, episodes=tracker)

    return _make


# ------------------------------------------------------------------ tracker (pure)


def test_tracker_splits_on_time_gap_and_visual_change():
    tr = EpisodeTracker(gap_s=60, split_sim=0.8, rejoin_sim=0.9)
    a = [_frame(i, 1000 + 5 * i) for i in range(4)]  # same view, 5s apart
    ids = [tr.assign(t) for t in a]
    assert len(set(ids)) == 1 and all(t.episode_id == ids[0] for t in a)

    gap = _frame(10, 1000 + 15 + 120)  # > gap_s later
    assert tr.assign(gap) != ids[0]

    turned = _frame(11, gap.ts + 5, angle=90)  # cos = 0 < split_sim
    assert tr.assign(turned) not in (ids[0], gap.episode_id)
    assert len(tr.open_episodes("dev1")) == 3


def test_tracker_rejoins_recent_episode_after_detour():
    tr = EpisodeTracker(gap_s=600, split_sim=0.8, rejoin_sim=0.9)
    desk = [tr.assign(_frame(i, 1000 + 5 * i, angle=0)) for i in range(3)]
    kitchen = tr.assign(_frame(5, 1030, angle=90))
    back = tr.assign(_frame(6, 1040, angle=2))
    assert kitchen != desk[0]
    assert back == desk[0]  # tight match to the older centroid → rejoin
    assert tr.open_episodes("dev1")[-1].id == desk[0]  # rejoined episode becomes current


def test_tracker_respects_preassigned_and_isolates_devices():
    tr = EpisodeTracker()
    t = _frame(0, 1000)
    t.episode_id = "ep-custom"
    assert tr.assign(t) == "ep-custom"
    assert tr.assign(_frame(1, 1001)) == "ep-custom"
    other = _frame(2, 1001, device="dev2")
    assert tr.assign(other) != "ep-custom"
    assert tr.open_episodes("dev2")[0].device_id == "dev2"


def test_tracker_seed_rebuilds_window():
    tr = EpisodeTracker(gap_s=60, rejoin_last=2)
    frames = [_frame(i, 1000 + 5 * i) for i in range(3)]
    for t in frames:
        t.episode_id = "ep-a"
    frames += [_frame(9, 1300)]
    frames[-1].episode_id = "ep-b"
    tr.seed("dev1", frames)
    assert [e.id for e in tr.open_episodes("dev1")] == ["ep-a", "ep-b"]
    assert tr.assign(_frame(10, 1305)) == "ep-b"


def _drift_then_jump() -> list[MemoryToken]:
    """A slow 5-degree-per-frame turn from 0 to 60 degrees, then a jump back to -5 degrees."""
    frames = [_frame(i, 1000 + 5 * i, angle=5 * i) for i in range(13)]
    return frames + [_frame(20, 1000 + 5 * 13, angle=-5)]


def test_split_compares_with_recent_window_not_whole_history():
    # the jump is 0.82 cosine to the whole-episode mean (~30 deg) but 0.61 to the last 6 frames (~47 deg)
    recent = EpisodeTracker(gap_s=60, split_sim=0.70, rejoin_sim=0.99, recent_n=6, max_s=0, move_m=0)
    ids = [recent.assign(t) for t in _drift_then_jump()]
    assert len(set(ids[:-1])) == 1  # the slow turn is one episode: each frame is close to the recent ones
    assert ids[-1] != ids[0]

    whole = EpisodeTracker(gap_s=60, split_sim=0.70, rejoin_sim=0.99, recent_n=10_000, max_s=0, move_m=0)
    assert len({whole.assign(t) for t in _drift_then_jump()}) == 1  # the old whole-mean rule never splits here


def test_max_length_caps_long_stays():
    tr = EpisodeTracker(gap_s=120, max_s=600, move_m=0)
    ids = [tr.assign(_frame(i, 1000 + 60 * i)) for i in range(25)]  # same view every minute for 24 min
    runs = [ids[0]] + [b for a, b in zip(ids, ids[1:]) if a != b]
    assert len(runs) == len(set(runs)) == 3  # 0-600, 660-1260, 1320-1440: split, never rejoined (would exceed cap)
    for ep in tr.open_episodes("dev1"):
        assert ep.end_ts - ep.start_ts <= 600

    off = EpisodeTracker(gap_s=120, max_s=0)
    assert len({off.assign(_frame(i, 1000 + 60 * i)) for i in range(25)}) == 1


def test_pose_movement_splits():
    tr = EpisodeTracker(gap_s=120, move_m=3.0)
    first = [_frame(i, 1000 + 5 * i) for i in range(4)]
    for i, t in enumerate(first):
        t.pose = Pose(x=0.2 * i, y=1.6, z=0.0, yaw=0.0)
    ids = [tr.assign(t) for t in first]
    near = _frame(5, 1025)
    near.pose = Pose(x=2.0, y=1.6, z=0.0, yaw=0.0)  # 1.7 m from the centroid (0.3, 1.6, 0)
    far = _frame(6, 1030)
    far.pose = Pose(x=6.0, y=1.6, z=0.0, yaw=0.0)  # same view, but another room
    assert tr.assign(near) == ids[0]
    assert tr.assign(far) != ids[0]
    assert tr.assign(_frame(7, 1035)) == far.episode_id  # frames without a pose never split on movement


def test_seed_restores_recent_window_and_start():
    frames = _drift_then_jump()
    seeded, jump = frames[:-1], frames[-1]
    for t in seeded:
        t.episode_id = "ep-turn"
    tr = EpisodeTracker(gap_s=60, split_sim=0.70, rejoin_sim=0.99, recent_n=6, max_s=0, move_m=0)
    tr.seed("dev1", seeded)
    (ep,) = tr.open_episodes("dev1")
    assert len(ep.recent) == 6 and ep.start_ts == 1000
    assert tr.assign(jump) != "ep-turn"  # judged against the last 6 seeded frames, as before the restart

    capped = EpisodeTracker(gap_s=60, max_s=60)
    capped.seed("dev1", seeded)  # the seeded episode already spans 60 s
    assert capped.assign(_frame(30, seeded[-1].ts + 5)) != "ep-turn"


# ------------------------------------------------------------------ store integration


async def test_store_assigns_episode_ids_and_persists(make_store, tmp_path):
    store = make_store(EpisodeTracker(gap_s=60))
    for i in range(3):
        await store.add(_frame(i, 1000 + 5 * i, tmp_path=tmp_path))
    await store.add_many([_frame(5, 2000, tmp_path=tmp_path), _frame(6, 2005, tmp_path=tmp_path)])
    toks = await store.range(None, None)
    eps = [t.episode_id for t in toks]
    assert all(eps)
    assert eps[0] == eps[1] == eps[2] and eps[3] == eps[4] and eps[0] != eps[3]
    # a fresh tracker (restart) is seeded from persisted frames and continues the open episode
    if isinstance(store, LanceDBStore):
        store = LanceDBStore(store.path, episodes=EpisodeTracker(gap_s=60))
    else:
        store.episodes = EpisodeTracker(gap_s=60)
    await store.add(_frame(7, 2010, tmp_path=tmp_path))
    assert (await store.get("f0007")).episode_id == eps[3]


async def test_store_without_tracker_leaves_episode_id_none(make_store):
    store = make_store(None)
    await store.add(_frame(0, 1000))
    assert (await store.get("f0000")).episode_id is None


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0, 1.0]


async def test_consolidator_writes_episode_summary(make_store, tmp_path):
    store = make_store(EpisodeTracker(gap_s=60))
    frames = [
        _frame(0, 1000, caption="laptop on a desk", ocr="MEGAMIND", entities=["laptop", "desk"], tmp_path=tmp_path),
        _frame(1, 1005, caption="laptop on a desk", entities=["laptop"], tmp_path=tmp_path),
        _frame(2, 1010, caption="coffee mug next to the laptop", entities=["mug", "laptop"], tmp_path=tmp_path),
        _frame(3, 1500, caption="kitchen sink", entities=["sink"], tmp_path=tmp_path),  # new, still open
    ]
    for f in frames:
        await store.add(f)
    job = Consolidator(store, MockSummarizer(), _Embedder(), gap_s=60, min_tokens=3)

    written = await job.run_once(now=1520)
    assert len(written) == 1
    ep = written[0]
    ep_id = (await store.get("f0000")).episode_id
    assert ep.kind == "episode" and ep.episode_id == ep_id and ep.id == summary_id(ep_id)
    assert ep.ts == 1000 and ep.device_id == "dev1"
    assert "3 moments" in ep.caption and "laptop" in ep.caption
    assert ep.ocr_text == "MEGAMIND"
    assert ep.entities[0] == "laptop"
    assert ep.text_vec == [0.0, 0.0, 0.0, 1.0]
    assert ep.image_vec is not None and abs(ep.image_vec[0] - 1.0) < 1e-6
    assert (tmp_path / f"{ep.id}.jpg").read_bytes() == b"jpeg"

    # idempotent; the open episode isn't summarised until it closes
    assert await job.run_once(now=1520) == []
    assert await job.run_once(now=1700) == []  # only 1 frame < min_tokens
    assert await store.count() == 5

    # summaries are reachable through the same primitives, and kinds separates them
    kw = await store.keyword_search("moments laptop", 5, kinds=("episode",))
    assert [t.id for t, _ in kw] == [ep.id]
    assert all(t.kind == "frame" for t, _ in await store.vector_search([0, 0, 0, 1], 10, kinds=("frame",)))
    assert [t.id for t in await store.range(None, None, kinds=("episode",))] == [ep.id]
    assert [t.id for t in await store.neighbors("f0001")] == ["f0000", "f0001", "f0002", "f0003"]
    # dedupe never sees the summary token
    hashes = await store.recent_hashes("dev1", 0)
    assert len(hashes) == 4

    # purge removes the summary + its copied thumbnail too
    assert await store.purge("dev1") == 5
    assert not (tmp_path / f"{ep.id}.jpg").exists()


async def test_consolidator_plans_without_vectors_and_returns_full_rows(make_store, tmp_path):
    store = make_store(EpisodeTracker(gap_s=60))
    for i, ts in enumerate((1000, 1005, 1010, 1500)):
        await store.add(_frame(i, ts, caption=f"c{i}", tmp_path=tmp_path))
    calls = []
    real = store.list_tokens

    async def spy(**kw):
        calls.append(kw.get("vectors", True))
        return await real(**kw)

    store.list_tokens = spy
    job = Consolidator(store, MockSummarizer(), _Embedder(), gap_s=60, min_tokens=3)

    (group,) = await job.pending(now=1520)
    assert [t.id for t in group] == ["f0000", "f0001", "f0002"]
    assert all(t.image_vec is not None for t in group)  # full rows are reloaded for summarising
    assert calls[-1] is True and not any(calls[:-1])  # planning never loads vectors (LanceDB: no token scan)

    calls.clear()
    await job.run_once(now=1520)
    assert await job.pending(now=1520) == []
    calls.clear()
    assert await job.pending(now=1520) == []
    assert not any(calls)  # steady state: nothing to summarise, no vectors loaded
