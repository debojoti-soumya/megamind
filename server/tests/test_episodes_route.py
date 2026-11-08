"""GET /v1/episodes: episodes from frame episode_ids plus their consolidated summaries, newest first."""

from datetime import datetime

from megamind.models import MemoryToken


async def _add(app, tid, ts, *, episode=None, device="dev1", kind="frame", caption="frame", entities=()):
    st = app.state
    await st.store.add(
        MemoryToken(
            id=tid,
            device_id=device,
            ts=ts,
            caption=caption,
            entities=list(entities),
            text_vec=await st.text_embedder.embed(caption),
            thumb_path=f"/nonexistent/{tid}.jpg",
            frame_hash=tid,
            episode_id=episode,
            kind=kind,
        )
    )


async def _seed(app):
    for i in range(3):
        await _add(app, f"a{i}", 1000.0 + i, episode="epA")
    await _add(app, "b0", 2000.0, episode="epB")
    await _add(app, "b1", 2001.0, episode="epB")
    await _add(app, "sumB", 2000.0, episode="epB", kind="episode", caption="made coffee", entities=["mug"])
    await _add(app, "c0", 3000.0, episode="epC", device="dev2")


async def test_episodes_newest_first_with_summaries(client):
    c, app = client
    await _seed(app)
    r = await c.get("/v1/episodes")
    assert r.status_code == 200, r.text
    eps = r.json()["episodes"]
    assert [e["episode_id"] for e in eps] == ["epC", "epB", "epA"]
    b, a = eps[1], eps[2]
    assert (b["start_ts"], b["end_ts"], b["n_frames"]) == (2000.0, 2001.0, 2)
    assert (b["summary"], b["entities"], b["summary_id"]) == ("made coffee", ["mug"], "sumB")
    assert b["thumb_url"] == "/v1/thumbs/sumB.jpg"
    assert (a["n_frames"], a["summary"], a["summary_id"], a["end_ts"]) == (3, "", None, 1002.0)


async def test_episodes_filters(client):
    c, app = client
    await _seed(app)
    ids = lambda r: [e["episode_id"] for e in r.json()["episodes"]]  # noqa: E731
    assert ids(await c.get("/v1/episodes", params={"device_id": "dev1"})) == ["epB", "epA"]
    assert ids(await c.get("/v1/episodes", params={"k": 1})) == ["epC"]
    window = {"from": datetime.fromtimestamp(1900).isoformat(), "to": datetime.fromtimestamp(2500).isoformat()}
    assert ids(await c.get("/v1/episodes", params=window)) == ["epB"]
    assert (await c.get("/v1/episodes", params={"from": "nope"})).status_code == 400
    assert (await c.get("/v1/episodes", params={"k": 0})).status_code == 422


async def test_episodes_empty(client):
    c, _ = client
    r = await c.get("/v1/episodes")
    assert r.status_code == 200 and r.json() == {"episodes": []}
