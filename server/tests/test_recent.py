"""GET /v1/memories/recent: newest-first listing without a query (Memory Theater cards)."""

import time

from megamind.models import MemoryToken


async def _add(app, i, ts, *, device="dev1", kind="frame", caption=None):
    st = app.state
    caption = f"frame {i}" if caption is None else caption
    await st.store.add(
        MemoryToken(
            id=f"tok{i:04d}",
            device_id=device,
            ts=ts,
            caption=caption,
            text_vec=await st.text_embedder.embed(caption),
            thumb_path="" if kind == "transcript" else f"/nonexistent/tok{i:04d}.jpg",
            frame_hash="" if kind == "transcript" else f"{i:016x}",
            kind=kind,
        )
    )


async def test_recent_newest_first_with_k(client):
    c, app = client
    now = time.time()
    for i in range(6):
        await _add(app, i, now - 600 + i * 60)
    r = await c.get("/v1/memories/recent", params={"k": 4})
    assert r.status_code == 200
    res = r.json()["results"]
    assert [h["id"] for h in res] == ["tok0005", "tok0004", "tok0003", "tok0002"]
    assert res[0]["thumb_url"] == "/v1/thumbs/tok0005.jpg"
    assert res[0]["kind"] == "frame"


async def test_recent_widens_window_to_old_memories(client):
    c, app = client
    now = time.time()
    await _add(app, 1, now - 10 * 86400)  # older than every bounded window but the last
    await _add(app, 2, now - 2 * 86400)
    await _add(app, 3, now - 60)
    res = (await c.get("/v1/memories/recent", params={"k": 3})).json()["results"]
    assert [h["id"] for h in res] == ["tok0003", "tok0002", "tok0001"]


async def test_recent_filters_device_and_kind(client):
    c, app = client
    now = time.time()
    await _add(app, 1, now - 30, device="a")
    await _add(app, 2, now - 20, device="b")
    await _add(app, 3, now - 10, device="a", kind="transcript", caption='Speech heard nearby: "hi"')
    res = (await c.get("/v1/memories/recent", params={"device_id": "a"})).json()["results"]
    assert [h["id"] for h in res] == ["tok0001"]  # default kind=frame
    res = await c.get("/v1/memories/recent", params=[("device_id", "a"), ("kind", "frame"), ("kind", "transcript")])
    assert [h["id"] for h in res.json()["results"]] == ["tok0003", "tok0001"]


async def test_recent_empty_and_bad_k(client):
    c, _ = client
    r = await c.get("/v1/memories/recent")
    assert r.status_code == 200 and r.json() == {"results": []}
    assert (await c.get("/v1/memories/recent", params={"k": 0})).status_code == 422
    assert (await c.get("/v1/memories/recent", params={"k": 201})).status_code == 422


async def test_search_route_unchanged(client):
    c, _ = client
    assert (await c.get("/v1/memories")).status_code == 422  # q is still required on the search route
