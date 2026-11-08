import io
import json
import random

import pytest
from PIL import Image

from megamind.replay import ReplayBuffer, find_ffmpeg, parse_range, render_mp4, replay_url
from megamind.replay.routes import RangeNotSatisfiable

HAVE_FFMPEG = find_ffmpeg() is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not available")


def _jpeg(seed: int, size=(96, 72)) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(size[0] * size[1])])
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def _post_frame(c, jpeg, ts, device="dev1"):
    return await c.post(
        "/v1/frames",
        files={"image": ("f.jpg", jpeg, "image/jpeg")},
        data={"meta": json.dumps({"device_id": device, "ts": ts})},
    )


# ---- buffer ----


def test_buffer_window_eviction_is_relative_to_newest_frame():
    b = ReplayBuffer(window_s=10.0)
    for t in range(100, 131):
        b.add("d", float(t), b"x" * 10)
    ts = [f.ts for f in b.frames("d", 0, 1e9)]
    assert ts == [float(t) for t in range(120, 131)]
    assert b.stats()["d"]["bytes"] == 110


def test_buffer_count_and_byte_bounds():
    b = ReplayBuffer(window_s=0, max_frames=5, max_bytes=10_000)
    for t in range(20):
        b.add("d", float(t), b"x" * 100)
    assert [f.ts for f in b.frames("d", 0, 100)] == [15.0, 16.0, 17.0, 18.0, 19.0]

    b = ReplayBuffer(window_s=0, max_frames=1000, max_bytes=350)
    for t in range(20):
        b.add("d", float(t), b"x" * 100)
    assert [f.ts for f in b.frames("d", 0, 100)] == [17.0, 18.0, 19.0]
    assert b.stats()["d"]["bytes"] == 300


def test_buffer_out_of_order_devices_and_resolve():
    b = ReplayBuffer()
    for t in (1.0, 3.0, 2.0, 5.0, 4.0):
        b.add("a", t, b"a")
    b.add("b", 2.5, b"b")
    assert [f.ts for f in b.frames("a", 2.0, 4.0)] == [2.0, 3.0, 4.0]
    assert b.frames("nope", 0, 10) == []
    assert b.resolve_device(0, 10) == "a"  # most frames in window
    assert b.resolve_device(2.4, 2.6) == "b"
    assert b.resolve_device(2.4, 2.6, "a") is None
    assert b.resolve_device(100, 200) is None
    assert b.newest_ts() == 5.0 and b.newest_ts("b") == 2.5 and b.newest_ts("nope") is None


def test_buffer_purge():
    b = ReplayBuffer()
    for t in range(10):
        b.add("a", float(t), b"a")
        b.add("b", float(t), b"b")
    assert b.purge("a", since_ts=5.0) == 5
    assert [f.ts for f in b.frames("a", 0, 100)] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert b.purge() == 15
    assert b.stats() == {}


def test_buffer_tap_never_raises():
    class Meta:
        device_id = "d"
        ts = "not a float"

    ReplayBuffer().tap(Meta(), b"x")  # logged + dropped


def test_buffer_spool_on_disk(tmp_path):
    b = ReplayBuffer(window_s=0, max_frames=3, spool_dir=tmp_path)
    for t in range(5):
        b.add("quest/3 abc", float(t), bytes([t]) * 4)
    files = sorted((tmp_path).rglob("*.jpg"))
    assert len(files) == 3  # evicted frames are unlinked
    assert b.frames("quest/3 abc", 4, 4)[0].read() == bytes([4]) * 4
    # rehydrate after "restart"
    b2 = ReplayBuffer(window_s=0, max_frames=2, spool_dir=tmp_path)
    assert [f.ts for f in b2.frames("quest/3 abc", 0, 10)] == [3.0, 4.0]
    assert len(list(tmp_path.rglob("*.jpg"))) == 2


def test_buffer_spool_dir_cannot_escape(tmp_path):
    spool = tmp_path / "spool"
    b = ReplayBuffer(spool_dir=spool)
    for dev in ("..", ".", "../../etc", "a/../.."):
        b.add(dev, 1.0, b"x")
    assert {p.parent.parent for p in spool.rglob("*.jpg")} == {spool}
    assert not list(tmp_path.glob("*.jpg"))  # nothing written next to the spool dir
    b2 = ReplayBuffer(spool_dir=spool)
    assert set(b2.stats()) == {"..", ".", "../../etc", "a/../.."}


def test_buffer_spool_repeated_ts_and_restart_keep_distinct_files(tmp_path):
    b = ReplayBuffer(window_s=0, spool_dir=tmp_path)
    b.add("d", 1.0, b"first")
    b2 = ReplayBuffer(window_s=0, spool_dir=tmp_path)  # restart, then replay the same footage again
    b2.add("d", 1.0, b"second")
    assert sorted(f.read() for f in b2.frames("d", 0, 2)) == [b"first", b"second"]
    b2.purge("d", since_ts=1.0)
    assert not list(tmp_path.rglob("*.jpg"))


def test_buffer_device_cap_drops_least_recently_fed(tmp_path):
    for spool in (None, tmp_path):
        b = ReplayBuffer(max_devices=2, spool_dir=spool)
        b.add("a", 1.0, b"a")
        b.add("b", 1.0, b"b")
        b.add("a", 2.0, b"a")  # a is now the most recently fed
        b.add("c", 1.0, b"c")
        assert set(b.stats()) == {"a", "c"}
    assert len(list(tmp_path.rglob("*.jpg"))) == 3
    assert set(ReplayBuffer(max_devices=1, spool_dir=tmp_path).stats()) == {"a"}  # newest frame wins


def test_buffer_rejects_non_finite_ts():
    b = ReplayBuffer()
    with pytest.raises(ValueError):
        b.add("d", float("nan"), b"x")
    assert b.stats() == {}


# ---- range parsing ----


@pytest.mark.parametrize(
    "header,expected",
    [
        (None, None),
        ("", None),
        ("bytes=0-99", (0, 99)),
        ("bytes=0-", (0, 999)),
        ("bytes=500-2000", (500, 999)),
        ("bytes=-100", (900, 999)),
        ("bytes=-5000", (0, 999)),
        ("bytes=10-5", None),  # last < first: ignore header
        ("bytes=0-1,5-6", None),  # multi-range: whole body
        ("items=0-1", None),
        ("bytes=-", None),
    ],
)
def test_parse_range(header, expected):
    assert parse_range(header, 1000) == expected


@pytest.mark.parametrize("header", ["bytes=1000-", "bytes=5000-6000", "bytes=-0"])
def test_parse_range_unsatisfiable(header):
    with pytest.raises(RangeNotSatisfiable):
        parse_range(header, 1000)


# ---- API ----


async def test_every_frame_is_buffered_even_when_filtered(client):
    c, app = client
    j = _jpeg(1)
    assert (await _post_frame(c, j, 1000.0)).json()["accepted"]
    assert (await _post_frame(c, j, 1001.0)).json()["reason"] == "duplicate"
    assert (await _post_frame(c, b"not a jpeg", 1002.0)).status_code == 400
    frames = app.state.replay.frames("dev1", 0, 1e9)
    assert [f.ts for f in frames] == [1000.0, 1001.0]
    assert frames[1].read() == j


async def test_replay_404s_and_400s(client):
    c, app = client
    r = await c.get("/v1/replay.mp4")
    assert r.status_code == 404  # nothing buffered at all
    await _post_frame(c, _jpeg(1), 1000.0)
    r = await c.get("/v1/replay.mp4", params={"device_id": "other", "from": "990", "to": "1010"})
    assert r.status_code == 404
    r = await c.get("/v1/replay.mp4", params={"device_id": "dev1", "from": "2000", "to": "2010"})
    assert r.status_code == 404
    r = await c.get("/v1/replay.mp4", params={"from": "yesterday"})
    assert r.status_code == 400 and "from" in r.json()["detail"]
    r = await c.get("/v1/replay.mp4", params={"from": "1010", "to": "1000"})
    assert r.status_code == 400
    r = await c.get("/v1/replay.mp4", params={"from": "0", "to": "100000"})
    assert r.status_code == 400 and "REPLAY_MAX_SPAN_S" in r.json()["detail"]
    for bad in ({"from": "nan", "to": "nan"}, {"from": "inf"}, {"to": "-inf"}):
        assert (await c.get("/v1/replay.mp4", params=bad)).status_code == 400


async def test_replay_ffmpeg_missing_is_503(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr("megamind.replay.routes.find_ffmpeg", lambda *_: None)
    await _post_frame(c, _jpeg(1), 1000.0)
    r = await c.get("/v1/replay.mp4", params={"device_id": "dev1", "from": "990", "to": "1010"})
    assert r.status_code == 503 and "ffmpeg" in r.json()["detail"]


async def test_chat_citations_get_replay_url(client):
    c, app = client
    await _post_frame(c, _jpeg(3), 2000.0)
    await app.state.worker.queue.join()
    r = await c.post("/v1/chat", json={"session_id": "s", "device_id": "dev1", "text": "mock caption"})
    cited = r.json()["cited"]
    assert cited and cited[0]["replay_url"] == replay_url("dev1", 1995.0, 2005.0)
    assert cited[0]["replay_url"].startswith("/v1/replay.mp4?device_id=dev1&from=1995.000&to=2005.000")

    # purge mirrors onto the buffer; memories gone too, so post a fresh token without replay frames
    r = await c.delete("/v1/memories", params={"device_id": "dev1"})
    assert r.status_code == 200
    assert app.state.replay.stats() == {}
    await _post_frame(c, _jpeg(4), 3000.0)
    await app.state.worker.queue.join()
    app.state.replay.purge()
    r = await c.post("/v1/chat", json={"session_id": "s2", "text": "mock caption"})
    cited = r.json()["cited"]
    # ring buffer empty: the kept keyframe still backs a (keyframe-rate) replay
    assert cited and cited[0]["replay_url"] == replay_url("dev1", 2995.0, 3005.0)
    # once the frame's pixels aged out (tier "text") there is nothing left to replay
    await c.post("/v1/admin/age", params={"hot_older_than_s": 0, "cold_older_than_s": 0})
    r = await c.post("/v1/chat", json={"session_id": "s3", "text": "mock caption"})
    cited = r.json()["cited"]
    assert cited and cited[0]["replay_url"] is None and cited[0]["thumb_url"] == ""


async def test_replay_disabled(client):
    c, app = client
    app.state.replay = None
    r = await c.get("/v1/replay.mp4", params={"from": "0", "to": "10"})
    assert r.status_code == 404 and "disabled" in r.json()["detail"]
    r = await c.post("/v1/chat", json={"session_id": "s", "text": "anything"})
    assert r.status_code == 200


# ---- rendering (needs ffmpeg) ----


@needs_ffmpeg
def test_render_mp4_holds_frames(tmp_path):
    frames = [(100.0, _jpeg(1, (97, 73))), (101.5, _jpeg(2, (97, 73))), (103.0, _jpeg(3, (97, 73)))]
    out = render_mp4(frames, tmp_path / "x.mp4", ffmpeg=find_ffmpeg(), fps=10)
    data = out.read_bytes()
    assert data[4:8] == b"ftyp" and len(data) > 500
    assert not list(tmp_path.glob(".render-*"))  # temp dir cleaned up


@needs_ffmpeg
async def test_render_skips_frames_evicted_from_spool(tmp_path):
    from megamind.replay import ReplayCache

    b = ReplayBuffer(window_s=0, spool_dir=tmp_path / "spool")
    for i in range(3):
        b.add("d", float(i), _jpeg(i))
    frames = b.frames("d", 0, 10)
    frames[0].path.unlink()  # evicted between planning and rendering
    cache = ReplayCache(tmp_path / "cache")
    p = await cache.get_or_render("k", frames, ffmpeg=find_ffmpeg(), fps=5, max_width=320)
    assert p.read_bytes()[4:8] == b"ftyp"
    for f in frames[1:]:
        f.path.unlink()
    with pytest.raises(FileNotFoundError):
        await cache.get_or_render("k2", frames, ffmpeg=find_ffmpeg(), fps=5, max_width=320)
    assert cache._locks == {}


@needs_ffmpeg
async def test_replay_endpoint_renders_caches_and_ranges(client):
    c, app = client
    for i in range(6):
        await _post_frame(c, _jpeg(i), 1000.0 + i)
    params = {"device_id": "dev1", "from": "999", "to": "1010"}
    r = await c.get("/v1/replay.mp4", params=params)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "video/mp4"
    assert r.headers["accept-ranges"] == "bytes"
    assert r.headers["x-replay-frames"] == "6" and r.headers["x-replay-start"] == "1000.000"
    body = r.content
    assert body[4:8] == b"ftyp" and int(r.headers["content-length"]) == len(body)
    cache = list(app.state.replay_cache.dir.glob("*.mp4"))
    assert len(cache) == 1

    r = await c.get("/v1/replay.mp4", params=params, headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and r.content == body[:100]
    assert r.headers["content-range"] == f"bytes 0-99/{len(body)}"
    r = await c.get("/v1/replay.mp4", params=params, headers={"Range": "bytes=-10"})
    assert r.status_code == 206 and r.content == body[-10:]
    r = await c.get("/v1/replay.mp4", params=params, headers={"Range": f"bytes={len(body)}-"})
    assert r.status_code == 416 and r.headers["content-range"] == f"bytes */{len(body)}"
    r = await c.head("/v1/replay.mp4", params=params)
    assert r.status_code == 200 and r.content == b"" and int(r.headers["content-length"]) == len(body)
    assert list(app.state.replay_cache.dir.glob("*.mp4")) == cache  # served from cache

    # a new frame inside the window changes the cache key
    await _post_frame(c, _jpeg(9), 1008.0)
    r = await c.get("/v1/replay.mp4", params=params)
    assert r.status_code == 200 and r.headers["x-replay-frames"] == "7"
    assert len(list(app.state.replay_cache.dir.glob("*.mp4"))) == 2

    # purge clears frames and the clip cache
    await c.delete("/v1/memories")
    assert list(app.state.replay_cache.dir.glob("*.mp4")) == []
    assert (await c.get("/v1/replay.mp4", params=params)).status_code == 404


@needs_ffmpeg
async def test_replay_falls_back_to_kept_frames_after_the_buffer_expires(client):
    c, app = client
    for i in range(3):
        assert (await _post_frame(c, _jpeg(20 + i), 1000.0 + 5 * i)).json()["accepted"]
    await app.state.worker.queue.join()
    params = {"device_id": "dev1", "from": "995", "to": "1015"}
    r = await c.get("/v1/replay.mp4", params=params)
    assert r.status_code == 200 and r.headers["x-replay-source"] == "buffer"

    app.state.replay.purge()  # older than REPLAY_WINDOW_S: the ring buffer no longer has these frames
    for p in (params, {"from": "995", "to": "1015"}):  # explicit device, then resolved from kept frames
        r = await c.get("/v1/replay.mp4", params=p)
        assert r.status_code == 200, r.text
        assert r.headers["x-replay-source"] == "keyframes" and r.headers["x-replay-device"] == "dev1"
        assert r.headers["x-replay-frames"] == "3" and r.headers["x-replay-start"] == "1000.000"
        assert r.content[4:8] == b"ftyp"

    await c.post("/v1/admin/age", params={"cold_older_than_s": 0})  # pixels aged out: nothing to replay
    assert (await c.get("/v1/replay.mp4", params=params)).status_code == 404
