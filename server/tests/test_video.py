"""Real headset video (megamind.video): POST /v1/video validation + storage, purge, retention, and replay/montage
preferring stored video over frames (synthetic clips made with ffmpeg; those tests skip without ffmpeg)."""

import io
import json
import random
import subprocess

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from megamind.config import Settings
from megamind.main import create_app
from megamind.models import Citation, MemoryToken
from megamind.replay import annotate_citations, find_ffmpeg
from megamind.video import VideoStore, render_video
from megamind.video.replay import prefer_video
from megamind.video.store import Probe, Segment, VideoClip, find_ffprobe

FFMPEG = find_ffmpeg()
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")
T0 = 1_000.0


def _ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-y", "-v", "error", *args], check=True, capture_output=True, timeout=60)


def _make_clip(path, seconds: float = 5.0, *, codec: str = "libx264", size: str = "160x120", rate: int = 30):
    _ffmpeg("-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}", "-t", f"{seconds}", "-c:v", codec,
            "-pix_fmt", "yuv420p", "-g", str(rate), str(path))  # fmt: skip
    return path


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    if FFMPEG is None:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("clips")
    return {
        "5s": _make_clip(d / "five.mp4").read_bytes(),
        "1s": _make_clip(d / "one.mp4", 1.0).read_bytes(),
        "3s": _make_clip(d / "three.mp4", 3.0).read_bytes(),
        "mpeg4": _make_clip(d / "mpeg4.mp4", 1.0, codec="mpeg4").read_bytes(),
        "mkv": _make_clip(d / "h264.mkv", 1.0).read_bytes(),
    }


def _settings(tmp_path, **kw) -> Settings:
    return Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend="memory",
        embed_provider="mock",
        enrich_provider="mock",
        filter_min_gap_s=0.0,
        consolidate_enabled=False,
        anthropic_api_key="",
        _env_file=None,
        **kw,
    )


@pytest.fixture
async def app_client(tmp_path):
    app = create_app(_settings(tmp_path))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app


def _jpeg(seed: int, size=(96, 72)) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(size[0] * size[1])])
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def _post_video(c, data: bytes, ts_start: float, device: str = "dev1", meta: dict | None = None):
    m = {"device_id": device, "ts_start": ts_start} if meta is None else meta
    return await c.post("/v1/video", files={"video": ("clip.mp4", data, "video/mp4")}, data={"meta": json.dumps(m)})


async def _post_frame(c, jpeg: bytes, ts: float, device: str = "dev1"):
    return await c.post(
        "/v1/frames",
        files={"image": ("f.jpg", jpeg, "image/jpeg")},
        data={"meta": json.dumps({"device_id": device, "ts": ts})},
    )


def _duration(data: bytes, tmp_path) -> float:
    p = tmp_path / "probe.mp4"
    p.write_bytes(data)
    ffprobe = find_ffprobe(FFMPEG)
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return float(out.strip())


def _fake_clip(store: VideoStore, device: str, ts: float, dur: float = 5.0, payload: bytes = b"x" * 100) -> VideoClip:
    """Index a clip without ffmpeg (bytes are not a real mp4: storage/purge/retention tests only)."""
    staged = store.staging_path(device, f"c{ts:.0f}")
    staged.write_bytes(payload)
    return store.commit(staged, f"c{ts:.0f}", device, ts, Probe("mov,mp4", "h264", 160, 120, dur, 30.0))


# ---- POST /v1/video ----


async def test_post_video_stores_clip_and_reports_stats(app_client, clips):
    c, app = app_client
    r = await _post_video(c, clips["5s"], T0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] and body["reason"] is None and body["video_id"]
    assert body["ts_start"] == T0 and abs(body["duration_s"] - 5.0) < 0.1
    assert (body["width"], body["height"]) == (160, 120)
    [clip] = app.state.video.clips("dev1")
    assert clip.id == body["video_id"] and clip.path.read_bytes() == clips["5s"]
    assert clip.sidecar().is_file() and not list(clip.path.parent.glob("*.part"))
    stats = (await c.get("/v1/stats")).json()["video"]
    assert stats["clips"] == 1 and stats["bytes"] == len(clips["5s"]) and stats["devices"] == 1


async def test_post_video_is_idempotent_for_retried_uploads(app_client, clips):
    c, app = app_client
    first = (await _post_video(c, clips["5s"], T0)).json()
    again = (await _post_video(c, clips["5s"], T0 + 0.01)).json()
    assert again["accepted"] and again["reason"] == "duplicate" and again["video_id"] == first["video_id"]
    assert len(app.state.video.clips("dev1")) == 1
    assert len(list(app.state.video.root.rglob("*.mp4"))) == 1


async def test_post_video_validation(app_client, clips):
    c, app = app_client
    for meta in ({"device_id": "dev1"}, {"device_id": "dev1", "ts_start": "nan"}, {"ts_start": 1.0}):
        assert (await _post_video(c, clips["1s"], 0, meta=meta)).status_code == 400
    r = await c.post("/v1/video", files={"video": ("c.mp4", b"x", "video/mp4")}, data={"meta": "{not json"})
    assert r.status_code == 400
    assert (await _post_video(c, b"", T0)).status_code == 400
    r = await _post_video(c, b"definitely not an mp4" * 50, T0)
    assert r.status_code == 415 and "decodable" in r.json()["detail"]
    r = await _post_video(c, clips["mpeg4"], T0)
    assert r.status_code == 415 and "H.264" in r.json()["detail"]
    r = await _post_video(c, clips["mkv"], T0)
    assert r.status_code == 415 and "mp4" in r.json()["detail"]

    app.state.video_settings.max_clip_s = 2.0
    r = await _post_video(c, clips["3s"], T0)
    assert r.status_code == 413 and "longer" in r.json()["detail"]
    app.state.video_settings.max_clip_bytes = 1000
    r = await _post_video(c, clips["3s"], T0)
    assert r.status_code == 413 and "larger" in r.json()["detail"]
    assert app.state.video.clips("dev1") == []
    assert not list(app.state.video.root.rglob("*.part")) and not list(app.state.video.root.rglob("*.mp4"))


async def test_post_video_rejects_unsafe_input_without_500(app_client, clips):
    c, app = app_client
    # a playlist must never be followed: the demuxer is pinned to mov, so this is simply not a video
    playlist = b"#EXTM3U\n#EXT-X-TARGETDURATION:5\n#EXTINF:5,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n"
    r = await _post_video(c, playlist, T0)
    assert r.status_code == 415 and str(app.state.video.root) not in r.json()["detail"]  # no server paths leak
    r = await _post_video(c, clips["1s"], T0, device="d" * 300)
    assert r.status_code == 400 and "device_id" in r.json()["detail"]
    r = await _post_video(c, clips["1s"], 1e300)  # finite but far in the future: bad meta, not a 500
    assert r.status_code == 400 and "in the future" in r.json()["detail"]
    r = await _post_video(c, clips["1s"], T0, device="../../x")  # odd id: stored inside the video dir, not a 500
    assert r.status_code == 200, r.text
    [clip] = app.state.video.clips("../../x")
    assert clip.path.parent.parent == app.state.video.root and clip.path.name == f"{clip.id}.mp4"


async def test_post_video_without_ffmpeg_is_503(app_client, monkeypatch):
    c, app = app_client

    def missing(*_a, **_k):
        raise FileNotFoundError("no ffprobe")

    monkeypatch.setattr("megamind.video.routes.probe", missing)
    r = await _post_video(c, b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64, T0)
    assert r.status_code == 503 and "ffmpeg" in r.json()["detail"]
    assert not list(app.state.video.root.rglob("*.part"))


async def test_post_video_disabled_and_privacy(app_client, clips):
    c, app = app_client
    app.state.settings.privacy_blur_faces = True
    r = await _post_video(c, clips["1s"], T0)
    assert r.status_code == 200
    assert r.json() == {"video_id": "", "accepted": False, "reason": "privacy", "ts_start": T0,
                        "duration_s": 0.0, "width": 0, "height": 0}  # fmt: skip
    assert app.state.video.clips("dev1") == []
    app.state.settings.privacy_blur_faces = False
    app.state.video = None
    r = await _post_video(c, clips["1s"], T0)
    assert r.status_code == 404 and "disabled" in r.json()["detail"]


# ---- store: purge, retention, cap, restart ----


async def test_delete_memories_purges_video(app_client):
    c, app = app_client
    store = app.state.video
    for ts in (T0, T0 + 5, T0 + 10):
        _fake_clip(store, "dev1", ts)
    _fake_clip(store, "dev2", T0)
    r = await c.delete("/v1/memories", params={"device_id": "dev1", "since": str(T0 + 7)})
    assert r.status_code == 200
    # the clip spanning `since` (T0+5..T0+10) goes too: it holds footage after since
    assert [cl.ts_start for cl in store.clips("dev1")] == [T0]
    assert len(store.clips("dev2")) == 1
    await c.delete("/v1/memories")
    assert store.stats()["clips"] == 0 and not list(store.root.rglob("*.mp4")) and not list(store.root.rglob("*.json"))


def test_upload_racing_a_purge_is_not_committed(tmp_path):
    store = VideoStore(tmp_path / "v")
    seq = store.seq
    store.purge("dev1", since_ts=T0)
    assert store.purged_since(seq, "dev1", T0 + 5)
    assert not store.purged_since(seq, "dev1", T0 - 1)  # ended before the purged window
    assert not store.purged_since(seq, "dev2", T0 + 5)
    assert not store.purged_since(store.seq, "dev1", T0 + 5)  # upload started after the purge


async def test_upload_racing_a_purge_is_409(app_client, clips, monkeypatch):
    c, app = app_client
    from megamind.video import routes

    real = routes.probe

    def probe_then_purge(*a, **k):
        p = real(*a, **k)
        app.state.video.purge()  # DELETE /v1/memories lands while the clip is being validated
        return p

    monkeypatch.setattr(routes, "probe", probe_then_purge)
    r = await _post_video(c, clips["1s"], T0)
    assert r.status_code == 409
    assert app.state.video.clips("dev1") == [] and not list(app.state.video.root.rglob("*.mp4"))


async def test_aging_drops_video_with_the_hot_tier(app_client):
    c, app = app_client
    store = app.state.video
    import time

    now = time.time()
    _fake_clip(store, "dev1", now - 10)
    old = _fake_clip(store, "dev1", now - 73 * 3600)  # older than HOT_RETENTION_H=72
    r = (await c.post("/v1/admin/age")).json()
    assert r["video_deleted"] == 1 and not old.path.exists()
    assert len(store.clips("dev1")) == 1
    r = (await c.post("/v1/admin/age", params={"hot_older_than_s": 0})).json()
    assert r["video_deleted"] == 1 and store.clips("dev1") == []


async def test_aging_honours_the_given_now(app_client):
    from megamind.store.media import run_aging

    c, app = app_client
    clip = _fake_clip(app.state.video, "dev1", T0)  # footage ends at T0 + 5
    stats = await run_aging(app.state, now=T0 + 6, hot_s=60.0)
    assert stats["video_deleted"] == 0 and clip.path.exists()
    stats = await run_aging(app.state, now=T0 + 66, hot_s=60.0)
    assert stats["video_deleted"] == 1 and not clip.path.exists()


def test_store_byte_cap_drops_oldest_and_restart_rehydrates(tmp_path):
    root = tmp_path / "v"
    store = VideoStore(root, max_bytes=250)
    a = _fake_clip(store, "dev1", T0)
    _fake_clip(store, "dev/../x", T0 + 5)
    assert store.stats()["clips"] == 2
    _fake_clip(store, "dev1", T0 + 10)  # 300 bytes > 250: the oldest goes
    assert not a.path.exists() and store.stats() == {"clips": 2, "seconds": 10.0, "bytes": 200, "devices": 2}
    assert all(p.is_relative_to(root) for p in root.rglob("*"))

    (root / "d_dev1" / "orphan.mp4.part").write_bytes(b"half")
    (root / "d_dev1" / "nosidecar.mp4").write_bytes(b"half")
    again = VideoStore(root, max_bytes=250)
    assert again.stats() == store.stats()
    assert [cl.ts_start for cl in again.clips("dev/../x")] == [T0 + 5]
    assert not (root / "d_dev1" / "orphan.mp4.part").exists() and not (root / "d_dev1" / "nosidecar.mp4").exists()


def test_segments_and_coverage(tmp_path):
    store = VideoStore(tmp_path / "v")
    _fake_clip(store, "d", T0, 4.9)
    _fake_clip(store, "d", T0 + 5, 5.0)
    segs = store.segments("d", T0 + 2, T0 + 8)
    assert [(s.start, s.end) for s in segs] == [(T0 + 2, T0 + 4.9), (T0 + 5, T0 + 8)]
    assert segs[0].offset == 2 and segs[1].offset == 0
    assert store.covered("d", T0 + 2, T0 + 8) == pytest.approx(5.9)
    assert store.resolve_device(T0, T0 + 1) == "d" and store.resolve_device(T0 + 20, T0 + 30) is None
    assert store.newest_ts("d") == T0 + 10 and store.newest_ts("nope") is None
    # the coverage rule against frame footage in the same window
    assert prefer_video(segs, [], T0, T0 + 10, 0.5)
    assert prefer_video(segs, [T0 + 2, T0 + 7], T0, T0 + 10, 0.5)
    tiny = [Segment(segs[0].clip, T0 + 2, T0 + 2.4)]
    assert not prefer_video(tiny, [], T0, T0 + 10, 0.5)  # < 0.5 s of video never wins
    assert not prefer_video([Segment(segs[0].clip, T0, T0 + 1)], [T0, T0 + 9], T0, T0 + 10, 0.5)


async def test_purge_cli_deletes_video(tmp_path):
    from megamind.store.purge import purge
    from megamind.video import VideoSettings, build_video_store

    s = _settings(tmp_path)
    store = build_video_store(s, VideoSettings(_env_file=None))
    clip = _fake_clip(store, "dev1", T0)
    await purge(s, device_id="dev1")
    assert not clip.path.exists() and not clip.sidecar().exists()


# ---- replay prefers stored video ----


@needs_ffmpeg
def test_render_video_cuts_and_holds_gaps(tmp_path, clips):
    store = VideoStore(tmp_path / "v")
    for i, ts in enumerate((T0, T0 + 5.1)):  # 0.1 s gap at the boundary, like the Quest's dropped frames
        staged = store.staging_path("d", f"c{i}")
        staged.write_bytes(clips["5s"])
        store.commit(staged, f"c{i}", "d", ts, Probe("mov,mp4", "h264", 160, 120, 5.0, 30.0))
    segs = store.segments("d", T0 + 2, T0 + 9)
    out = render_video(segs, tmp_path / "o.mp4", ffmpeg=FFMPEG, fps=30, max_width=960)
    assert _duration(out.read_bytes(), tmp_path) == pytest.approx(7.0, abs=0.1)
    assert not list(tmp_path.glob(".video-*"))


@needs_ffmpeg
async def test_replay_prefers_stored_video_and_falls_back_to_frames(app_client, clips, tmp_path):
    c, app = app_client
    for i in range(10):
        await _post_frame(c, _jpeg(i), T0 + i)
    params = {"device_id": "dev1", "from": str(T0), "to": str(T0 + 10)}
    r = await c.get("/v1/replay.mp4", params=params)
    assert r.status_code == 200 and r.headers["x-replay-source"] == "buffer"  # no video yet: unchanged

    for ts in (T0, T0 + 5):
        assert (await _post_video(c, clips["5s"], ts)).status_code == 200
    r = await c.get("/v1/replay.mp4", params=params)
    assert r.status_code == 200, r.text
    h = r.headers
    assert h["x-replay-source"] == "video" and h["x-replay-device"] == "dev1" and h["x-replay-clips"] == "2"
    assert h["x-replay-start"] == f"{T0:.3f}" and h["x-replay-frames"] == "300"
    assert h["accept-ranges"] == "bytes" and r.content[4:8] == b"ftyp"
    assert _duration(r.content, tmp_path) == pytest.approx(10.0, abs=0.15)
    r2 = await c.get("/v1/replay.mp4", params=params, headers={"Range": "bytes=0-99"})
    assert r2.status_code == 206 and r2.content == r.content[:100]
    r = await c.get("/v1/replay.mp4", params={"from": str(T0 + 1), "to": str(T0 + 4)})  # device resolved
    assert r.headers["x-replay-source"] == "video" and r.headers["x-replay-start"] == f"{T0 + 1:.3f}"

    # face blur on: real video is never served, frames as before
    app.state.settings.privacy_blur_faces = True
    r = await c.get("/v1/replay.mp4", params=params)
    assert r.headers["x-replay-source"] == "buffer"
    app.state.settings.privacy_blur_faces = False

    # a clip deleted under the cache (aged out between plan and render): falls back instead of failing
    app.state.replay_cache.clear()
    for cl in app.state.video.clips("dev1"):
        cl.path.unlink()
    r = await c.get("/v1/replay.mp4", params=params)
    assert r.status_code == 200 and r.headers["x-replay-source"] == "buffer"

    app.state.video.purge()
    r = await c.get("/v1/replay.mp4", params=params)
    assert r.status_code == 200 and r.headers["x-replay-source"] == "buffer"
    assert r.headers["x-replay-frames"] == "10" and r.headers["x-replay-start"] == f"{T0:.3f}"


@needs_ffmpeg
async def test_replay_uses_frames_when_video_covers_too_little(app_client, clips):
    c, app = app_client
    for i in range(10):
        await _post_frame(c, _jpeg(i), T0 + i)
    await _post_video(c, clips["1s"], T0 + 3)  # 1 s of video in 10 s of frames
    r = await c.get("/v1/replay.mp4", params={"device_id": "dev1", "from": str(T0), "to": str(T0 + 10)})
    assert r.status_code == 200 and r.headers["x-replay-source"] == "buffer"


@needs_ffmpeg
async def test_replay_from_video_alone(app_client, clips, tmp_path):
    c, app = app_client
    assert (await c.get("/v1/replay.mp4")).status_code == 404  # nothing at all
    await _post_video(c, clips["5s"], T0)
    r = await c.get("/v1/replay.mp4")  # defaults: to = newest footage (the clip's end), from = to - 30 s
    assert r.status_code == 200, r.text
    assert r.headers["x-replay-source"] == "video" and r.headers["x-replay-start"] == f"{T0:.3f}"
    assert _duration(r.content, tmp_path) == pytest.approx(5.0, abs=0.15)

    cited = [Citation(id="x", ts=T0 + 2, caption="", thumb_url="")]
    await annotate_citations(app.state, cited)  # no buffered or kept frames: the video backs replay_url
    assert cited[0].replay_url and "device_id=dev1" in cited[0].replay_url


async def test_replay_without_video_module_is_unchanged(app_client):
    c, app = app_client
    app.state.video_replay = None
    app.state.video_plan = None
    await _post_frame(c, _jpeg(1), T0)
    r = await c.get("/v1/replay.mp4", params={"device_id": "other", "from": "990", "to": "1010"})
    assert r.status_code == 404


# ---- montage prefers stored video ----


@needs_ffmpeg
async def test_montage_uses_video_clips_when_covered(app_client, clips, tmp_path):
    c, app = app_client
    st = app.state
    for i in range(2):
        p = st.settings.thumb_dir / f"t{i}.jpg"
        p.write_bytes(_jpeg(i))
        await st.store.add(MemoryToken(id=f"t{i}", device_id="dev1", ts=T0 + 2 + 30 * i,
                                       caption=f"scene number {i} at the desk", thumb_path=str(p),
                                       frame_hash=f"{i:016x}"))  # fmt: skip
    await _post_video(c, clips["5s"], T0)  # covers the first moment (T0+1 .. T0+5) only
    await _post_video(c, clips["1s"], T0 + 4.5)
    r = await c.post("/v1/montage", json={"time_from": T0, "time_to": T0 + 100, "max_clips": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [cl["source"] for cl in body["clips"]] == ["video", "timelapse"]
    assert body["clips"][0]["duration_s"] == pytest.approx(4.0)
    data = (await c.get(body["url"])).content
    total = sum(cl["duration_s"] for cl in body["clips"])
    assert _duration(data, tmp_path) == pytest.approx(total, abs=0.3)

    st.settings.privacy_blur_faces = True  # unblurred sources skipped: stored thumbnails only
    r = await c.post("/v1/montage", json={"time_from": T0, "time_to": T0 + 100, "max_clips": 2})
    assert [cl["source"] for cl in r.json()["clips"]] == ["timelapse", "timelapse"]


@needs_ffmpeg
async def test_montage_render_failing_after_a_purge_is_409(app_client, clips, monkeypatch):
    c, app = app_client
    st = app.state
    p = st.settings.thumb_dir / "t0.jpg"
    p.write_bytes(_jpeg(0))
    await st.store.add(MemoryToken(id="t0", device_id="dev1", ts=T0 + 2, caption="scene at the desk",
                                   thumb_path=str(p), frame_hash="0" * 16))  # fmt: skip
    await _post_video(c, clips["5s"], T0)

    async def purged_mid_render(*a, **kw):  # DELETE /v1/memories removed the clip while ffmpeg was cutting it
        st.montage_cache.clear()
        raise RuntimeError("ffmpeg failed: No such file or directory")

    monkeypatch.setattr(st.montage_cache, "get_or_render", purged_mid_render)
    r = await c.post("/v1/montage", json={"time_from": T0, "time_to": T0 + 100, "max_clips": 1})
    assert r.status_code == 409, r.text
