import io
import random
import subprocess
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from megamind.agent.montage import (
    ClipSpec,
    MontagePlan,
    MontagePlanner,
    MontageSettings,
    PlannedClip,
    _labels,
    evenly,
    offline_plan,
    overlay_vf,
)
from megamind.config import Settings
from megamind.main import create_app
from megamind.models import MemoryToken
from megamind.replay import find_ffmpeg

HAVE_FFMPEG = find_ffmpeg() is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not available")
T0 = 1_731_000_000.0


def _jpeg(seed: int, size=(96, 72)) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", size, (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _tok(i: int, caption: str, ts: float | None = None, thumb: str = "", ocr: str = "", kind="frame") -> MemoryToken:
    return MemoryToken(
        id=f"t{i:03d}",
        device_id="dev1",
        ts=T0 + 10 * i if ts is None else ts,
        caption=caption,
        ocr_text=ocr,
        thumb_path=thumb,
        frame_hash=f"{i:016x}" if thumb else "",
        kind=kind,
    )


@pytest.fixture
async def app_client(tmp_path):
    settings = Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend="memory",
        embed_provider="mock",
        consolidate_enabled=False,
        anthropic_api_key="",
        _env_file=None,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app


async def _seed(app, n=6, with_thumbs=True, buffer=False):
    st = app.state
    toks = []
    for i in range(n):
        thumb = ""
        if with_thumbs:
            p = st.settings.thumb_dir / f"t{i:03d}.jpg"
            p.write_bytes(_jpeg(i))
            thumb = str(p)
        toks.append(_tok(i, f"scene number {i} at the desk", thumb=thumb))
    for t in toks:
        await st.store.add(t)
    if buffer:
        for k in range(10 * n * 2):  # 2 fps for the whole span
            st.replay.add("dev1", T0 - 5 + k * 0.5, _jpeg(1000 + k))
    return toks


# ---- plan ----


def test_evenly_spaced_selection():
    assert evenly(list(range(10)), 3) == [0, 4, 9] or evenly(list(range(10)), 3) == [0, 5, 9]
    assert evenly([1, 2], 5) == [1, 2]
    assert evenly([1, 2, 3], 1) == [2]
    assert evenly([], 3) == []


def test_offline_plan_skips_empty_and_duplicate_moments():
    cands = [
        _tok(3, "kitchen counter with keys"),
        _tok(1, "(no visual information)"),
        _tok(2, "kitchen counter with keys"),  # duplicate caption
        _tok(4, "", ocr="EXIT 12"),
        _tok(5, "whiteboard with a sprint plan written on it in blue marker"),
    ]
    p = offline_plan(cands, 8, tz="UTC")
    labels = _labels(cands)
    chosen = [labels[c.label].id for c in p.clips]
    assert chosen == ["t002", "t004", "t005"]  # chronological; t001 empty, t003 duplicate
    assert [c.title for c in p.clips][:2] == ["kitchen counter with keys", "EXIT 12"]
    assert p.clips[2].title.endswith("…") and len(p.clips[2].title) <= 43
    assert "3 moments" in p.narration
    assert offline_plan(cands, 8) == p  # deterministic


def test_offline_plan_falls_back_to_time_titles():
    p = offline_plan([_tok(1, "")], 4, tz="UTC")
    assert len(p.clips) == 1 and p.clips[0].title.startswith("Moment at ")


class _FakeMessages:
    def __init__(self, plan_input=None, exc=None):
        self.plan_input, self.exc, self.calls = plan_input, exc, []

    async def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="montage_plan", input=self.plan_input)])


def _planner(messages) -> MontagePlanner:
    return MontagePlanner(
        Settings(_env_file=None), MontageSettings(_env_file=None), client=SimpleNamespace(messages=messages)
    )


async def test_claude_planner_used_and_sanitised():
    cands = [_tok(i, f"moment {i}") for i in range(1, 6)]
    msgs = _FakeMessages(
        {
            "title": "Your morning",
            "narration": "Two moments.",
            "clips": [
                {"label": "S4", "title": "Later one", "duration": 3},
                {"label": "S99", "title": "hallucinated"},
                {"label": "S2", "title": "Earlier one"},
                {"label": "S2", "title": "dup"},
            ],
        }
    )
    plan, who = await _planner(msgs).plan("my morning", cands, 4)
    assert who == "claude"
    assert [c.label for c in plan.clips] == ["S2", "S4"]
    assert plan.title == "Your morning"
    call = msgs.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "montage_plan"}
    assert "[S1]" in call["messages"][0]["content"] and "Target clips: 4" in call["messages"][0]["content"]


async def test_claude_planner_failure_falls_back_offline():
    cands = [_tok(i, f"moment {i}") for i in range(1, 4)]
    plan, who = await _planner(_FakeMessages(exc=RuntimeError("boom"))).plan("x", cands, 2)
    assert who == "offline" and len(plan.clips) == 2
    plan, who = await _planner(_FakeMessages({"title": "t", "clips": "nope", "narration": ""})).plan("x", cands, 2)
    assert who == "offline"


async def test_claude_planner_retries_with_auto_when_forced_tool_rejected():
    import anthropic
    import httpx

    class _RejectForced(_FakeMessages):
        async def create(self, **kw):
            if kw["tool_choice"]["type"] == "tool":
                self.calls.append(kw)
                req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                msg = 'tool_choice: type "tool" and "any" are not supported for this model.'
                raise anthropic.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)
            return await super().create(**kw)

    cands = [_tok(i, f"moment {i}") for i in range(1, 4)]
    msgs = _RejectForced({"title": "t", "narration": "n", "clips": [{"label": "S2", "title": "two"}]})
    plan, who = await _planner(msgs).plan("x", cands, 2)
    assert who == "claude" and [c.label for c in plan.clips] == ["S2"]
    assert [c["tool_choice"]["type"] for c in msgs.calls] == ["tool", "auto"]
    assert "Call montage_plan" in msgs.calls[1]["messages"][0]["content"]


def test_planner_offline_without_key():
    p = MontagePlanner(Settings(anthropic_api_key="", _env_file=None), MontageSettings(_env_file=None))
    assert not p.enabled
    p = MontagePlanner(
        Settings(anthropic_api_key="k", _env_file=None), MontageSettings(planner="offline", _env_file=None)
    )
    assert not p.enabled


def test_overlay_escapes_and_normalises_size():
    c = ClipSpec(_tok(1, "x"), "it's 50% done: ok", "9:05 AM", ((0.0, b""), (1.0, b"")), 1.0, "replay")
    vf = overlay_vf(c, MontageSettings(width=640, height=360, _env_file=None))
    assert vf.startswith("scale=640:360:force_original_aspect_ratio=decrease,pad=640:360")
    assert "text='it’s 50% done\\: ok'" in vf and "text='9\\:05 AM'" in vf
    assert "fade=t=out:st=1.70" in vf
    assert "drawtext" not in overlay_vf(c, MontageSettings(_env_file=None), text=False)


# ---- HTTP ----


async def test_montage_requires_query_or_range(app_client):
    c, _ = app_client
    r = await c.post("/v1/montage", json={"device_id": "dev1"})
    assert r.status_code == 400
    r = await c.post("/v1/montage", json={"time_from": T0 + 10, "time_to": T0})
    assert r.status_code == 400
    r = await c.post("/v1/montage", json={"time_from": "not a time"})
    assert r.status_code == 400
    # Unity JsonUtility shape: nulls arrive as "" and an unset int as 0
    r = await c.post("/v1/montage", json={"device_id": "", "query": "", "time_from": "", "time_to": "", "max_clips": 0})
    assert r.status_code == 400


async def test_montage_503_without_ffmpeg(app_client, monkeypatch):
    c, app = app_client
    await _seed(app)
    monkeypatch.setattr("megamind.agent.montage.find_ffmpeg", lambda *_: None)
    r = await c.post("/v1/montage", json={"time_from": T0 - 1, "time_to": T0 + 100})
    assert r.status_code == 503


async def test_montage_404_when_nothing_matches(app_client):
    c, _ = app_client
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    r = await c.post("/v1/montage", json={"time_from": T0, "time_to": T0 + 100})
    assert r.status_code == 404


async def test_get_montage_supports_range_and_purge_clears(app_client):
    c, app = app_client
    cache = app.state.montage_cache
    mid = "0123456789abcdef0123"
    cache.dir.mkdir(parents=True, exist_ok=True)
    cache.path(mid).write_bytes(bytes(range(100)))

    r = await c.get(f"/v1/montages/{mid}.mp4")
    assert r.status_code == 200 and r.headers["accept-ranges"] == "bytes" and len(r.content) == 100
    assert r.headers["content-type"] == "video/mp4"
    r = await c.get(f"/v1/montages/{mid}.mp4", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))
    assert r.headers["content-range"] == "bytes 10-19/100"
    r = await c.get(f"/v1/montages/{mid}.mp4", headers={"Range": "bytes=-5"})
    assert r.status_code == 206 and r.content == bytes(range(95, 100))
    r = await c.get(f"/v1/montages/{mid}.mp4", headers={"Range": "bytes=500-"})
    assert r.status_code == 416 and r.headers["content-range"] == "bytes */100"
    assert (await c.head(f"/v1/montages/{mid}.mp4")).status_code == 200
    assert (await c.get("/v1/montages/../../etc.mp4")).status_code == 404
    assert (await c.get("/v1/montages/ffff.mp4")).status_code == 404

    assert (await c.delete("/v1/memories")).status_code == 200
    assert not cache.path(mid).exists()
    assert (await c.get(f"/v1/montages/{mid}.mp4")).status_code == 404


class _StubPlanner:
    """Mock planner: returns a fixed plan over the given candidates."""

    def __init__(self):
        self.calls = []

    async def plan(self, request, cands, n_clips):
        self.calls.append((request, len(cands), n_clips))
        return MontagePlan(
            title="Desk day",
            narration="Two desk moments.",
            clips=[PlannedClip(label="S2", title="Second scene", duration=3), PlannedClip(label="S5", title="Fifth")],
        ), "claude"


def _duration(ffmpeg: str, data: bytes, tmp_path) -> float:
    p = tmp_path / "m.mp4"
    p.write_bytes(data)
    out = subprocess.run([ffmpeg, "-i", str(p)], capture_output=True, text=True).stderr
    h, m, s = out.split("Duration: ", 1)[1].split(",", 1)[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


@needs_ffmpeg
@pytest.mark.parametrize("buffer", [False, True])
async def test_render_montage_end_to_end(app_client, tmp_path, buffer):
    c, app = app_client
    await _seed(app, buffer=buffer)
    app.state.montage_planner = stub = _StubPlanner()
    body = {"device_id": "dev1", "time_from": T0 - 1, "time_to": str(T0 + 100), "query": "", "max_clips": 0}
    r = await c.post("/v1/montage", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["title"] == "Desk day" and j["planner"] == "claude" and j["narration"] == "Two desk moments."
    assert [cl["token_id"] for cl in j["clips"]] == ["t001", "t004"]
    assert {cl["source"] for cl in j["clips"]} == {"replay" if buffer else "timelapse"}
    assert j["clips"][0]["thumb_url"] == "/v1/thumbs/t001.jpg"
    assert stub.calls == [("highlights of this period", 6, 8)]
    assert j["url"] == f"/v1/montages/{j['id']}.mp4"

    v = await c.get(j["url"])
    assert v.status_code == 200 and v.content[4:8] == b"ftyp"
    expect = sum(cl["duration_s"] for cl in j["clips"])
    assert abs(_duration(find_ffmpeg(), v.content, tmp_path) - expect) < 0.6
    r2 = await c.get(j["url"], headers={"Range": "bytes=0-99"})
    assert r2.status_code == 206 and r2.content == v.content[:100]

    # same plan -> same id, served from cache
    j2 = (await c.post("/v1/montage", json=body)).json()
    assert j2["id"] == j["id"]


@needs_ffmpeg
async def test_query_montage_with_offline_planner(app_client):
    c, app = app_client
    await _seed(app)
    assert not app.state.montage_planner.enabled
    r = await c.post("/v1/montage", json={"query": "desk scene", "max_clips": 2})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["planner"] == "offline" and 1 <= len(j["clips"]) <= 2
    assert (await c.get(j["url"])).status_code == 200


# ---- review fixes: no ffmpeg needed (render stubbed) ----


def _fake_render(app, monkeypatch):
    """Skip ffmpeg: the cache "renders" a placeholder file, so request handling is tested offline."""
    monkeypatch.setattr("megamind.agent.montage.find_ffmpeg", lambda *_: "ffmpeg")
    rendered = []

    def _render(frames, p, *_a):
        rendered.append(list(frames))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    app.state.montage_cache._render = _render
    return rendered


async def test_range_montage_samples_whole_span_beyond_scan_limit(app_client, monkeypatch):
    from megamind.agent import montage

    c, app = app_client
    monkeypatch.setattr(montage, "RANGE_SCAN_LIMIT", 20)
    for i in range(200):  # 200 tokens, 10 s apart; one scan would only see the oldest 20
        await app.state.store.add(_tok(i, f"distinct scene {i}"))
    cands = await montage.candidates(app.state, "", T0 - 1, T0 + 2000, "dev1", 8)
    assert len(cands) == 8
    assert cands[0].ts < T0 + 200 and cands[-1].ts > T0 + 1700  # spans the range, not just its start


async def test_purge_during_montage_discards_render(app_client, monkeypatch):
    c, app = app_client
    await _seed(app)
    _fake_render(app, monkeypatch)

    class _PurgingPlanner(_StubPlanner):
        async def plan(self, request, cands, n_clips):
            assert (await c.delete("/v1/memories")).status_code == 200  # wearer deletes mid-request
            return await super().plan(request, cands, n_clips)

    app.state.montage_planner = _PurgingPlanner()
    r = await c.post("/v1/montage", json={"time_from": T0 - 1, "time_to": T0 + 100})
    assert r.status_code in (404, 409)  # thumbnails already gone (404) or render discarded (409)
    assert not list(app.state.montage_cache.dir.glob("*.mp4"))


async def test_purge_after_candidates_discards_render(app_client, monkeypatch):
    """Clip frames were read before the purge: the render must not survive in the cache."""
    c, app = app_client
    await _seed(app)
    _fake_render(app, monkeypatch)
    app.state.montage_planner = _StubPlanner()
    cache = app.state.montage_cache
    orig = cache.get_or_render

    async def _get_or_render(key, *a, **kw):
        cache.clear()  # DELETE /v1/memories lands while ffmpeg runs
        return await orig(key, *a, **kw)

    cache.get_or_render = _get_or_render
    r = await c.post("/v1/montage", json={"time_from": T0 - 1, "time_to": T0 + 100})
    assert r.status_code == 409
    assert not list(cache.dir.glob("*.mp4"))


@pytest.mark.parametrize("blur", [False, True])
async def test_face_blur_uses_blurred_thumbnails_not_replay_frames(app_client, monkeypatch, blur):
    c, app = app_client
    await _seed(app, buffer=True)
    rendered = _fake_render(app, monkeypatch)
    app.state.montage_planner = _StubPlanner()
    monkeypatch.setattr(app.state.settings, "privacy_blur_faces", blur)
    r = await c.post("/v1/montage", json={"time_from": T0 - 1, "time_to": T0 + 100})
    assert r.status_code == 200, r.text
    assert {cl["source"] for cl in r.json()["clips"]} == {"timelapse" if blur else "replay"}
    assert len(rendered) == 1
    r2 = await c.get(r.json()["url"])
    assert r2.status_code == 200 and r2.content[4:8] == b"ftyp"


def test_overlay_forces_even_size():
    c = ClipSpec(_tok(1, "x"), "t", "9:05 AM", ((0.0, b""), (1.0, b"")), 1.0, "replay")
    assert overlay_vf(c, MontageSettings(width=641, height=361, _env_file=None)).startswith("scale=640:360:")
