"""Startup warmup (megamind/warmup.py): background, non-blocking, failure-tolerant, never a paid model call."""

from __future__ import annotations

import asyncio
import io
import json
import random
import time
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from megamind import warmup as wu
from megamind.config import Settings
from megamind.main import create_app
from megamind.retrieval.hybrid import HybridRetriever


def _settings(tmp_path, **kw) -> Settings:
    base = dict(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend="lancedb",
        lancedb_path=tmp_path / "lancedb",
        embed_provider="mock",
        filter_min_gap_s=0.0,
        enrich_provider="mock",
        consolidate_enabled=False,
        anthropic_api_key="",
        openai_api_key="",
        _env_file=None,
    )
    return Settings(**(base | kw))


def _jpeg(seed: int) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", (96, 72))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(96 * 72)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def _post_frame(c, seed: int, ts: float):
    r = await c.post(
        "/v1/frames",
        files={"image": ("f.jpg", _jpeg(seed), "image/jpeg")},
        data={"meta": json.dumps({"device_id": "dev1", "ts": ts})},
    )
    assert r.status_code == 202


class _FakeAnthropic:
    """count_tokens is the only call warmup may make; messages.create would be a paid turn."""

    def __init__(self, fail: bool = False) -> None:
        self.counted: list[dict] = []
        self.created = 0
        self.fail = fail
        outer = self

        class _Messages:
            async def count_tokens(self, **kw):
                if outer.fail:
                    raise RuntimeError("401 invalid x-api-key")
                outer.counted.append(kw)
                return SimpleNamespace(input_tokens=8)

            async def create(self, **kw):
                outer.created += 1
                raise AssertionError("warmup must not make a paid Claude call")

        self.messages = _Messages()


class _FakeOpenAI:
    def __init__(self) -> None:
        self.listed = 0
        self.audio = SimpleNamespace(transcriptions=object(), speech=object())
        outer = self

        class _Models:
            async def list(self):
                outer.listed += 1
                return []

        self.models = _Models()


async def test_off_by_flag_starts_nothing(tmp_path):
    app = create_app(_settings(tmp_path, warmup=False))
    async with app.router.lifespan_context(app):
        assert app.state.warmup_task is None
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            assert (await c.get("/v1/health")).json()["warmup"] == "off"


async def test_startup_and_requests_do_not_wait_for_warmup(tmp_path, monkeypatch):
    """A warmup step that hangs must not delay startup, /v1/health or /v1/chat, and shutdown cancels it."""
    release = asyncio.Event()

    async def hang():
        await release.wait()

    monkeypatch.setattr(wu, "_steps", lambda st, settings: {"slow": hang})
    app = create_app(_settings(tmp_path, warmup=True, warmup_timeout_s=60))
    t0 = time.perf_counter()
    async with app.router.lifespan_context(app):
        assert time.perf_counter() - t0 < 5
        task = app.state.warmup_task
        assert task is not None and not task.done()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            t1 = time.perf_counter()
            h = await c.get("/v1/health")
            assert h.status_code == 200 and h.json()["warmup"] == "running"
            r = await c.post("/v1/chat", json={"session_id": "s", "text": "what did I see?", "device_id": "dev1"})
            assert r.status_code == 200
            assert time.perf_counter() - t1 < 5
        assert not task.done()
    assert task.cancelled()  # stopped with the app, no stray task


async def test_failures_and_timeouts_are_logged_not_raised(tmp_path, monkeypatch, caplog):
    async def boom():
        raise RuntimeError("provider down")

    async def slow():
        await asyncio.sleep(10)

    async def fine():
        return None

    monkeypatch.setattr(wu, "_steps", lambda st, settings: {"boom": boom, "slow": slow, "fine": fine})
    app = create_app(_settings(tmp_path, warmup=True, warmup_timeout_s=0.2))
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.warmup_task, 5)
        state = app.state.warmup
        assert state.status == "done (failed: boom, slow)"
        assert state.steps["boom"] == "failed (RuntimeError)"
        assert state.steps["slow"].startswith("failed (timeout")
        assert state.steps["fine"].startswith("ok ")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            h = (await c.get("/v1/health")).json()
            assert h["status"] == "ok" and h["warmup"] == "done (failed: boom, slow)"
    assert "warmup boom failed" in caplog.text


async def test_real_steps_with_mock_providers(tmp_path):
    """The default steps run against a mock-provider app with memories: retrieval + reader warm, nothing fails."""
    app = create_app(_settings(tmp_path, warmup=False))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            for i in range(3):
                await _post_frame(c, i + 1, 1000.0 + 10 * i)
            await app.state.worker.queue.join()
        st = await wu.run_warmup(app.state, app.state.settings)
        assert st.status == "done", st.steps
        assert st.steps["retrieval"].startswith("ok ")
        assert st.steps["anthropic"] == "skipped (no Claude key)"
        assert st.steps["stt"] == st.steps["tts"] == "skipped (mock)"
        assert st.steps["image_embed"] == "skipped (no image embedder)"


async def test_empty_store_is_fine(tmp_path):
    app = create_app(_settings(tmp_path, warmup=False))
    async with app.router.lifespan_context(app):
        st = await wu.run_warmup(app.state, app.state.settings)
        assert st.status == "done"
        assert st.steps["retrieval"].startswith("ok (empty store)")


async def test_provider_clients_get_only_free_calls(tmp_path):
    app = create_app(_settings(tmp_path, warmup=False))
    async with app.router.lifespan_context(app):
        claude, stt, tts = _FakeAnthropic(), _FakeOpenAI(), _FakeOpenAI()
        app.state.agent._client = claude
        app.state.stt = SimpleNamespace(client=stt)
        app.state.tts = SimpleNamespace(client=tts)
        st = await wu.run_warmup(app.state, app.state.settings)
        assert st.status == "done", st.steps
        assert claude.counted and claude.counted[0]["model"] == app.state.settings.claude_model
        assert claude.created == 0
        assert stt.listed == 1 and tts.listed == 1


async def test_provider_failure_marks_step_but_others_finish(tmp_path):
    app = create_app(_settings(tmp_path, warmup=False))
    async with app.router.lifespan_context(app):
        app.state.agent._client = _FakeAnthropic(fail=True)
        st = await wu.run_warmup(app.state, app.state.settings)
        assert st.status == "done (failed: anthropic)"
        assert st.steps["retrieval"].startswith("ok")


async def test_vlm_reader_is_never_called(tmp_path):
    """A hosted reader (Sail VLM) would be a paid call: warmup retrieves but does not read."""
    app = create_app(_settings(tmp_path, warmup=False))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await _post_frame(c, 7, 1000.0)
            await app.state.worker.queue.join()

        class VLMReader:
            calls = 0

            async def read(self, *a, **kw):
                VLMReader.calls += 1
                raise AssertionError("paid reader call")

        app.state.reader = VLMReader()
        st = await wu.run_warmup(app.state, app.state.settings)
        assert st.status == "done" and VLMReader.calls == 0


def test_hybrid_is_found_through_wrappers_without_touching_them(tmp_path):
    hybrid = HybridRetriever.__new__(HybridRetriever)

    class Hier:  # e.g. HierRetriever.base / ComboRetriever.hybrid / UCCRetriever.fallback
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, name):  # like LiveRetriever: forwards unknown names
            raise AssertionError(f"forwarded {name}")

    assert wu._hybrid_of(Hier(inner=Hier(base=hybrid))) is hybrid
    assert wu._hybrid_of(Hier(hybrid=Hier(fallback=hybrid))) is hybrid
    assert wu._hybrid_of(Hier(inner=None)) is None


async def test_health_does_no_provider_work(tmp_path):
    """Clients poll /v1/health every 4 s: it must not embed, retrieve or call a model."""
    app = create_app(_settings(tmp_path, warmup=False))
    async with app.router.lifespan_context(app):
        calls = []
        emb = app.state.text_embedder
        orig = emb.embed

        async def counting(text):
            calls.append(text)
            return await orig(text)

        emb.embed = counting
        app.state.agent._client = _FakeAnthropic()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            for _ in range(3):
                assert (await c.get("/v1/health")).status_code == 200
        assert calls == [] and app.state.agent._client.counted == []


@pytest.mark.parametrize("flag", ["true", "false"])
def test_env_flag(monkeypatch, flag):
    monkeypatch.setenv("WARMUP", flag)
    assert Settings(_env_file=None).warmup is (flag == "true")
