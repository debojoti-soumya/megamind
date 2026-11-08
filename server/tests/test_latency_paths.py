"""Latency paths: stream event order (transcript -> retrieval -> delta -> audio -> final), sentence-streamed TTS,
the "right now" fast path, broadened recall_first rules, mixed-resolution reader images, opt-in model routing."""

from __future__ import annotations

import base64
import io
import json
import time

from PIL import Image

from megamind.agent.claude_agent import ClaudeAgent, fast_model_ok, now_fast_path, recall_first
from megamind.agent.reader import ClaudeVisionReader
from megamind.config import Settings
from megamind.models import MemoryToken, RecallResult
from megamind.retrieval.planner import plan_query
from tests.test_chat_stream import (
    ScriptedAnthropic,
    _message,
    _parse_sse,
    _two_rounds,
    _use_script,
    app_client,  # noqa: F401 - fixture
    text,
)

NOW = time.time()


def _plan(q: str):
    from datetime import UTC, datetime

    return plan_query(q, now=datetime.fromtimestamp(NOW, UTC), tz="UTC")


# ---- recall_first / fast path / routing rules ----


def test_recall_first_broadened_but_safe():
    for q in (
        "summarize what I saw at the market",
        "tell me where I left my wallet",
        "show me what the menu said",
        "remind me what the whiteboard said",
        "list the books on the shelf",
        "describe the poster in the hallway",
        "tell me about the receipt from the cafe",
        "what about the receipt",
    ):
        assert recall_first(q, _plan(q)), q
    for q in ("tell me a joke", "show me", "tell me about yourself", "what about it", "and the receipt?", "thanks"):
        assert not recall_first(q, _plan(q)), q
    # "what about X" keeps Claude's pick when the previous search was time-bounded (the window would be lost)
    q = "what about the receipt"
    assert not recall_first(q, _plan(q), {"query": "coffee", "time_from": "2026-01-01T09:00:00+00:00"})
    assert recall_first(q, _plan(q), {"query": "coffee"})


def test_now_fast_path_selection():
    for q in ("what am I looking at right now?", "what does the sign in front of me say?", "what's happening"):
        assert now_fast_path(q, _plan(q), NOW), q
    for q in (
        "where are my keys now?",  # trailing "now" only
        "is my mug still on the desk right now?",  # a change vs earlier
        "where did I put my keys just now?",  # object location: may have left the view
        "what was I doing this morning?",
        "what did I see yesterday",
    ):
        assert not now_fast_path(q, _plan(q), NOW), q


def test_fast_model_only_for_short_standalone_lookups():
    assert fast_model_ok("where did I leave my keys?", _plan("where did I leave my keys?"))
    for q in ("summarize my day", "where did I leave my keys this morning?", "and where after that?"):
        assert not fast_model_ok(q, _plan(q)), q


# ---- /v1/chat/stream event order ----


async def test_stream_order_transcript_retrieval_delta_final(app_client):  # noqa: F811
    c, app = app_client
    script = ScriptedAnthropic(_two_rounds())
    _use_script(app, script)
    wav = b"RIFF" + b"\0" * 40
    r = await c.post(
        "/v1/chat/stream", files={"audio": ("q.wav", wav, "audio/wav")}, data={"session_id": "s", "device_id": "dev1"}
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "transcript" and events[0][1] == {"text": "[mock transcript]"}
    assert names.count("transcript") == 1 and names[-1] == "final"
    assert names.index("retrieval") < names.index("delta") < len(names) - 1
    assert events[-1][1]["transcript"] == "[mock transcript]"


async def test_stream_typed_question_has_no_transcript(app_client):  # noqa: F811
    c, app = app_client
    _use_script(app, ScriptedAnthropic(_two_rounds()))
    r = await c.post("/v1/chat/stream", json={"session_id": "s", "text": "where are my keys?"})
    names = [n for n, _ in _parse_sse(r.text)]
    assert "transcript" not in names and names[-1] == "final"


# ---- fast path in the agent ----


class LiveOnlyRecall:
    def __init__(self, live: RecallResult | None):
        self.live = live
        self.calls: list[dict] = []
        self.now_calls = 0

    async def recall_now(self, query, *, device_id=None, now=None):
        self.now_calls += 1
        return self.live

    async def __call__(self, **kw):
        self.calls.append(kw)
        return RecallResult(evidence="No memories match.", confidence="not_found", cited=[], all_retrieved=[])


async def test_now_question_skips_retrieval_and_uses_one_call():
    from megamind.models import Citation

    cite = Citation(id="now_1_a", ts=NOW - 1, caption="LIVE NOW-1: raw camera frame", thumb_url="/v1/live/now_1_a.jpg")
    recall = LiveOnlyRecall(
        RecallResult(evidence="[live view only] x", confidence="medium", cited=[cite], all_retrieved=[cite])
    )
    script = ScriptedAnthropic([_message([text("A laptop on a desk. <cites>now_1_a</cites>")])])
    settings = Settings(_env_file=None, anthropic_api_key="k", recall_prefetch=True, recall_first=True)
    agent = ClaudeAgent(settings, recall, client=script.client())
    resp, stats = await agent.chat_with_stats("s", "what am I looking at right now?", "dev1", now=NOW)
    assert recall.now_calls == 1 and recall.calls == [] and len(script.requests) == 1
    assert stats.extra["fast_path"] == "now" and [c.id for c in resp.cited] == ["now_1_a"]
    assert script.requests[0]["model"] == settings.claude_model  # routing is opt-in

    # no live frames -> the normal search
    recall2 = LiveOnlyRecall(None)
    script2 = ScriptedAnthropic([_message([text("I don't have a memory of that.")])])
    agent2 = ClaudeAgent(settings, recall2, client=script2.client())
    await agent2.chat_with_stats("s", "what am I looking at right now?", "dev1", now=NOW)
    assert recall2.now_calls == 1 and len(recall2.calls) == 1

    # routing on: the live turn goes to the fast model
    script3 = ScriptedAnthropic([_message([text("A laptop. <cites>now_1_a</cites>")])])
    s3 = settings.model_copy(update={"claude_route_fast": True})
    await ClaudeAgent(s3, recall, client=script3.client()).chat_with_stats("s", "what am I looking at right now?", "d")
    assert script3.requests[0]["model"] == s3.claude_fast_model


async def test_recall_now_uses_live_items_only(tmp_path):
    from megamind.agent.reader import CaptionOnlyReader
    from megamind.agent.tools import LIVE_ONLY_PREFIX, RecallMemory
    from megamind.ingest.enrich import MockTextEmbedder
    from megamind.store.memory_store import InMemoryStore

    frame = MemoryToken(
        id="now_5_b", device_id="d", ts=NOW - 1, caption="LIVE NOW-1: raw camera frame", thumb_path="", frame_hash=""
    )

    class Live:
        async def now_items(self, retrieved, device_id, now):
            assert retrieved == []
            return [frame]

    class Ret:
        live = Live()

        async def retrieve(self, *a, **k):
            raise AssertionError("no search on the fast path")

    rm = RecallMemory(InMemoryStore(), MockTextEmbedder(), None, CaptionOnlyReader(), retriever=Ret())
    r = await rm.recall_now("what am I looking at right now", device_id="d", now=NOW)
    assert r is not None and r.evidence.startswith(LIVE_ONLY_PREFIX) and [c.id for c in r.all_retrieved] == ["now_5_b"]


# ---- mixed-resolution images ----


async def test_claude_reader_sends_top_full_rest_small(tmp_path):
    toks = []
    for i in range(5):
        p = tmp_path / f"m{i}.jpg"
        Image.new("RGB", (1280, 960), (i * 40, 90, 160)).save(p, format="JPEG")
        toks.append(
            MemoryToken(id=f"m{i}", device_id="d", ts=1000.0 + i, caption=f"c{i}", thumb_path=str(p), frame_hash="")
        )

    def sizes(out):
        res = []
        for _, uri in out.images:
            with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as im:
                res.append(max(im.size))
        return res

    s = Settings(_env_file=None, reader_provider="claude", reader_full_images=2, reader_small_px=512)
    out = await ClaudeVisionReader(s).read("q", toks, now=2000.0)
    assert [i for i, _ in out.images] == ["m0", "m1", "m2", "m3", "m4"] and sizes(out) == [960, 960, 512, 512, 512]
    assert all(f"c{i}" in out.evidence for i in range(5))
    s0 = s.model_copy(update={"reader_small_px": 0})
    assert [i for i, _ in (await ClaudeVisionReader(s0).read("q", toks)).images] == ["m0", "m1"]
    s_all = s.model_copy(update={"reader_full_images": 0})
    assert sizes(await ClaudeVisionReader(s_all).read("q", toks)) == [960] * 5
    json.dumps([])  # keep import used
