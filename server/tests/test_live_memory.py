"""Live short-term memory: NOW items from the replay buffer / fresh transcripts for "right now" questions."""

from __future__ import annotations

import io
import json
import time

import pytest
from PIL import Image
from test_claude_agent import ScriptedAnthropic, _message, text, tool_use

from megamind.agent.claude_agent import ClaudeAgent
from megamind.agent.reader import CaptionOnlyReader
from megamind.agent.stt import STT
from megamind.config import Settings
from megamind.live import (
    LIVE_REASON,
    LiveMemory,
    LiveMemorySettings,
    LiveRetriever,
    _frame_id,
    is_now_question,
    wrap_retriever,
)
from megamind.models import MemoryToken
from megamind.replay import ReplayBuffer
from megamind.retrieval.base import RetrievalResult, RetrievedToken, Retriever
from megamind.retrieval.planner import QueryPlan


def _jpeg(color=(200, 40, 40), size=(320, 240)) -> bytes:
    """A sharp red/white checkerboard (passes the blur filter; repeats are dhash duplicates)."""
    img = Image.new("RGB", size, (255, 255, 255))
    for x in range(0, size[0], 16):
        for y in range(0, size[1], 16):
            if (x // 16 + y // 16) % 2:
                img.paste(color, (x, y, x + 16, y + 16))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def _post_frame(c, jpeg, ts, device="dev1"):
    r = await c.post(
        "/v1/frames",
        files={"image": ("f.jpg", jpeg, "image/jpeg")},
        data={"meta": json.dumps({"device_id": device, "ts": ts})},
    )
    assert r.status_code in (200, 202), r.text
    return r.json()


async def _post_burst(c, t0: float, n: int = 5, device="dev1") -> list[float]:
    """n identical frames 1 s apart: the keyframe filter keeps only the first (duplicates), so the newest
    frames exist only in the replay buffer, exactly the "enrichment hasn't caught up" situation."""
    ts = [t0 + i for i in range(n)]
    accepted = [(await _post_frame(c, _jpeg(), t, device))["accepted"] for t in ts]
    assert accepted == [True] + [False] * (n - 1)
    return ts


def _fallback_agent(app) -> ClaudeAgent:
    return ClaudeAgent(Settings(_env_file=None, anthropic_api_key=""), app.state.recall)


# ---- detection ----


@pytest.mark.parametrize(
    "q",
    [
        "what am I looking at right now?",
        "What is this I'm holding",
        "what did I just see",
        "what was that a moment ago",
        "what's in front of me",
        "who just walked past",
        "what am I currently reading",
        "what do I see now?",
        "current view",
    ],
)
def test_now_questions(q):
    assert is_now_question(q)


@pytest.mark.parametrize(
    "q",
    [
        "where did I leave my keys?",
        "what was on the whiteboard this morning",
        "what was I looking at yesterday",
        "I just want to know where my mug is",
        "how many coffees did I have today",
        "",
        None,
    ],
)
def test_not_now_questions(q):
    assert not is_now_question(q)


# ---- end to end through the app (store + replay buffer + mock agent fallback) ----


async def test_right_now_question_gets_now_items_and_valid_citations(client):
    c, app = client
    t0 = 1_762_000_000.0
    ts = await _post_burst(c, t0)
    now = ts[-1] + 1.0

    res = await app.state.recall(query="what am I looking at right now", device_id="dev1", now=now)
    live = [x for x in res.all_retrieved if x.id.startswith("now_")]
    assert [x.ts for x in live] == [ts[-1], ts[-2], ts[-3]]  # newest 3, NOW-1 first
    assert [x.caption.split(":")[0] for x in live] == ["LIVE NOW-1", "LIVE NOW-2", "LIVE NOW-3"]
    assert all(x.thumb_url == f"/v1/live/{x.id}.jpg" for x in live)
    retrieved = {x.id for x in res.all_retrieved}
    assert res.cited and {x.id for x in res.cited} <= retrieved
    assert res.cited[0].id == live[0].id  # caption-only reader cites the top-ranked items
    assert res.confidence != "not_found"

    # the mock agent path (no Claude key): same evidence, cited ids valid
    resp, stats = await _fallback_agent(app).chat_with_stats("s", "what am I looking at right now?", "dev1", now=now)
    assert stats.path == "fallback"
    assert resp.cited and resp.cited[0].id == _frame_id("dev1", ts[-1])
    assert "LIVE NOW-1" in resp.answer

    # the cited frame is served as an image while it is buffered
    r = await c.get(live[0].thumb_url)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.content)).size == (320, 240)


async def test_chat_endpoint_now_items_get_replay_urls(client):
    c, app = client
    app.state.agent = _fallback_agent(app)
    ts = await _post_burst(c, time.time() - 4.0)
    body = {"session_id": "s", "device_id": "dev1", "text": "what am I looking at right now"}
    r = await c.post("/v1/chat", json=body)
    assert r.status_code == 200, r.text
    cited = r.json()["cited"]
    assert cited[0]["id"] == _frame_id("dev1", ts[-1])
    assert cited[0]["thumb_url"].startswith("/v1/live/")
    assert cited[0]["replay_url"] and "device_id=dev1" in cited[0]["replay_url"]


async def test_non_now_question_has_no_now_items(client):
    c, app = client
    ts = await _post_burst(c, 1_762_000_000.0)
    res = await app.state.recall(query="where did I leave my keys", device_id="dev1", now=ts[-1] + 1)
    assert not [x for x in res.all_retrieved if x.id.startswith("now_")]


async def test_stale_buffer_is_not_now(client):
    c, app = client
    ts = await _post_burst(c, 1_762_000_000.0)
    res = await app.state.recall(query="what am I looking at right now", device_id="dev1", now=ts[-1] + 3600)
    assert not [x for x in res.all_retrieved if x.id.startswith("now_")]


async def test_purge_hides_live_frames(client):
    c, app = client
    ts = await _post_burst(c, 1_762_000_000.0)
    res = await app.state.recall(query="what am I looking at right now", device_id="dev1", now=ts[-1] + 1)
    url = next(x.thumb_url for x in res.all_retrieved if x.id.startswith("now_"))
    assert (await c.get(url)).status_code == 200
    assert (await c.delete("/v1/memories")).status_code == 200
    assert (await c.get(url)).status_code == 404
    assert (await c.get("/v1/live/now_1_nope.jpg")).status_code == 404


class _STT(STT):
    def __init__(self, text: str) -> None:
        self.text = text

    async def transcribe(self, wav: bytes) -> str:
        return self.text


async def test_fresh_transcript_is_a_now_item_once(client):
    from test_audio import _post_audio, _wav

    c, app = client
    t0 = 1_762_000_000.0
    ts = await _post_burst(c, t0)
    app.state.audio_stt = _STT("the parking code is four four two one")
    r = await _post_audio(c, _wav(2.0), ts_start=ts[-1] - 2.0)
    assert r.status_code == 200, r.text
    tid = r.json()["token_ids"][0]

    res = await app.state.recall(query="what did I just hear right now", device_id="dev1", now=ts[-1] + 1)
    ids = [x.id for x in res.all_retrieved]
    assert ids.count(tid) == 1  # deduped whether the retriever found it or live memory added it
    assert sum(i.startswith("now_") for i in ids) == 3

    # an old transcript (outside ~30 s) is not pulled in by live memory
    r = await _post_audio(c, _wav(2.0), ts_start=t0 - 600)
    old = r.json()["token_ids"][0]
    out = await app.state.live_memory.now_items([], "dev1", ts[-1] + 1)
    assert tid in [t.id for t in out] and old not in [t.id for t in out]
    assert next(t for t in out if t.id == tid).caption.startswith("LIVE NOW-")


async def test_claude_path_uses_raw_question_behind_keyword_query(client):
    """Claude rewrites "what am I holding right now" to keywords; the raw question still triggers NOW items."""
    c, app = client
    ts = await _post_burst(c, 1_762_000_000.0)
    now = ts[-1] + 1
    fid = _frame_id("dev1", ts[-1])
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "object in hand", "k": 6})], stop_reason="tool_use"),
            _message([text(f"You're holding something red. <cites>{fid}</cites>")]),
        ]
    )
    settings = Settings(_env_file=None, anthropic_api_key="k", recall_prefetch=False, reader_skip_on_ocr=True)
    agent = ClaudeAgent(settings, app.state.recall, client=script.client())
    resp, stats = await agent.chat_with_stats("s", "what am I holding right now?", "dev1", now=now)
    assert [x.id for x in resp.cited] == [fid]
    tool_result = json.dumps(script.requests[1]["messages"])
    assert "LIVE NOW-1" in tool_result


# ---- units ----


class _Inner(Retriever):
    def __init__(self, tokens: list[RetrievedToken]) -> None:
        self.tokens = tokens
        self.cfg = "inner-cfg"

    async def retrieve(self, query, **kw) -> RetrievalResult:
        plan = QueryPlan(query=query, search_text=query, query_type="generic")
        return RetrievalResult(plan=plan, tokens=list(self.tokens))


class _State:
    def __init__(self, replay: ReplayBuffer | None, store=None) -> None:
        self.replay = replay
        self.store = store


def _tok(tid: str, ts: float, kind="frame") -> MemoryToken:
    return MemoryToken(id=tid, device_id="d", ts=ts, caption=tid, thumb_path="", frame_hash="", kind=kind)


async def test_frames_already_retrieved_are_deduped(tmp_path):
    buf = ReplayBuffer()
    for t in range(100, 106):
        buf.add("d", float(t), _jpeg())
    live = LiveMemory(_State(buf), LiveMemorySettings(_env_file=None), tmp_path)
    enriched = RetrievedToken(_tok("t_105", 105.0), 0.9)
    out = await live.now_items([enriched], "d", now=106.0)
    assert [t.ts for t in out] == [104.0, 103.0]  # 105 is already evidence
    assert all(t.thumb_path and (tmp_path / f"{t.id}.jpg").is_file() for t in out)


async def test_frame_gap_and_cache_bound(tmp_path):
    buf = ReplayBuffer()
    for i in range(30):  # 10 fps
        buf.add("d", 100.0 + i / 10, _jpeg())
    cfg = LiveMemorySettings(_env_file=None, cache_max_files=4)
    live = LiveMemory(_State(buf), cfg, tmp_path)
    out = await live.now_items([], None, now=103.0)  # device resolved to the most recently fed one
    assert [round(t.ts, 1) for t in out] == [102.9, 101.9, 100.9]
    for n in (102.0, 101.0):
        await live.now_items([], "d", now=n)
    assert len(list(tmp_path.glob("*.jpg"))) <= 4


async def test_live_retriever_ranks_now_first_and_keeps_order(tmp_path):
    buf = ReplayBuffer()
    for t in (100.0, 101.0, 102.0):
        buf.add("d", t, _jpeg())
    inner = _Inner([RetrievedToken(_tok("old", 50.0), 0.7), RetrievedToken(_tok("older", 40.0), 0.9)])
    lr = LiveRetriever(inner, LiveMemory(_State(buf), LiveMemorySettings(_env_file=None), tmp_path))
    from datetime import UTC, datetime

    res = await lr.retrieve("what am I looking at right now", device_id="d", now=datetime.fromtimestamp(103.0, UTC))
    assert [r.token.ts for r in res.tokens] == sorted(r.token.ts for r in res.tokens)  # chronological
    ranked = res.ranked
    assert [r.token.ts for r in ranked[:3]] == [102.0, 101.0, 100.0]
    assert all(r.reasons == [LIVE_REASON] for r in ranked[:3])
    assert lr.cfg == "inner-cfg"  # attribute passthrough

    res = await lr.retrieve("keys", device_id="d", now=datetime.fromtimestamp(103.0, UTC))
    assert [r.token.id for r in res.tokens] == ["old", "older"]


def test_opt_out_flag(tmp_path, monkeypatch):
    inner = _Inner([])
    settings = Settings(_env_file=None, thumb_dir=tmp_path / "thumbs")

    class S:
        live_settings = LiveMemorySettings(_env_file=None, enabled=False)

    assert wrap_retriever(S(), settings, inner) is inner
    monkeypatch.setenv("LIVE_MEMORY_ENABLED", "false")
    assert LiveMemorySettings(_env_file=None).enabled is False
    S.live_settings = LiveMemorySettings(_env_file=None, enabled=True)
    assert isinstance(wrap_retriever(S(), settings, inner), LiveRetriever)


async def test_no_replay_buffer_is_harmless(tmp_path):
    live = LiveMemory(_State(None), LiveMemorySettings(_env_file=None), tmp_path)
    assert await live.now_items([], "d", now=1.0) == []
    assert live.lookup("now_1_x") is None


def test_caption_only_reader_unchanged_interface():
    # NOW items are plain MemoryTokens to the reader: nothing reader-specific was added
    assert CaptionOnlyReader().read.__code__.co_varnames[:3] == ("self", "query", "tokens")
