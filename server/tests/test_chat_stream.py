"""POST /v1/chat/stream (SSE) against recorded Anthropic responses, streamed and non-streamed (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import anthropic
import httpx2
import pytest
from httpx import ASGITransport, AsyncClient

from megamind.agent.claude_agent import (
    EPHEMERAL,
    NOT_FOUND_ANSWER,
    ClaudeAgent,
    _TrailerFilter,
    render_static_system_prompt,
)
from megamind.config import Settings
from megamind.ingest.enrich import MockTextEmbedder
from megamind.main import create_app
from megamind.models import ChatResponse, MemoryToken

NOW = time.time()


def _message(content: list[dict], stop_reason: str = "end_turn", usage: dict | None = None) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 100, "output_tokens": 20},
    }


def text(s: str) -> dict:
    return {"type": "text", "text": s}


def tool_use(args: dict, tid: str = "toolu_1") -> dict:
    return {"type": "tool_use", "id": tid, "name": "recall_memory", "input": args}


def _sse_events(msg: dict, chunk: int = 7) -> list[tuple[str, bytes]]:
    """A Messages API response as the streaming wire format, text split into `chunk`-char deltas."""

    def ev(name: str, data: dict) -> tuple[str, bytes]:
        return name, f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

    start = {**msg, "content": [], "stop_reason": None, "usage": {**msg["usage"], "output_tokens": 1}}
    out = [ev("message_start", {"type": "message_start", "message": start})]
    for i, b in enumerate(msg["content"]):
        if b["type"] == "text":
            started = {"type": "content_block_start", "index": i, "content_block": text("")}
            out.append(ev("content_block_start", started))
            for j in range(0, len(b["text"]), chunk):
                delta = {"type": "text_delta", "text": b["text"][j : j + chunk]}
                out.append(ev("content_block_delta", {"type": "content_block_delta", "index": i, "delta": delta}))
        else:
            block = {**b, "input": {}}
            out.append(ev("content_block_start", {"type": "content_block_start", "index": i, "content_block": block}))
            delta = {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}
            out.append(ev("content_block_delta", {"type": "content_block_delta", "index": i, "delta": delta}))
        out.append(ev("content_block_stop", {"type": "content_block_stop", "index": i}))
    md = {"stop_reason": msg["stop_reason"], "stop_sequence": None}
    out.append(ev("message_delta", {"type": "message_delta", "delta": md, "usage": {"output_tokens": 20}}))
    out.append(ev("message_stop", {"type": "message_stop"}))
    return out


def _sse_body(msg: dict, chunk: int = 7) -> bytes:
    return b"".join(data for _, data in _sse_events(msg, chunk))


class ScriptedAnthropic:
    """Replays Messages API responses as JSON, or as SSE when the request asks to stream; records request bodies."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        nxt = self.responses.pop(0)
        if nxt.get("_status"):
            err = {"type": "error", "error": {"type": "api_error", "message": "x"}}
            return httpx2.Response(nxt["_status"], json=err)
        if body.get("stream"):
            return httpx2.Response(200, content=self.sse(nxt), headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=nxt)

    def sse(self, msg: dict):
        return _sse_body(msg)

    def client(self) -> anthropic.AsyncAnthropic:
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))
        return anthropic.AsyncAnthropic(api_key="test", http_client=http, max_retries=0)


class PacedAnthropic(ScriptedAnthropic):
    """Like the real API: the SSE body is produced over time, one text delta every `gap_s`; `produced` records when."""

    def __init__(self, responses: list[dict], gap_s: float = 0.03):
        super().__init__(responses)
        self.gap_s = gap_s
        self.produced: list[float] = []

    def sse(self, msg: dict):
        events = _sse_events(msg)

        async def gen():
            for name, data in events:
                if name == "content_block_delta" and b"text_delta" in data:
                    await asyncio.sleep(self.gap_s)
                    self.produced.append(time.perf_counter())
                yield data

        return gen()


def _agent_settings(**cfg) -> Settings:
    cfg.setdefault("recall_prefetch", True)
    cfg.setdefault("recall_first", False)
    cfg.setdefault("reader_skip_on_ocr", True)
    return Settings(_env_file=None, **cfg)


@pytest.fixture
async def app_client(tmp_path):
    settings = Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend="memory",
        embed_provider="mock",
        enrich_provider="mock",
        consolidate_enabled=False,
        _env_file=None,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        emb = MockTextEmbedder()
        for tid, ts, cap in [
            ("t_keys", NOW - 20 * 60, "Keys on the kitchen counter next to the coffee machine"),
            ("t_mug", NOW - 5 * 60, "A blue coffee mug on a wooden desk beside a laptop"),
        ]:
            await app.state.store.add(
                MemoryToken(
                    id=tid,
                    device_id="dev1",
                    ts=ts,
                    caption=cap,
                    text_vec=await emb.embed(cap),
                    thumb_path=f"/tmp/{tid}.jpg",
                    frame_hash="0" * 16,
                )
            )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app


def _use_script(app, script: ScriptedAnthropic | None, **cfg) -> ClaudeAgent:
    key = "k" if script is not None else ""
    agent = ClaudeAgent(
        _agent_settings(anthropic_api_key=key, **cfg), app.state.recall, client=script.client() if script else None
    )
    app.state.agent = agent
    return agent


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def _stream(c, body: dict) -> list[tuple[str, dict]]:
    r = await c.post("/v1/chat/stream", json=body)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    return _parse_sse(r.text)


def _same_call(q: dict) -> dict:
    """A request body minus `stream` and the clock sentence, wherever it is (separate turns can straddle a second)."""
    out = {k: v for k, v in q.items() if k != "stream"}
    return json.loads(re.sub(r"The wearer's local time is [^\"\\\\]*", "CLOCK", json.dumps(out)))


def _breakpoints(q: dict) -> list[str]:
    """Where the cache_control markers are in a request body."""
    out = []
    if isinstance(q["system"], list):
        out += [f"system[{i}]" for i, b in enumerate(q["system"]) if b.get("cache_control")]
    for m_i, m in enumerate(q["messages"]):
        if isinstance(m["content"], list):
            marked = [(b_i, b["type"]) for b_i, b in enumerate(m["content"]) if b.get("cache_control")]
            out += [f"messages[{m_i}][{b_i}]:{t}" for b_i, t in marked]
    return out


def _unmarked(msgs: list[dict]) -> list[dict]:
    return json.loads(re.sub(r', "cache_control": \{"type": "ephemeral"\}', "", json.dumps(msgs)))


async def _asgi_stream(app, body: dict) -> list[tuple[float, bytes]]:
    """POST /v1/chat/stream straight into the ASGI app, timestamping every body chunk as the server sends it
    (httpx's ASGITransport collects the whole body first, so it cannot show pacing)."""
    raw = json.dumps(body).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/stream",
        "raw_path": b"/v1/chat/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"test"), (b"content-type", b"application/json")],
        "server": ("test", 80),
        "client": ("127.0.0.1", 1),
    }
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await asyncio.Event().wait()  # the client never disconnects

    chunks: list[tuple[float, bytes]] = []

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            chunks.append((time.perf_counter(), message["body"]))

    await app(scope, receive, send)
    return chunks


ANSWER = "Your keys are on the kitchen counter, about twenty minutes ago. <cites>t_keys</cites>"


def _two_rounds(preamble: str | None = None) -> list[dict]:
    first = ([text(preamble)] if preamble else []) + [tool_use({"query": "keys kitchen"})]
    return [_message(first, stop_reason="tool_use"), _message([text(ANSWER)])]


def test_trailer_filter_strips_cites_across_any_chunking():
    s = "Keys are on the counter. <cites>t_keys,t_mug</cites>"
    for n in range(1, len(s) + 1):
        f = _TrailerFilter()
        out = "".join(f.feed(s[i : i + n]) for i in range(0, len(s), n)) + f.flush()
        assert out == "Keys are on the counter. ", n
    f = _TrailerFilter()
    assert f.feed("a < b <c") == "a < b "  # "<c" might start "<cites>": held back
    assert f.feed("at") + f.flush() == "<cat"


async def test_stream_event_order_and_shape(app_client):
    c, app = app_client
    script = ScriptedAnthropic(_two_rounds(preamble="Let me check."))
    _use_script(app, script)
    events = await _stream(c, {"session_id": "s", "device_id": "dev1", "text": "where are my keys?"})
    names = [n for n, _ in events]

    # preamble streamed then withdrawn when the round turned into a tool call; candidates before answer text
    r_i = names.index("reset")
    assert set(names[:r_i]) == {"delta"} and "".join(d["text"] for _, d in events[:r_i]) == "Let me check."
    assert names[r_i + 1] == "retrieval"
    events = events[r_i:]
    names = names[r_i:]
    assert set(names[2:-1]) == {"delta"} and len(names[2:-1]) > 3  # chunked
    assert names[-1] == "final" and names.count("reset") == 1

    retrieval = events[1][1]
    assert retrieval["confidence"] in ("high", "medium", "low", "not_found")
    assert retrieval["candidates"][0]["id"] == "t_keys"
    for cand in retrieval["candidates"]:
        assert set(cand) == {"id", "ts", "caption", "thumb_url", "replay_url"}
        assert cand["thumb_url"] == f"/v1/thumbs/{cand['id']}.jpg"

    streamed = "".join(d["text"] for n, d in events[2:-1])
    final = events[-1][1]
    assert "<cites" not in streamed
    assert streamed.strip() == final["answer"] == "Your keys are on the kitchen counter, about twenty minutes ago."
    assert ChatResponse.model_validate(final).cited[0].id == "t_keys"

    # same Messages API requests as the non-streaming path, plus stream=true
    assert all(r.get("stream") is True for r in script.requests)


async def test_final_event_equals_post_chat_body(app_client):
    c, app = app_client
    body = {"session_id": "s", "device_id": "dev1", "text": "where are my keys?"}

    plain = ScriptedAnthropic(_two_rounds())
    _use_script(app, plain)
    r = await c.post("/v1/chat", json=body)
    assert r.status_code == 200
    chat_json = r.json()

    streamed = ScriptedAnthropic(_two_rounds())
    _use_script(app, streamed)
    events = await _stream(c, body)
    assert events[-1][0] == "final"
    assert events[-1][1] == chat_json
    assert list(events[-1][1]) == list(ChatResponse.model_fields)

    # identical model requests apart from the stream flag (same pipeline, same answer)
    assert [_same_call(q) for q in streamed.requests] == [_same_call(q) for q in plain.requests]
    assert all("stream" not in q for q in plain.requests)


async def test_stream_no_key_fallback_matches_post_chat(app_client):
    c, app = app_client
    body = {"session_id": "s", "text": "keys kitchen counter"}
    _use_script(app, None)
    chat_json = (await c.post("/v1/chat", json=body)).json()
    events = await _stream(c, body)
    assert [n for n, _ in events] == ["retrieval", "delta", "final"]
    assert events[1][1]["text"] == chat_json["answer"]
    assert events[-1][1] == chat_json


async def test_stream_api_failure_resets_then_falls_back_like_post_chat(app_client):
    c, app = app_client
    body = {"session_id": "s", "text": "keys kitchen counter"}
    _use_script(app, ScriptedAnthropic([_message([tool_use({"query": "keys"})], "tool_use"), {"_status": 500}]))
    chat_json = (await c.post("/v1/chat", json=body)).json()

    _use_script(app, ScriptedAnthropic([_message([tool_use({"query": "keys"})], "tool_use"), {"_status": 500}]))
    events = await _stream(c, body)
    names = [n for n, _ in events]
    assert names[0] == "retrieval"  # Claude's tool call ran before the second call failed
    assert names[-2:] == ["delta", "final"] and "error" not in names
    assert events[-1][1] == chat_json and events[-1][1]["answer"] != NOT_FOUND_ANSWER


async def test_stream_mid_turn_exception_is_a_final_error_event(app_client):
    c, app = app_client
    _use_script(app, ScriptedAnthropic(_two_rounds()))

    async def boom(**_kw):
        raise RuntimeError("retriever exploded")

    app.state.agent.recall = boom
    events = await _stream(c, {"session_id": "s", "text": "where are my keys?"})
    assert events[-1] == ("error", {"detail": "RuntimeError: retriever exploded"})
    assert "final" not in [n for n, _ in events]


async def test_stream_bad_requests_fail_like_post_chat(app_client):
    c, _ = app_client
    for path in ("/v1/chat", "/v1/chat/stream"):
        assert (await c.post(path, json={"text": "hi"})).status_code == 422
        assert (await c.post(path, json={"session_id": "s"})).status_code == 400
        assert (await c.post(path, content=b"x", headers={"content-type": "text/plain"})).status_code == 415


async def test_post_chat_unchanged_non_streaming_requests(app_client):
    c, app = app_client
    script = ScriptedAnthropic(_two_rounds())
    _use_script(app, script)
    r = await c.post("/v1/chat", json={"session_id": "s", "text": "where are my keys?"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    assert r.json()["answer"] == "Your keys are on the kitchen counter, about twenty minutes ago."
    assert len(script.requests) == 2 and all("stream" not in q for q in script.requests)


async def test_stream_forwards_deltas_as_the_api_produces_them(app_client):
    """Regression for batched answers: a delta must leave the server while the API is still writing the rest
    (TTS_STREAM default on, mock TTS), not be flushed together with `final`."""
    c, app = app_client
    script = PacedAnthropic(_two_rounds())
    _use_script(app, script)
    chunks = await _asgi_stream(app, {"session_id": "s", "device_id": "dev1", "text": "where are my keys?"})
    deltas = [t for t, data in chunks if data.startswith(b"event: delta")]
    assert chunks[-1][1].startswith(b"event: final") and len(deltas) >= 5 and len(script.produced) >= len(deltas)
    assert deltas[0] < script.produced[-1], "first delta was held until the API finished"
    assert deltas[0] - script.produced[0] < 0.1  # forwarded at once ...
    assert deltas[-1] - deltas[0] > 0.5 * (script.produced[-1] - script.produced[0])  # ... and spread like the source


async def test_stream_logs_per_call_delivery_timing(app_client, caplog):
    c, app = app_client
    _use_script(app, ScriptedAnthropic([_message([text(ANSWER)])]), recall_first=True)
    with caplog.at_level("INFO", logger="megamind.agent"):
        await _stream(c, {"session_id": "s", "text": "where are my keys?"})
    lines = [r.getMessage() for r in caplog.records if " stream: " in r.getMessage()]
    assert len(lines) == 1
    pat = r"first_event_ms=(\d+) first_text_ms=(\d+) text_deltas=(\d+) last_text_ms=(\d+) total_ms=(\d+)"
    m = re.search(pat, lines[0])
    assert m and int(m.group(3)) > 3 and int(m.group(1)) <= int(m.group(2)) <= int(m.group(4)) <= int(m.group(5))
    turn = [r.getMessage() for r in caplog.records if "llm_calls=" in r.getMessage()]
    assert len(turn) == 1 and "cache_read=0 cache_write=0" in turn[0]


CACHE_USAGE = {
    "input_tokens": 100,
    "output_tokens": 20,
    "cache_creation_input_tokens": 700,
    "cache_read_input_tokens": 1500,
}


async def test_prompt_cache_breakpoints_read_back_across_rounds_and_turns(app_client):
    c, app = app_client
    rounds = [
        _message([tool_use({"query": "keys kitchen"})], "tool_use", usage=CACHE_USAGE),
        _message([text(ANSWER)], usage=CACHE_USAGE),
    ]
    script = ScriptedAnthropic(rounds)
    agent = _use_script(app, script)  # default on
    resp, stats = await agent.chat_with_stats("s", "where are my keys?", "dev1")
    assert resp.answer == "Your keys are on the kitchen counter, about twenty minutes ago."
    assert stats.extra["cache_creation_input_tokens"] == 1400 and stats.extra["cache_read_input_tokens"] == 3000
    q1, q2 = script.requests
    for q in (q1, q2):
        assert "cache_control" not in q and len(_breakpoints(q)) <= 4

    # the system prompt has no clock (identical bytes every turn) and is marked: tools + system cache together
    static = render_static_system_prompt("America/Los_Angeles")
    assert q1["system"] == q2["system"] == [{"type": "text", "text": static, "cache_control": EPHEMERAL}]
    assert "local time is" not in static and "America/Los_Angeles" in static and "ISO-8601" in static
    # the clock rides with the question, after the (cached) history
    clock, question = q1["messages"][0]["content"]
    assert clock["text"].startswith("The wearer's local time is ") and "in UTC that is" in clock["text"]
    assert "cache_control" not in clock
    assert question == {"type": "text", "text": "where are my keys?", "cache_control": EPHEMERAL}
    assert _breakpoints(q1) == ["system[0]", "messages[0][1]:text"]
    # the tool round: the marker moves to the tool result (the API finds the question's entry behind it), sent as
    # text by CLAUDE_TOOL_RESULTS_AS_TEXT (default): ["tool result:", payload]
    assert _breakpoints(q2) == ["system[0]", "messages[2][1]:text"]
    assert q2["messages"][0]["content"] == [clock, {"type": "text", "text": "where are my keys?"}]
    assert [m["role"] for m in q2["messages"]] == ["user", "assistant", "user"]

    # next turn: a marker on the last block of the previous turn's answer, so this turn reads back all of turn 1
    script2 = ScriptedAnthropic([_message([text("About twenty minutes ago. <cites>t_keys</cites>")])])
    agent._client = script2.client()
    resp2, _ = await agent.chat_with_stats("s", "when was that?", "dev1")
    assert resp2.answer == "About twenty minutes ago." and resp2.cited[0].id == "t_keys"
    (q3,) = script2.requests
    assert _breakpoints(q3) == ["system[0]", "messages[3][0]:text", "messages[4][1]:text"]
    assert q3["messages"][3]["content"][0]["text"] == ANSWER  # turn 1's final assistant message
    assert _unmarked(q3["messages"][:3]) == _unmarked(q2["messages"])  # turn 1 re-sent byte for byte (prefix)
    assert q3["messages"][4]["content"][0]["text"].startswith("The wearer's local time is ")
    # the stored history keeps the clock blocks (when each question was asked) but never a marker
    assert "cache_control" not in json.dumps(agent.histories["s"])
    assert agent.histories["s"][0][0]["content"][1] == {"type": "text", "text": "where are my keys?"}

    # recall_first (the common path): one user message = clock, question, label, result (#178: no tool_result,
    # which Sonnet answers in a burst) -> marked on its last block, by position
    script3 = ScriptedAnthropic([_message([text(ANSWER)])])
    _use_script(app, script3, recall_first=True)
    await c.post("/v1/chat", json={"session_id": "r", "text": "where are my keys?"})
    (q4,) = script3.requests
    assert len(q4["messages"]) == 1 and "tool_result" not in json.dumps(q4["messages"])
    blocks = q4["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text"] * 4 and blocks[2]["text"].startswith("recall_memory(")
    assert blocks[0]["text"].startswith("The wearer's local time is ") and blocks[1]["text"] == "where are my keys?"
    assert _breakpoints(q4) == ["system[0]", "messages[0][3]:text"]
    assert app.state.agent.histories["r"][0][0] == _unmarked(q4["messages"])[0]  # stored as sent, minus the marker

    # streaming sends the same bodies apart from `stream`
    plain, streamed = ScriptedAnthropic(_two_rounds()), ScriptedAnthropic(_two_rounds())
    _use_script(app, plain)
    await c.post("/v1/chat", json={"session_id": "t", "text": "where are my keys?"})
    _use_script(app, streamed)
    await _stream(c, {"session_id": "t", "text": "where are my keys?"})
    assert [_same_call(q) for q in streamed.requests] == [_same_call(q) for q in plain.requests]


async def test_prompt_cache_off_sends_the_pre_caching_request_bodies(app_client):
    c, app = app_client
    body = {"session_id": "s", "text": "where are my keys?"}
    script = ScriptedAnthropic(_two_rounds())
    _use_script(app, script, claude_prompt_cache=False)
    r = await c.post("/v1/chat", json=body)
    assert r.json()["answer"] == "Your keys are on the kitchen counter, about twenty minutes ago."
    assert all("cache_control" not in json.dumps(q) for q in script.requests)
    assert script.requests[0]["messages"] == [{"role": "user", "content": "where are my keys?"}]
    assert isinstance(script.requests[0]["system"], str) and "in UTC that is" in script.requests[0]["system"]
    assert app.state.agent.histories["s"][-1][0] == {"role": "user", "content": "where are my keys?"}


def test_ui_and_xr_stream_with_post_chat_fallback():
    root = Path(__file__).resolve().parents[1] / "megamind"
    for src in ((root / "ui" / "index.html").read_text(), (root / "xr" / "static" / "app.js").read_text()):
        assert "/v1/chat/stream" in src
        assert "'/v1/chat'" in src or '"/v1/chat"' in src  # fallback for servers without the stream
        assert all(ev in src for ev in ("'delta'", "'reset'", "'final'", "'error'")) or all(
            ev in src for ev in ('"delta"', '"reset"', '"final"', '"error"')
        )


async def test_stream_never_shows_a_tool_call_written_as_text(app_client):
    # the model imitates the text-flattened history and writes `recall_memory({...})` instead of a tool_use block:
    # nothing of it is streamed (so TTS never speaks it), the call runs as a tool round, the answer follows
    c, app = app_client
    script = ScriptedAnthropic([_message([text('recall_memory({"query": "keys kitchen"})')]), _message([text(ANSWER)])])
    _use_script(app, script)
    events = await _stream(c, {"session_id": "s1", "device_id": "dev1", "text": "where did I leave my keys?"})

    deltas = "".join(d["text"] for e, d in events if e == "delta")
    assert "recall_memory" not in deltas and deltas.startswith("Your keys are on the kitchen counter")
    assert not [e for e, _ in events if e == "reset"]  # nothing wrong was shown, so nothing to retract
    final = next(d for e, d in events if e == "final")
    assert final["answer"].startswith("Your keys are on the kitchen counter")
    assert [x["id"] for x in final["cited"]] == ["t_keys"] and len(final["tool_calls"]) == 1
    assert len(script.requests) == 2
