"""POST /v1/chat/stream (SSE) against recorded Anthropic responses, streamed and non-streamed (httpx.MockTransport)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic
import httpx2
import pytest
from httpx import ASGITransport, AsyncClient

from megamind.agent.claude_agent import NOT_FOUND_ANSWER, ClaudeAgent, _TrailerFilter
from megamind.config import Settings
from megamind.ingest.enrich import MockTextEmbedder
from megamind.main import create_app
from megamind.models import ChatResponse, MemoryToken

NOW = time.time()


def _message(content: list[dict], stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def text(s: str) -> dict:
    return {"type": "text", "text": s}


def tool_use(args: dict, tid: str = "toolu_1") -> dict:
    return {"type": "tool_use", "id": tid, "name": "recall_memory", "input": args}


def _sse_body(msg: dict, chunk: int = 7) -> bytes:
    """A Messages API response as the streaming wire format, text split into `chunk`-char deltas."""

    def ev(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"

    start = {**msg, "content": [], "stop_reason": None, "usage": {"input_tokens": 100, "output_tokens": 1}}
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
    return "".join(out).encode()


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
            return httpx2.Response(200, content=_sse_body(nxt), headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=nxt)

    def client(self) -> anthropic.AsyncAnthropic:
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))
        return anthropic.AsyncAnthropic(api_key="test", http_client=http, max_retries=0)


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
    assert [{k: v for k, v in q.items() if k != "stream"} for q in streamed.requests] == plain.requests
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


async def test_prompt_cache_only_on_the_tool_picking_call(app_client):
    c, app = app_client
    body = {"session_id": "s", "text": "where are my keys?"}

    script = ScriptedAnthropic(_two_rounds())
    _use_script(app, script)
    await c.post("/v1/chat", json=body)
    assert script.requests[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in script.requests[1]  # the answer round only reads it

    script = ScriptedAnthropic(_two_rounds())
    _use_script(app, script, claude_prompt_cache=False)
    await c.post("/v1/chat", json=body)
    assert all("cache_control" not in q for q in script.requests)

    # recall_first: the first call already carries a tool result (usually the only call): no cache write
    script = ScriptedAnthropic([_message([text(ANSWER)])])
    _use_script(app, script, recall_first=True)
    await c.post("/v1/chat", json=body)
    assert len(script.requests) == 1 and "cache_control" not in script.requests[0]


def test_ui_and_xr_stream_with_post_chat_fallback():
    root = Path(__file__).resolve().parents[1] / "megamind"
    for src in ((root / "ui" / "index.html").read_text(), (root / "xr" / "static" / "app.js").read_text()):
        assert "/v1/chat/stream" in src
        assert "'/v1/chat'" in src or '"/v1/chat"' in src  # fallback for servers without the stream
        assert all(ev in src for ev in ("'delta'", "'reset'", "'final'", "'error'")) or all(
            ev in src for ev in ('"delta"', '"reset"', '"final"', '"error"')
        )
