"""Claude orchestrator tests with recorded Anthropic Messages API responses (httpx.MockTransport)."""

from __future__ import annotations

import json

import anthropic
import httpx2 as httpx
import pytest

from megamind.agent.claude_agent import MAX_TOOL_ROUNDS, MAX_TURNS, NOT_FOUND_ANSWER, ClaudeAgent
from megamind.agent.reader import CaptionOnlyReader
from megamind.agent.tools import RecallMemory
from megamind.config import Settings
from megamind.ingest.enrich import MockTextEmbedder
from megamind.models import MemoryToken
from megamind.store.memory_store import InMemoryStore

NOW = 1_731_000_000.0


def _message(content: list[dict], stop_reason: str = "end_turn", in_tok: int = 100, out_tok: int = 20) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def text(s: str) -> dict:
    return {"type": "text", "text": s}


def tool_use(args: dict, tid: str = "toolu_1") -> dict:
    return {"type": "tool_use", "id": tid, "name": "recall_memory", "input": args}


class ScriptedAnthropic:
    """Replays a list of Messages API responses; records every request body."""

    def __init__(self, responses: list[dict | Exception]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if isinstance(nxt, dict) and nxt.get("_status"):
            err = {"type": "error", "error": {"type": "api_error", "message": "x"}}
            return httpx.Response(nxt["_status"], json=err)
        return httpx.Response(200, json=nxt)

    def client(self) -> anthropic.AsyncAnthropic:
        http = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        return anthropic.AsyncAnthropic(api_key="test", http_client=http, max_retries=0)


async def _store_with_tokens() -> InMemoryStore:
    store = InMemoryStore()
    emb = MockTextEmbedder()
    specs = [
        ("t_keys", NOW - 20 * 60, "Keys on the kitchen counter next to the coffee machine", ""),
        ("t_board", NOW - 60 * 60, "A whiteboard in a meeting room", "Standup 10am\nShip demo Sunday"),
        ("t_mug", NOW - 5 * 60, "A blue coffee mug on a wooden desk beside a laptop", ""),
    ]
    for tid, ts, cap, ocr in specs:
        await store.add(
            MemoryToken(
                id=tid,
                device_id="dev1",
                ts=ts,
                caption=cap,
                ocr_text=ocr,
                entities=[],
                text_vec=await emb.embed(f"{cap} {ocr}"),
                thumb_path=f"/tmp/{tid}.jpg",
                frame_hash="0" * 16,
            )
        )
    return store


@pytest.fixture
async def recall():
    store = await _store_with_tokens()
    return RecallMemory(store, MockTextEmbedder(), None, CaptionOnlyReader())


def _agent(recall: RecallMemory, script: ScriptedAnthropic | None, key: str = "k") -> ClaudeAgent:
    settings = Settings(anthropic_api_key=key, _env_file=None)
    return ClaudeAgent(settings, recall, client=script.client() if script else None)


async def test_no_key_uses_fallback(recall):
    agent = _agent(recall, None, key="")
    assert not agent.enabled
    resp, stats = await agent.chat_with_stats("s", "keys kitchen counter", "dev1")
    assert stats.path == "fallback"
    assert resp.tool_calls[0].name == "recall_memory"
    assert resp.cited and resp.cited[0].id == "t_keys"


async def test_single_tool_call_with_claude_cites(recall):
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "keys", "k": 6})], stop_reason="tool_use", in_tok=500, out_tok=40),
            _message([text("Your keys are on the kitchen counter, about twenty minutes ago. <cites>t_keys</cites>")]),
        ]
    )
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s1", "where did I leave my keys?", "dev1")

    assert resp.answer == "Your keys are on the kitchen counter, about twenty minutes ago."
    assert "<cites>" not in resp.answer
    assert [c.id for c in resp.cited] == ["t_keys"]
    assert resp.tool_calls == [resp.tool_calls[0]] and resp.tool_calls[0].args == {"query": "keys", "k": 6}
    assert resp.tool_calls[0].confidence == "low"  # CaptionOnlyReader
    assert stats.model_calls == 2 and stats.tool_calls == 1 and stats.cites_from == "claude"
    assert stats.input_tokens == 600 and stats.output_tokens == 60

    # request shape: system prompt, tool schema, tool_result fed back
    first, second = script.requests
    assert first["model"] == "claude-sonnet-4-6"
    assert first["tools"][0]["name"] == "recall_memory"
    assert "Current time (UTC)" in first["system"]
    assert first["messages"] == [{"role": "user", "content": "where did I leave my keys?"}]
    assert second["messages"][1]["role"] == "assistant"
    assert second["messages"][1]["content"][0]["type"] == "tool_use"
    tr = second["messages"][2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "toolu_1"
    payload = json.loads(tr["content"])
    assert payload["confidence"] == "low"
    assert payload["cited"][0]["id"] == "t_keys"
    assert "ago" in payload["cited"][0]["when"]


async def test_multi_call_widens_window_and_history_is_kept(recall):
    script = ScriptedAnthropic(
        [
            _message(
                [tool_use({"query": "whiteboard", "time_from": "2030-01-01T00:00:00+00:00", "k": 4}, "toolu_a")],
                stop_reason="tool_use",
            ),
            _message([tool_use({"query": "whiteboard", "time_from": None, "k": 12}, "toolu_b")], "tool_use"),
            _message([text("The whiteboard said standup at ten and ship the demo Sunday. <cites>t_board</cites>")]),
            _message([text("You looked at it about an hour ago.")]),
        ]
    )
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s2", "what did the whiteboard say?", "dev1")
    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0].confidence == "not_found"  # window in the future -> nothing
    assert resp.tool_calls[1].confidence == "low"
    assert [c.id for c in resp.cited] == ["t_board"]
    assert stats.model_calls == 3

    # second turn: history carried into the request, tool blocks included
    resp2, _ = await agent.chat_with_stats("s2", "when was that?", "dev1")
    assert resp2.answer == "You looked at it about an hour ago."
    assert resp2.cited == []  # no trailer, no tool call -> nothing to cite
    msgs = script.requests[-1]["messages"]
    assert msgs[0]["content"] == "what did the whiteboard say?"
    assert msgs[-1] == {"role": "user", "content": "when was that?"}
    assert any(m["role"] == "user" and isinstance(m["content"], list) for m in msgs)  # tool_result kept


async def test_not_found_answer_drops_citations(recall):
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "bicycle"})], stop_reason="tool_use"),
            _message([text(f"{NOT_FOUND_ANSWER} <cites>t_keys</cites>")]),
        ]
    )
    agent = _agent(recall, script)
    resp = await agent.chat("s3", "where is my bicycle?", "dev1")
    assert resp.answer == NOT_FOUND_ANSWER
    assert resp.cited == []


async def test_reader_citations_used_when_claude_omits_trailer(recall):
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "mug"})], stop_reason="tool_use"),
            _message([text("Your mug is on the desk beside the laptop.")]),
        ]
    )
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s4", "where is my mug?", "dev1")
    assert stats.cites_from == "reader"
    assert resp.cited and resp.cited[0].id == "t_mug"


async def test_tool_round_cap(recall):
    script = ScriptedAnthropic(
        [_message([tool_use({"query": "x"}, f"toolu_{i}")], stop_reason="tool_use") for i in range(MAX_TOOL_ROUNDS + 1)]
    )
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s5", "loop forever", "dev1")
    assert stats.model_calls == MAX_TOOL_ROUNDS + 1
    assert stats.tool_calls == MAX_TOOL_ROUNDS
    assert resp.answer == NOT_FOUND_ANSWER


async def test_bad_tool_args_are_reported_not_raised(recall):
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "keys", "time_from": "yesterday-ish"})], stop_reason="tool_use"),
            _message([text(NOT_FOUND_ANSWER)]),
        ]
    )
    agent = _agent(recall, script)
    resp = await agent.chat("s6", "keys?", "dev1")
    assert resp.tool_calls[0].confidence == "not_found"
    payload = json.loads(script.requests[1]["messages"][2]["content"][0]["content"])
    assert payload["evidence"].startswith("Tool error")


async def test_api_error_degrades_to_fallback(recall):
    script = ScriptedAnthropic([{"_status": 500}])
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s7", "keys kitchen counter", "dev1")
    assert stats.path == "fallback" and stats.error
    assert resp.cited and resp.cited[0].id == "t_keys"


async def test_connection_error_degrades_to_fallback(recall):
    script = ScriptedAnthropic([httpx.ConnectError("boom")])
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s8", "keys kitchen counter", "dev1")
    assert stats.path == "fallback"
    assert resp.answer


async def test_history_capped(recall):
    script = ScriptedAnthropic([_message([text(f"reply {i}")]) for i in range(MAX_TURNS + 3)])
    agent = _agent(recall, script)
    for i in range(MAX_TURNS + 3):
        await agent.chat("s9", f"q{i}", "dev1")
    assert len(agent.histories["s9"]) == MAX_TURNS
    assert script.requests[-1]["messages"][0]["content"] == "q2"  # 13 chats, last request carries the 9 previous turns
    agent.reset("s9")
    assert "s9" not in agent.histories


def test_claude_model_default_is_not_the_retiring_sonnet_4_5_and_is_overridable(monkeypatch):
    from megamind.config import Settings

    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    assert Settings(_env_file=None).claude_model == "claude-sonnet-4-6"
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    assert Settings(_env_file=None).claude_model == "claude-sonnet-5-5"
