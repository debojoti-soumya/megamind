"""Object markers: "where is X" -> X's box in a cited frame (ChatResponse.markers)."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from openai import AsyncOpenAI
from test_claude_agent import ScriptedAnthropic, _message
from test_live_memory import _fallback_agent, _post_burst

from megamind.agent.locate import LocateSettings, Locator, _to_marker, build_locator, is_where_question
from megamind.config import Settings
from megamind.live import _frame_id
from megamind.models import Citation


def report(**kw) -> dict:
    return _message([{"type": "tool_use", "id": "toolu_1", "name": "report_location", "input": kw}], "tool_use")


def _cfg(**kw) -> LocateSettings:
    return LocateSettings(_env_file=None, **{"provider": "anthropic", **kw})


class ScriptedOpenAI:
    """Replays chat.completions responses whose tool call reports `args`; records request bodies."""

    def __init__(self, responses: list[dict | int]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        nxt = self.responses.pop(0)
        if isinstance(nxt, int):
            return httpx.Response(nxt, json={"error": {"message": "x", "type": "server_error"}})
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "report_location", "arguments": json.dumps(nxt)},
        }
        msg = {"role": "assistant", "content": None, "tool_calls": [call]}
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4.1",
                "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
            },
        )

    def client(self) -> AsyncOpenAI:
        http = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        return AsyncOpenAI(api_key="test", http_client=http, max_retries=0)


def _settings() -> Settings:
    return Settings(_env_file=None, anthropic_api_key="")


@pytest.mark.parametrize(
    "q",
    [
        "where is my pen right now?",
        "Where's the remote",
        "wheres my phone",
        "find my keys",
        "can you locate the mug",
        "show me the charger",
        "point to the scissors",
    ],
)
def test_where_questions(q):
    assert is_where_question(q)


@pytest.mark.parametrize("q", ["what am I looking at right now?", "what did I eat for lunch", "", None])
def test_not_where_questions(q):
    assert not is_where_question(q)


def _cit(id_="now_a", ts=100.0) -> Citation:
    return Citation(id=id_, ts=ts, caption="c", thumb_url=f"/v1/live/{id_}.jpg")


def test_to_marker_normalizes_and_clamps():
    shown = [(_cit(), "dev1", 320, 240)]
    m = _to_marker(
        {"found": True, "image": 0, "label": "blue pen", "box": [160, 120, 400, 60], "confidence": 0.9}, shown, 0.5
    )
    assert m is not None
    assert m.box == [0.5, 0.25, 1.0, 0.5]  # clamped to the frame and reordered
    assert (m.x, m.y) == (0.75, 0.375)
    assert (m.label, m.token_id, m.device_id, m.ts, m.confidence) == ("blue pen", "now_a", "dev1", 100.0, 0.9)


@pytest.mark.parametrize(
    "out",
    [
        None,
        {"found": False},
        {"found": True},
        {"found": True, "image": 3, "box": [0, 0, 10, 10]},
        {"found": True, "box": [0, 0, 10, 10], "confidence": 0.2},
        {"found": True, "box": [5, 5, 5, 9], "confidence": 0.9},
        {"found": True, "box": [0, 0, 10, 10]},
        {"found": True, "box": ["a", 0, 1, 1]},
    ],
)
def test_to_marker_rejects(out):
    assert _to_marker(out, [(_cit(), "dev1", 320, 240)], 0.5) is None


def test_to_marker_needs_image_index_when_several_frames():
    shown = [(_cit("now_a"), "dev1", 320, 240), (_cit("now_b"), "dev1", 320, 240)]
    out = {"found": True, "box": [0, 0, 10, 10], "confidence": 0.9}
    assert _to_marker(out, shown, 0.5) is None
    assert _to_marker({**out, "image": 1}, shown, 0.5).token_id == "now_b"


def test_candidates_prefer_live_then_newest():
    loc = Locator(_settings(), _cfg(max_frames=2, max_age_s=600), load=None)
    cited = [_cit("t_old", 10.0), _cit("t_new", 950.0), _cit("now_b", 990.0), _cit("t_mid", 900.0)]
    assert [c.id for c in loc.candidates(cited, now=1000.0)] == ["now_b", "t_new"]
    assert [c.id for c in loc.candidates(cited, now=2000.0)] == []


async def test_disabled_without_client_or_for_other_questions():
    assert not Locator(_settings(), _cfg(), load=None).enabled
    fake = ScriptedAnthropic([])
    loc = Locator(_settings(), _cfg(), load=None, client=fake.client())
    assert await loc.markers("what am I looking at right now?", "a desk", [_cit()]) == []
    assert fake.requests == []


async def test_chat_returns_marker_for_live_frame(client):
    c, app = client
    app.state.agent = _fallback_agent(app)
    ts = await _post_burst(c, time.time() - 4.0)
    fake = ScriptedAnthropic([report(found=True, image=0, label="pen", box=[32, 24, 96, 72], confidence=0.8)])
    app.state.locator = build_locator(app.state, _settings(), _cfg(), client=fake.client())

    r = await c.post("/v1/chat", json={"session_id": "s", "device_id": "dev1", "text": "where is my pen right now?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cited"][0]["id"] == _frame_id("dev1", ts[-1])
    (m,) = body["markers"]
    assert m["token_id"] == body["cited"][0]["id"] and m["device_id"] == "dev1" and m["ts"] == ts[-1]
    assert m["box"] == [0.1, 0.1, 0.3, 0.3] and (m["x"], m["y"]) == (0.2, 0.2) and m["label"] == "pen"

    (req,) = fake.requests
    assert req["tool_choice"] == {"type": "tool", "name": "report_location"}
    content = req["messages"][0]["content"]
    images = [b for b in content if b["type"] == "image"]
    assert 1 <= len(images) <= 2
    assert content[0]["text"] == "snapshot 0 (320x240 px):"
    assert "where is my pen right now?" in content[-1]["text"]


async def test_chat_survives_locator_failure_and_not_found(client):
    c, app = client
    app.state.agent = _fallback_agent(app)
    await _post_burst(c, time.time() - 4.0)
    fake = ScriptedAnthropic([{"_status": 500}, report(found=False)])
    app.state.locator = build_locator(app.state, _settings(), _cfg(), client=fake.client())
    for _ in range(2):
        r = await c.post("/v1/chat", json={"session_id": "s", "device_id": "dev1", "text": "where's the pen right now"})
        assert r.status_code == 200, r.text
        assert r.json()["markers"] == [] and r.json()["cited"]
    assert len(fake.requests) == 2


async def test_non_where_chat_makes_no_locate_call(client):
    c, app = client
    app.state.agent = _fallback_agent(app)
    await _post_burst(c, time.time() - 4.0)
    fake = ScriptedAnthropic([])
    app.state.locator = build_locator(app.state, _settings(), _cfg(), client=fake.client())
    body = {"session_id": "s", "device_id": "dev1", "text": "what am I looking at right now"}
    r = await c.post("/v1/chat", json=body)
    assert r.status_code == 200 and r.json()["markers"] == [] and fake.requests == []


async def test_openai_provider_is_the_default_and_returns_marker(client):
    c, app = client
    app.state.agent = _fallback_agent(app)
    ts = await _post_burst(c, time.time() - 4.0)
    assert LocateSettings(_env_file=None).provider == "openai"
    fake = ScriptedOpenAI(
        [{"found": True, "image": 0, "label": "pen", "box": [160, 0, 320, 120], "confidence": 0.8}, 500]
    )
    loc = build_locator(app.state, _settings(), LocateSettings(_env_file=None), client=fake.client())
    assert loc.provider == "openai" and loc.model == "gpt-4.1"
    app.state.locator = loc

    body = {"session_id": "s", "device_id": "dev1", "text": "where is my pen right now?"}
    r = await c.post("/v1/chat", json=body)
    assert r.status_code == 200, r.text
    (m,) = r.json()["markers"]
    assert m["box"] == [0.5, 0.0, 1.0, 0.5] and m["ts"] == ts[-1] and m["device_id"] == "dev1"
    req = fake.requests[0]
    assert req["model"] == "gpt-4.1"
    assert req["tool_choice"] == {"type": "function", "function": {"name": "report_location"}}
    parts = req["messages"][0]["content"]
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    r = await c.post("/v1/chat", json=body)  # provider error -> no markers, turn still answers
    assert r.status_code == 200 and r.json()["markers"] == [] and r.json()["answer"]


def test_provider_needs_its_own_key():
    s = Settings(_env_file=None, anthropic_api_key="sk-ant", openai_api_key="")
    assert not Locator(s, LocateSettings(_env_file=None, provider="openai"), load=None).enabled
    assert Locator(s, LocateSettings(_env_file=None, provider="anthropic"), load=None).enabled
    s = Settings(_env_file=None, anthropic_api_key="", openai_api_key="sk-oa")
    assert Locator(s, LocateSettings(_env_file=None), load=None).enabled
    assert not Locator(s, LocateSettings(_env_file=None, enabled=False), load=None).enabled
