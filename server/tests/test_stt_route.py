"""POST /v1/stt: transcript only (no agent turn, no TTS) + optional server-side voice-command matching."""

from __future__ import annotations

import io
import json
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from megamind.agent.stt import STT, MockSTT
from megamind.agent.voice import match_command, normalize_command, parse_hints, stt_label
from megamind.config import Settings
from megamind.main import create_app

HINTS = ["open menu", "close menu", "pause recording", "start recording", "bigger text", "show controls"]


def _wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class ScriptedSTT(STT):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def transcribe(self, wav: bytes) -> str:
        self.calls += 1
        return self.text


class BrokenSTT(STT):
    async def transcribe(self, wav: bytes) -> str:
        raise RuntimeError("provider down")


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
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app


def test_normalize_matches_headset_matcher():
    assert normalize_command("Hey Megamind, please open the menu!") == "open menu"
    assert normalize_command("Pause  recording.") == "pause recording"
    assert normalize_command("...") == ""


def test_match_command_exact_fuzzy_and_rejections():
    assert match_command("Open the menu.", HINTS).model_dump() == {"phrase": "open menu", "score": 1.0}
    m = match_command("paws recording", HINTS)
    assert m is not None and m.phrase == "pause recording" and 0.8 <= m.score < 1.0
    assert match_command("where did I leave my keys this morning", HINTS) is None  # too long: a question
    assert match_command("what time is it", HINTS) is None  # nothing close
    assert match_command("", HINTS) is None
    assert match_command("open menu", []) is None


@pytest.mark.parametrize(
    ("transcript", "hints"),
    [
        ("start recording", ["stop recording"]),
        ("unmute", ["mute"]),
        ("mute", ["unmute"]),
        ("disable captions", ["enable captions"]),
        ("lock panel", ["unlock panel"]),
        ("open menu", ["close menu"]),
        ("show controls", ["hide controls"]),
        ("menu", ["open menu"]),
    ],
)
def test_match_command_never_picks_a_different_command(transcript, hints):
    assert match_command(transcript, hints) is None


@pytest.mark.parametrize(
    ("transcript", "phrase"),
    [("stopped recording", "stop recording"), ("pen menu", "open menu"), ("paws recording", "pause recording")],
)
def test_match_command_tolerates_mishearings(transcript, phrase):
    m = match_command(transcript, [phrase, "start recording", "close menu"])
    assert m is not None and m.phrase == phrase and 0.8 <= m.score < 1.0


def test_parse_hints_json_or_separated():
    assert parse_hints(json.dumps(["open menu", " close menu "])) == ["open menu", "close menu"]
    assert parse_hints("open menu|close menu\npause recording") == ["open menu", "close menu", "pause recording"]
    assert parse_hints(None) == [] and parse_hints("  ") == []
    for bad in ("[1, 2]", "[not json", json.dumps([f"h{i}" for i in range(201)])):
        with pytest.raises(Exception) as ei:
            parse_hints(bad)
        assert ei.value.status_code == 400


def test_stt_label():
    assert stt_label(MockSTT()) == "mock"
    s = ScriptedSTT("x")
    s.model = "gpt-transcribe"
    assert stt_label(s) == "scripted:gpt-transcribe"


async def test_stt_route_transcript_duration_model(app_client):
    c, app = app_client
    r = await c.post("/v1/stt", files={"audio": ("q.wav", _wav(0.5), "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json() == {"transcript": "[mock transcript]", "duration_s": 0.5, "model": "mock", "command": None}
    # non-WAV bytes (e.g. m4a) still go to the provider; duration is unknown
    app.state.stt = stt = ScriptedSTT("where are my keys")
    r = await c.post("/v1/stt", files={"audio": ("q.m4a", b"\x00\x00\x00\x18ftypM4A ", "audio/mp4")})
    assert r.status_code == 200 and r.json()["transcript"] == "where are my keys"
    assert r.json()["duration_s"] is None and stt.calls == 1


async def test_stt_route_command_hints(app_client):
    c, app = app_client
    app.state.stt = ScriptedSTT("Please pause recording.")
    r = await c.post(
        "/v1/stt",
        files={"audio": ("q.wav", _wav(), "audio/wav")},
        data={"command_hints": json.dumps(HINTS)},
    )
    assert r.status_code == 200, r.text
    assert r.json()["command"] == {"phrase": "pause recording", "score": 1.0}
    app.state.stt = ScriptedSTT("where did I leave my keys?")
    files = {"audio": ("q.wav", _wav(), "audio/wav")}
    r = await c.post("/v1/stt", files=files, data={"command_hints": "|".join(HINTS)})
    assert r.status_code == 200 and r.json()["command"] is None
    r = await c.post("/v1/stt", files=files, data={"command_hints": "[1]"})
    assert r.status_code == 400


async def test_stt_route_errors(app_client):
    c, app = app_client
    assert (await c.post("/v1/stt", json={"text": "x"})).status_code == 415
    r = await c.post("/v1/stt", data={"command_hints": "x"}, files={"x": ("x.txt", b"n", "text/plain")})
    assert r.status_code == 400
    assert (await c.post("/v1/stt", files={"audio": ("q.wav", b"", "audio/wav")})).status_code == 400
    app.state.stt = BrokenSTT()
    r = await c.post("/v1/stt", files={"audio": ("q.wav", _wav(), "audio/wav")})
    assert r.status_code == 502 and "provider down" in r.json()["detail"]


async def test_stt_route_requires_token_when_auth_is_on(tmp_path):
    settings = Settings(
        api_token="sekrit",
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
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            files = {"audio": ("q.wav", _wav(), "audio/wav")}
            assert (await c.post("/v1/stt", files=files)).status_code == 401
            r = await c.post("/v1/stt", files=files, headers={"Authorization": "Bearer sekrit"})
            assert r.status_code == 200 and r.json()["transcript"] == "[mock transcript]"
