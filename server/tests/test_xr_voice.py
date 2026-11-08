"""Voice questions in the Memory Theater (GET /xr): the page serves the voice module and its button, the clip it
sends reaches the existing STT through `POST /v1/chat` (no second STT stack), and the recorder's state machine
(hold-to-talk, tap-to-toggle, errors) behaves (node harness, skipped without node)."""

import io
import re
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from megamind.agent.stt import STT
from megamind.xr import STATIC_DIR

HARNESS = Path(__file__).parent / "fixtures" / "xr_voice_harness.mjs"


async def test_xr_serves_voice_module(client):
    c, _ = client
    app_js = (await c.get("/xr/static/app.js")).text
    assert 'from "./voice.js"' in app_js
    r = await c.get("/xr/static/voice.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
    for name in ("export class VoiceRecorder", "export function mountVoiceButton", "export function encodeWav"):
        assert name in r.text


async def test_xr_page_has_voice_button_next_to_ask_box(client):
    c, _ = client
    html = (await c.get("/xr")).text
    ask_end = html.index("</form>")
    btn = html.index('id="xr-voice-btn"')
    assert ask_end < btn < html.index('id="xr-enter"'), "mic button sits right after the ask box"
    assert re.search(r'<button type="button" class="voice-btn[^"]*" id="xr-voice-btn"', html)
    assert 'id="xr-voice-tip"' in html
    css = (await c.get("/xr/static/xr.css")).text
    # big enough for a Quest controller ray
    assert re.search(r"\.voice-btn \{[^}]*min-width: 60px;[^}]*min-height: 60px;", css)


def test_xr_voice_reuses_chat_endpoint_and_no_transcribe_route():
    js = "".join(p.read_text() for p in STATIC_DIR.glob("*.js"))
    assert not re.search(r"[\"'`]/(?:v1|api)/transcribe", js), "no second transcription endpoint"
    app_js = (STATIC_DIR / "app.js").read_text()
    # the clip goes to the same /v1/chat the typed question uses, as multipart `audio`
    assert app_js.count('"/v1/chat"') == 1
    assert 'fd.append("audio", wav, "question.wav")' in app_js
    assert 'fd.append("session_id", SESSION)' in app_js


class RecordingSTT(STT):
    def __init__(self, text: str) -> None:
        self.text = text
        self.clips: list[bytes] = []

    async def transcribe(self, wav: bytes) -> str:
        self.clips.append(wav)
        return self.text


def _wav(seconds: float = 0.6, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


@pytest.mark.parametrize("device", [None, "quest3-xr"])
async def test_voice_question_goes_through_existing_stt(client, device):
    """What voice.js + app.js send (multipart session_id/device_id/audio=question.wav) is transcribed by the app's
    configured STT and answered in the same request; the response carries the transcript the theater shows."""
    c, app = client
    stt = RecordingSTT("where did I leave my keys?")
    app.state.stt = stt
    clip = _wav()
    data = {"session_id": "xr-test"}
    if device:
        data["device_id"] = device
    r = await c.post("/v1/chat", data=data, files={"audio": ("question.wav", clip, "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript"] == "where did I leave my keys?"
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["cited"], list)
    assert stt.clips == [clip]


async def test_voice_question_nothing_heard_returns_empty_transcript(client):
    """The theater turns an empty transcript into "Didn't catch that"; the server must hand it back as ""."""
    c, app = client
    app.state.stt = RecordingSTT("")
    r = await c.post("/v1/chat", data={"session_id": "xr-test"}, files={"audio": ("question.wav", _wav(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["transcript"] == ""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_voice_recorder_state_machine(tmp_path):
    mod = tmp_path / "voice.mjs"
    mod.write_text((STATIC_DIR / "voice.js").read_text())
    r = subprocess.run(["node", str(HARNESS), mod.as_uri()], capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().endswith("ok")
