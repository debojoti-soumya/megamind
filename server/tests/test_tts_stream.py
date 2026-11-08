"""TTS_STREAM: `audio` events per sentence on POST /v1/chat/stream (agent/speech.py), mock TTS only."""

from __future__ import annotations

import asyncio
import json
import time
import wave

import httpx2
import pytest
from test_chat_stream import (
    ANSWER,
    ScriptedAnthropic,
    _message,
    _sse_body,
    _stream,
    _two_rounds,
    _use_script,
    text,
    tool_use,
)
from test_chat_stream import app_client as _app_client

from megamind.agent import tts as tts_mod
from megamind.agent.speech import SentenceBuffer, StreamingSpeaker, concat_wavs
from megamind.models import ChatResponse

app_client = _app_client  # the fixture from test_chat_stream (store with two memories + Anthropic script)

TWO = "Your keys are on the kitchen counter. You left them there about twenty minutes ago."


class SlowFirstTTS(tts_mod._FileTTS):
    """Clip i takes `delays[i]` s (first sentence slowest), 16 kHz, 100 samples per 10 chars: ordering must hold."""

    def __init__(self, audio_dir, delays=(0.2, 0.0, 0.0)):
        super().__init__(audio_dir)
        self.delays, self.calls = list(delays), []

    async def _render(self, text: str) -> bytes:
        i = len(self.calls)
        self.calls.append(text)
        await asyncio.sleep(self.delays[i] if i < len(self.delays) else 0)
        return tts_mod.pcm16_to_wav(b"\x01\x00" * (10 * len(text)), 16000)


def test_sentence_buffer_incremental_matches_split_sentences():
    for n in (1, 3, 7, 50):
        b = SentenceBuffer()
        out = []
        for i in range(0, len(TWO), n):
            out += b.feed(TWO[i : i + n])
        out.append(b.flush())
        assert out == tts_mod.split_sentences(TWO), n
    b = SentenceBuffer()
    assert b.feed("Yes. It was on the kitchen counter next to the coffee machine. And") == [
        "Yes. It was on the kitchen counter next to the coffee machine."
    ]
    assert b.flush() == "And"
    assert SentenceBuffer().feed("The total was $23.47 at checkout. ") == []  # no next sentence start yet


def test_concat_wavs(tmp_path):
    for i, n in enumerate((100, 50)):
        (tmp_path / f"p{i}.wav").write_bytes(tts_mod.pcm16_to_wav(b"\x01\x00" * n, 16000))
    assert concat_wavs([tmp_path / "p0.wav", tmp_path / "p1.wav"], tmp_path / "out.wav")
    with wave.open(str(tmp_path / "out.wav")) as w:
        assert (w.getframerate(), w.getnframes()) == (16000, 150 + 2400)
    (tmp_path / "p2.wav").write_bytes(tts_mod.pcm16_to_wav(b"\x01\x00" * 10, 24000))
    assert not concat_wavs([tmp_path / "p0.wav", tmp_path / "p2.wav"], tmp_path / "out2.wav")


async def test_speaker_emits_in_order_and_concatenates(tmp_path):
    events = []

    async def emit(name, data):
        events.append((name, data))

    sp = StreamingSpeaker(SlowFirstTTS(tmp_path), tmp_path, emit)
    for i in range(0, len(TWO), 5):
        await sp.feed(TWO[i : i + 5])
    await asyncio.sleep(0.05)
    assert events == []  # clip 1 is done but clip 0 is not: nothing out of order
    url = await sp.finish(TWO)
    assert [(d["seq"], d["text"]) for _, d in events] == [(0, TWO.split(". ")[0] + "."), (1, TWO.split(". ")[1])]
    assert all(d["url"] == f"/v1/audio/{sp.audio_id}-{d['seq']}.wav" for _, d in events)
    assert url == f"/v1/audio/{sp.audio_id}.wav"
    with wave.open(str(tmp_path / f"{sp.audio_id}.wav")) as w:
        assert w.getnframes() == 10 * len(events[0][1]["text"]) + 10 * len(events[1][1]["text"]) + 2400


async def test_speaker_reset_drops_clips_and_mismatch_respeaks(tmp_path):
    events = []

    async def emit(name, data):
        events.append((name, data))

    tts = SlowFirstTTS(tmp_path, delays=(0.0, 0.0, 0.0))
    sp = StreamingSpeaker(tts, tmp_path, emit)
    await sp.feed("Let me check that for you right now. One")
    await asyncio.sleep(0.01)
    assert len(events) == 1
    old = sp.audio_id
    await sp.reset()
    await sp.feed(TWO)
    url = await sp.finish(TWO)
    assert url != f"/v1/audio/{old}.wav" and [d["seq"] for _, d in events] == [0, 0, 1]
    # the final answer is not what was streamed (e.g. NOT_FOUND replacement): one whole-answer call instead
    sp = StreamingSpeaker(tts, tmp_path, emit)
    await sp.feed(TWO)
    url = await sp.finish("Sorry, I couldn't find that.")
    assert url == f"/v1/audio/{sp.audio_id}.wav" and tts.calls[-1] == "Sorry, I couldn't find that."


async def test_speaker_each_seq_once_when_clips_finish_between_sentences(tmp_path):
    """Claude pauses between sentences and TTS is instant: the sender drains and restarts; no clip is re-sent."""
    events = []

    async def emit(name, data):
        events.append(data["seq"])

    sp = StreamingSpeaker(SlowFirstTTS(tmp_path, delays=(0.0,) * 4), tmp_path, emit)
    sentences = ["First sentence of the answer is here.", "Second one follows it now.", "Third one closes it all."]
    for s in sentences:
        await sp.feed(s + " ")  # completes the previous sentence's boundary -> its clip starts and finishes...
        await asyncio.sleep(0.03)  # ...before this sentence ends, so the sender drains and has to restart
    url = await sp.finish(" ".join(sentences))
    assert events == [0, 1, 2] and url is not None


async def test_speaker_failed_clip_respeaks_whole_answer(tmp_path):
    class FlakyTTS(SlowFirstTTS):
        async def _render(self, text: str) -> bytes:
            if len(self.calls) == 1:  # the second clip's provider call fails
                self.calls.append(text)
                raise RuntimeError("tts down")
            return await super()._render(text)

    tts = FlakyTTS(tmp_path, delays=(0.0, 0.0, 0.0))
    sp = StreamingSpeaker(tts, tmp_path, lambda n, d: asyncio.sleep(0))
    await sp.feed(TWO)
    url = await sp.finish(TWO)
    assert url == f"/v1/audio/{sp.audio_id}.wav" and tts.calls[-1] == TWO  # not a silently shorter file
    with wave.open(str(tmp_path / f"{sp.audio_id}.wav")) as w:
        assert w.getnframes() == 10 * len(TWO)


async def test_mock_tts_streams_no_audio(tmp_path):
    sp = StreamingSpeaker(tts_mod.MockTTS(), tmp_path, lambda n, d: asyncio.sleep(0))
    await sp.feed(TWO)
    assert await sp.finish(TWO) is None


@pytest.mark.parametrize("stream_on", [True, False])
async def test_chat_stream_audio_events(app_client, stream_on):
    c, app = app_client
    app.state.settings.tts_stream = stream_on
    app.state.tts = SlowFirstTTS(app.state.settings.audio_dir)
    two = TWO + " <cites>t_keys</cites>"
    _use_script(app, ScriptedAnthropic([_message([text(two)])]))
    events = await _stream(c, {"session_id": "s", "device_id": "dev1", "text": "where are my keys?"})
    names = [n for n, _ in events]
    audio = [d for n, d in events if n == "audio"]
    final = events[-1][1]
    assert names[-1] == "final"
    if not stream_on:
        assert audio == [] and final["audio_url"].endswith(".wav")
        return
    assert [a["seq"] for a in audio] == [0, 1] and set(audio[0]) == {"seq", "url", "text"}
    assert " ".join(a["text"] for a in audio) == final["answer"] == TWO
    assert names.index("audio") < names.index("final") and names.count("audio") == 2
    for a in audio:
        r = await c.get(a["url"])
        assert r.status_code == 200 and r.content[:4] == b"RIFF"
    r = await c.get(final["audio_url"])
    assert r.status_code == 200 and final["audio_url"] != audio[0]["url"]
    with wave.open(str(app.state.settings.audio_dir / final["audio_url"].rsplit("/", 1)[1])) as w:
        assert w.getnframes() == 10 * sum(len(a["text"]) for a in audio) + 2400  # both clips + 150 ms gap


async def test_chat_stream_reset_cancels_preamble_audio(app_client):
    c, app = app_client
    app.state.tts = SlowFirstTTS(app.state.settings.audio_dir, delays=(0.0, 0.0, 0.0))
    _use_script(app, ScriptedAnthropic(_two_rounds(preamble="Let me check my memory for that.")))
    events = await _stream(c, {"session_id": "s", "device_id": "dev1", "text": "where are my keys?"})
    names = [n for n, _ in events]
    r_i = names.index("reset")
    assert "audio" not in names[:r_i] or all(n != "audio" for n in names[r_i:]) is False
    after = [d for n, d in events[r_i:] if n == "audio"]
    assert [a["seq"] for a in after] == [0] and after[0]["text"] == ANSWER.split(" <cites>")[0]
    assert ChatResponse.model_validate(events[-1][1]).audio_url.endswith(".wav")


class GappedAnthropic(ScriptedAnthropic):
    """Streams each SSE event `gap` s apart, like the live API writing an answer token by token."""

    gap = 0.03

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        events = _sse_body(self.responses.pop(0)).split(b"\n\n")

        async def body():
            for e in events:
                if e:
                    yield e + b"\n\n"
                    await asyncio.sleep(self.gap)

        return httpx2.Response(200, content=body(), headers={"content-type": "text/event-stream"})


async def test_recall_first_answer_streams_and_first_audio_overlaps_the_call(app_client):
    """#138: on the recall_first path the answer must stream (no trailing tool_result), so sentence 0's clip is
    ready while Claude still writes sentence 1, one sentence's TTS time before `final`."""
    c, app = app_client
    tts_s = 0.1
    script = GappedAnthropic([_message([text(TWO + " <cites>t_keys</cites>")])])
    agent = _use_script(app, script, recall_first=True)
    tts = SlowFirstTTS(app.state.settings.audio_dir, delays=(tts_s, tts_s))
    t0, seen = time.perf_counter(), {}

    async def put(name, data):
        seen.setdefault(name, time.perf_counter() - t0)

    speaker = StreamingSpeaker(tts, app.state.settings.audio_dir, put)

    async def on_event(name, data):
        seen.setdefault(name, time.perf_counter() - t0)
        if name == "delta":
            await speaker.feed(data["text"])

    resp, stats = await agent.chat_with_stats("s", "where did I leave my keys?", "dev1", on_event=on_event)
    t_call = time.perf_counter() - t0
    assert stats.extra["prefetch"] == "first" and len(script.requests) == 1
    assert "tool_result" not in json.dumps(script.requests[0]["messages"])
    url = await speaker.finish(resp.answer)
    t_final = time.perf_counter() - t0
    assert url is not None and resp.answer == TWO
    assert seen["delta"] < t_call - 5 * GappedAnthropic.gap  # text arrives while the call is still running
    assert seen["audio"] < t_call  # clip 0 is playable before Claude has finished
    assert t_final - seen["audio"] >= tts_s * 0.9  # ... a whole sentence's TTS ahead of `final`


@pytest.mark.parametrize("as_text", [True, False])
async def test_two_search_answer_is_sent_tool_results_as_text(app_client, as_text):
    """#138: Claude's own searches; with CLAUDE_TOOL_RESULTS_AS_TEXT the answer call carries no tool blocks (so
    Sonnet streams it), and the first delta and first audio still precede `final`."""
    c, app = app_client
    uses = [tool_use({"query": "front door lock"}, "toolu_1"), tool_use({"query": "stove off"}, "toolu_2")]
    script = GappedAnthropic([_message(uses, stop_reason="tool_use"), _message([text(TWO)])])
    agent = _use_script(app, script, recall_first=False, recall_prefetch=False, claude_tool_results_as_text=as_text)
    tts = SlowFirstTTS(app.state.settings.audio_dir, delays=(0.05, 0.05))
    t0, seen = time.perf_counter(), {}

    async def put(name, data):
        seen.setdefault(name, time.perf_counter() - t0)

    speaker = StreamingSpeaker(tts, app.state.settings.audio_dir, put)

    async def on_event(name, data):
        seen.setdefault(name, time.perf_counter() - t0)
        if name == "delta":
            await speaker.feed(data["text"])

    q = "did I lock the front door and turn off the stove before leaving?"
    resp, stats = await agent.chat_with_stats("s", q, "dev1", on_event=on_event)
    t_call = time.perf_counter() - t0
    assert await speaker.finish(resp.answer) is not None
    t_final = time.perf_counter() - t0
    assert [t.args["query"] for t in resp.tool_calls] == ["front door lock", "stove off"]
    answer_req = json.dumps(script.requests[1]["messages"])
    assert ("tool_result" in answer_req) is not as_text and ("tool_use" in answer_req) is not as_text
    if as_text:
        calls = [b["text"] for b in script.requests[1]["messages"][1]["content"]]
        assert calls == ['recall_memory({"query": "front door lock"})', 'recall_memory({"query": "stove off"})']
        assert "tool result:" in answer_req
    assert "tool_result" in json.dumps(agent.histories["s"])  # history keeps the real blocks
    assert seen["delta"] < t_call < t_final and seen["audio"] < t_final
