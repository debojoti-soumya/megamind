import io
import json
import math
import struct
import wave

import pytest

from megamind.agent.stt import STT, MockSTT, Segment, wav_duration
from megamind.config import Settings
from megamind.ingest.audio import merge_segments
from megamind.ingest.transcribe import (
    OpenAIWhisperSTT,
    build_audio_stt,
    ensure_wav,
    have_ffmpeg,
    to_wav16k,
)

T0 = 1_762_000_000.0


def _wav(seconds: float, rate: int = 16000, freq: float = 440.0) -> bytes:
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        samples = (int(3000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n))
        w.writeframes(b"".join(struct.pack("<h", x) for x in samples))
    return buf.getvalue()


async def _post_audio(c, data, ts_start=T0, device="dev1", filename="a.wav", ctype="audio/wav"):
    return await c.post(
        "/v1/audio",
        files={"audio": (filename, data, ctype)},
        data={"meta": json.dumps({"device_id": device, "ts_start": ts_start})},
    )


class TextOnlySTT(STT):
    """A provider with no timestamps: exercises the STT.transcribe_segments default."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def transcribe(self, wav: bytes) -> str:
        return self.text


async def test_audio_transcript_becomes_searchable_token(client):
    c, app = client
    app.state.audio_stt = MockSTT(
        script=[
            "hi I'm Priya from the robotics lab",
            "nice to meet you",
            "the demo is on Friday",
            "let's grab lunch on Tuesday",
            "bring the headset",
        ],
        segment_s=5.0,
    )
    r = await _post_audio(c, _wav(25.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["duration_s"] == pytest.approx(25.0)
    # 5 mock segments of 5 s merged into <= 15 s windows -> [0,15) and [15,25)
    assert len(body["token_ids"]) == 2
    assert "Priya" in body["transcript"]

    r = await c.get("/v1/memories", params={"q": "Priya"})
    assert r.status_code == 200
    hits = r.json()["results"]
    assert [h["id"] for h in hits] == [body["token_ids"][0]]
    assert hits[0]["kind"] == "transcript"
    assert hits[0]["ts"] == pytest.approx(T0)
    assert hits[0]["thumb_url"] == ""
    assert "Priya" in hits[0]["caption"]

    second = await app.state.store.get(body["token_ids"][1])
    assert second.ts == pytest.approx(T0 + 15.0) and second.kind == "transcript"
    assert second.caption == 'Speech heard nearby: "let\'s grab lunch on Tuesday bring the headset"'


async def test_recall_memory_finds_transcript(client):
    c, app = client
    app.state.audio_stt = TextOnlySTT("this is Marcus, he runs the coffee cart downstairs")
    await _post_audio(c, _wav(3.0), ts_start=T0)
    res = await app.state.recall(query="who is Marcus", k=4, now=T0 + 600)
    assert res.confidence != "not_found"
    assert res.all_retrieved and "Marcus" in res.all_retrieved[0].caption
    assert res.all_retrieved[0].thumb_url == ""


async def test_audio_fallback_single_segment(client):
    c, app = client
    app.state.audio_stt = TextOnlySTT("met Dana at the bakery on Fifth")
    r = await _post_audio(c, _wav(4.0), ts_start=T0 + 100)
    assert r.status_code == 200, r.text
    assert len(r.json()["token_ids"]) == 1

    hits = (await c.get("/v1/memories", params={"q": "Dana bakery"})).json()["results"]
    assert len(hits) == 1 and hits[0]["ts"] == pytest.approx(T0 + 100)
    assert hits[0]["kind"] == "transcript"


async def test_audio_silence_creates_no_tokens(client):
    c, app = client
    app.state.audio_stt = TextOnlySTT("   ")
    r = await _post_audio(c, _wav(2.0))
    assert r.status_code == 200
    assert r.json()["token_ids"] == [] and r.json()["transcript"] == ""
    assert await app.state.store.count() == 0


async def test_audio_tokens_do_not_disturb_frames_and_purge(client):
    c, app = client
    r = await _post_audio(c, _wav(6.0))  # default mock: "[mock transcript]" segments
    assert r.status_code == 200 and r.json()["token_ids"]
    assert await app.state.store.recent_hashes("dev1", 0.0) == []  # transcripts never enter the frame dedupe
    assert (await c.delete("/v1/memories", params={"device_id": "dev1"})).json()["deleted"] == 1


async def test_audio_bad_requests(client):
    c, _ = client
    r = await c.post("/v1/audio", files={"audio": ("a.wav", _wav(1.0), "audio/wav")}, data={"meta": "{not json"})
    assert r.status_code == 400
    r = await c.post(
        "/v1/audio", files={"audio": ("a.wav", _wav(1.0), "audio/wav")}, data={"meta": json.dumps({"device_id": "d"})}
    )
    assert r.status_code == 400  # ts_start missing
    r = await _post_audio(c, b"")
    assert r.status_code == 400
    r = await _post_audio(c, b"definitely not audio" * 10, filename="a.bin", ctype="application/octet-stream")
    assert r.status_code == 415  # ffmpeg missing or unable to decode -> same answer
    r = await c.post(
        "/v1/audio",
        files={"audio": ("a.wav", _wav(1.0), "audio/wav")},
        data={"meta": '{"device_id": "d", "ts_start": NaN}'},  # json.loads accepts NaN; the contract does not
    )
    assert r.status_code == 400


def _lying_wav(n_frames: int = 100) -> bytes:
    """A tiny WAV whose header claims ~2^31 frames at 1 Hz (68 years) but carries only `n_frames`."""
    fmt = struct.pack("<IHHIIHH", 16, 1, 1, 1, 2, 2, 16)
    data = b"\x00\x00" * n_frames
    return b"RIFF" + struct.pack("<I", 0x7FFFFFFF) + b"WAVEfmt " + fmt + b"data" + struct.pack("<I", 0x7FFFFFF0) + data


async def test_audio_limits(client, monkeypatch):
    c, app = client
    from megamind.ingest import audio as audio_mod
    from megamind.ingest import transcribe

    # duration comes from the samples present, not the header: 100 frames at 1 Hz -> 100 s, 20 mock segments
    r = await _post_audio(c, _lying_wav())
    assert r.status_code == 200, r.text
    assert r.json()["duration_s"] == pytest.approx(100.0)

    monkeypatch.setattr(transcribe, "MAX_AUDIO_S", 5.0)
    assert (await _post_audio(c, _wav(6.0))).status_code == 413  # too long
    assert (await _post_audio(c, _wav(4.0))).status_code == 200

    monkeypatch.setattr(audio_mod, "MAX_AUDIO_BYTES", 1000)
    assert (await _post_audio(c, _wav(1.0))).status_code == 413  # too many bytes


async def test_default_stt_segments_fallback():
    stt = TextOnlySTT("hello there")
    assert await stt.transcribe_segments(_wav(3.0)) == [Segment(0.0, 3.0, "hello there")]
    assert await stt.transcribe_segments(b"not a wav") == [Segment(0.0, 0.0, "hello there")]
    assert await TextOnlySTT("").transcribe_segments(_wav(1.0)) == []


async def test_mock_stt_segments_deterministic():
    segs = await MockSTT().transcribe_segments(_wav(12.0))
    assert [(s.start, s.end) for s in segs] == [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]
    assert {s.text for s in segs} == {"[mock transcript]"}
    assert await MockSTT().transcribe(_wav(1.0)) == "[mock transcript]"  # chat path unchanged


def test_merge_segments_windows():
    segs = [Segment(i * 4.0, i * 4.0 + 3.5, f"w{i}") for i in range(8)] + [Segment(40.0, 41.0, "  ")]
    out = merge_segments(segs, 15.0)
    assert [o.text for o in out] == ["w0 w1 w2", "w3 w4 w5", "w6 w7"]
    assert all(o.end - o.start <= 15.0 for o in out)
    assert merge_segments([Segment(0.0, 30.0, "long monologue")], 15.0) == [Segment(0.0, 30.0, "long monologue")]
    assert merge_segments([], 15.0) == []


def test_wav_duration():
    assert wav_duration(_wav(1.5, rate=8000)) == pytest.approx(1.5)
    assert wav_duration(b"RIFF....garbage") is None
    assert wav_duration(_lying_wav(100)) == pytest.approx(100.0)  # header claims 2^31 frames
    assert wav_duration(_wav(1.0)[:-1000]) == pytest.approx(1.0 - 500 / 16000)  # truncated upload


class _FakeTranscriptions:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    async def create(self, **kw):
        self.calls.append(kw)
        return self.resp


class _FakeClient:
    def __init__(self, resp):
        self.audio = type("A", (), {})()
        self.audio.transcriptions = _FakeTranscriptions(resp)


async def test_openai_verbose_json_segments():
    resp = {
        "text": "hello Sam see you at noon",
        "segments": [
            {"start": 0.0, "end": 2.1, "text": " hello Sam", "no_speech_prob": 0.01},
            {"start": 2.1, "end": 4.0, "text": " Thanks for watching!", "no_speech_prob": 0.95},
            {"start": 4.0, "end": 6.5, "text": " see you at noon", "no_speech_prob": 0.05},
        ],
    }
    client = _FakeClient(resp)
    stt = OpenAIWhisperSTT(Settings(_env_file=None), client=client)
    segs = await stt.transcribe_segments(_wav(7.0))
    assert segs == [Segment(0.0, 2.1, "hello Sam"), Segment(4.0, 6.5, "see you at noon")]
    call = client.audio.transcriptions.calls[0]
    assert call["model"] == "whisper-1" and call["response_format"] == "verbose_json"
    assert call["timestamp_granularities"] == ["segment"]


async def test_openai_non_whisper_model_falls_back_to_text():
    client = _FakeClient({"text": "just text"})
    stt = OpenAIWhisperSTT(Settings(audio_stt_model="gpt-4o-mini-transcribe", _env_file=None), client=client)
    assert await stt.transcribe_segments(_wav(2.0)) == [Segment(0.0, 2.0, "just text")]
    assert client.audio.transcriptions.calls[0]["response_format"] == "json"


def test_build_audio_stt_selection():
    chat = MockSTT()
    assert build_audio_stt(Settings(_env_file=None), chat) is chat
    assert isinstance(build_audio_stt(Settings(audio_stt_provider="mock", _env_file=None), chat), MockSTT)
    # unconfigured / unknown providers degrade to the chat STT instead of failing startup
    assert build_audio_stt(Settings(audio_stt_provider="openai", openai_api_key="", _env_file=None), chat) is chat
    assert build_audio_stt(Settings(audio_stt_provider="nope", _env_file=None), chat) is chat
    s = Settings(stt_provider="openai", openai_api_key="sk-test", _env_file=None)
    assert isinstance(build_audio_stt(s, chat), OpenAIWhisperSTT)


def test_build_audio_stt_faster_whisper_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "faster_whisper":
            raise ImportError("no faster_whisper")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    chat = MockSTT()
    assert build_audio_stt(Settings(audio_stt_provider="faster-whisper", _env_file=None), chat) is chat


needs_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg not on PATH")


@needs_ffmpeg
def test_ffmpeg_resamples_low_rate_headset_audio():
    out = to_wav16k(_wav(2.0, rate=8000), enhance=True)
    assert out is not None
    with wave.open(io.BytesIO(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.getnframes() / 16000 == pytest.approx(2.0, abs=0.05)
    assert to_wav16k(b"garbage" * 20) is None


@needs_ffmpeg
def test_ensure_wav_decodes_non_wav(tmp_path):
    import subprocess

    src = tmp_path / "a.wav"
    src.write_bytes(_wav(1.0))
    flac = tmp_path / "a.flac"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), str(flac)], check=True)
    wav = ensure_wav(flac.read_bytes())
    assert wav is not None and wav_duration(wav) == pytest.approx(1.0, abs=0.05)
    same = _wav(0.5)
    assert ensure_wav(same) is same


@needs_ffmpeg
def test_ffmpeg_decode_is_capped(tmp_path, monkeypatch):
    """Compressed audio can decode to hours of PCM; ffmpeg stops just past MAX_AUDIO_S."""
    import subprocess

    from megamind.ingest import transcribe

    src = tmp_path / "a.wav"
    src.write_bytes(_wav(5.0))
    flac = tmp_path / "a.flac"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), str(flac)], check=True)
    monkeypatch.setattr(transcribe, "MAX_AUDIO_S", 2.0)
    wav = to_wav16k(flac.read_bytes())
    assert wav is not None and wav_duration(wav) == pytest.approx(3.0, abs=0.05)


def test_faster_whisper_provider_builds_lazily():
    pytest.importorskip("faster_whisper")
    if not have_ffmpeg():
        pytest.skip("ffmpeg not on PATH")
    from megamind.ingest.transcribe import FasterWhisperSTT

    stt = build_audio_stt(Settings(audio_stt_provider="faster-whisper", _env_file=None), MockSTT())
    assert isinstance(stt, FasterWhisperSTT) and stt.model == "small.en"  # no model download until first use


# ------------------------------------------------------------------ speech merge (TranscriptSink)


async def test_audio_chunks_close_together_merge_into_one_token(client):
    c, app = client
    app.state.audio_stt = TextOnlySTT("so the plan is that we meet")
    first = (await _post_audio(c, _wav(4.0), ts_start=T0)).json()
    app.state.audio_stt = TextOnlySTT("at the north entrance on Tuesday")
    second = (await _post_audio(c, _wav(4.0), ts_start=T0 + 9.0)).json()  # starts 5 s after the first ends
    assert second["token_ids"] == first["token_ids"]  # the merged-into token's id
    assert second["transcript"] == "at the north entrance on Tuesday"  # this chunk's words only

    (tid,) = first["token_ids"]
    tok = await app.state.store.get(tid)
    assert tok.caption == 'Speech heard nearby: "so the plan is that we meet at the north entrance on Tuesday"'
    assert tok.ts == pytest.approx(T0) and tok.kind == "transcript"
    assert tok.text_vec == pytest.approx(await app.state.text_embedder.embed(tok.caption), abs=1e-6)  # re-embedded
    assert await app.state.store.count() == 1
    hits = (await c.get("/v1/memories", params={"q": "plan north entrance"})).json()["results"]
    assert [h["id"] for h in hits] == [tid]


async def test_audio_chunks_far_apart_stay_separate(client):
    c, app = client
    app.state.audio_stt = TextOnlySTT("first thing")
    a = (await _post_audio(c, _wav(4.0), ts_start=T0)).json()["token_ids"]
    b = (await _post_audio(c, _wav(4.0), ts_start=T0 + 24.0)).json()["token_ids"]  # 20 s after the first ends
    assert a != b and await app.state.store.count() == 2
    other = (await _post_audio(c, _wav(4.0), ts_start=T0 + 28.0, device="dev2")).json()["token_ids"]
    assert other != b  # merging is per device


async def test_audio_merge_respects_max_token_span(client):
    c, app = client
    app.state.audio_stt = TextOnlySTT("still talking")
    ids = [
        (await _post_audio(c, _wav(10.0), ts_start=T0 + 12.0 * i)).json()["token_ids"][0] for i in range(7)
    ]  # 10 s chunks, 2 s apart: merge until the token would span > 60 s
    assert ids[:5] == [ids[0]] * 5  # [T0, T0+58]
    assert ids[5] != ids[0] and ids[6] == ids[5]


async def test_audio_merge_only_continues_previous_chunk(client):
    c, app = client
    app.state.audio_stt = MockSTT(script=["one", "two", "three", "four"], segment_s=5.0)
    r1 = (await _post_audio(c, _wav(20.0), ts_start=T0)).json()  # windows [0,15) [15,20)
    assert len(r1["token_ids"]) == 2  # within a chunk AUDIO_WINDOW_S still decides
    r2 = (await _post_audio(c, _wav(20.0), ts_start=T0 + 22.0)).json()
    assert r2["token_ids"][0] == r1["token_ids"][1]  # the boundary utterance continues the last token
    assert len(r2["token_ids"]) == 2 and r2["token_ids"][1] not in r1["token_ids"]


async def test_purge_drops_merge_target(client):
    c, app = client
    app.state.audio_stt = TextOnlySTT("my PIN is something private")
    (old,) = (await _post_audio(c, _wav(4.0), ts_start=T0)).json()["token_ids"]
    assert (await c.delete("/v1/memories", params={"device_id": "dev1"})).json()["deleted"] == 1
    app.state.audio_stt = TextOnlySTT("anyway")
    (new,) = (await _post_audio(c, _wav(4.0), ts_start=T0 + 6.0)).json()["token_ids"]
    assert new != old and await app.state.store.get(old) is None
    assert (await app.state.store.get(new)).caption == 'Speech heard nearby: "anyway"'


async def test_transcript_sink_unit():
    from megamind.ingest.audio import TranscriptSink
    from megamind.ingest.enrich import MockTextEmbedder
    from megamind.store.memory_store import InMemoryStore

    store = InMemoryStore()
    sink = TranscriptSink(store, MockTextEmbedder(), merge_s=10, max_token_s=60)
    assert await sink.add("d", T0, T0 + 1, "   ") is None
    a = await sink.add("d", T0, T0 + 2, "hello there")
    same = await sink.add("d", T0 + 3, T0 + 3, "hello there")  # the same cue re-sent with the next frame
    assert same.id == a.id and same.caption == a.caption
    late = await sink.add("d", T0 - 30, T0 - 28, "earlier words")  # out of order: never merged
    assert late.id != a.id
    nxt = await sink.add("d", T0 + 8, T0 + 9, "general Kenobi")  # target stays the newest token
    assert nxt.id == a.id and nxt.caption == 'Speech heard nearby: "hello there general Kenobi"'
    off = TranscriptSink(store, MockTextEmbedder(), merge_s=0)
    x = await off.add("d", T0, T0 + 1, "a")
    assert (await off.add("d", T0 + 1, T0 + 2, "b")).id != x.id
