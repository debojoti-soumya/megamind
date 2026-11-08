"""GET /v1/turns (every chat turn's question/answer + times) and tools/make_captions.py (demo-video captions)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from megamind.agent.turn_log import Turn, TurnLog
from megamind.config import Settings
from megamind.main import create_app

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import make_captions  # noqa: E402


def _settings(tmp_path, **kw) -> Settings:
    return Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend="memory",
        embed_provider="mock",
        enrich_provider="mock",
        anthropic_api_key="",
        consolidate_enabled=False,
        _env_file=None,
        **kw,
    )


async def _client(settings: Settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


@pytest.fixture
async def client(tmp_path):
    async for c in _client(_settings(tmp_path)):
        yield c


async def test_chat_and_stream_turns_are_listed_with_times(client):
    t0 = time.time()
    a = (
        await client.post("/v1/chat", json={"session_id": "s", "text": "where are my keys?", "device_id": "q1"})
    ).json()
    async with client.stream(
        "POST", "/v1/chat/stream", json={"session_id": "s", "text": "what did I eat?", "device_id": "q2"}
    ) as r:
        async for _ in r.aiter_bytes():
            pass
    turns = (await client.get("/v1/turns")).json()["turns"]
    assert [t["question"] for t in turns] == ["where are my keys?", "what did I eat?"]
    assert turns[0]["answer"] == a["answer"] and turns[0]["device_id"] == "q1" and turns[0]["voice"] is False
    for t in turns:
        assert t0 <= t["started_at"] <= t["ended_at"] <= time.time()
    only_q2 = (await client.get("/v1/turns", params={"device_id": "q2"})).json()["turns"]
    assert [t["question"] for t in only_q2] == ["what did I eat?"]
    later = (await client.get("/v1/turns", params={"since": time.time() + 60})).json()["turns"]
    assert later == []


async def test_turn_log_survives_a_restart(tmp_path):
    async for c in _client(_settings(tmp_path)):
        await c.post("/v1/chat", json={"session_id": "s", "text": "where are my keys?"})
    assert (tmp_path / "turns.jsonl").exists()
    async for c in _client(_settings(tmp_path)):
        assert [t["question"] for t in (await c.get("/v1/turns")).json()["turns"]] == ["where are my keys?"]


async def test_turn_log_off_keeps_nothing(tmp_path):
    async for c in _client(_settings(tmp_path, turn_log=False)):
        await c.post("/v1/chat", json={"session_id": "s", "text": "where are my keys?"})
        assert (await c.get("/v1/turns")).json() == {"turns": []}
    assert not (tmp_path / "turns.jsonl").exists()


def test_turn_log_skips_corrupt_lines(tmp_path):
    p = tmp_path / "turns.jsonl"
    good = Turn(started_at=1.0, ended_at=2.0, question="q", answer="a")
    p.write_text("{not json\n" + good.model_dump_json() + "\n")
    assert TurnLog(p).query() == [good]


def _turn(start: float, end: float, q: str, a: str) -> dict:
    return {"started_at": start, "ended_at": end, "question": q, "answer": a}


def test_captions_follow_the_turns_relative_to_the_video_start():
    start = 1_000.0
    turns = [
        _turn(1_010.0, 1_012.0, "where are my keys?", "In the bowl by the door, ten minutes ago."),
        _turn(1_015.0, 1_016.5, "what colour was the mug?", "Blue."),
    ]
    cues = make_captions.build_cues(turns, start, duration=30.0)
    assert [c.text.split(":")[0] for c in cues] == ["Q", "A", "Q", "A"]
    q1, a1, q2, a2 = cues
    assert (q1.start, q1.end) == (9.0, 12.0)  # 1 s lead before the server got it, until the answer is ready
    assert a1.start == 12.0 and a1.end == 14.0  # cut short where the next question starts
    assert a1.text.endswith("(2.0 s)") and a2.text == "A: Blue. (1.5 s)"
    assert a2.end == 16.5 + 2.5  # short answers stay up at least 2.5 s


def test_captions_clip_to_the_video_and_drop_turns_outside_it():
    turns = [_turn(900.0, 902.0, "before", "x"), _turn(1_028.0, 1_029.0, "at the end", "y y y")]
    cues = make_captions.build_cues(turns, 1_000.0, duration=30.0, latency=False)
    assert [c.text for c in cues] == ["Q: at the end", "A: y y y"]
    assert cues[-1].end == 30.0


def test_srt_format_and_wrapping():
    cues = make_captions.build_cues([_turn(3_661.5, 3_662.0, "q", "word " * 20)], 0.0, lead_s=0.0, latency=False)
    srt = make_captions.to_srt(cues)
    assert srt.startswith("1\n01:01:01,500 --> 01:01:02,000\nQ: q\n\n2\n01:01:02,000 --> ")
    assert all(len(line) <= make_captions.WRAP for line in srt.splitlines())


def test_video_start_from_the_quest_file_name():
    got = make_captions.start_from_name("com.oculus.vrshell-20261108-093012.mp4", "America/Los_Angeles")
    assert got == make_captions.parse_start("2026-11-08T09:30:12-08:00", "UTC")
    assert make_captions.start_from_name("clip.mp4", "UTC") is None


def test_end_to_end_srt_and_burned_video(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    video = tmp_path / "quest.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=6", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    turns = tmp_path / "turns.jsonl"
    turns.write_text(
        "\n".join(Turn(**_turn(101.0, 102.0, "where are my keys?", "Bowl.")).model_dump_json() for _ in [0])
    )
    out = tmp_path / "out.mp4"
    make_captions.main([str(video), "--turns", str(turns), "--start", "100", "--burn", str(out)])
    assert "Q: where are my keys?" in video.with_suffix(".srt").read_text()
    assert out.stat().st_size > 0
