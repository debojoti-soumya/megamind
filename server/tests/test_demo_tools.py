"""tools/demo_check.py and tools/rehearse.py against httpx.MockTransport (no server, no adb, no feeder)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import demo_check  # noqa: E402
import rehearse  # noqa: E402

LIVE = {
    "claude": "claude-sonnet-5-5",
    "embed": "openai:text-embedding-3-small",
    "stt": "openai",
    "tts": "openai",
    "image": "openclip:ViT-B-16-SigLIP",
}


def _server(health=None, turns_status=200, last_frame_age=3.0, chat_answer="Your keys are in the bowl."):
    asked: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/v1/health":
            return httpx.Response(
                200, json=health or {"status": "ok", "store_count": 42, "models": LIVE, "warmup": "done"}
            )
        if path == "/v1/turns":
            return httpx.Response(turns_status, json={"turns": []})
        if path == "/v1/devices":
            now = time.time()
            return httpx.Response(
                200,
                json={
                    "devices": [
                        {"device_id": "old", "count": 1, "last_ts": now - 999, "last_frame_ts": now - 999},
                        {"device_id": "quest3", "count": 9, "last_ts": now, "last_frame_ts": now - last_frame_age},
                    ]
                },
            )
        if path == "/v1/chat":
            asked.append(json.loads(req.content))
            return httpx.Response(200, json={"answer": chat_answer, "cited": [{"id": "a"}], "transcript": "x"})
        return httpx.Response(404)

    return httpx.MockTransport(handler), asked


def _adb(level=80, state="device"):
    def run(cmd):
        if cmd == ["adb", "devices"]:
            return f"List of devices attached\n2G0YC1ZF\t{state}\n\n"
        assert cmd[:3] == ["adb", "-s", "2G0YC1ZF"]
        return f"Current Battery Service state:\n  AC powered: false\n  level: {level}\n  scale: 100\n"

    return run


def test_demo_check_all_green(capsys):
    transport, asked = _server()
    assert demo_check.main(["--server", "http://s"], transport=transport, run=_adb()) == 0
    out = capsys.readouterr().out
    assert "ready to record" in out and "FAIL" not in out
    assert "battery   80%" in out and "quest3: last frame" in out
    assert asked == [{"session_id": "demo-check", "device_id": "quest3", "text": "what did I just see?"}]


@pytest.mark.parametrize(
    "kwargs, adb, needle",
    [
        ({"turns_status": 404}, _adb(), "captions need the #196 deploy"),
        ({"last_frame_age": 120}, _adb(), "last frame 120 s ago"),
        ({}, _adb(level=12), "12% - charge before the take"),
        ({}, _adb(state="unauthorized"), "accept the USB debugging prompt"),
        (
            {"health": {"status": "ok", "store_count": 1, "models": {**LIVE, "image": "none"}, "warmup": "done"}},
            _adb(),
            "image     none - colour/object questions get weaker",
        ),
        (
            {
                "health": {
                    "status": "ok",
                    "store_count": 1,
                    "models": {**LIVE, "stt": "mock (openai configured, no key)"},
                }
            },
            _adb(),
            "FAIL  stt",
        ),
    ],
)
def test_demo_check_flags_problems(capsys, kwargs, adb, needle):
    transport, _ = _server(**kwargs)
    assert demo_check.main(["--server", "http://s"], transport=transport, run=adb) == 1
    out = capsys.readouterr().out
    assert needle in out and "problem(s)" in out


def test_demo_check_server_only_skips_adb_and_question(capsys):
    transport, asked = _server()

    def no_adb(cmd):
        raise AssertionError("adb must not run")

    assert demo_check.main(["--server", "http://s", "--no-adb", "--no-ask"], transport=transport, run=no_adb) == 0
    assert asked == []


def test_demo_check_bad_token_stops_early(capsys):
    transport = httpx.MockTransport(lambda req: httpx.Response(401, json={"detail": "missing or invalid API token"}))
    assert demo_check.main(["--server", "http://s", "--no-adb"], transport=transport) == 1
    assert "token rejected" in capsys.readouterr().out


def test_demo_check_missing_adb_is_a_warning():
    def missing(cmd):
        raise FileNotFoundError("adb")

    assert [c.status for c in demo_check.check_adb(missing)] == [demo_check.WARN]


def test_parse_questions_and_score():
    qs = rehearse.parse_questions(
        "# card\nwhere did I park? :: level 2 | blue pillar\n\ndid I see a hydrant? :: NONE  # absent\nwhat now?\n"
    )
    assert [(q.text, q.expected, q.none) for q in qs] == [
        ("where did I park?", ["level 2", "blue pillar"], False),
        ("did I see a hydrant?", [], True),
        ("what now?", None, False),
    ]
    assert rehearse.score("You parked on Level 2 by the lift.", qs[0]) is True
    assert rehearse.score("Level 3.", qs[0]) is False
    assert rehearse.score("I don't have a memory of a fire hydrant.", qs[1]) is True
    assert rehearse.score("Yes, a red hydrant by the cafe.", qs[1]) is False
    assert rehearse.score("anything", qs[2]) is None


def test_wait_drained_needs_an_empty_queue_and_a_stable_count():
    stats = iter([{"worker_queue": 5, "tokens": 1}, {"worker_queue": 0, "tokens": 9}, {"worker_queue": 0, "tokens": 9}])
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=next(stats)))
    with httpx.Client(base_url="http://s", transport=transport) as http:
        assert rehearse.wait_drained(http, 60, sleep=lambda s: None) is True


def test_rehearse_feeds_asks_scores_and_cleans_up(tmp_path, monkeypatch, capsys):
    video = tmp_path / "walk.mp4"
    video.write_bytes(b"")
    card = tmp_path / "q.txt"
    card.write_text("where did I park? :: level 2\nwhat colour was the dog? :: brown\n")
    monkeypatch.setattr(rehearse, "video_duration", lambda v: 600.0)
    fed: list[tuple] = []
    calls: list[tuple[str, str, dict | None]] = []
    answers = iter(["On level 2, row D.", "It was black."])

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else None
        calls.append((req.method, req.url.path, body or dict(req.url.params)))
        if req.url.path == "/v1/stats":
            return httpx.Response(200, json={"worker_queue": 0, "tokens": 7})
        if req.url.path == "/v1/chat":
            return httpx.Response(200, json={"answer": next(answers), "cited": []})
        if req.method == "DELETE":
            return httpx.Response(200, json={"deleted": 7})
        return httpx.Response(404)

    t0 = time.time()
    rc = rehearse.main(
        [str(video), str(card), "--server", "http://s", "--device-id", "rehearse-1", "--token", "tok"],
        transport=httpx.MockTransport(handler),
        feed=lambda *a: fed.append(a),
    )
    assert rc == 0
    server, token, _, device, start, fps, width = fed[0]
    assert (server, token, device, fps, width) == ("http://s", "tok", "rehearse-1", 1.0, 1280)
    assert t0 - 600 - 6 <= start <= time.time() - 600 - 5  # the walk ends just before the questions
    chats = [b for m, p, b in calls if p == "/v1/chat"]
    assert [c["device_id"] for c in chats] == ["rehearse-1", "rehearse-1"]
    assert len({c["session_id"] for c in chats}) == 2  # a fresh session per question
    assert ("DELETE", "/v1/memories", {"device_id": "rehearse-1"}) in calls
    out = capsys.readouterr().out
    assert "1/2 passed" in out and "FAIL" in out and "It was black." in out


def test_rehearse_keep_and_skip_feed(tmp_path):
    card = tmp_path / "q.txt"
    card.write_text("what now?\n")
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(f"{req.method} {req.url.path}")
        return httpx.Response(200, json={"answer": "walking", "cited": []})

    def no_feed(*a):
        raise AssertionError("must not feed")

    args = [str(tmp_path / "walk.mp4"), str(card), "--server", "http://s", "--device-id", "d", "--skip-feed", "--keep"]
    assert rehearse.main(args, transport=httpx.MockTransport(handler), feed=no_feed) == 0
    assert paths == ["POST /v1/chat"]
    with pytest.raises(SystemExit):
        rehearse.main([str(tmp_path / "walk.mp4"), str(card), "--skip-feed"])
