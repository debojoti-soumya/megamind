#!/usr/bin/env python3
"""Dry-run the demo route before the shoot. Film the walk on a phone, write the question card, then

    python tools/rehearse.py walk.mp4 questions.txt --server $U        # token: MEGAMIND_API_TOKEN

1. feeds the clip to the real server as a fresh device (tools/feeder.py), stamped so the walk ends just now
2. waits until enrichment has drained (GET /v1/stats)
3. asks every question against that device only, each in a new session
4. prints answer / pass / seconds, then deletes the rehearsal memories (--keep leaves them)

questions.txt, one question per line, "#" starts a comment:

    where did I park? :: level 2 | blue pillar
    what was the name of the cafe on the corner? :: blue bottle
    did I see a fire hydrant? :: NONE
    what was I doing just now?

After "::" come the accepted answers: the question passes if any one is in the answer (case-insensitive).
NONE passes when the answer says it has no memory of it. No "::" just prints the answer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

FEEDER = Path(__file__).resolve().parent / "feeder.py"
DECLINE = re.compile(
    r"\b(don'?t|do not|didn'?t|did not|no memory|no record|not see|haven'?t|couldn'?t|could not|can'?t find|"
    r"cannot find|not sure|no sign)\b",
    re.I,
)


@dataclass
class Question:
    text: str
    expected: list[str] | None = None  # None: unscored
    none: bool = False  # the right answer is "I don't remember that"


@dataclass
class Result:
    question: Question
    answer: str
    seconds: float
    cited: int
    passed: bool | None


def parse_questions(text: str) -> list[Question]:
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        q, sep, exp = (s.strip() for s in line.partition("::"))
        alts = [a.strip() for a in exp.split("|") if a.strip()] if sep else []
        if alts == ["NONE"]:
            out.append(Question(q, [], none=True))
        else:
            out.append(Question(q, alts or None))
    return out


def score(answer: str, q: Question) -> bool | None:
    if q.none:
        return bool(DECLINE.search(answer))
    if not q.expected:
        return None
    a = answer.lower()
    return any(e.lower() in a for e in q.expected)


def video_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def feed_video(server: str, token: str, video: Path, device_id: str, start: float, fps: float, width: int) -> None:
    cmd = [sys.executable, str(FEEDER), "--server", server, "--source", str(video), "--device-id", device_id]
    cmd += ["--start", f"{start:.3f}", "--fps", str(fps), "--width", str(width), "--quality", "80"]
    subprocess.run(cmd, env={**os.environ, "MEGAMIND_API_TOKEN": token}, check=True)


def wait_drained(http: httpx.Client, timeout_s: float, poll_s: float = 2.0, sleep=time.sleep) -> bool:
    """True once the enrichment queue is empty and the token count held still for one poll."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        s = http.get("/v1/stats").json()
        if s.get("worker_queue", 0) == 0 and s.get("tokens") == last:
            return True
        last = s.get("tokens") if s.get("worker_queue", 0) == 0 else None
        sleep(poll_s)
    return False


def ask(http: httpx.Client, device_id: str, q: Question, session: str) -> Result:
    t0 = time.monotonic()
    r = http.post("/v1/chat", json={"session_id": session, "device_id": device_id, "text": q.text})
    r.raise_for_status()
    dt = time.monotonic() - t0
    body = r.json()
    answer = " ".join(body.get("answer", "").split())
    return Result(q, answer, dt, len(body.get("cited", [])), score(answer, q))


def report(results: list[Result]) -> str:
    mark = {True: "PASS", False: "FAIL", None: "  - "}
    lines = [f"{mark[r.passed]}  {r.seconds:4.1f} s  {r.question.text}\n        -> {r.answer[:160]}" for r in results]
    scored = [r for r in results if r.passed is not None]
    ok = sum(bool(r.passed) for r in scored)
    secs = [r.seconds for r in results]
    if secs:
        lines.append(
            f"{ok}/{len(scored)} passed, median {statistics.median(secs):.1f} s, slowest {max(secs):.1f} s"
            + ("  (keep the PASS questions for the take)" if scored else "")
        )
    return "\n".join(lines)


Feed = Callable[[str, str, Path, str, float, float, int], None]


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None, feed: Feed = feed_video) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", type=Path, help="phone video of the route")
    ap.add_argument("questions", type=Path, help="question card, see the module docstring")
    ap.add_argument("--server", default=os.environ.get("MEGAMIND_SERVER", "http://localhost:8000"))
    ap.add_argument("--token", default=os.environ.get("MEGAMIND_API_TOKEN", ""), help="env MEGAMIND_API_TOKEN")
    ap.add_argument("--device-id", default=None, help="default rehearse-<time>: a fresh id per run")
    ap.add_argument("--fps", type=float, default=1.0, help="frames per second sent, like the headset")
    ap.add_argument("--width", type=int, default=1280, help="frame width sent, like the headset")
    ap.add_argument("--wait", type=float, default=300.0, help="max seconds to wait for enrichment")
    ap.add_argument("--keep", action="store_true", help="keep the rehearsal memories (default: delete them)")
    ap.add_argument("--skip-feed", action="store_true", help="ask only: the clip was fed already (needs --device-id)")
    args = ap.parse_args(argv)
    if args.skip_feed and not args.device_id:
        raise SystemExit("--skip-feed needs the --device-id of the earlier feed")
    questions = parse_questions(args.questions.read_text())
    if not questions:
        raise SystemExit(f"no questions in {args.questions}")
    # a fresh device every run: the keyframe filter drops frames older than a device's newest kept one
    device_id = args.device_id or f"rehearse-{int(time.time())}"
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    with httpx.Client(base_url=args.server, timeout=90.0, transport=transport, headers=headers) as http:
        if not args.skip_feed:
            duration = video_duration(args.video)
            start = time.time() - duration - 5  # the walk ends just now: "just now" / "most recently" questions work
            print(f"feeding {args.video} ({duration:.0f} s) as {device_id}")
            feed(args.server, args.token, args.video, device_id, start, args.fps, args.width)
            print("waiting for enrichment...")
            if not wait_drained(http, args.wait):
                print(f"warning: enrichment still busy after {args.wait:.0f} s; answers may miss the last frames")
        results = []
        for i, q in enumerate(questions):
            results.append(ask(http, device_id, q, f"{device_id}-q{i}"))
            print(report(results[-1:]).split("\n")[0], flush=True)
        if not args.keep:
            n = http.delete("/v1/memories", params={"device_id": device_id}).json()
            print(f"deleted rehearsal memories of {device_id}: {n}")
    print()
    print(report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
