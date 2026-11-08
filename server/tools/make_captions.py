"""Caption a recording of the headset view with the turns the server really answered (`GET /v1/turns`).

    python tools/make_captions.py quest.mp4 --server https://...modal.run --token $API_TOKEN --burn quest-captioned.mp4
    python tools/make_captions.py quest.mp4 --turns data/turns.jsonl --start 2026-11-08T09:30:12-08:00

Each turn becomes two cues: "Q: ..." from just before the request reached the server until the answer was ready,
then "A: ... (1.9 s)" for as long as it takes to read. The video's start time comes from `--start`, else the
container's `creation_time` tag, else a `YYYYMMDD-HHMMSS` stamp in the file name (in `--tz`). If captions run
early or late, nudge them with `--offset` (seconds; positive = later). Writes `<video>.srt`; `--burn` also renders
them into a copy of the video with ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STAMP = re.compile(r"(\d{8})[-_](\d{6})")
WRAP = 42


@dataclass
class Cue:
    start: float  # seconds into the video
    end: float
    text: str


def parse_start(value: str, tz: str) -> float:
    try:
        return float(value)
    except ValueError:
        pass
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.timestamp()


def start_from_name(name: str, tz: str) -> float | None:
    """`com.oculus.vrshell-20261108-093012.mp4` -> epoch s, the stamp read in `tz` (the headset's clock)."""
    m = STAMP.search(name)
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo(tz))
    return dt.timestamp()


def probe(video: Path) -> tuple[float | None, float | None]:
    """(creation_time epoch s, duration s) from ffprobe; None where missing."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:format_tags=creation_time",
                "-of",
                "json",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    fmt = json.loads(out).get("format", {})
    created = fmt.get("tags", {}).get("creation_time")
    duration = fmt.get("duration")
    return (parse_start(created, "UTC") if created else None), (float(duration) if duration else None)


def video_start(video: Path, start: str | None, tz: str) -> float:
    if start:
        return parse_start(start, tz)
    created, _ = probe(video)
    if created is not None:
        return created
    named = start_from_name(video.name, tz)
    if named is not None:
        return named
    sys.exit(f"can't tell when {video.name} started recording: pass --start (epoch s or ISO time)")


def read_time(text: str) -> float:
    return min(8.0, max(2.5, len(text.split()) / 2.8))


def build_cues(
    turns: list[dict],
    start: float,
    duration: float | None = None,
    lead_s: float = 1.0,
    latency: bool = True,
) -> list[Cue]:
    """Turns (epoch-s `started_at`/`ended_at`) -> cues relative to the video `start`, clipped to the video."""
    turns = sorted(turns, key=lambda t: t["started_at"])
    cues: list[Cue] = []
    for i, t in enumerate(turns):
        q0, q1 = t["started_at"] - lead_s - start, t["ended_at"] - start
        a1 = q1 + read_time(t["answer"])
        if i + 1 < len(turns):
            a1 = min(a1, turns[i + 1]["started_at"] - lead_s - start)
        took = f" ({t['ended_at'] - t['started_at']:.1f} s)" if latency else ""
        for c0, c1, text in ((q0, q1, f"Q: {t['question']}"), (q1, a1, f"A: {t['answer']}{took}")):
            c0 = max(c0, 0.0)
            if duration is not None:
                c1 = min(c1, duration)
            if c1 - c0 >= 0.3:
                cues.append(Cue(c0, c1, "\n".join(textwrap.wrap(text, WRAP))))
    return cues


def _ts(s: float) -> str:
    ms = round(s * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    return "".join(f"{i}\n{_ts(c.start)} --> {_ts(c.end)}\n{c.text}\n\n" for i, c in enumerate(cues, 1))


def fetch_turns(server: str, token: str | None, since: float, until: float, device_id: str | None) -> list[dict]:
    q = {"since": since, "until": until, "limit": 1000} | ({"device_id": device_id} if device_id else {})
    req = urllib.request.Request(f"{server.rstrip('/')}/v1/turns?{urllib.parse.urlencode(q)}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["turns"]


def load_turns(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def burn(video: Path, srt: Path, out: Path) -> None:
    style = "FontName=DejaVu Sans,FontSize=16,Outline=2,Shadow=0,MarginV=36,BorderStyle=1"
    sub = str(srt).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", f"subtitles='{sub}':force_style='{style}'"]
    subprocess.run([*cmd, "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-c:a", "copy", str(out)], check=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--server", default=os.environ.get("MEGAMIND_SERVER"), help="base URL (env MEGAMIND_SERVER)")
    src.add_argument("--turns", type=Path, help="a turns.jsonl instead of asking the server")
    ap.add_argument("--token", default=os.environ.get("MEGAMIND_TOKEN"), help="API_TOKEN (env MEGAMIND_TOKEN)")
    ap.add_argument("--device-id")
    ap.add_argument("--start", help="when the recording started: epoch s or ISO time (default: from the file)")
    ap.add_argument("--tz", default="America/Los_Angeles", help="zone of --start / the file-name stamp")
    ap.add_argument("--offset", type=float, default=0.0, help="shift every caption by this many seconds")
    ap.add_argument("--lead", type=float, default=1.0, help="show the question this long before the server got it")
    ap.add_argument("--no-latency", action="store_true", help="don't append '(1.9 s)' to answers")
    ap.add_argument("--out", type=Path, help="SRT path (default <video>.srt)")
    ap.add_argument("--burn", type=Path, help="also write a copy of the video with the captions rendered in")
    a = ap.parse_args(argv)

    start = video_start(a.video, a.start, a.tz) - a.offset
    _, duration = probe(a.video)
    if a.turns:
        turns = load_turns(a.turns)
    elif a.server:
        turns = fetch_turns(a.server, a.token, start, start + (duration or 6 * 3600), a.device_id)
    else:
        sys.exit("pass --server (or set MEGAMIND_SERVER) or --turns")
    if a.device_id:
        turns = [t for t in turns if t.get("device_id") == a.device_id]
    cues = build_cues(turns, start, duration, lead_s=a.lead, latency=not a.no_latency)
    out = a.out or a.video.with_suffix(".srt")
    out.write_text(to_srt(cues))
    print(f"{len(cues)} captions from {len(turns)} turns -> {out}")
    if not cues:
        print("no turn falls inside the video: check --start / --offset / --device-id", file=sys.stderr)
    if a.burn:
        burn(a.video, out, a.burn)
        print(f"captioned video -> {a.burn}")


if __name__ == "__main__":
    main()
