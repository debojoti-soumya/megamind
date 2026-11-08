#!/usr/bin/env python3
"""Emulate the headset: push frames from a webcam or video file to POST /v1/frames.

    python tools/feeder.py --server http://localhost:8000 --source 0 --fps 1
    python tools/feeder.py --server http://localhost:8000 --source walk.mp4 --fps 1 --realtime
    python tools/feeder.py --server http://localhost:8000 --source walk.mp4 --fps 1 --audio   # + speech -> /v1/audio
    python tools/feeder.py --source walk.mp4 --start 1767600000 --srt walk.srt   # clock time + subtitles

--start EPOCH stamps a video file's first frame with that unix time (implies --realtime). --srt FILE sends each
subtitle cue once, as FrameMeta.transcript on the first sampled frame at/after the cue's start (cue times are
relative to the video). Servers without FrameMeta.transcript ignore the field.

Requires: pip install opencv-python-headless httpx  (and ffmpeg on PATH for --audio)
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import httpx

Cue = tuple[float, float, str]  # (start_s, end_s, text) relative to the video


def _secs(t: str) -> float:
    h, m, s = t.strip().replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


_CUE_RE = re.compile(r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)[^\n]*\n(.*)", re.S)


def parse_srt(text: str) -> list[tuple[float, float, list[str]]]:
    """SRT text -> [(start_s, end_s, [non-empty text lines])], in file order. Malformed blocks are skipped."""
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        m = _CUE_RE.search(block)
        if m:
            lines = [ln.strip() for ln in m[3].splitlines() if ln.strip()]
            if lines:
                out.append((_secs(m[1]), _secs(m[2]), lines))
    return out


def load_srt(path: str | Path) -> list[Cue]:
    cues = parse_srt(Path(path).read_text(encoding="utf-8", errors="ignore"))
    return sorted((a, b, " ".join(" ".join(lines).split())) for a, b, lines in cues)


def due_cues(cues: list[Cue], offset_s: float) -> tuple[str | None, list[Cue]]:
    """Cues starting at or before `offset_s` (video seconds) -> (their text joined, the cues still pending).
    Each cue goes out exactly once: on the first sampled frame at/after its start."""
    due = [c for c in cues if c[0] <= offset_s]
    if not due:
        return None, cues
    return " ".join(c[2] for c in due), [c for c in cues if c[0] > offset_s]


def extract_audio(path: str, rate: int = 16000) -> bytes | None:
    """The file's audio track as 16 kHz mono s16 PCM (no header); None if there is none / ffmpeg is missing."""
    if shutil.which("ffmpeg") is None:
        print("--audio: ffmpeg not found on PATH; sending frames only")
        return None
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "a.wav"
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le"]
        r = subprocess.run([*cmd, str(out)], capture_output=True)
        if r.returncode or not out.is_file():
            print(f"--audio: no decodable audio track in {path}; sending frames only")
            return None
        with wave.open(str(out)) as w:
            return w.readframes(w.getnframes())


class AudioSender:
    """Posts the audio track to /v1/audio in `chunk_s` chunks, interleaved with the frames on the video timeline:
    chunk [t0, t1) is sent once frames past t1 have gone out, with ts_start = start + t0 (same clock as frames)."""

    def __init__(self, http, pcm: bytes, start: float, device_id: str, chunk_s: float, rate: int = 16000) -> None:
        self.http, self.pcm, self.start, self.device_id = http, pcm, start, device_id
        self.chunk_s, self.rate = chunk_s, rate
        self.duration = len(pcm) / (2 * rate)
        self.next_t = 0.0
        self.chunks = 0
        self.token_ids: set[str] = set()  # a chunk may continue (merge into) the last token

    def pump(self, video_t: float) -> None:
        while self.next_t + self.chunk_s <= min(video_t, self.duration):
            self._send(self.next_t, self.next_t + self.chunk_s)

    def flush(self) -> None:
        while self.duration - self.next_t >= 0.5:
            self._send(self.next_t, min(self.next_t + self.chunk_s, self.duration))

    def _send(self, t0: float, t1: float) -> None:
        self.next_t = t1
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.rate)
            w.writeframes(self.pcm[int(t0 * self.rate) * 2 : int(t1 * self.rate) * 2])
        meta = {"device_id": self.device_id, "ts_start": self.start + t0}
        try:
            r = self.http.post(
                "/v1/audio",
                files={"audio": ("chunk.wav", buf.getvalue(), "audio/wav")},
                data={"meta": json.dumps(meta)},
                timeout=120.0,  # transcription is synchronous
            )
            r.raise_for_status()
            body = r.json()
            self.chunks += 1
            self.token_ids.update(body.get("token_ids") or [])
            if body.get("transcript"):
                print(f"\n[audio {t0:6.1f}s] {body['transcript']}")
        except Exception as exc:  # noqa: BLE001
            print(f"\naudio upload failed ({t0:.1f}-{t1:.1f}s): {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--device-id", default="feeder")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--realtime", action="store_true", help="for video files: use the video's own timestamps")
    ap.add_argument(
        "--audio",
        action="store_true",
        help="video files: also send the audio track to /v1/audio (needs ffmpeg; implies --realtime timestamps)",
    )
    ap.add_argument("--audio-chunk", type=float, default=15.0, help="seconds of audio per /v1/audio request")
    ap.add_argument(
        "--start", type=float, default=None, help="video files: unix ts of the first frame (implies --realtime)"
    )
    ap.add_argument("--srt", default=None, help="video files: subtitles sent once per cue as FrameMeta.transcript")
    args = ap.parse_args()

    import cv2  # tools extra; imported here so the helpers above stay importable without it

    src: int | str = int(args.source) if args.source.isdigit() else args.source
    if args.audio:
        if not isinstance(src, str):
            raise SystemExit("--audio needs a video file as --source (webcam audio is not supported)")
        args.realtime = True  # frames and audio must share the video timeline
    if (args.start is not None or args.srt) and not isinstance(src, str):
        raise SystemExit("--start/--srt need a video file as --source")
    if args.start is not None:
        args.realtime = True
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(video_fps / args.fps))) if isinstance(src, str) else 1
    period = 1.0 / args.fps
    start = time.time() if args.start is None else args.start
    cues = load_srt(args.srt) if args.srt else []
    if cues and args.audio:
        print("note: --srt and --audio both send speech; the server will store it twice")
    sent = accepted = 0
    idx = 0

    pcm = extract_audio(args.source) if args.audio else None

    with httpx.Client(base_url=args.server, timeout=10.0) as http:
        audio = AudioSender(http, pcm, start, args.device_id, args.audio_chunk) if pcm else None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if isinstance(src, str) and idx % step:
                continue
            h, w = frame.shape[:2]
            if w > args.width:
                frame = cv2.resize(frame, (args.width, int(h * args.width / w)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                continue
            ts = start + idx / video_fps if (args.realtime and isinstance(src, str)) else time.time()
            meta: dict = {"device_id": args.device_id, "ts": ts}
            if cues:
                said, cues = due_cues(cues, idx / video_fps)
                if said:
                    meta["transcript"] = said
            try:
                r = http.post(
                    "/v1/frames",
                    files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                    data={"meta": json.dumps(meta)},
                )
                body = r.json()
                sent += 1
                accepted += bool(body.get("accepted"))
                print(f"\r{sent} sent, {accepted} accepted  last={body.get('reason') or 'ok'}   ", end="", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"\nupload failed: {exc}")
            if audio is not None:
                audio.pump(idx / video_fps)
            if not isinstance(src, str):
                time.sleep(max(0.0, period - 0.05))
        if audio is not None:
            audio.flush()
    print()
    if audio is not None:
        print(f"audio: {audio.chunks} chunks sent, {len(audio.token_ids)} transcript tokens")


if __name__ == "__main__":
    main()
