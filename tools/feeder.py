#!/usr/bin/env python3
"""Emulate the headset: push frames from a webcam or video file to POST /v1/frames.

    python tools/feeder.py --server http://localhost:8000 --source 0 --fps 1
    python tools/feeder.py --server http://localhost:8000 --source walk.mp4 --fps 1 --realtime
    python tools/feeder.py --server http://localhost:8000 --source walk.mp4 --fps 1 --audio   # + speech -> /v1/audio

Requires: pip install opencv-python-headless httpx  (and ffmpeg on PATH for --audio)
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import cv2
import httpx


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
        self.chunks = self.tokens = 0

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
            self.tokens += len(body.get("token_ids") or [])
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
    args = ap.parse_args()

    src: int | str = int(args.source) if args.source.isdigit() else args.source
    if args.audio:
        if not isinstance(src, str):
            raise SystemExit("--audio needs a video file as --source (webcam audio is not supported)")
        args.realtime = True  # frames and audio must share the video timeline
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(video_fps / args.fps))) if isinstance(src, str) else 1
    period = 1.0 / args.fps
    start = time.time()
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
            meta = {"device_id": args.device_id, "ts": ts}
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
        print(f"audio: {audio.chunks} chunks sent, {audio.tokens} transcript tokens")


if __name__ == "__main__":
    main()
