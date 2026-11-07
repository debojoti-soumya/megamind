#!/usr/bin/env python3
"""Emulate the headset: push frames from a webcam or video file to POST /v1/frames.

    python tools/feeder.py --server http://localhost:8000 --source 0 --fps 1
    python tools/feeder.py --server http://localhost:8000 --source walk.mp4 --fps 1 --realtime

Requires: pip install opencv-python-headless httpx
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--device-id", default="feeder")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--realtime", action="store_true", help="for video files: use the video's own timestamps")
    args = ap.parse_args()

    src: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(video_fps / args.fps))) if isinstance(src, str) else 1
    period = 1.0 / args.fps
    start = time.time()
    sent = accepted = 0
    idx = 0

    with httpx.Client(base_url=args.server, timeout=10.0) as http:
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
            if not isinstance(src, str):
                time.sleep(max(0.0, period - 0.05))
    print()


if __name__ == "__main__":
    main()
