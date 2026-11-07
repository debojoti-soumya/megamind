#!/usr/bin/env python3
"""CLI chat client.   python tools/ask.py --server http://localhost:8000 "where are my keys?" """
from __future__ import annotations

import argparse
import json
import time

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--session", default="cli")
    ap.add_argument("--device-id", default=None)
    ap.add_argument("--audio", help="path to a 16 kHz mono WAV to send instead of text")
    args = ap.parse_args()

    with httpx.Client(base_url=args.server, timeout=60.0) as http:
        t0 = time.time()
        if args.audio:
            with open(args.audio, "rb") as f:
                r = http.post(
                    "/v1/chat",
                    files={"audio": ("q.wav", f, "audio/wav")},
                    data={"session_id": args.session, **({"device_id": args.device_id} if args.device_id else {})},
                )
        else:
            if not args.text:
                raise SystemExit("text or --audio required")
            r = http.post(
                "/v1/chat",
                json={"session_id": args.session, "device_id": args.device_id, "text": args.text},
            )
        r.raise_for_status()
        body = r.json()
        print(f"[{time.time() - t0:.1f}s] transcript: {body['transcript']}")
        print(f"answer: {body['answer']}")
        for c in body.get("cited", []):
            print(f"  cited {c['id']} @ {time.strftime('%H:%M:%S', time.localtime(c['ts']))}: {c['caption'][:100]}")
        if body.get("tool_calls"):
            print("tool_calls:", json.dumps(body["tool_calls"], indent=1))
        if body.get("audio_url"):
            print("audio:", args.server + body["audio_url"])


if __name__ == "__main__":
    main()
