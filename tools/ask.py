#!/usr/bin/env python3
"""CLI chat client.   python tools/ask.py --server http://localhost:8000 "where are my keys?"

Without --device-id it asks the most recently active device from GET /v1/devices (prints which one); pass
--device-id "" to search every device.
"""

from __future__ import annotations

import argparse
import json
import time

import httpx


def latest_device(http: httpx.Client) -> str | None:
    """The most recently active device from GET /v1/devices, or None (no devices, or an older server)."""
    try:
        r = http.get("/v1/devices")
        r.raise_for_status()
        devices = r.json().get("devices") or []
    except (httpx.HTTPError, ValueError):
        return None
    return devices[0]["device_id"] if devices else None


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--session", default="cli")
    ap.add_argument(
        "--device-id",
        default=None,
        help='default: most recent device from /v1/devices; "" = all devices',
    )
    ap.add_argument("--audio", help="path to a 16 kHz mono WAV to send instead of text")
    args = ap.parse_args(argv)
    if not args.audio and not args.text:
        raise SystemExit("text or --audio required")

    with httpx.Client(base_url=args.server, timeout=60.0, transport=transport) as http:
        device_id = args.device_id
        if device_id is None:
            device_id = latest_device(http)
            print(f"device: {device_id} (most recent, from /v1/devices)" if device_id else "device: all devices")
        device_id = device_id or None
        t0 = time.time()
        if args.audio:
            with open(args.audio, "rb") as f:
                r = http.post(
                    "/v1/chat",
                    files={"audio": ("q.wav", f, "audio/wav")},
                    data={
                        "session_id": args.session,
                        **({"device_id": device_id} if device_id else {}),
                    },
                )
        else:
            r = http.post(
                "/v1/chat",
                json={
                    "session_id": args.session,
                    "device_id": device_id,
                    "text": args.text,
                },
            )
        r.raise_for_status()
        body = r.json()
        if body.get("warning"):
            print(f"warning: {body['warning']}")
        print(f"[{time.time() - t0:.1f}s] transcript: {body['transcript']}")
        print(f"answer: {body['answer']}")
        for c in body.get("cited", []):
            when = time.strftime("%H:%M:%S", time.localtime(c["ts"]))
            thumb = args.server.rstrip("/") + (c.get("thumb_url") or f"/v1/thumbs/{c['id']}.jpg")
            print(f"  cited {c['id']} @ {when}: {c['caption'][:100]}\n    thumb: {thumb}")
        if body.get("tool_calls"):
            print("tool_calls:", json.dumps(body["tool_calls"], indent=1))
        if body.get("audio_url"):
            print("audio:", args.server + body["audio_url"])


if __name__ == "__main__":
    main()
