#!/usr/bin/env python3
"""Pre-take check for the demo: one PASS / WARN / FAIL line per thing that has to work before you hit record.

    python tools/demo_check.py --server https://<workspace>--megamind-web.modal.run   # token: MEGAMIND_API_TOKEN
    python tools/demo_check.py --server ... --no-adb --no-ask                          # server only, no paid call

Checks:
- server health and warmup
- the models really serving: Claude, STT, TTS, text embeddings and SigLIP image vectors, none mocked
- the turn log (GET /v1/turns), which the demo captions come from
- the headset over adb: connected, authorised, battery (ADB_SERIAL picks one of several)
- frames from the headset arrived in the last --fresh seconds
- one timed test question

Exits 1 if anything FAILs. The test question is a real turn, so it shows up in GET /v1/turns; run this before you
start recording, not during the take.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
LIVE_ROLES = ("claude", "embed", "stt", "tts")


@dataclass
class Check:
    status: str
    name: str
    detail: str


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def check_health(http: httpx.Client) -> list[Check]:
    try:
        r = http.get("/v1/health")
    except httpx.HTTPError as exc:
        return [Check(FAIL, "server", f"unreachable: {exc}")]
    if r.status_code == 401:
        return [Check(FAIL, "server", "token rejected (set MEGAMIND_API_TOKEN or --token)")]
    if r.status_code != 200:
        return [Check(FAIL, "server", f"/v1/health -> {r.status_code}")]
    h = r.json()
    warmup = h.get("warmup", "off")
    out = [
        Check(
            PASS if h.get("status") == "ok" else FAIL, "server", f"{h.get('status')}, {h.get('store_count')} memories"
        ),
        Check(
            WARN if warmup == "running" or "failed" in warmup else PASS,
            "warmup",
            warmup + (" (first question may be slow)" if warmup == "running" else ""),
        ),
    ]
    models = h.get("models", {})
    for role in LIVE_ROLES:
        label = models.get(role, "?")
        bad = label.startswith(("mock", "fallback", "?"))
        out.append(Check(FAIL if bad else PASS, role, label))
    image = models.get("image")
    if image is None:
        out.append(Check(WARN, "image", "server does not report image vectors (older build)"))
    else:
        bad = image.startswith(("none", "mock"))
        out.append(
            Check(FAIL if bad else PASS, "image", image + (" - colour/object questions get weaker" if bad else ""))
        )
    return out


def check_turns(http: httpx.Client) -> Check:
    try:
        r = http.get("/v1/turns", params={"limit": 1})
    except httpx.HTTPError as exc:
        return Check(FAIL, "turn log", f"unreachable: {exc}")
    if r.status_code == 404:
        return Check(FAIL, "turn log", "no GET /v1/turns on this server: captions need the #196 deploy")
    if r.status_code != 200:
        return Check(FAIL, "turn log", f"/v1/turns -> {r.status_code}")
    turns = r.json().get("turns", [])
    last = time.strftime("%H:%M:%S", time.localtime(turns[-1]["ended_at"])) if turns else "none yet"
    return Check(PASS, "turn log", f"on (last turn {last})")


def pick_device(devices: list[dict], device_id: str | None) -> dict | None:
    if device_id:
        return next((d for d in devices if d["device_id"] == device_id), None)
    with_frames = [d for d in devices if d.get("last_frame_ts")]
    return max(with_frames, key=lambda d: d["last_frame_ts"], default=None)


def check_uploads(http: httpx.Client, device_id: str | None, fresh_s: float, now: float) -> tuple[Check, str | None]:
    try:
        r = http.get("/v1/devices")
        r.raise_for_status()
    except httpx.HTTPError as exc:
        return Check(FAIL, "uploads", f"/v1/devices failed: {exc}"), device_id
    dev = pick_device(r.json().get("devices", []), device_id)
    if dev is None or not dev.get("last_frame_ts"):
        return Check(FAIL, "uploads", f"no frames stored from {device_id or 'any device'} yet"), device_id
    age = now - dev["last_frame_ts"]
    if age <= fresh_s:
        return Check(PASS, "uploads", f"{dev['device_id']}: last frame {age:.0f} s ago"), dev["device_id"]
    return (
        Check(
            FAIL,
            "uploads",
            f"{dev['device_id']}: last frame {age:.0f} s ago (capture paused? Wi-Fi/hotspot? look at the REC strip)",
        ),
        dev["device_id"],
    )


Run = Callable[[list[str]], str]


def run_cmd(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False).stdout


def check_adb(run: Run = run_cmd, serial: str | None = None) -> list[Check]:
    try:
        listing = run(["adb", "devices"])
    except FileNotFoundError:
        return [Check(WARN, "adb", "adb not installed: skipped (headset battery unknown)")]
    except subprocess.TimeoutExpired:
        return [Check(FAIL, "adb", "adb devices timed out")]
    rows = [ln.split("\t") for ln in listing.splitlines()[1:] if "\t" in ln]
    if serial:
        rows = [r for r in rows if r[0] == serial]
    if not rows:
        return [Check(FAIL, "adb", f"no headset{' ' + serial if serial else ''} (USB cable, or adb connect <ip>)")]
    if rows[0][1].strip() == "unauthorized":
        return [Check(FAIL, "adb", f"{rows[0][0]} unauthorized: accept the USB debugging prompt in the headset")]
    if len(rows) > 1:
        return [Check(FAIL, "adb", f"{len(rows)} devices: set ADB_SERIAL to one of {', '.join(r[0] for r in rows)}")]
    sn = rows[0][0]
    out = [Check(PASS, "adb", f"{sn} connected")]
    m = re.search(r"level:\s*(\d+)", run(["adb", "-s", sn, "shell", "dumpsys", "battery"]))
    if m is None:
        out.append(Check(WARN, "battery", "unknown"))
    else:
        level = int(m.group(1))
        status = FAIL if level < 20 else WARN if level < 50 else PASS
        out.append(Check(status, "battery", f"{level}%" + (" - charge before the take" if status != PASS else "")))
    return out


def check_ask(http: httpx.Client, device_id: str | None, question: str, warn_s: float, fail_s: float) -> Check:
    t0 = time.monotonic()
    try:
        r = http.post("/v1/chat", json={"session_id": "demo-check", "device_id": device_id, "text": question})
        r.raise_for_status()
    except httpx.HTTPError as exc:
        return Check(FAIL, "question", f"/v1/chat failed: {exc}")
    dt = time.monotonic() - t0
    body = r.json()
    answer = " ".join(body.get("answer", "").split())
    status = FAIL if dt > fail_s or body.get("warning") else WARN if dt > warn_s else PASS
    detail = f"{dt:.1f} s: {answer[:90]}"
    if body.get("warning"):
        detail += f" (warning: {body['warning']})"
    return Check(status, "question", detail)


def run_checks(args: argparse.Namespace, http: httpx.Client, run: Run = run_cmd) -> list[Check]:
    checks = check_health(http)
    if checks[0].status == FAIL:
        return checks
    checks.append(check_turns(http))
    if not args.no_adb:
        checks += check_adb(run, os.environ.get("ADB_SERIAL") or None)
    uploads, device_id = check_uploads(http, args.device_id, args.fresh, time.time())
    checks.append(uploads)
    if not args.no_ask:
        checks.append(check_ask(http, device_id, args.question, args.warn_s, args.fail_s))
    return checks


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None, run: Run = run_cmd) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default=os.environ.get("MEGAMIND_SERVER", "http://localhost:8000"))
    ap.add_argument("--token", default=os.environ.get("MEGAMIND_API_TOKEN", ""), help="env MEGAMIND_API_TOKEN")
    ap.add_argument("--device-id", default=None, help="headset device id (default: the one that uploaded last)")
    ap.add_argument("--fresh", type=float, default=20.0, help="newest frame must be at most this many seconds old")
    ap.add_argument("--question", default="what did I just see?")
    ap.add_argument("--warn-s", type=float, default=3.0)
    ap.add_argument("--fail-s", type=float, default=8.0)
    ap.add_argument("--no-adb", action="store_true", help="skip the headset adb/battery checks")
    ap.add_argument("--no-ask", action="store_true", help="skip the test question (no paid call, no turn logged)")
    args = ap.parse_args(argv)
    with httpx.Client(
        base_url=args.server, timeout=30.0, transport=transport, headers=auth_headers(args.token)
    ) as http:
        checks = run_checks(args, http, run)
    for c in checks:
        print(f"{c.status}  {c.name:<9} {c.detail}")
    failed = sum(c.status == FAIL for c in checks)
    print("ready to record" if not failed else f"{failed} problem(s): fix before recording")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
