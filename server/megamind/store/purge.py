"""Retention / privacy CLI: delete memory tokens (+ thumbnails and real video clips) from the configured store.

    python -m megamind.store.purge                     # everything
    python -m megamind.store.purge --device quest-abc  # one device
    python -m megamind.store.purge --since 2h          # only tokens captured in the last 2 hours
    python -m megamind.store.purge --since 2026-11-07T09:00:00 --dry-run

`--since` accepts an ISO-8601 timestamp (local time if naive) or a relative age `<N>[s|m|h|d]`.
Mirrors `DELETE /v1/memories?device_id=&since=`; this CLI exists so the wearer can wipe data with the
server stopped (LanceDB is embedded, so both must not write concurrently).
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from datetime import datetime

from megamind.config import Settings
from megamind.store import build_store

_REL = re.compile(r"^(\d+(?:\.\d+)?)([smhd])$")
_UNIT_S = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_since(value: str | None, *, now: float | None = None) -> float | None:
    """`None` → `None`; `"90m"` → now-90 min; ISO string → its timestamp."""
    if not value:
        return None
    m = _REL.match(value.strip())
    if m:
        return (now if now is not None else time.time()) - float(m.group(1)) * _UNIT_S[m.group(2)]
    return datetime.fromisoformat(value.strip()).timestamp()


async def purge(
    settings: Settings,
    *,
    device_id: str | None = None,
    since: str | None = None,
    delete_thumbs: bool = True,
    dry_run: bool = False,
) -> int:
    store = build_store(settings)
    since_ts = parse_since(since)
    if dry_run:
        rows = await store.range(since_ts, None, limit=1_000_000, device_id=device_id)
        return len(rows)
    if delete_thumbs:  # frames aged into AV1 chunks: rewrite those chunks without the purged frames
        from megamind.store.media import MediaStore

        await MediaStore(settings, store).purge_window(device_id, since_ts)
        from megamind.video import VideoSettings, build_video_store

        build_video_store(settings, VideoSettings()).purge(device_id, since_ts)  # real video clips (POST /v1/video)
    return await store.purge(device_id, since_ts=since_ts, delete_thumbs=delete_thumbs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="only this device_id (default: all devices)")
    ap.add_argument("--since", help="only tokens with ts >= this (ISO-8601 or relative like 30m / 2h / 1d)")
    ap.add_argument("--keep-thumbs", action="store_true", help="delete rows but leave thumbnail JPEGs on disk")
    ap.add_argument("--dry-run", action="store_true", help="count matching tokens, delete nothing")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    a = ap.parse_args(argv)
    try:
        parse_since(a.since)
    except ValueError:
        ap.error(f"--since: expected ISO-8601 or <N>[s|m|h|d], got {a.since!r}")
    settings = Settings()
    if settings.store_backend != "lancedb":
        print(f"note: STORE_BACKEND={settings.store_backend!r} keeps nothing on disk; nothing to purge")
    scope = f"device={a.device or '*'} since={a.since or 'beginning'} backend={settings.store_backend}"
    if not a.dry_run and not a.yes:
        if input(f"Delete memories ({scope})? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1
    n = asyncio.run(
        purge(settings, device_id=a.device, since=a.since, delete_thumbs=not a.keep_thumbs, dry_run=a.dry_run)
    )
    print(f"{'would delete' if a.dry_run else 'deleted'} {n} token(s) ({scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
