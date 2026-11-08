"""Per-device ring buffer of *every* received frame JPEG (moment replay).

The keyframe filter keeps ~1 frame in 10, which is right for memory tokens but leaves nothing to
replay. This buffer sits next to it, fed from `POST /v1/frames` before the filter runs, and keeps
the raw JPEGs for a sliding window. Bounds are per device: frames older than `window_s` (measured
against the newest frame's timestamp, so replaying recorded footage works) are dropped first, then
the oldest frames until both `max_frames` and `max_bytes` hold.

With `spool_dir` set, JPEGs live on disk (one file per frame) instead of in RAM, and the buffer is
rehydrated from that directory on startup.
"""

from __future__ import annotations

import bisect
import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, unquote

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferedFrame:
    ts: float
    size: int
    data: bytes | None = None  # in-memory mode
    path: Path | None = None  # spool mode

    def read(self) -> bytes:
        if self.data is not None:
            return self.data
        assert self.path is not None
        return self.path.read_bytes()


@dataclass
class _Ring:
    items: list[BufferedFrame] = field(default_factory=list)  # sorted by ts
    nbytes: int = 0


def _key(f: BufferedFrame) -> float:
    return f.ts


class ReplayBuffer:
    def __init__(
        self,
        window_s: float = 900.0,
        max_frames: int = 9000,
        max_bytes: int = 256 << 20,
        spool_dir: Path | None = None,
    ) -> None:
        self.window_s = window_s
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self.spool_dir = Path(spool_dir) if spool_dir else None
        self._rings: dict[str, _Ring] = {}
        self._seq = itertools.count()
        if self.spool_dir is not None and self.spool_dir.is_dir():
            self._rehydrate()

    # ---- writes ----

    def add(self, device_id: str, ts: float, jpeg: bytes) -> None:
        ring = self._rings.setdefault(device_id, _Ring())
        if self.spool_dir is not None:
            d = self.spool_dir / quote(device_id, safe="")
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{ts:.6f}_{next(self._seq)}.jpg"
            p.write_bytes(jpeg)
            frame = BufferedFrame(ts=ts, size=len(jpeg), path=p)
        else:
            frame = BufferedFrame(ts=ts, size=len(jpeg), data=jpeg)
        self._insert(ring, frame)
        self._evict(ring)

    def tap(self, meta, jpeg: bytes) -> None:
        """Frame-ingest hook (`app.state.frame_taps`): never let replay break ingest."""
        try:
            self.add(meta.device_id, float(meta.ts), jpeg)
        except Exception:  # noqa: BLE001
            log.exception("replay buffer: dropping frame")

    def purge(self, device_id: str | None = None, since_ts: float | None = None) -> int:
        """Mirror of `DELETE /v1/memories`: drop frames (all devices by default; only `ts >= since_ts`)."""
        n = 0
        for dev in [device_id] if device_id is not None else list(self._rings):
            ring = self._rings.get(dev)
            if ring is None:
                continue
            cut = 0 if since_ts is None else bisect.bisect_left(ring.items, since_ts, key=_key)
            dropped, ring.items = ring.items[cut:], ring.items[:cut]
            for f in dropped:
                self._release(ring, f)
            n += len(dropped)
            if not ring.items:
                del self._rings[dev]
        return n

    # ---- reads ----

    def frames(self, device_id: str, t0: float, t1: float) -> list[BufferedFrame]:
        ring = self._rings.get(device_id)
        if ring is None:
            return []
        lo = bisect.bisect_left(ring.items, t0, key=_key)
        hi = bisect.bisect_right(ring.items, t1, key=_key)
        return ring.items[lo:hi]

    def count(self, device_id: str, t0: float, t1: float) -> int:
        return len(self.frames(device_id, t0, t1))

    def resolve_device(self, t0: float, t1: float, device_id: str | None = None) -> str | None:
        """`device_id` if it has frames in [t0, t1]; with no device, the one with the most frames there."""
        if device_id is not None:
            return device_id if self.count(device_id, t0, t1) else None
        best, best_n = None, 0
        for dev in self._rings:
            n = self.count(dev, t0, t1)
            if n > best_n:
                best, best_n = dev, n
        return best

    def newest_ts(self, device_id: str | None = None) -> float | None:
        rings = [self._rings[device_id]] if device_id in self._rings else [] if device_id else self._rings.values()
        ts = [r.items[-1].ts for r in rings if r.items]
        return max(ts) if ts else None

    def stats(self) -> dict[str, dict[str, float]]:
        return {
            dev: {"frames": len(r.items), "bytes": r.nbytes, "from": r.items[0].ts, "to": r.items[-1].ts}
            for dev, r in self._rings.items()
            if r.items
        }

    # ---- internals ----

    def _insert(self, ring: _Ring, frame: BufferedFrame) -> None:
        if not ring.items or frame.ts >= ring.items[-1].ts:
            ring.items.append(frame)
        else:  # out-of-order arrival
            bisect.insort_right(ring.items, frame, key=_key)
        ring.nbytes += frame.size

    def _evict(self, ring: _Ring) -> None:
        items = ring.items
        cut = 0
        if items and self.window_s > 0:
            cut = bisect.bisect_left(items, items[-1].ts - self.window_s, key=_key)
        nbytes = ring.nbytes - sum(f.size for f in items[:cut])
        while cut < len(items) and (len(items) - cut > self.max_frames or nbytes > self.max_bytes):
            nbytes -= items[cut].size
            cut += 1
        if cut:
            dropped, ring.items = items[:cut], items[cut:]
            for f in dropped:
                self._release(ring, f)

    def _release(self, ring: _Ring, f: BufferedFrame) -> None:
        ring.nbytes -= f.size
        if f.path is not None:
            f.path.unlink(missing_ok=True)

    def _rehydrate(self) -> None:
        assert self.spool_dir is not None
        n = 0
        for d in sorted(p for p in self.spool_dir.iterdir() if p.is_dir()):
            ring = self._rings.setdefault(unquote(d.name), _Ring())
            for p in d.glob("*.jpg"):
                try:
                    ts = float(p.stem.split("_", 1)[0])
                except ValueError:
                    continue
                self._insert(ring, BufferedFrame(ts=ts, size=p.stat().st_size, path=p))
                n += 1
            self._evict(ring)
        if n:
            log.info("replay buffer: rehydrated %d frames from %s", n, self.spool_dir)
