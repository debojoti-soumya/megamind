"""Keyframe filter v2: dark -> blur -> too_soon -> dedupe -> per-minute cap, with pose and
settled-frame triggers.

Pure PIL/NumPy, ~3-4 ms per 960 px frame (one grey decode, one 320 px downscale). Per-device state
is keyed on frame timestamps (not wall clock) so video replays through tools/feeder.py are deterministic.

Decision order for a frame (h = dHash, prev = previous frame of this device, kept = last kept):
  dark      mean luma < filter_min_luma
  blurry    sharpness() < filter_min_sharpness (motion blur, out of focus)
  keep      first frame of the device, or gap since kept >= filter_heartbeat_s ("heartbeat"),
            or |dyaw| >= filter_pose_yaw_deg / |dpos| >= filter_pose_move_m since kept ("pose"),
            or settled: prev was rejected for blur/change, hamming(h, prev) <= filter_settle_max_distance
            and h differs from kept ("settled" - the first stable frame after a scene change)
  too_soon  gap since kept < filter_min_gap_s
  duplicate hamming < filter_dhash_max_distance within filter_dhash_window_s of a kept frame,
            or < filter_dhash_long_distance within filter_dhash_long_window_s (static scene re-entered),
            or < filter_pose_still_distance while the head barely moved since kept (pose-assisted)
  budget    more than filter_max_per_min kept in the last minute (token bucket; heartbeat/pose exempt)
  keep      "novel"
"""

from __future__ import annotations

import io
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from PIL import Image

from megamind.config import Settings
from megamind.models import FrameMeta, Pose

Reason = str  # one of docs/api.md reason values; see megamind.models.FrameAccepted


def decode_frame(jpeg: bytes) -> Image.Image:
    """Decode an uploaded frame straight to greyscale - all the filter needs.

    For JPEGs `draft("L")` makes libjpeg emit only the Y plane (no chroma upsampling or RGB
    conversion); other formats decode normally and are converted where needed.
    """
    img = Image.open(io.BytesIO(jpeg))
    img.draft("L", img.size)
    img.load()
    return img


def _grey(img: Image.Image) -> Image.Image:
    return img if img.mode == "L" else img.convert("L")


def dhash(img: Image.Image, size: int = 8) -> str:
    g = _grey(img).resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{size * size // 4}x}"


def hamming(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def mean_luma(img: Image.Image) -> float:
    return float(np.asarray(_grey(img).resize((32, 32)), dtype=np.float32).mean())


def _small_grey(img: Image.Image, width: int = 320) -> np.ndarray:
    g = _grey(img)
    w, h = g.size
    if w > width:
        g = g.resize((width, max(1, round(h * width / w))), Image.Resampling.BILINEAR)
    return np.asarray(g, dtype=np.float32)


def _sharpness(a: np.ndarray) -> float:
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    dxx = a[:, :-2] + a[:, 2:] - 2.0 * a[:, 1:-1]
    dyy = a[:-2, :] + a[2:, :] - 2.0 * a[1:-1, :]
    return 2.0 * float(min(dxx.var(), dyy.var()))


def sharpness(img: Image.Image, width: int = 320) -> float:
    """Blur score on a `width` px grey copy: 2 * min(var(d2/dx2), var(d2/dy2)). Low = blurry.

    Taking the weaker axis catches directional motion blur (a horizontal pan keeps vertical
    detail sharp, so a plain Laplacian variance barely drops). Typical values at 320 px:
    sharp indoor passthrough 100-600, motion-blurred pan < 30, flat wall < 5.
    """
    return _sharpness(_small_grey(img, width))


def frame_metrics(img: Image.Image) -> tuple[str, float, float]:
    """(dhash, mean luma, sharpness) from one grey conversion and one 320 px downscale.

    Luma is the mean of the 320 px copy (same value as `mean_luma` to within ~0.1 grey level).
    """
    g = _grey(img)
    a = _small_grey(g)
    return dhash(g), float(a.mean()), _sharpness(a)


def pose_delta(a: Pose | None, b: Pose | None) -> tuple[float, float]:
    """(|yaw change| in degrees, translation in metres); (0, 0) when either pose is unknown."""
    if a is None or b is None:
        return 0.0, 0.0
    dyaw = abs((a.yaw - b.yaw + 180.0) % 360.0 - 180.0)
    return dyaw, math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


@dataclass
class FilterDecision:
    accepted: bool
    reason: Reason | None
    frame_hash: str
    trigger: str | None = None  # why an accepted frame was kept: first|heartbeat|pose|settled|novel
    sharpness: float = 0.0


@dataclass
class _DeviceState:
    last_kept_ts: float | None = None
    last_kept_hash: str | None = None
    last_kept_pose: Pose | None = None
    prev_hash: str | None = None
    prev_unstable: bool = False  # previous frame was rejected as blurry, or changed but came too soon
    tokens: float = 0.0
    tokens_ts: float | None = None


class KeyframeFilter:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._state: dict[str, _DeviceState] = {}
        self.counts: Counter[str] = Counter()  # reject reasons and keep triggers, for logs/stats

    @property
    def history_window_s(self) -> float:
        """How far back the caller should fetch kept (ts, hash) pairs for dedupe."""
        return max(self.s.filter_dhash_window_s, self.s.filter_dhash_long_window_s)

    def reset(self, device_id: str | None = None) -> None:
        if device_id is None:
            self._state.clear()
        else:
            self._state.pop(device_id, None)

    def _new_state(self) -> _DeviceState:
        return _DeviceState(tokens=float(self.s.filter_max_per_min))

    def _refill(self, st: _DeviceState, ts: float) -> None:
        if st.tokens_ts is not None and ts > st.tokens_ts:
            refill = (ts - st.tokens_ts) * self.s.filter_max_per_min / 60.0
            st.tokens = min(float(self.s.filter_max_per_min), st.tokens + refill)
        st.tokens_ts = ts if st.tokens_ts is None else max(st.tokens_ts, ts)

    def decide(
        self,
        img: Image.Image | None,
        meta: FrameMeta,
        recent: list[tuple[float, str]],
        metrics: tuple[str, float, float] | None = None,
    ) -> FilterDecision:
        """`recent` = (ts, dhash) of frames already kept for this device (store + in-flight).

        Pass `metrics` (from `frame_metrics`, e.g. computed in a worker thread) instead of `img`
        to keep this call cheap; the stateful part always runs on the caller's thread.
        """
        st = self._state.setdefault(meta.device_id, self._new_state())
        d = self._decide(metrics or frame_metrics(img), meta, recent, st)
        st.prev_hash = d.frame_hash
        st.prev_unstable = (not d.accepted) and (
            d.reason == "blurry" or (d.reason == "too_soon" and self._changed(d.frame_hash, st))
        )
        if d.accepted:
            st.last_kept_ts = meta.ts
            st.last_kept_hash = d.frame_hash
            st.last_kept_pose = meta.pose or st.last_kept_pose
            if d.trigger not in ("heartbeat", "pose"):
                st.tokens = max(0.0, st.tokens - 1.0)
        self.counts[d.trigger if d.accepted else d.reason] += 1
        return d

    def _changed(self, h: str, st: _DeviceState) -> bool:
        return st.last_kept_hash is None or hamming(h, st.last_kept_hash) >= self.s.filter_dhash_max_distance

    def _decide(
        self, metrics: tuple[str, float, float], meta: FrameMeta, recent: list[tuple[float, str]], st: _DeviceState
    ) -> FilterDecision:
        s = self.s
        h, luma, sharp = metrics
        if luma < s.filter_min_luma:
            return FilterDecision(False, "dark", h)
        if sharp < s.filter_min_sharpness:
            return FilterDecision(False, "blurry", h, sharpness=sharp)

        self._refill(st, meta.ts)
        last_kept_ts = st.last_kept_ts
        if recent:
            newest = max(t for t, _ in recent)
            last_kept_ts = newest if last_kept_ts is None else max(last_kept_ts, newest)
        if last_kept_ts is None:
            return FilterDecision(True, None, h, "first", sharp)

        gap = meta.ts - last_kept_ts
        if s.filter_heartbeat_s > 0 and gap >= s.filter_heartbeat_s:
            return FilterDecision(True, None, h, "heartbeat", sharp)
        dyaw, dmove = pose_delta(st.last_kept_pose, meta.pose)
        turned = s.filter_pose_yaw_deg > 0 and dyaw >= s.filter_pose_yaw_deg
        moved = s.filter_pose_move_m > 0 and dmove >= s.filter_pose_move_m
        if turned or moved:
            return FilterDecision(True, None, h, "pose", sharp)

        settled = (
            st.prev_unstable
            and st.prev_hash is not None
            and hamming(h, st.prev_hash) <= s.filter_settle_max_distance
            and self._changed(h, st)
            and gap >= s.filter_settle_min_gap_s
        )
        if gap < s.filter_min_gap_s and not settled:
            return FilterDecision(False, "too_soon", h, sharpness=sharp)

        for t, fh in recent:
            age = meta.ts - t
            dist = hamming(h, fh)
            if age <= s.filter_dhash_window_s and dist < s.filter_dhash_max_distance:
                return FilterDecision(False, "duplicate", h, sharpness=sharp)
            if age <= s.filter_dhash_long_window_s and dist < s.filter_dhash_long_distance:
                return FilterDecision(False, "duplicate", h, sharpness=sharp)
        if (
            st.last_kept_hash is not None
            and meta.pose is not None
            and st.last_kept_pose is not None
            and dyaw < s.filter_pose_still_yaw_deg
            and dmove < s.filter_pose_still_move_m
            and hamming(h, st.last_kept_hash) < s.filter_pose_still_distance
        ):
            return FilterDecision(False, "duplicate", h, sharpness=sharp)

        if s.filter_max_per_min > 0 and st.tokens < 1.0:
            return FilterDecision(False, "budget", h, sharpness=sharp)
        return FilterDecision(True, None, h, "settled" if settled else "novel", sharp)
