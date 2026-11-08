"""Keyframe filter v2 (docs/stream1-memory-engine.pdf §3).

Order: dark -> blur -> forced -> min gap -> heartbeat/pose triggers -> novelty -> budget.
Novelty is measured against recently *kept* frames only, so slow pans cannot drift forever.
State is per device and uses frame timestamps (not wall clock), so video replays are deterministic.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from megamind.config import Settings
from megamind.models import FrameMeta, Pose

_KEPT_HISTORY = 8


def dhash(img: Image.Image, size: int = 8) -> str:
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{size * size // 4}x}"


def hamming(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def mean_luma(img: Image.Image) -> float:
    return float(np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32).mean())


def sharpness(img: Image.Image, width: int = 160) -> float:
    """Variance of the 4-neighbour Laplacian on a small grey copy; low = motion blur / out of focus."""
    w, h = img.size
    g = img.convert("L").resize((width, max(1, round(h * width / max(w, 1)))), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4.0 * a[1:-1, 1:-1]
    return float(lap.var())


def pose_delta(a: Pose | None, b: Pose | None) -> tuple[float, float]:
    """(yaw change in degrees, translation in metres); zeros when either pose is unknown."""
    if a is None or b is None:
        return 0.0, 0.0
    dyaw = abs((a.yaw - b.yaw + 180.0) % 360.0 - 180.0)
    return dyaw, math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


@dataclass
class FilterDecision:
    accepted: bool
    reason: str | None
    frame_hash: str
    trigger: str | None = None
    sharpness: float = 0.0


@dataclass
class _DeviceState:
    last_kept_ts: float | None = None
    last_kept_pose: Pose | None = None
    kept: deque[tuple[float, str]] = field(default_factory=lambda: deque(maxlen=_KEPT_HISTORY))
    tokens: float = 0.0
    tokens_ts: float | None = None


class KeyframeFilter:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._state: dict[str, _DeviceState] = {}
        self.counts: Counter[str] = Counter()

    def has_state(self, device_id: str) -> bool:
        return device_id in self._state

    def seed(self, device_id: str, recent: list[tuple[float, str]]) -> None:
        """Restore novelty state after a restart from (ts, hash) of stored tokens."""
        st = self._state.setdefault(device_id, self._new_state())
        for ts, h in sorted(recent)[-_KEPT_HISTORY:]:
            st.kept.append((ts, h))
            st.last_kept_ts = ts

    def reset(self, device_id: str | None = None) -> None:
        if device_id is None:
            self._state.clear()
        else:
            self._state.pop(device_id, None)

    def _new_state(self) -> _DeviceState:
        return _DeviceState(tokens=float(self.s.filter_budget_burst))

    def _refill(self, st: _DeviceState, ts: float) -> None:
        if st.tokens_ts is not None and ts > st.tokens_ts:
            st.tokens = min(
                float(self.s.filter_budget_burst),
                st.tokens + (ts - st.tokens_ts) * self.s.filter_budget_per_min / 60.0,
            )
        st.tokens_ts = ts if st.tokens_ts is None else max(st.tokens_ts, ts)

    def _keep(self, st: _DeviceState, meta: FrameMeta, h: str, trigger: str, sharp: float) -> FilterDecision:
        st.last_kept_ts = meta.ts
        st.last_kept_pose = meta.pose or st.last_kept_pose
        st.kept.append((meta.ts, h))
        st.tokens = max(0.0, st.tokens - 1.0)
        return FilterDecision(True, None, h, trigger, sharp)

    def decide(self, img: Image.Image, meta: FrameMeta, can_merge: bool = False) -> FilterDecision:
        """can_merge: the caller can attach this frame's transcript to the device's last kept token."""
        d = self._decide(img, meta, can_merge)
        self.counts[d.trigger if d.accepted else d.reason] += 1
        return d

    def _decide(self, img: Image.Image, meta: FrameMeta, can_merge: bool) -> FilterDecision:
        s = self.s
        h = dhash(img)
        keep_anyway = meta.force or bool(meta.transcript)
        last = self._state[meta.device_id].last_kept_ts if meta.device_id in self._state else None
        if (
            meta.transcript
            and not meta.force
            and can_merge
            and last is not None
            and 0 <= meta.ts - last < s.filter_speech_merge_s
        ):
            return FilterDecision(False, "merged", h)  # continuous talk: text joins the last token, no new frame
        if mean_luma(img) < s.filter_min_luma and not keep_anyway:
            return FilterDecision(False, "dark", h)
        sharp = sharpness(img)
        if sharp < s.filter_min_sharpness and not keep_anyway:
            return FilterDecision(False, "blurry", h, sharpness=sharp)

        st = self._state.setdefault(meta.device_id, self._new_state())
        self._refill(st, meta.ts)
        if keep_anyway:
            return self._keep(st, meta, h, "forced" if meta.force else "speech", sharp)
        if st.last_kept_ts is None:
            return self._keep(st, meta, h, "novel", sharp)

        gap = meta.ts - st.last_kept_ts
        if gap < s.filter_min_gap_s:
            return FilterDecision(False, "too_soon", h, sharpness=sharp)
        if s.filter_heartbeat_s > 0 and gap >= s.filter_heartbeat_s:
            return self._keep(st, meta, h, "heartbeat", sharp)
        dyaw, dmove = pose_delta(st.last_kept_pose, meta.pose)
        if dyaw >= s.filter_pose_yaw_deg or dmove >= s.filter_pose_move_m:
            return self._keep(st, meta, h, "pose", sharp)

        window = [fh for t, fh in st.kept if meta.ts - t <= s.filter_dhash_window_s]
        if any(hamming(h, fh) < s.filter_dhash_max_distance for fh in window):
            return FilterDecision(False, "duplicate", h, sharpness=sharp)
        if s.filter_budget_per_min > 0 and st.tokens < 1.0:
            return FilterDecision(False, "budget", h, sharpness=sharp)
        return self._keep(st, meta, h, "novel", sharp)
