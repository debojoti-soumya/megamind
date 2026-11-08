"""Online episode segmentation: groups consecutive frame tokens into `episode_id`s.

A frame joins the device's current episode unless
- the time gap exceeds `gap_s`,
- its image vector's cosine to the mean of the episode's last `recent_n` image vectors drops below
  `split_sim` (a recent window, not the whole centroid: the frame is judged against what the wearer was just
  looking at, so a gradual turn stays one episode while a cut to a different view splits even late in a long
  episode whose whole-history mean has averaged into something every view resembles),
- the episode already spans more than `max_s` (0 = off), so a long stay is still closed and summarised, or
- the wearer is more than `move_m` from the episode's pose centroid (0 = off).
A short detour back to one of the last `rejoin_last` episodes rejoins it when the frame is close to that
episode's whole centroid (`rejoin_sim`), subject to the same gap / length / pose limits. Pure and synchronous;
`MemoryStore.add` calls `assign()` for frames that arrive without an `episode_id`, so ingest never
has to know about episodes. State is per device and rebuilt from persisted frames via `seed()`.

The recent-N / max-length / pose rules and the 0.70 / 0.80 thresholds (SigLIP B/16) are ported from PR #54,
where the whole-episode mean never split on 6.9 h of EgoLife footage and only the length cap produced
bounded episodes.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from megamind.models import MemoryToken


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def episode_id_for(device_id: str, start_ts: float) -> str:
    return f"ep-{device_id}-{int(round(start_ts * 1000))}"


@dataclass
class OpenEpisode:
    id: str
    device_id: str
    start_ts: float
    end_ts: float
    n: int = 0
    vec_sum: list[float] | None = None
    ids: list[str] = field(default_factory=list)
    recent: deque[list[float]] = field(default_factory=lambda: deque(maxlen=6))
    pose_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pose_n: int = 0

    def centroid(self) -> list[float] | None:
        return self.vec_sum

    def recent_mean(self) -> list[float] | None:
        """Sum of the last `recent_n` image vectors (direction == mean; cosine ignores the scale)."""
        if not self.recent:
            return None
        return [sum(col) for col in zip(*self.recent)]

    def pose_centroid(self) -> tuple[float, float, float] | None:
        if not self.pose_n:
            return None
        x, y, z = (v / self.pose_n for v in self.pose_sum)
        return x, y, z

    def absorb(self, t: MemoryToken) -> None:
        self.start_ts, self.end_ts = min(self.start_ts, t.ts), max(self.end_ts, t.ts)
        self.n += 1
        self.ids.append(t.id)
        if t.image_vec is not None:
            if self.vec_sum is None or len(self.vec_sum) != len(t.image_vec):
                self.vec_sum = list(t.image_vec)
                self.recent.clear()
            else:
                self.vec_sum = [a + b for a, b in zip(self.vec_sum, t.image_vec)]
            self.recent.append(list(t.image_vec))
        if t.pose is not None:
            self.pose_sum = [a + b for a, b in zip(self.pose_sum, (t.pose.x, t.pose.y, t.pose.z))]
            self.pose_n += 1


class EpisodeTracker:
    def __init__(
        self,
        *,
        gap_s: float = 120.0,
        split_sim: float = 0.70,
        rejoin_sim: float = 0.80,
        rejoin_last: int = 4,
        recent_n: int = 6,
        max_s: float = 1800.0,
        move_m: float = 3.0,
    ) -> None:
        self.gap_s = gap_s
        self.split_sim = split_sim
        self.rejoin_sim = rejoin_sim
        self.rejoin_last = max(1, rejoin_last)
        self.recent_n = max(1, recent_n)
        self.max_s = max_s
        self.move_m = move_m
        self._open: dict[str, list[OpenEpisode]] = {}

    def _new(self, episode_id: str, device_id: str, ts: float) -> OpenEpisode:
        return OpenEpisode(episode_id, device_id, ts, ts, recent=deque(maxlen=self.recent_n))

    # ------------------------------------------------------------------ state

    def knows(self, device_id: str) -> bool:
        return device_id in self._open

    def reset(self, device_id: str | None = None) -> None:
        if device_id is None:
            self._open.clear()
        else:
            self._open.pop(device_id, None)

    def open_episodes(self, device_id: str) -> list[OpenEpisode]:
        return list(self._open.get(device_id, []))

    def seed(self, device_id: str, tokens: Iterable[MemoryToken]) -> None:
        """Rebuild the open-episode window for `device_id` from already-assigned frames (oldest first)."""
        recent: list[OpenEpisode] = []
        by_id: dict[str, OpenEpisode] = {}
        for t in sorted(tokens, key=lambda t: (t.ts, t.id)):
            if t.device_id != device_id or t.kind != "frame" or not t.episode_id:
                continue
            ep = by_id.get(t.episode_id)
            if ep is None:
                ep = by_id[t.episode_id] = self._new(t.episode_id, device_id, t.ts)
            else:
                recent.remove(ep)
            ep.absorb(t)
            recent.append(ep)
        self._open[device_id] = recent[-self.rejoin_last :]

    # ------------------------------------------------------------------ assignment

    def _fits(self, ep: OpenEpisode, t: MemoryToken, min_sim: float, *, recent: bool = False) -> bool:
        if t.ts > ep.end_ts + self.gap_s or t.ts < ep.start_ts - self.gap_s:
            return False
        if self.max_s > 0 and max(ep.end_ts, t.ts) - min(ep.start_ts, t.ts) > self.max_s:
            return False
        c = ep.recent_mean() if recent else ep.centroid()
        if t.image_vec is not None and c is not None and len(c) == len(t.image_vec):
            if cosine(t.image_vec, c) < min_sim:
                return False
        pc = ep.pose_centroid()
        if self.move_m > 0 and t.pose is not None and pc is not None:
            if math.dist((t.pose.x, t.pose.y, t.pose.z), pc) > self.move_m:
                return False
        return True

    def assign(self, t: MemoryToken) -> str:
        """Set `t.episode_id` (if unset) and return it. Also absorbs pre-assigned frames into state."""
        recent = self._open.setdefault(t.device_id, [])
        if t.episode_id:
            target = next((ep for ep in recent if ep.id == t.episode_id), None)
            if target is None:
                target = self._new(t.episode_id, t.device_id, t.ts)
                recent.append(target)
        else:
            target = recent[-1] if recent and self._fits(recent[-1], t, self.split_sim, recent=True) else None
            if target is None:
                target = next((ep for ep in reversed(recent[:-1]) if self._fits(ep, t, self.rejoin_sim)), None)
            if target is None:
                target = self._new(episode_id_for(t.device_id, t.ts), t.device_id, t.ts)
                recent.append(target)
        recent.remove(target)
        recent.append(target)
        del recent[: -self.rejoin_last]
        target.absorb(t)
        t.episode_id = target.id
        return target.id
