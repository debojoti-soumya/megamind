"""Online episode segmentation: groups consecutive frame tokens into `episode_id`s.

A frame joins the device's current episode unless the time gap exceeds `gap_s` or its image vector's
cosine to the episode centroid drops below `split_sim`. A short detour back to one of the last
`rejoin_last` episodes rejoins it when the match is tighter (`rejoin_sim`). Pure and synchronous;
`MemoryStore.add` calls `assign()` for frames that arrive without an `episode_id`, so ingest never
has to know about episodes. State is per device and rebuilt from persisted frames via `seed()`.
"""

from __future__ import annotations

import math
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

    def centroid(self) -> list[float] | None:
        return self.vec_sum

    def absorb(self, t: MemoryToken) -> None:
        self.start_ts, self.end_ts = min(self.start_ts, t.ts), max(self.end_ts, t.ts)
        self.n += 1
        self.ids.append(t.id)
        if t.image_vec is not None:
            if self.vec_sum is None or len(self.vec_sum) != len(t.image_vec):
                self.vec_sum = list(t.image_vec)
            else:
                self.vec_sum = [a + b for a, b in zip(self.vec_sum, t.image_vec)]


class EpisodeTracker:
    def __init__(
        self,
        *,
        gap_s: float = 120.0,
        split_sim: float = 0.80,
        rejoin_sim: float = 0.85,
        rejoin_last: int = 4,
    ) -> None:
        self.gap_s = gap_s
        self.split_sim = split_sim
        self.rejoin_sim = rejoin_sim
        self.rejoin_last = max(1, rejoin_last)
        self._open: dict[str, list[OpenEpisode]] = {}

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
                ep = by_id[t.episode_id] = OpenEpisode(t.episode_id, device_id, t.ts, t.ts)
            else:
                recent.remove(ep)
            ep.absorb(t)
            recent.append(ep)
        self._open[device_id] = recent[-self.rejoin_last :]

    # ------------------------------------------------------------------ assignment

    def _fits(self, ep: OpenEpisode, t: MemoryToken, min_sim: float) -> bool:
        if t.ts > ep.end_ts + self.gap_s or t.ts < ep.start_ts - self.gap_s:
            return False
        c = ep.centroid()
        if t.image_vec is not None and c is not None and len(c) == len(t.image_vec):
            return cosine(t.image_vec, c) >= min_sim
        return True

    def assign(self, t: MemoryToken) -> str:
        """Set `t.episode_id` (if unset) and return it. Also absorbs pre-assigned frames into state."""
        recent = self._open.setdefault(t.device_id, [])
        if t.episode_id:
            target = next((ep for ep in recent if ep.id == t.episode_id), None)
            if target is None:
                target = OpenEpisode(t.episode_id, t.device_id, t.ts, t.ts)
                recent.append(target)
        else:
            target = recent[-1] if recent and self._fits(recent[-1], t, self.split_sim) else None
            if target is None:
                target = next((ep for ep in reversed(recent[:-1]) if self._fits(ep, t, self.rejoin_sim)), None)
            if target is None:
                target = OpenEpisode(episode_id_for(t.device_id, t.ts), t.device_id, t.ts, t.ts)
                recent.append(target)
        recent.remove(target)
        recent.append(target)
        del recent[: -self.rejoin_last]
        target.absorb(t)
        t.episode_id = target.id
        return target.id
