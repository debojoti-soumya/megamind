"""Online episode segmentation (docs/stream1-memory-engine.pdf §4.2).

A token joins the current episode unless: the time gap exceeds `episode_gap_s`, its image vector
falls below `episode_split_sim` cosine to the episode centroid, or the wearer moved more than
`episode_move_m` from the episode's pose centroid. A short detour back to one of the last few
episodes rejoins it. Titles are zero-shot scene labels from the image embedder's text tower,
falling back to the most frequent entities.
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter, deque
from dataclasses import dataclass, field

from ulid import ULID

from megamind.config import Settings
from megamind.ingest.enrich import ImageEmbedder
from megamind.models import MemoryEpisode, MemoryToken, Pose

SCENE_LABELS = [
    "a kitchen", "a living room", "a bedroom", "a bathroom", "an office", "a desk with a computer",
    "a meeting room", "a whiteboard", "a classroom or lecture hall", "a hallway", "a staircase",
    "a street", "a parking lot", "a car interior", "a shop or supermarket", "a restaurant or cafe",
    "a park", "a garden", "a gym", "a lab or workshop", "a library or bookshelf", "a dining table",
    "a laptop screen", "a phone screen", "a poster or sign", "people talking", "a hackathon venue",
    "outdoors in nature", "a store shelf with products", "a doorway or entrance",
]  # fmt: skip


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _rep_score(t: MemoryToken) -> float:
    return math.log1p(t.sharpness) + 0.02 * min(len(t.ocr_text), 200) + (0.5 if t.caption else 0.0)


@dataclass
class _Open:
    ep: MemoryEpisode
    vec_sum: list[float] | None = None
    pose_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    pose_n: int = 0
    rep_score: float = -1.0
    entities: Counter[str] = field(default_factory=Counter)
    recent: deque[list[float]] = field(default_factory=deque)

    def recent_mean(self) -> list[float] | None:
        if not self.recent:
            return None
        return [sum(col) for col in zip(*self.recent)]

    @property
    def centroid(self) -> list[float] | None:
        return self.vec_sum

    def pose_centroid(self) -> Pose | None:
        if not self.pose_n:
            return None
        x, y, z, yaw = (v / self.pose_n for v in self.pose_sum)
        return Pose(x=x, y=y, z=z, yaw=yaw)


class EpisodeTracker:
    def __init__(
        self, settings: Settings, image_embedder: ImageEmbedder | None, labels: list[str] | None = None
    ) -> None:
        self.s = settings
        self.image_embedder = image_embedder
        self.labels = labels or SCENE_LABELS
        self._label_vecs: list[list[float]] | None = None
        self._open: dict[str, list[_Open]] = {}
        self._lock = asyncio.Lock()

    def reset(self, device_id: str | None = None) -> None:
        if device_id is None:
            self._open.clear()
        else:
            self._open.pop(device_id, None)

    async def _labels(self) -> list[list[float]] | None:
        if self.image_embedder is None:
            return None
        if self._label_vecs is None:
            self._label_vecs = [await self.image_embedder.embed_text(f"a photo of {lbl}") for lbl in self.labels]
        return self._label_vecs

    def _fits(self, o: _Open, t: MemoryToken, min_sim: float, recent: bool = False) -> bool:
        if abs(t.ts - o.ep.end_ts) > self.s.episode_gap_s and not (o.ep.start_ts <= t.ts <= o.ep.end_ts):
            return False
        if self.s.episode_max_s > 0 and t.ts - o.ep.start_ts > self.s.episode_max_s:
            return False
        ref = o.recent_mean() if recent else o.centroid
        if t.image_vec is not None and ref is not None:
            if _cos(t.image_vec, ref) < min_sim:
                return False
        pc = o.pose_centroid()
        if t.pose is not None and pc is not None:
            if math.dist((t.pose.x, t.pose.y, t.pose.z), (pc.x, pc.y, pc.z)) > self.s.episode_move_m:
                return False
        return True

    async def assign(self, t: MemoryToken) -> MemoryEpisode:
        """Set `t.episode_id` and return the updated episode (caller persists both)."""
        async with self._lock:
            recent = self._open.setdefault(t.device_id, [])
            current = recent[-1:] if recent else []
            target = next((o for o in current if self._fits(o, t, self.s.episode_split_sim, recent=True)), None)
            if target is None:  # rejoining an older episode after a detour needs a closer match
                target = next((o for o in reversed(recent[:-1]) if self._fits(o, t, self.s.episode_rejoin_sim)), None)
            if target is None:
                target = _Open(
                    MemoryEpisode(
                        id=f"ep_{ULID()}", device_id=t.device_id, start_ts=t.ts, end_ts=t.ts, rep_token_id=t.id
                    )
                )
                recent.append(target)
                del recent[: -self.s.episode_rejoin_last]
            else:
                recent.remove(target)
                recent.append(target)
            await self._update(target, t)
            t.episode_id = target.ep.id
            return target.ep.model_copy(deep=True)

    async def _update(self, o: _Open, t: MemoryToken) -> None:
        e = o.ep
        e.start_ts, e.end_ts = min(e.start_ts, t.ts), max(e.end_ts, t.ts)
        e.n_tokens += 1
        if t.image_vec is not None:
            o.vec_sum = list(t.image_vec) if o.vec_sum is None else [a + b for a, b in zip(o.vec_sum, t.image_vec)]
            o.recent.append(list(t.image_vec))
            while len(o.recent) > max(1, self.s.episode_recent_n):
                o.recent.popleft()
        if t.pose is not None:
            o.pose_sum = [a + b for a, b in zip(o.pose_sum, (t.pose.x, t.pose.y, t.pose.z, t.pose.yaw))]
            o.pose_n += 1
            e.pose_centroid = o.pose_centroid()
        o.entities.update(x.lower() for x in t.entities)
        e.entities = [w for w, _ in o.entities.most_common(10)]
        score = _rep_score(t)
        if score > o.rep_score:
            o.rep_score, e.rep_token_id = score, t.id
        labels = await self._labels()
        if labels is not None and o.centroid is not None:
            best = max(range(len(labels)), key=lambda i: _cos(o.centroid, labels[i]))
            e.title = self.labels[best]
        elif e.entities:
            e.title = ", ".join(e.entities[:3])
        elif t.caption and not e.title:
            e.title = t.caption[:60]
