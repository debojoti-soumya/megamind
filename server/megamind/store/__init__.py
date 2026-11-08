from megamind.config import Settings
from megamind.store.base import MemoryStore, Scored, VectorField
from megamind.store.episodes import EpisodeTracker
from megamind.store.memory_store import InMemoryStore
from megamind.store.rrf import fuse_hits, rrf

__all__ = [
    "EpisodeTracker",
    "InMemoryStore",
    "MemoryStore",
    "Scored",
    "VectorField",
    "build_store",
    "fuse_hits",
    "rrf",
]


def build_episode_tracker(settings: Settings) -> EpisodeTracker:
    return EpisodeTracker(
        gap_s=settings.episode_gap_s,
        split_sim=settings.episode_split_sim,
        rejoin_sim=settings.episode_rejoin_sim,
        rejoin_last=settings.episode_rejoin_last,
        recent_n=settings.episode_recent_n,
        max_s=settings.episode_max_s,
        move_m=settings.episode_move_m,
    )


def build_store(settings: Settings) -> MemoryStore:
    episodes = build_episode_tracker(settings)
    if settings.store_backend == "lancedb":
        from megamind.store.lancedb_store import LanceDBStore

        return LanceDBStore(settings.lancedb_path, episodes=episodes, vector_dtype=settings.lancedb_vector_dtype)
    return InMemoryStore(episodes=episodes)
