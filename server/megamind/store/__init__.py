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


def build_store(settings: Settings, *, text_dim: int | None = None, image_dim: int | None = None) -> MemoryStore:
    """`text_dim` / `image_dim`: the configured embedders' widths, so LanceDB's vector columns don't depend on
    which kind of token (e.g. an audio transcript without an image vector) happens to be written first."""
    episodes = build_episode_tracker(settings)
    if settings.store_backend == "lancedb":
        from megamind.store.lancedb_store import LanceDBStore

        return LanceDBStore(
            settings.lancedb_path,
            text_dim=text_dim,
            image_dim=image_dim,
            episodes=episodes,
            vector_dtype=settings.lancedb_vector_dtype,
        )
    return InMemoryStore(episodes=episodes)
