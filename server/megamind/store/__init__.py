from megamind.config import Settings
from megamind.store.base import MemoryStore
from megamind.store.fusion import FusionWeights
from megamind.store.memory_store import InMemoryStore


def fusion_weights(settings: Settings) -> FusionWeights:
    return FusionWeights(
        bm25=settings.search_w_bm25,
        text=settings.search_w_text,
        image=settings.search_w_image,
        image_visual_mode=settings.search_w_image_visual,
        per_episode_cap=settings.search_per_episode_cap,
    )


def build_store(settings: Settings, text_dim: int, image_dim: int | None) -> MemoryStore:
    if settings.store_backend == "lancedb":
        from megamind.store.lancedb_store import LanceDBStore

        return LanceDBStore(settings.lancedb_path, text_dim, image_dim, fusion_weights(settings))
    return InMemoryStore(fusion_weights(settings))
