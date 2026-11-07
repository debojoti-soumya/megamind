from megamind.config import Settings
from megamind.store.base import MemoryStore, Scored, VectorField
from megamind.store.memory_store import InMemoryStore
from megamind.store.rrf import rrf

__all__ = ["InMemoryStore", "MemoryStore", "Scored", "VectorField", "build_store", "rrf"]


def build_store(settings: Settings) -> MemoryStore:
    if settings.store_backend == "lancedb":
        from megamind.store.lancedb_store import LanceDBStore

        return LanceDBStore(settings.lancedb_path)
    return InMemoryStore()
