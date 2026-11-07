from megamind.config import Settings
from megamind.store.base import MemoryStore
from megamind.store.memory_store import InMemoryStore


def build_store(settings: Settings) -> MemoryStore:
    if settings.store_backend == "lancedb":
        from megamind.store.lancedb_store import LanceDBStore

        return LanceDBStore(settings.lancedb_path)
    return InMemoryStore()
