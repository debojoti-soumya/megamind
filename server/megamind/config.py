from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-5"
    claude_timeout_s: float = 30.0
    claude_max_retries: int = 2

    reader_base_url: str = "http://localhost:8001/v1"
    reader_api_key: str = "none"
    reader_model: str = "Qwen/Qwen3.6-35B-A3B"
    enrich_base_url: str = ""
    enrich_model: str = ""

    embed_provider: str = "mock"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1024  # text-embedding-3-* Matryoshka truncation; frozen per LanceDB table
    image_embed_provider: str = "none"  # none | mock | openclip (needs the local-embeddings extra)
    image_embed_model: str = "ViT-B-16-SigLIP"  # 768-d; ViT-B-32 is 512-d
    image_embed_pretrained: str = "webli"
    openai_api_key: str = ""
    voyage_api_key: str = ""

    stt_provider: str = "mock"
    tts_provider: str = "mock"
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""

    store_backend: str = "memory"
    retriever_backend: str = "hybrid"  # "hybrid" (ours) | "ucc" (Universal Constructor; falls back to hybrid on error)
    ucc_api_key: str = ""
    ucc_base_url: str = "https://api.universalconstructor.co/v1"
    ucc_corpus: str = "megamind"
    ucc_timeout_s: float = 4.0  # per recall; past it the hybrid answers
    ucc_candidates: int = 0  # >0: UCC reranks the hybrid's top-N instead of the whole time window
    ucc_sync_interval_s: float = 30.0
    ucc_keep_alive_s: int | None = None  # corpus retention; None keeps it until deleted
    wearer_tz: str = "America/Los_Angeles"  # IANA zone used to resolve "this morning" in recall_memory
    lancedb_path: Path = Path("./data/lancedb")
    thumb_dir: Path = Path("./data/thumbs")
    audio_dir: Path = Path("./data/audio")

    filter_min_gap_s: float = 2.0
    filter_dhash_max_distance: int = 6
    filter_dhash_window_s: float = 30.0
    filter_min_luma: int = 20
    filter_min_sharpness: float = 40.0  # ingest.filter.sharpness() at 320 px; calibrate on real passthrough frames
    filter_heartbeat_s: float = 30.0  # always keep a frame after this long without one (0 = off)
    filter_pose_yaw_deg: float = 20.0  # keep when the head turned this much since the last kept frame (0 = off)
    filter_pose_move_m: float = 0.5  # keep when the head moved this far since the last kept frame (0 = off)
    filter_pose_still_yaw_deg: float = 5.0  # pose-assisted dedupe: head barely moved ...
    filter_pose_still_move_m: float = 0.15
    filter_pose_still_distance: int = 10  # ... and the hash is this close -> duplicate
    filter_settle_max_distance: int = 4  # settled-frame rule: hamming to the previous (rejected) frame
    filter_settle_min_gap_s: float = 0.5
    filter_dhash_long_distance: int = 3  # near-identical to a frame kept within the long window -> duplicate
    filter_dhash_long_window_s: float = 600.0
    filter_max_per_min: int = 20  # per-device cap on kept frames (token bucket; heartbeat/pose exempt)

    episode_gap_s: float = 120.0
    episode_split_sim: float = 0.80
    episode_rejoin_sim: float = 0.85
    episode_rejoin_last: int = 4
    consolidate_enabled: bool = True  # background Consolidator in the server lifespan
    consolidate_interval_s: float = 600.0
    consolidate_min_tokens: int = 3
    summarizer_provider: str = "mock"

    @property
    def enrich_url(self) -> str:
        return self.enrich_base_url or self.reader_base_url

    @property
    def enrich_model_name(self) -> str:
        return self.enrich_model or self.reader_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
