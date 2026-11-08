from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-5"

    reader_base_url: str = "http://localhost:8001/v1"
    reader_api_key: str = "none"
    reader_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    reader_provider: str = "caption"  # caption (no model) | vlm (READER_BASE_URL) | local (transformers VLM)
    reader_max_frames: int = 4
    local_reader_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    local_reader_split: bool = False  # image splitting: ~5x more vision tokens, slower on CPU
    enrich_base_url: str = ""
    enrich_model: str = ""

    enrich_provider: str = "auto"  # auto (vlm if ENRICH_BASE_URL set) | vlm | local (tesseract) | mock
    embed_provider: str = "mock"  # text: mock | openai | voyage | local
    embed_model: str = "text-embedding-3-small"
    local_text_model: str = "BAAI/bge-small-en-v1.5"
    image_embed_provider: str = "none"  # none | mock | siglip
    image_embed_model: str = "ViT-B-16-SigLIP"
    image_embed_pretrained: str = "webli"
    openai_api_key: str = ""
    voyage_api_key: str = ""

    stt_provider: str = "mock"
    tts_provider: str = "mock"
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""

    store_backend: str = "memory"
    lancedb_path: Path = Path("./data/lancedb")
    thumb_dir: Path = Path("./data/thumbs")
    chunk_dir: Path = Path("./data/chunks")
    hot_retention_h: float = 72.0
    cold_retention_days: float = 100.0
    cold_width: int = 960
    cold_crf: int = 50
    age_interval_s: float = 3600.0  # 0 disables the background aging loop
    audio_dir: Path = Path("./data/audio")

    filter_min_gap_s: float = 2.0
    filter_dhash_max_distance: int = 6
    filter_dhash_window_s: float = 30.0
    filter_min_luma: int = 20
    filter_min_sharpness: float = 15.0
    filter_heartbeat_s: float = 30.0
    filter_speech_merge_s: float = 10.0  # speech within this of the last kept frame joins that token
    filter_pose_yaw_deg: float = 25.0
    filter_pose_move_m: float = 0.75
    filter_budget_per_min: float = 6.0
    filter_budget_burst: int = 6

    episode_gap_s: float = 120.0
    episode_rejoin_sim: float = 0.80  # stricter bar to rejoin a non-current recent episode
    episode_split_sim: float = 0.70  # SigLIP-B/16, calibrated on tools/make_day.py footage
    episode_move_m: float = 3.0
    episode_rejoin_last: int = 4
    episode_recent_n: int = 6  # continuing an episode compares to its last N frames, not the whole-episode mean
    episode_max_s: float = 1800.0  # cap so a long stay in one place still yields queryable chunks

    search_w_bm25: float = 1.0
    search_w_text: float = 1.0
    search_w_image: float = 1.5
    search_w_image_visual: float = 1.5
    search_calibrate_image: bool = True  # scale image hits by the embedder's match probability
    search_per_episode_cap: int = 3
    search_w_summary: float = 1.0  # each of summary-vector and summary-BM25 rankings

    summary_window_s: float = 120.0  # 0 disables sliding-window summaries
    summary_step_s: float = 60.0

    @property
    def enrich_url(self) -> str:
        return self.enrich_base_url or self.reader_base_url

    @property
    def enrich_model_name(self) -> str:
        return self.enrich_model or self.reader_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
