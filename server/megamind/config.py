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
    enrich_base_url: str = ""
    enrich_model: str = ""

    embed_provider: str = "mock"
    embed_model: str = "text-embedding-3-small"
    openai_api_key: str = ""
    voyage_api_key: str = ""

    stt_provider: str = "mock"
    tts_provider: str = "mock"
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""

    store_backend: str = "memory"
    lancedb_path: Path = Path("./data/lancedb")
    thumb_dir: Path = Path("./data/thumbs")
    audio_dir: Path = Path("./data/audio")

    filter_min_gap_s: float = 2.0
    filter_dhash_max_distance: int = 6
    filter_dhash_window_s: float = 30.0
    filter_min_luma: int = 20

    @property
    def enrich_url(self) -> str:
        return self.enrich_base_url or self.reader_base_url

    @property
    def enrich_model_name(self) -> str:
        return self.enrich_model or self.reader_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
