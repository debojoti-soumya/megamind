from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_token: str = ""  # shared secret for a public deployment; empty = no auth (megamind/auth.py)

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"  # claude-sonnet-4-5* retires 2026-11-30; override via CLAUDE_MODEL
    claude_timeout_s: float = 30.0
    claude_max_retries: int = 2

    reader_base_url: str = "http://localhost:8001/v1"
    reader_api_key: str = "none"
    reader_model: str = "google/gemma-4-31B-it"  # Qwen3.6-35B-A3B is flex-only on Sail (research/reports/04)
    reader_fallback_model: str = "nvidia/Gemma-4-31B-IT-NVFP4"  # retry model on 429/503/timeout; "" disables
    reader_provider: str = "auto"  # auto (vlm iff READER_API_KEY set) | vlm | mock
    reader_timeout_s: float = 7.0  # per attempt, capped by the deadline; Sail Gemma asap calls take ~3.1 s
    reader_hedge_s: float = 3.25  # start the fallback model if the primary hasn't answered by then; 0 = only on failure
    reader_deadline_s: float = 7.0  # wall budget for all attempts of one call (hedge + healthy p95 3.25 s + 0.5 s)
    reader_max_retries: int = 1
    reader_max_images: int = 12  # per call (Sail cap 20; report 04 says <=12 for latency); more -> chunk + merge
    reader_thumb_max_px: int = 512
    reader_max_tokens: int = 300
    reader_json_mode: str = "schema"  # schema (json_schema) | object (json_object) | off; downgraded on HTTP 400
    reader_completion_window: str = ""  # Sail: asap | balanced; sent as X-Sail-Completion-Window when set
    reader_skip_on_ocr: bool = True  # Claude path: skip the VLM when every query keyword is in a top hit's OCR
    recall_prefetch: bool = True  # run recall_memory on the raw question while Claude picks its tool call
    recall_first: bool = (
        True  # standalone recall questions: the prefetch is Claude's first tool result (no picking call)
    )
    # opt-in prompt caching within a turn that starts with Claude's tool-picking call: a cache breakpoint on the
    # question, so the tool round re-reads tools + system + history. Never changes answers; off (default) = the
    # request bodies are exactly as before
    claude_prompt_cache: bool = False
    # demo convenience: "latest" = a chat whose device_id has no memories searches the most recently active device
    # instead (reported as device_id_used + warning); "" (default) = search the requested device and only warn
    chat_default_device: Literal["", "latest"] = ""
    reader_extra_body: str = ""  # JSON merged into the request, e.g. {"chat_template_kwargs":{"enable_thinking":false}}
    enrich_provider: str = "auto"  # auto (vlm iff an enrich/reader key is set, else local if installed, else mock)
    enrich_base_url: str = ""  # "" -> reader_base_url (and the reader's extra body / completion window)
    enrich_model: str = ""  # "" -> reader_model
    enrich_api_key: str = ""  # "" -> reader_api_key, but only when enrich_base_url is "" (same endpoint)
    enrich_timeout_s: float = 20.0  # per attempt; one retry on 429/5xx/timeout, then the local/empty fallback
    enrich_max_tokens: int = 400

    embed_provider: str = "mock"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1024  # text-embedding-3-* Matryoshka truncation; frozen per LanceDB table
    local_embed_model: str = "BAAI/bge-small-en-v1.5"  # EMBED_PROVIDER=local (sentence-transformers); its own dim
    image_embed_provider: str = "none"  # none | mock | openclip (needs the local-embeddings extra)
    image_embed_model: str = "ViT-B-16-SigLIP"  # 768-d; ViT-B-32 is 512-d
    image_embed_pretrained: str = "webli"
    openai_api_key: str = ""
    voyage_api_key: str = ""

    stt_provider: str = "mock"
    stt_model: str = "whisper-1"
    tts_provider: str = "mock"
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""

    store_backend: str = "memory"
    retriever_backend: str = "hybrid"  # "hybrid" (ours) | "ucc" (Universal Constructor; falls back to hybrid on error)
    # | "hier": LLM day/hour/block notes searched coarse-to-fine over the hybrid's candidates (retrieval/hier.py)
    hier_model: str = "gpt-4.1"  # ranks blocks and frame tokens at recall time
    hier_note_model: str = "gpt-4.1-mini"  # writes the block/hour/day notes
    hier_base_url: str = "https://api.openai.com/v1"
    hier_api_key: str = ""  # "" -> openai_api_key
    hier_cache_path: str = "data/hier_notes.json"  # "" = keep notes in memory only
    ucc_api_key: str = ""
    ucc_base_url: str = "https://api.universalconstructor.co/v1"
    ucc_corpus: str = "megamind"
    ucc_timeout_s: float = 2.0  # per recall; the hybrid races UCC and answers past it
    ucc_candidates: int = 0  # >0: UCC reranks the hybrid's top-N instead of the whole time window
    ucc_sync_interval_s: float = 30.0
    ucc_keep_alive_s: int | None = None  # corpus retention; None keeps it until deleted
    wearer_tz: str = "America/Los_Angeles"  # IANA zone used to resolve "this morning" in recall_memory
    lancedb_path: Path = Path("./data/lancedb")
    lancedb_vector_dtype: Literal["float16", "float32"] = "float16"  # new tables only; existing tables keep theirs
    search_calibrate_image: bool = True  # scale image hits by SigLIP sigmoid(scale*cos+bias) in RRF (no-op for CLIP)
    thumb_dir: Path = Path("./data/thumbs")
    audio_dir: Path = Path("./data/audio")
    # tiered media (store/media.py): hot JPEG -> cold AV1 chunk per episode -> text only
    chunk_dir: Path | None = None  # AV1 chunks; default <thumb_dir>/../chunks
    hot_retention_h: float = 72.0  # frames older than this are re-encoded into AV1 chunks
    cold_retention_days: float = 100.0  # then their pixels are dropped (text, OCR and vectors are kept)
    cold_width: int = 960  # chunk frame width (px, never upscaled)
    cold_crf: int = 50
    media_age_interval_s: float = 3600.0  # background aging pass; 0 = off (POST /v1/admin/age still works)
    privacy_blur_faces: bool = False  # blur bystander faces in stored thumbnails (needs the privacy extra)
    privacy_blur_faces_vlm: bool = False  # also blur before the caption VLM / image embedder; implies the above

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
    episode_split_sim: float = 0.70  # vs the mean of the last episode_recent_n image vectors (SigLIP B/16, #54)
    episode_rejoin_sim: float = 0.80  # vs a recent episode's whole centroid, to rejoin it after a detour
    episode_rejoin_last: int = 4
    episode_recent_n: int = 6
    episode_max_s: float = 1800.0  # split episodes longer than this so long stays still get summarised (0 = off)
    episode_move_m: float = 3.0  # split when the wearer is this far from the episode's pose centroid (0 = off)
    consolidate_enabled: bool = True  # background Consolidator in the server lifespan
    consolidate_interval_s: float = 600.0
    consolidate_min_tokens: int = 3
    summarizer_provider: str = "mock"

    # ambient audio (POST /v1/audio): "" = openai if STT_PROVIDER=openai + key, else the chat STT | mock | openai |
    # faster-whisper (needs the local-stt extra + ffmpeg)
    audio_stt_provider: str = ""
    audio_stt_model: str = ""  # "" -> whisper-1 (openai) / small.en (faster-whisper)
    audio_window_s: float = 15.0  # transcript segments are merged into memory tokens spanning at most this long
    audio_enhance: bool = True  # band-pass + loudnorm when ffmpeg decodes (low-rate headset mics)
    speech_merge_s: float = 10.0  # speech starting <= this long after the device's last transcript token joins it
    speech_max_token_s: float = 60.0  # ... unless the merged token would span longer than this
    speech_link_s: float = 15.0  # retrieval: pair a transcript hit with the nearest frame within +-N s (0 = off)

    @field_validator("chunk_dir", mode="before")
    @classmethod
    def _blank_path_is_unset(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v  # CHUNK_DIR= in .env -> default

    def provider_key(self, provider: str) -> str | None:
        """The API key `provider` authenticates with ("" if unset), or None for keyless/unknown providers."""
        keys = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "voyage": self.voyage_api_key,
            "deepgram": self.deepgram_api_key,
            "elevenlabs": self.elevenlabs_api_key,
            "vlm": "" if self.reader_api_key == "none" else self.reader_api_key,
            "enrich": self.enrich_key,
        }
        return keys.get(provider.lower())

    @property
    def enrich_url(self) -> str:
        return self.enrich_base_url or self.reader_base_url

    @property
    def enrich_model_name(self) -> str:
        return self.enrich_model or self.reader_model

    @property
    def enrich_key(self) -> str:
        """Bearer key for the enrich endpoint; the reader key is only reused for the reader's own endpoint."""
        key = self.enrich_api_key or ("" if self.enrich_base_url else self.reader_api_key)
        return "" if key == "none" else key


@lru_cache
def get_settings() -> Settings:
    return Settings()
