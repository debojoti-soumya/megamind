"""Pydantic schemas. Must stay in sync with docs/api.md and the C# DTOs in headset/."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low", "not_found"]
SearchMode = Literal["semantic", "visual", "hybrid"]
RejectReason = Literal["duplicate", "too_soon", "dark", "blurry", "budget"]
Trigger = Literal["novel", "pose", "heartbeat", "forced"]
Tier = Literal["hot", "cold", "text"]


class Pose(BaseModel):
    x: float
    y: float
    z: float
    yaw: float


class FrameMeta(BaseModel):
    device_id: str
    ts: float
    pose: Pose | None = None
    force: bool = False
    source: str | None = None


class FrameAccepted(BaseModel):
    frame_id: str
    accepted: bool
    reason: RejectReason | None = None


class MemoryToken(BaseModel):
    id: str
    device_id: str
    ts: float
    caption: str = ""
    ocr_text: str = ""
    entities: list[str] = Field(default_factory=list)
    pose: Pose | None = None
    text_vec: list[float] | None = None
    image_vec: list[float] | None = None
    thumb_path: str
    frame_hash: str
    episode_id: str | None = None
    transcript: str = ""
    source: str = "quest"
    trigger: Trigger | None = None
    sharpness: float = 0.0
    enriched: bool = False
    tier: Tier = "hot"
    chunk_id: str | None = None
    chunk_offset: int | None = None

    def search_text(self) -> str:
        return "\n".join(p for p in (self.caption, self.ocr_text, " ".join(self.entities), self.transcript) if p)

    @property
    def thumb_url(self) -> str:
        return f"/v1/thumbs/{self.id}.jpg"


class MemoryHit(BaseModel):
    id: str
    ts: float
    caption: str
    ocr_text: str = ""
    entities: list[str] = Field(default_factory=list)
    thumb_url: str
    score: float
    transcript: str = ""
    episode_id: str | None = None
    tier: Tier = "hot"

    @classmethod
    def from_token(cls, t: MemoryToken, score: float) -> MemoryHit:
        return cls(
            id=t.id,
            ts=t.ts,
            caption=t.caption,
            ocr_text=t.ocr_text,
            entities=t.entities,
            thumb_url=t.thumb_url,
            score=score,
            transcript=t.transcript,
            episode_id=t.episode_id,
            tier=t.tier,
        )


class SearchResponse(BaseModel):
    results: list[MemoryHit]


class MemoryEpisode(BaseModel):
    id: str
    device_id: str
    start_ts: float
    end_ts: float
    n_tokens: int = 0
    title: str = ""
    summary: str = ""
    rep_token_id: str
    entities: list[str] = Field(default_factory=list)
    pose_centroid: Pose | None = None

    @property
    def rep_thumb_url(self) -> str:
        return f"/v1/thumbs/{self.rep_token_id}.jpg"


class EpisodeHit(BaseModel):
    id: str
    start_ts: float
    end_ts: float
    n_tokens: int
    title: str
    summary: str = ""
    entities: list[str] = Field(default_factory=list)
    rep_token_id: str
    rep_thumb_url: str
    score: float | None = None
    top_hits: list[MemoryHit] = Field(default_factory=list)

    @classmethod
    def from_episode(cls, e: MemoryEpisode, score: float | None = None, top_hits=None) -> EpisodeHit:
        return cls(
            id=e.id,
            start_ts=e.start_ts,
            end_ts=e.end_ts,
            n_tokens=e.n_tokens,
            title=e.title,
            summary=e.summary,
            entities=e.entities,
            rep_token_id=e.rep_token_id,
            rep_thumb_url=e.rep_thumb_url,
            score=score,
            top_hits=top_hits or [],
        )


class EpisodeListResponse(BaseModel):
    episodes: list[EpisodeHit]


class Citation(BaseModel):
    id: str
    ts: float
    caption: str
    thumb_url: str


class RecallResult(BaseModel):
    evidence: str
    confidence: Confidence
    cited: list[Citation]
    all_retrieved: list[Citation]


class ToolCallRecord(BaseModel):
    name: str
    args: dict
    confidence: Confidence | None = None


class ChatRequest(BaseModel):
    session_id: str
    device_id: str | None = None
    text: str | None = None


class ChatResponse(BaseModel):
    transcript: str
    answer: str
    audio_url: str | None = None
    cited: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    store_count: int
    models: dict[str, str]
