"""Pydantic schemas. Must stay in sync with docs/api.md and the C# DTOs in headset/."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low", "not_found"]
SearchMode = Literal["semantic", "visual", "hybrid"]
TokenKind = Literal["frame", "episode"]


class Pose(BaseModel):
    x: float
    y: float
    z: float
    yaw: float


class FrameMeta(BaseModel):
    device_id: str
    ts: float
    pose: Pose | None = None


class FrameAccepted(BaseModel):
    frame_id: str
    accepted: bool
    reason: Literal["duplicate", "too_soon", "dark", "blurry", "budget"] | None = None


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
    kind: TokenKind = "frame"

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
        )


class SearchResponse(BaseModel):
    results: list[MemoryHit]


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
