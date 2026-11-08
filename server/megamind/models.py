"""Pydantic schemas. Must stay in sync with docs/api.md and the C# DTOs in headset/."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low", "not_found"]
SearchMode = Literal["semantic", "visual", "hybrid"]
TokenKind = Literal["frame", "episode", "transcript"]
Tier = Literal["hot", "cold", "text"]  # where a frame's pixels live: JPEG / AV1 chunk / none (text + vectors only)


class Pose(BaseModel):
    x: float
    y: float
    z: float
    yaw: float


class FrameMeta(BaseModel):
    device_id: str
    ts: float
    pose: Pose | None = None
    # text heard around this frame (e.g. subtitles / external ASR); stored as a kind="transcript" token, never
    # attached to the frame and never changes the keep decision. "" / null = none.
    transcript: str | None = None


class FrameAccepted(BaseModel):
    frame_id: str
    accepted: bool
    reason: Literal["duplicate", "too_soon", "dark", "blurry", "budget"] | None = None


class AudioMeta(BaseModel):
    device_id: str
    ts_start: float = Field(allow_inf_nan=False)  # unix seconds of the clip's first sample
    pose: Pose | None = None


class AudioAccepted(BaseModel):
    audio_id: str
    duration_s: float = 0.0
    transcript: str = ""
    token_ids: list[str] = Field(default_factory=list)


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
    tier: Tier = "hot"
    chunk_id: str | None = None  # tier="cold": the AV1 chunk holding this frame ...
    chunk_offset: int | None = None  # ... and its frame index inside it

    @property
    def has_image(self) -> bool:
        """False for tokens without pixels: kind="transcript", or frames aged to tier="text"."""
        if self.kind == "transcript" or self.tier == "text":
            return False
        if self.tier == "cold":
            return bool(self.chunk_id) and self.chunk_offset is not None
        return bool(self.thumb_path)

    @property
    def thumb_url(self) -> str:
        """`/v1/thumbs/{id}.jpg` on the hot and cold tiers; empty when the token has no image."""
        return f"/v1/thumbs/{self.id}.jpg" if self.has_image else ""


class MemoryHit(BaseModel):
    id: str
    ts: float
    caption: str
    ocr_text: str = ""
    entities: list[str] = Field(default_factory=list)
    thumb_url: str
    score: float
    kind: TokenKind = "frame"

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
            kind=t.kind,
        )


class SearchResponse(BaseModel):
    results: list[MemoryHit]


class PurgeResponse(BaseModel):
    deleted: int


class Citation(BaseModel):
    id: str
    ts: float
    caption: str
    thumb_url: str  # /v1/thumbs/{id}.jpg; /v1/live/{id}.jpg for live NOW frames (id "now_..."); "" = no image
    replay_url: str | None = None  # /v1/replay.mp4 clip around ts; null when no frames are buffered


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


class ChatStreamRetrieval(BaseModel):
    """`POST /v1/chat/stream` event `retrieval`: candidates from one recall (cited first), before the answer."""

    confidence: Confidence
    candidates: list[Citation] = Field(default_factory=list)


class ChatStreamDelta(BaseModel):
    """`POST /v1/chat/stream` event `delta`: the next piece of answer text (append)."""

    text: str


class ChatStreamError(BaseModel):
    """`POST /v1/chat/stream` event `error`: the turn failed after the stream started; no `final` follows."""

    detail: str


class MontageRequest(BaseModel):
    """`POST /v1/montage`: a query, a time range, or both (time bounds: unix seconds or ISO-8601; "" = unset)."""

    device_id: str | None = None
    query: str | None = None
    time_from: float | str | None = None
    time_to: float | str | None = None
    max_clips: int | None = Field(None, ge=0, le=24)  # None/0 = MONTAGE_MAX_CLIPS


class MontageClip(BaseModel):
    token_id: str
    ts: float
    title: str
    duration_s: float
    source: Literal["replay", "timelapse"]  # buffered frames (#58) or a ~1 fps run of stored keyframes
    thumb_url: str = ""


class MontageResponse(BaseModel):
    id: str
    url: str  # /v1/montages/{id}.mp4
    title: str
    narration: str = ""
    planner: Literal["claude", "offline"] = "offline"
    clips: list[MontageClip] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    store_count: int
    models: dict[str, str]
