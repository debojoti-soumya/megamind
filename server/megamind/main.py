from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from megamind.agent.claude_agent import ClaudeAgent
from megamind.agent.reader import build_reader
from megamind.agent.router import router as agent_router
from megamind.agent.stt import build_stt
from megamind.agent.tools import RecallMemory
from megamind.agent.tts import build_tts
from megamind.config import Settings, get_settings
from megamind.ingest.audio import router as audio_router
from megamind.ingest.enrich import build_enricher, build_image_embedder, build_text_embedder
from megamind.ingest.filter import KeyframeFilter
from megamind.ingest.privacy import build_face_blur
from megamind.ingest.router import router as ingest_router
from megamind.ingest.transcribe import build_audio_stt
from megamind.ingest.worker import EnrichmentWorker
from megamind.models import HealthResponse
from megamind.replay import install as install_replay
from megamind.retrieval import build_retriever
from megamind.store import build_store
from megamind.store.consolidate import Consolidator, build_summarizer
from megamind.ui.router import router as ui_router

logging.basicConfig(level=logging.INFO)


_LOCAL_DEPS = {"local": ("torch", "sentence_transformers"), "openclip": ("torch", "open_clip")}  # .[local-embeddings]


def active_provider(settings: Settings, configured: str, impl: object, active: str) -> str:
    """Health label for what is really serving a role: `active` when a real backend was built, otherwise
    e.g. "mock (openai configured, no key)" so an empty key can't masquerade as a live provider."""
    configured = configured.lower()
    if not type(impl).__name__.startswith(("Mock", "CaptionOnly")):
        return active
    if configured in ("", "mock"):
        return "mock"
    if settings.provider_key(configured) == "":
        why = "no key"
    elif any(importlib.util.find_spec(m) is None for m in _LOCAL_DEPS.get(configured, ())):
        why = "not installed"
    else:
        why = "not implemented"
    return f"mock ({configured} configured, {why})"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.thumb_dir.mkdir(parents=True, exist_ok=True)
        settings.audio_dir.mkdir(parents=True, exist_ok=True)
        st = app.state
        st.settings = settings
        st.store = build_store(settings)
        st.keyframe_filter = KeyframeFilter(settings)
        st.pending_hashes = {}
        st.text_embedder = build_text_embedder(settings)
        st.image_embedder = build_image_embedder(settings)
        st.worker = EnrichmentWorker(
            st.store,
            build_enricher(settings),
            st.text_embedder,
            st.image_embedder,
            face_blur=build_face_blur(settings),
        )
        st.reader = build_reader(settings)
        st.retriever, st.ucc_indexer = build_retriever(settings, st.store, st.text_embedder, st.image_embedder)
        st.recall = RecallMemory(
            st.store, st.text_embedder, st.image_embedder, st.reader, retriever=st.retriever, tz=settings.wearer_tz
        )
        st.agent = ClaudeAgent(settings, st.recall)
        st.stt = build_stt(settings)
        st.tts = build_tts(settings)
        st.audio_stt = build_audio_stt(settings, st.stt)
        st.consolidator = Consolidator(
            st.store,
            build_summarizer(settings),
            st.text_embedder,
            gap_s=settings.episode_gap_s,
            min_tokens=settings.consolidate_min_tokens,
        )
        await st.worker.start()
        if st.ucc_indexer is not None:
            await st.ucc_indexer.start()
        st.consolidate_task = None
        if settings.consolidate_enabled:
            st.consolidate_task = asyncio.create_task(st.consolidator.run_forever(settings.consolidate_interval_s))
        yield
        if st.ucc_indexer is not None:
            await st.ucc_indexer.stop()
        if st.consolidate_task is not None:
            st.consolidate_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await st.consolidate_task
        await st.worker.stop()

    app = FastAPI(title="Megamind", version="0.1.0", lifespan=lifespan)
    app.include_router(ingest_router, prefix="/v1")
    app.include_router(audio_router, prefix="/v1")
    app.include_router(agent_router, prefix="/v1")
    app.include_router(ui_router)
    # the browser panel may be served from a different origin (e.g. file:// or another laptop)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    install_replay(app, settings)  # moment replay: frame tap + GET /v1/replay.mp4

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        st = app.state
        return HealthResponse(
            status="ok",
            store_count=await st.store.count(),
            models={
                "claude": settings.claude_model
                if st.agent.enabled
                else f"fallback ({settings.claude_model} configured, no key)",
                "reader": active_provider(
                    settings,
                    "vlm" if settings.reader_provider.lower() == "auto" else settings.reader_provider,
                    st.reader,
                    settings.reader_model,
                ),
                "embed": active_provider(
                    settings,
                    settings.embed_provider,
                    st.text_embedder,
                    f"{settings.embed_provider}:{settings.embed_model}",
                ),
                "stt": active_provider(settings, settings.stt_provider, st.stt, settings.stt_provider),
                "tts": active_provider(settings, settings.tts_provider, st.tts, settings.tts_provider),
            },
        )

    return app


app = create_app()
