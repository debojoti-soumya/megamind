from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from megamind.agent.claude_agent import ClaudeAgent
from megamind.agent.reader import build_reader
from megamind.agent.router import router as agent_router
from megamind.agent.stt import build_stt
from megamind.agent.tools import RecallMemory
from megamind.agent.tts import build_tts
from megamind.config import Settings, get_settings
from megamind.ingest.enrich import build_enricher, build_image_embedder, build_text_embedder
from megamind.ingest.filter import KeyframeFilter
from megamind.ingest.router import router as ingest_router
from megamind.ingest.worker import EnrichmentWorker
from megamind.models import HealthResponse
from megamind.store import build_store

logging.basicConfig(level=logging.INFO)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.thumb_dir.mkdir(parents=True, exist_ok=True)
        settings.audio_dir.mkdir(parents=True, exist_ok=True)
        st = app.state
        st.settings = settings
        st.text_embedder = build_text_embedder(settings)
        st.image_embedder = build_image_embedder(settings)
        st.store = build_store(settings, st.text_embedder.dim, st.image_embedder.dim if st.image_embedder else None)
        st.keyframe_filter = KeyframeFilter(settings)
        st.worker = EnrichmentWorker(st.store, build_enricher(settings), st.text_embedder, st.image_embedder)
        st.reader = build_reader(settings)
        st.recall = RecallMemory(st.store, st.text_embedder, st.image_embedder, st.reader)
        st.agent = ClaudeAgent(settings, st.recall)
        st.stt = build_stt(settings)
        st.tts = build_tts(settings)
        await st.worker.start()
        yield
        await st.worker.stop()

    app = FastAPI(title="Megamind", version="0.1.0", lifespan=lifespan)
    app.include_router(ingest_router, prefix="/v1")
    app.include_router(agent_router, prefix="/v1")

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        st = app.state
        return HealthResponse(
            status="ok",
            store_count=await st.store.count(),
            models={
                "claude": settings.claude_model,
                "reader": settings.reader_model,
                "embed": f"{settings.embed_provider}:{settings.embed_model}",
            },
        )

    return app


app = create_app()
