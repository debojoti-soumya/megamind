from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from megamind.agent.claude_agent import ClaudeAgent
from megamind.agent.collage import agent_tool as collage_tool
from megamind.agent.collage import install as install_collage
from megamind.agent.locate import build_locator
from megamind.agent.montage import install as install_montage
from megamind.agent.reader import build_reader
from megamind.agent.router import router as agent_router
from megamind.agent.stt import build_stt
from megamind.agent.tools import RecallMemory
from megamind.agent.tts import build_tts
from megamind.agent.voice import router as voice_router
from megamind.auth import TokenAuth
from megamind.config import Settings, get_settings
from megamind.devices import router as devices_router
from megamind.ingest.audio import TranscriptSink
from megamind.ingest.audio import router as audio_router
from megamind.ingest.enrich import build_enricher
from megamind.ingest.episodes_route import router as episodes_router
from megamind.ingest.filter import KeyframeFilter
from megamind.ingest.privacy import build_face_blur
from megamind.ingest.recent import router as recent_router
from megamind.ingest.router import router as ingest_router
from megamind.ingest.transcribe import build_audio_stt
from megamind.ingest.worker import EnrichmentWorker
from megamind.live import install as install_live
from megamind.live import wrap_retriever as wrap_live_retriever
from megamind.models import HealthResponse
from megamind.replay import install as install_replay
from megamind.retrieval import build_retriever
from megamind.retrieval.hier import HierRetriever
from megamind.store import build_store
from megamind.store.consolidate import Consolidator, build_summarizer
from megamind.store.embeddings import build_image_embedder, build_text_embedder
from megamind.store.media import MediaStore, aging_loop
from megamind.ui.router import router as ui_router
from megamind.video import install as install_video
from megamind.warmup import WarmupState, start_warmup, stop_warmup
from megamind.xr import install as install_xr

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


_LOCAL_DEPS = {"local": ("torch", "sentence_transformers"), "openclip": ("torch", "open_clip")}  # .[local-embeddings]


def active_provider(settings: Settings, configured: str, impl: object, active: str) -> str:
    """Health label for what is really serving a role: `active` when a real backend was built, otherwise
    e.g. "mock (openai configured, no key)" so an empty key can't masquerade as a live provider."""
    configured = configured.lower()
    if not type(impl).__name__.startswith(("Mock", "CaptionOnly")):
        degraded = getattr(impl, "degraded", None)
        return f"{active} (degraded: {degraded})" if degraded else active
    if configured in ("", "mock"):
        return "mock"
    if settings.provider_key(configured) == "":
        why = "no key"
    elif any(importlib.util.find_spec(m) is None for m in _LOCAL_DEPS.get(configured, ())):
        why = "not installed"
    else:
        why = "not implemented"
    return f"mock ({configured} configured, {why})"


def _image_label(settings: Settings, image_embedder: object | None) -> str:
    """Health label for image vectors: "openclip:ViT-B-16-SigLIP", "mock", or "none (...)" when they are off."""
    configured = settings.image_embed_provider.lower()
    if image_embedder is None:
        return "none" if configured in ("", "none") else f"none ({configured} configured, not installed)"
    return active_provider(settings, configured, image_embedder, f"{configured}:{settings.image_embed_model}")


def image_calibration(settings: Settings, image_embedder: object | None) -> tuple[float, float] | None:
    """SigLIP (scale, bias) for calibrated image fusion, or None (toggle off, no embedder, or a CLIP model)."""
    if image_embedder is None or not settings.search_calibrate_image:
        return None
    cal = getattr(image_embedder, "calibration", None)
    if cal is not None:
        log.info("image fusion calibrated: p = sigmoid(%.2f * cos %+.2f)", *cal)
    return cal


def _enrich_label(settings: Settings, enricher: object) -> str:
    """Health label for the enricher: "vlm:<model>" / "local" (+ degraded reason), or mock and why."""
    configured = settings.enrich_provider.lower()
    if type(enricher).__name__ == "LocalEnricher":
        active = "local"
    else:
        active = f"vlm:{settings.enrich_model_name}"
    if configured == "auto" and type(enricher).__name__.startswith("Mock"):
        return "mock (auto: no enrich/reader key, no image embedder)"
    # vlm without a key reports "mock (enrich configured, no key)" via provider_key("enrich")
    return active_provider(settings, "enrich" if configured in ("auto", "vlm") else configured, enricher, active)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t_start = time.perf_counter()
        settings.thumb_dir.mkdir(parents=True, exist_ok=True)
        settings.audio_dir.mkdir(parents=True, exist_ok=True)
        st = app.state
        st.settings = settings
        st.text_embedder = build_text_embedder(settings)
        st.image_embedder = build_image_embedder(settings)
        st.store = build_store(
            settings,
            text_dim=getattr(st.text_embedder, "dim", None),
            image_dim=getattr(st.image_embedder, "dim", None),
        )
        st.keyframe_filter = KeyframeFilter(settings)
        st.pending_hashes = {}
        st.store.image_calibration = image_calibration(settings, st.image_embedder)
        st.transcripts = TranscriptSink(
            st.store,
            st.text_embedder,
            merge_s=settings.speech_merge_s,
            max_token_s=settings.speech_max_token_s,
        )
        st.face_blur = build_face_blur(settings)  # thumbnails (worker) and replay clips
        st.worker = EnrichmentWorker(
            st.store,
            build_enricher(settings, st.image_embedder),
            st.text_embedder,
            st.image_embedder,
            face_blur=st.face_blur,
        )
        st.media = MediaStore(settings, st.store, lock=st.worker.add_lock)  # hot JPEG -> cold AV1 -> text
        st.worker.media = st.media
        st.reader = build_reader(settings, image_loader=st.media.read)
        st.retriever, st.ucc_indexer = build_retriever(settings, st.store, st.text_embedder, st.image_embedder)
        hier = st.retriever if isinstance(st.retriever, HierRetriever) else getattr(st.retriever, "hier", None)
        st.retriever = wrap_live_retriever(st, settings, st.retriever)  # NOW items for "right now" questions
        st.recall = RecallMemory(
            st.store, st.text_embedder, st.image_embedder, st.reader, retriever=st.retriever, tz=settings.wearer_tz
        )
        st.agent = ClaudeAgent(settings, st.recall, collage=collage_tool(st))
        st.stt = build_stt(settings)
        st.tts = build_tts(settings)
        st.locator = build_locator(st, settings)  # "where is X": X's box in a cited frame (ChatResponse.markers)
        st.audio_stt = build_audio_stt(settings, st.stt)
        st.consolidator = Consolidator(
            st.store,
            build_summarizer(settings),
            st.text_embedder,
            gap_s=settings.episode_gap_s,
            min_tokens=settings.consolidate_min_tokens,
            image_loader=st.media.read,
            thumb_dir=settings.thumb_dir,
        )
        await st.worker.start()
        if st.ucc_indexer is not None:
            await st.ucc_indexer.start()
        st.consolidate_task = None
        if settings.consolidate_enabled:
            st.consolidate_task = asyncio.create_task(st.consolidator.run_forever(settings.consolidate_interval_s))
        st.hier_task = None
        if hier is not None and settings.hier_refresh_s > 0:
            st.hier_task = asyncio.create_task(hier.run_forever(settings.hier_refresh_s))
        st.aging_task = None
        if settings.media_age_interval_s > 0:
            st.aging_task = asyncio.create_task(aging_loop(st, settings.media_age_interval_s))
        st.warmup_task = start_warmup(st, settings)  # background: never delays startup or the first request
        log.info("startup ready in %.2f s (warmup %s)", time.perf_counter() - t_start, st.warmup.status)
        yield
        await stop_warmup(st.warmup_task)
        if st.hier_task is not None:
            st.hier_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await st.hier_task
        if st.aging_task is not None:
            st.aging_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await st.aging_task
        if st.ucc_indexer is not None:
            await st.ucc_indexer.stop()
        if st.consolidate_task is not None:
            st.consolidate_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await st.consolidate_task
        await st.worker.stop()

    app = FastAPI(title="Megamind", version="0.1.0", lifespan=lifespan)
    app.include_router(recent_router, prefix="/v1")  # GET /v1/memories/recent
    app.include_router(episodes_router, prefix="/v1")  # GET /v1/episodes
    app.include_router(ingest_router, prefix="/v1")
    app.include_router(audio_router, prefix="/v1")
    app.include_router(agent_router, prefix="/v1")
    app.include_router(voice_router, prefix="/v1")  # POST /v1/stt
    app.include_router(devices_router, prefix="/v1")  # GET /v1/devices
    app.include_router(ui_router)
    # the browser panel may be served from a different origin (e.g. file:// or another laptop)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    install_replay(app, settings)  # moment replay: frame tap + GET /v1/replay.mp4
    install_video(app, settings)  # real headset video: POST /v1/video; replay + montage prefer it when present
    install_live(app, settings)  # live short-term memory: GET /v1/live/{id}.jpg (+ retriever wrap in lifespan)
    install_xr(app)  # Memory Theater: static WebXR page at GET /xr
    install_montage(app, settings)  # highlight reel: POST /v1/montage + GET /v1/montages/{id}.mp4
    install_collage(app, settings)  # photo montage: POST /v1/collage + GET /v1/collages/{id}.jpg + chat tool
    if settings.api_token:
        app.add_middleware(TokenAuth, token=settings.api_token)  # outermost: nothing runs before the check

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        st = app.state
        store_degraded = getattr(st.store, "degraded", None)
        return HealthResponse(
            status="degraded" if store_degraded else "ok",
            store_count=await st.store.count(),
            warmup=getattr(st, "warmup", WarmupState()).status,
            models={
                "store": f"{settings.store_backend} (degraded: {store_degraded})"
                if store_degraded
                else settings.store_backend,
                "claude": settings.claude_model
                if st.agent.enabled
                else f"fallback ({settings.claude_model} configured, no key)",
                "reader": f"claude ({settings.claude_model} reads the snapshots)"
                if type(st.reader).__name__ == "ClaudeVisionReader"
                else active_provider(
                    settings,
                    "vlm" if settings.reader_provider.lower() == "auto" else settings.reader_provider,
                    st.reader,
                    settings.reader_model,
                ),
                "embed": active_provider(
                    settings,
                    settings.embed_provider,
                    st.text_embedder,
                    f"{settings.embed_provider}:{getattr(st.text_embedder, 'model', settings.embed_model)}",
                ),
                "enrich": _enrich_label(settings, st.worker.enricher),
                "stt": active_provider(settings, settings.stt_provider, st.stt, settings.stt_provider),
                "tts": active_provider(settings, settings.tts_provider, st.tts, settings.tts_provider),
                "image": _image_label(settings, st.image_embedder),
            },
        )

    return app


app = create_app()
