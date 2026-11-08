"""Megamind server on Modal: public HTTPS for the headset, no hotspot / Tailscale / firewall (deploy/README.md).

    modal deploy deploy/modal_app.py                         # live (secrets megamind-providers + megamind-auth)
    MEGAMIND_MIN_CONTAINERS=1 modal deploy deploy/modal_app.py   # live, one container kept warm (demo window)
    MEGAMIND_PROFILE=mock modal deploy deploy/modal_app.py   # keyless smoke test (secret megamind-auth only)

One CPU container (`max_containers=1`): the in-process workers (enrichment, consolidator, UCC sync, media aging) and
LanceDB are single-writer. All state (LanceDB, thumbnails, AV1 chunks, TTS audio, replay/montage caches) lives on a
Modal Volume mounted at /data, so memories survive restarts and redeploys.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import modal

SERVER_DIR = Path(__file__).resolve().parent.parent / "server"
DATA = "/data"
# Read at deploy time and baked into the image env, so the container re-imports this file with the same value.
PROFILE = os.environ.get("MEGAMIND_PROFILE", "live")
if PROFILE not in ("live", "mock"):
    raise SystemExit(f"MEGAMIND_PROFILE must be live or mock, not {PROFILE!r}")
# Warm containers kept even when idle: 0 = scale to zero (cold start ~10-15 s); 1 for a demo window (~1 vCPU-h per h).
MIN_CONTAINERS = int(os.environ.get("MEGAMIND_MIN_CONTAINERS", "0"))
if MIN_CONTAINERS not in (0, 1):
    raise SystemExit("MEGAMIND_MIN_CONTAINERS must be 0 or 1 (max_containers=1)")

COMMON_ENV = {
    "MEGAMIND_PROFILE": PROFILE,
    "STORE_BACKEND": "lancedb",
    "LANCEDB_PATH": f"{DATA}/lancedb",
    "THUMB_DIR": f"{DATA}/thumbs",  # chunks / replay / montages default to siblings of this dir, i.e. on the Volume
    "AUDIO_DIR": f"{DATA}/audio",
    "IMAGE_EMBED_PROVIDER": "none",  # SigLIP needs CPU torch (~1 GB image, slow cold start); opt-in by editing here
}
PROFILE_ENV = {
    # mirrors server/.env.example; keys come from the megamind-providers secret
    "live": {
        "EMBED_PROVIDER": "openai",
        "RETRIEVER_BACKEND": "hybrid",  # combo/hier are opt-in: on a 9k-memory week UCC timed out and hier took 16 s p50
        "READER_PROVIDER": "auto",
        "READER_BASE_URL": "https://api.sailresearch.com/v1",
        "ENRICH_PROVIDER": "auto",
        "STT_PROVIDER": "openai",
        "TTS_PROVIDER": "openai",
        "SUMMARIZER_PROVIDER": "mock",
        # recall: LLM block notes + pick/rank over the hybrid's candidates (bake-off #152); falls back to the
        # hybrid on any LLM error and when OPENAI_API_KEY is absent. Notes live on the Volume, so a cold start
        # reuses them; missing ones are written at query time (<=3 inline, a larger backlog in the background).
        "RETRIEVER_BACKEND": "hier",
        "HIER_MODEL": "gpt-4.1-mini",  # block pick; the model sweep in #152 picked this pair
        "HIER_RANK_MODEL": "gpt-4.1",  # token rank
        "HIER_NOTE_MODEL": "gpt-4.1-mini",
        "HIER_CACHE_PATH": f"{DATA}/hier_notes.json",
        "HIER_REFRESH_S": "300",  # background note refresh while the container is up
    },
    # no paid calls at all, even if a key leaks into the environment
    "mock": {
        "EMBED_PROVIDER": "mock",
        "RETRIEVER_BACKEND": "hybrid",
        "READER_PROVIDER": "mock",
        "ENRICH_PROVIDER": "mock",
        "STT_PROVIDER": "mock",
        "TTS_PROVIDER": "mock",
        "AUDIO_STT_PROVIDER": "mock",
        "SUMMARIZER_PROVIDER": "mock",
        "MONTAGE_PLANNER": "offline",
        "ANTHROPIC_API_KEY": "",
        "RETRIEVER_BACKEND": "hybrid",  # hier / ucc would call OpenAI / UCC
    },
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "fonts-dejavu-core")  # ffmpeg >= 5.1 (store/media.py uses -fps_mode, #114)
    .add_local_dir(
        SERVER_DIR,
        "/app/server",
        copy=True,
        ignore=[".venv", ".env", "data", "tests", "tools", "**/__pycache__", "*.egg-info", ".pytest_cache"],
    )
    .run_commands("pip install --no-cache-dir /app/server", "ffmpeg -hide_banner -version | head -1")
    .env(COMMON_ENV | PROFILE_ENV[PROFILE])
)

# a separate Volume per profile: mock embeddings are 256-d, OpenAI 1024-d, and a LanceDB table's dim is frozen
volume = modal.Volume.from_name(
    "megamind-data" if PROFILE == "live" else "megamind-data-mock", create_if_missing=True, version=2
)
secrets = [modal.Secret.from_name("megamind-auth")]  # API_TOKEN: the URL is public, so auth is not optional here
if PROFILE == "live":
    secrets.insert(0, modal.Secret.from_name("megamind-providers"))

app = modal.App("megamind")


@app.function(
    image=image,
    volumes={DATA: volume},
    secrets=secrets,
    cpu=1.0,
    memory=2048,
    min_containers=MIN_CONTAINERS,  # MEGAMIND_MIN_CONTAINERS=1 for a demo window, 0 afterwards
    max_containers=1,  # single writer: LanceDB + in-process background workers
    scaledown_window=600,
    timeout=600,  # per request; /v1/chat/stream and replay renders stay well below this
)
@modal.concurrent(max_inputs=64)  # one container serves the headset, the browser panel and SSE streams at once
@modal.asgi_app()
def web():
    from megamind.main import create_app

    fastapi_app = create_app()
    inner = fastapi_app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(a):
        async with inner(a):
            yield
        # workers are stopped: persist their last writes now (Modal also commits in the background and at exit)
        await volume.commit.aio()

    fastapi_app.router.lifespan_context = lifespan
    return fastapi_app
