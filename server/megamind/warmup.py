"""Startup warmup: pay the first-use costs before the first question instead of during it.

A fresh process pays a lot of one-off costs on its first turn: lazy SDK imports and pydantic models, LanceDB opening
the table and loading its FTS/vector indices, torch's first SigLIP pass, PIL/JPEG codecs and the image cache in the
reader, plus the first TLS handshakes. `start_warmup` runs one cheap pass over every piece on the question path in a
background task. It never blocks startup or a request, never raises (each step is time-boxed, failures are logged),
and never makes a paid model call. The only billed request is one tiny embedding (a few tokens) when the text
embedder is hosted. Off with `WARMUP=false`.

Steps (concurrent):
- `anthropic`: `messages.count_tokens` on the agent's client (free) and the streaming helpers' import.
- `retrieval`: one hybrid retrieval: the query embedding (opens the embedder's HTTP connection), LanceDB opening its
  table and FTS/vector indices, then the reader's image loading and encoding on the hits when the reader is local
  (Claude reads the images; no reader-model call).
- `image_embed`: one SigLIP/CLIP text encode in a worker thread (torch's first pass is the slow one).
- `stt` / `tts`: the OpenAI SDK's audio resources plus a free `models.list` on each client.

`app.state.warmup` holds the status ("off", "running", "done" or "done (failed: embed, tts)") and the per-step
timings. `GET /v1/health` reports the status and never waits for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

WARMUP_QUERY = "what did I see on the table?"  # ordinary words: exercises the planner, vector and BM25 lists


@dataclass
class WarmupState:
    status: str = "off"  # off | running | done | done (failed: a, b)
    steps: dict[str, str] = field(default_factory=dict)  # step -> "ok 120 ms" | "skipped (...)" | "failed (...)"
    started: float | None = None
    elapsed_s: float | None = None


def _hybrid_of(retriever: Any) -> Any | None:
    """The plain HybridRetriever inside whatever wraps it (live wrapper, hier, combo, UCC fallback), or None.
    Warmup only ever calls the hybrid: hier/combo/UCC recalls can make paid LLM or corpus calls."""
    from megamind.retrieval.hybrid import HybridRetriever

    seen: set[int] = set()
    stack = [retriever]
    while stack:
        r = stack.pop()
        if r is None or id(r) in seen:
            continue
        seen.add(id(r))
        if isinstance(r, HybridRetriever):
            return r
        d = getattr(r, "__dict__", {})  # not getattr: LiveRetriever.__getattr__ forwards unknown names
        stack.extend(d.get(name) for name in ("inner", "hybrid", "base", "fallback"))
    return None


async def _anthropic(st: Any, settings: Any) -> str | None:
    client = getattr(getattr(st, "agent", None), "_client", None)
    if client is None:
        return "skipped (no Claude key)"
    import anthropic.lib.streaming  # noqa: F401  (the /v1/chat/stream path)

    await client.messages.count_tokens(model=settings.claude_model, messages=[{"role": "user", "content": "hi"}])
    return None


async def _openai_client(client: Any, *resources: str) -> str | None:
    if client is None:
        return "skipped (mock)"
    for path in resources:  # the SDK imports each resource module on first attribute access
        obj = client
        for part in path.split("."):
            obj = getattr(obj, part)
    await client.models.list()
    return None


async def _embed(st: Any) -> str | None:
    emb = getattr(st, "text_embedder", None)
    if emb is None:
        return "skipped (no embedder)"
    q = getattr(emb, "embed_query", None) or emb.embed
    await q(WARMUP_QUERY)
    return None


async def _image_embed(st: Any) -> str | None:
    emb = getattr(st, "image_embedder", None)
    if emb is None:
        return "skipped (no image embedder)"
    await emb.embed_text("a photo of a table")
    return None


async def _retrieval(st: Any) -> str | None:
    hybrid = _hybrid_of(getattr(st, "retriever", None))
    if hybrid is None:
        return await _embed(st)
    res = await hybrid.retrieve(WARMUP_QUERY, k=12)
    reader = getattr(st, "reader", None)
    name = type(reader).__name__
    # local readers only: a VLM reader (Sail) would be a paid call
    local = name in ("ClaudeVisionReader", "CaptionOnlyReader") or name.startswith("Mock")
    if res.tokens and reader is not None and local:
        await reader.read(WARMUP_QUERY, res.memory_tokens())
    return None if res.tokens else "ok (empty store)"


def _steps(st: Any, settings: Any) -> dict[str, Callable[[], Awaitable[str | None]]]:
    return {
        "anthropic": lambda: _anthropic(st, settings),
        "image_embed": lambda: _image_embed(st),
        "retrieval": lambda: _retrieval(st),
        "stt": lambda: _openai_client(getattr(getattr(st, "stt", None), "client", None), "audio.transcriptions"),
        "tts": lambda: _openai_client(getattr(getattr(st, "tts", None), "client", None), "audio.speech"),
    }


async def run_warmup(st: Any, settings: Any, state: WarmupState | None = None) -> WarmupState:
    """Run every step concurrently, each bounded by `settings.warmup_timeout_s`. Never raises (except cancel)."""
    state = state or WarmupState()
    state.status, state.started = "running", time.perf_counter()
    timeout = float(getattr(settings, "warmup_timeout_s", 20.0))

    async def one(name: str, step: Callable[[], Awaitable[str | None]]) -> bool:
        t0 = time.perf_counter()
        try:
            note = await asyncio.wait_for(step(), timeout)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            state.steps[name] = f"failed (timeout {timeout:.0f} s)"
            log.warning("warmup %s: timed out after %.0f s", name, timeout)
            return False
        except Exception as e:  # noqa: BLE001  (warmup must never take the server down)
            state.steps[name] = f"failed ({type(e).__name__})"
            log.warning("warmup %s failed: %s: %s", name, type(e).__name__, e)
            return False
        ms = (time.perf_counter() - t0) * 1000
        state.steps[name] = note if note and note.startswith("skipped") else f"{note or 'ok'} {ms:.0f} ms"
        return True

    steps = _steps(st, settings)
    oks = await asyncio.gather(*(one(n, s) for n, s in steps.items()))
    failed = [n for n, ok in zip(steps, oks, strict=True) if not ok]
    state.elapsed_s = time.perf_counter() - state.started
    state.status = f"done (failed: {', '.join(failed)})" if failed else "done"
    log.info(
        "warmup %s in %.2f s: %s", state.status, state.elapsed_s, ", ".join(f"{k} {v}" for k, v in state.steps.items())
    )
    return state


def start_warmup(st: Any, settings: Any) -> asyncio.Task | None:
    """Schedule `run_warmup` in the background (returns at once). `st.warmup` is the live status."""
    st.warmup = WarmupState()
    if not getattr(settings, "warmup", True):
        return None
    st.warmup.status = "running"
    return asyncio.create_task(run_warmup(st, settings, st.warmup), name="megamind-warmup")


async def stop_warmup(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
