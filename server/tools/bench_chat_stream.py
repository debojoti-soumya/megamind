"""Where does chat latency go? POST /v1/chat vs POST /v1/chat/stream against an in-process uvicorn on localhost
(a real socket: httpx's ASGITransport buffers whole responses, which would hide the streaming).

    uv run python tools/bench_chat_stream.py            # simulated Claude (fixed TTFT + token rate), offline
    uv run python tools/bench_chat_stream.py --real     # real Claude via ANTHROPIC_API_KEY (server/.env); bills

Reports, per endpoint: total (what /v1/chat makes the client wait), and for the stream the time to the first
`retrieval` event (candidate thumbnails), the first `delta` (first answer words) and `final`. The agent's own
split (Claude vs tool time, model calls, cache-read tokens) comes from TurnStats. Retrieval/reader/TTS use whatever
the settings build (mocks without keys), so with --real only the Claude part is real unless those keys are set too.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import anthropic
import httpx2
import uvicorn
from httpx import AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megamind.agent.claude_agent import ClaudeAgent  # noqa: E402
from megamind.config import Settings  # noqa: E402
from megamind.ingest.enrich import MockTextEmbedder  # noqa: E402
from megamind.main import create_app  # noqa: E402
from megamind.models import MemoryToken  # noqa: E402

QUESTIONS = ["where did I leave my keys?", "what was on the whiteboard?", "where is my coffee mug?"]
MEMORIES = [
    ("Keys on the kitchen counter next to the coffee machine", ""),
    ("A whiteboard in a meeting room", "Standup 10am Ship demo Sunday"),
    ("A blue coffee mug on a wooden desk beside a laptop", ""),
]
ANSWER = "Your keys are on the kitchen counter next to the coffee machine, about twenty minutes ago. <cites>m0</cites>"


def _sim_transport(ttft_s: float, tok_per_s: float) -> httpx2.AsyncBaseTransport:
    """Simulated Messages API: a tool call on the first (tool-picking) call, then a text answer; streamed when asked."""

    def message(content: list[dict], stop: str) -> dict:
        usage = {"input_tokens": 1200, "output_tokens": 30}
        return {
            "id": "msg_sim",
            "type": "message",
            "role": "assistant",
            "model": "sim",
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": usage,
        }

    async def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        has_result = isinstance(last, list) and any(b.get("type") == "tool_result" for b in last)
        if has_result:
            msg = message([{"type": "text", "text": ANSWER}], "end_turn")
        else:
            use = {"type": "tool_use", "id": "toolu_sim", "name": "recall_memory", "input": {"query": "keys"}}
            msg = message([use], "tool_use")
        words = ANSWER.split(" ") if has_result else ["{}"]
        if not body.get("stream"):
            await asyncio.sleep(ttft_s + len(words) / tok_per_s)
            return httpx2.Response(200, json=msg)

        async def sse():
            def ev(name, data):
                return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

            await asyncio.sleep(ttft_s)
            yield ev("message_start", {"type": "message_start", "message": {**msg, "content": [], "stop_reason": None}})
            b = msg["content"][0]
            start = {"type": "text", "text": ""} if b["type"] == "text" else {**b, "input": {}}
            yield ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": start})
            for i, w in enumerate(words):
                await asyncio.sleep(1 / tok_per_s)
                if b["type"] == "text":
                    delta = {"type": "text_delta", "text": w if i == 0 else " " + w}
                else:
                    delta = {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}
                yield ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta})
            yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
            md = {"stop_reason": msg["stop_reason"], "stop_sequence": None}
            yield ev("message_delta", {"type": "message_delta", "delta": md, "usage": {"output_tokens": 30}})
            yield ev("message_stop", {"type": "message_stop"})

        return httpx2.Response(200, content=sse(), headers={"content-type": "text/event-stream"})

    return httpx2.MockTransport(handler)


async def _seed(app) -> None:
    emb, now = MockTextEmbedder(), time.time()
    for i, (cap, ocr) in enumerate(MEMORIES):
        await app.state.store.add(
            MemoryToken(
                id=f"m{i}",
                device_id="dev1",
                ts=now - 600 * (i + 1),
                caption=cap,
                ocr_text=ocr,
                text_vec=await emb.embed(f"{cap} {ocr}"),
                thumb_path=f"/tmp/m{i}.jpg",
                frame_hash="0" * 16,
            )
        )


async def _stream(c, body: dict) -> dict[str, float]:
    t0, marks = time.perf_counter(), {}
    async with c.stream("POST", "/v1/chat/stream", json=body) as r:
        buf = ""
        async for chunk in r.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                name = block.split("\n", 1)[0].removeprefix("event: ")
                marks.setdefault(name, (time.perf_counter() - t0) * 1000)
    return marks


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="call the real Claude API (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--n", type=int, default=3, help="rounds over the questions")
    ap.add_argument("--ttft", type=float, default=0.6, help="simulated time to first token, s")
    ap.add_argument("--tps", type=float, default=60.0, help="simulated output tokens per second")
    ap.add_argument("--recall-first", action="store_true", help="RECALL_FIRST=true (skip the tool-picking call)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        base = dict(
            thumb_dir=Path(tmp) / "t",
            audio_dir=Path(tmp) / "a",
            store_backend="memory",
            embed_provider="mock",
            enrich_provider="mock",
            consolidate_enabled=False,
        )
        settings = Settings(**base) if args.real else Settings(_env_file=None, **base)
        if args.real and not settings.anthropic_api_key:
            sys.exit("--real needs ANTHROPIC_API_KEY (environment or server/.env)")
        app = create_app(settings)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        serving = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        try:
            await _seed(app)
            agent_settings = settings.model_copy(update={"recall_first": args.recall_first, "anthropic_api_key": "k"})
            client = None
            if not args.real:
                http = httpx2.AsyncClient(transport=_sim_transport(args.ttft, args.tps))
                client = anthropic.AsyncAnthropic(api_key="sim", http_client=http, max_retries=0)
            else:
                agent_settings = agent_settings.model_copy(update={"anthropic_api_key": settings.anthropic_api_key})
            app.state.agent = ClaudeAgent(agent_settings, app.state.recall, client=client)
            stats_log = []
            orig = app.state.agent.chat_with_stats

            async def spy(*a, **kw):
                resp, st = await orig(*a, **kw)
                stats_log.append(st)
                return resp, st

            app.state.agent.chat_with_stats = spy
            chat_ms, marks = [], []
            async with AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=120) as c:
                for k in range(args.n):
                    for q in QUESTIONS:
                        body = {"session_id": f"b{k}-{q}", "device_id": "dev1", "text": q}
                        t0 = time.perf_counter()
                        r = await c.post("/v1/chat", json=body)
                        r.raise_for_status()
                        chat_ms.append((time.perf_counter() - t0) * 1000)
                        marks.append(await _stream(c, {**body, "session_id": body["session_id"] + "-s"}))
        finally:
            server.should_exit = True
            await serving

    def med(xs):
        return f"{statistics.median(xs):7.0f} ms" if xs else "      n/a"

    mode = "real Claude" if args.real else f"simulated Claude (TTFT {args.ttft:.2f}s, {args.tps:.0f} tok/s)"
    print(f"{mode}, recall_first={args.recall_first}, {len(chat_ms)} questions per endpoint (medians)")
    print(f"  POST /v1/chat            total           {med(chat_ms)}")
    for name in ("retrieval", "delta", "final"):
        print(f"  POST /v1/chat/stream     first {name:<9} {med([m[name] for m in marks if name in m])}")
    print(
        f"  agent: llm {med([s.llm_ms for s in stats_log])}, tools {med([s.tool_ms for s in stats_log])}, "
        f"model calls {statistics.median([s.model_calls for s in stats_log]):.0f}, "
        f"cache-read tokens {sum(s.extra.get('cache_read_tokens', 0) for s in stats_log)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
