"""Paid-call metering for the retrieval eval: an httpx transport that caches OpenAI / UCC answers on disk, counts
calls and dollars, and enforces a total budget across runs (a ledger in the cache dir).

- Cacheable calls: OpenAI `/chat/completions` + `/embeddings`, UCC `/queries/select` with facet columns (the calls
  that run a model). Uploads, corpus info and fields-only selects pass through uncounted.
- A cache hit replays the recorded latency (`replay_latency`), so latency columns and UCC/LLM timeouts behave as in
  the live run while costing nothing; its cost still counts towards the backend's per-question figures (`all`), but
  not towards the ledger (`live`).
- Before a live call the ledger is checked against `max_usd` / `max_ucc`; past either, `BudgetExceeded` (a
  RuntimeError, which every backend treats as a failed call and falls back on) is raised instead of spending.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# USD per 1M tokens: (input, cached input, output); https://openai.com/api/pricing (2026-11)
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "text-embedding-3-small": (0.02, 0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.13, 0.0),
}


class BudgetExceeded(RuntimeError):
    pass


def openai_cost(model: str, usage: dict) -> float:
    key = next((m for m in sorted(PRICES, key=len, reverse=True) if model.startswith(m)), None)
    if key is None:
        return 0.0
    pin, pcached, pout = PRICES[key]
    prompt = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    out = int(usage.get("completion_tokens") or 0)
    return ((prompt - cached) * pin + cached * pcached + out * pout) / 1e6


@dataclass
class Tally:
    calls: int = 0  # paid API calls (chat + embeddings + UCC facet selects)
    llm_calls: int = 0  # chat completions
    ucc_queries: int = 0
    usd: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    def add(self, e: dict) -> None:
        self.calls += 1
        self.llm_calls += e["kind"] == "chat"
        self.ucc_queries += e["kind"] == "ucc"
        self.usd += e.get("usd", 0.0)
        u = e.get("usage") or {}
        self.prompt_tokens += int(u.get("prompt_tokens") or u.get("total_tokens") or 0)
        self.cached_tokens += int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        self.completion_tokens += int(u.get("completion_tokens") or 0)

    def minus(self, o: Tally) -> Tally:
        return Tally(**{k: getattr(self, k) - getattr(o, k) for k in self.__dataclass_fields__})

    def copy(self) -> Tally:
        return Tally(**self.__dict__)


@dataclass
class Meter:
    cache_dir: Path | None = None  # None: no cache, no ledger (tests)
    max_usd: float = 3.0
    max_ucc: int = 300
    replay_latency: bool = True
    all: Tally = field(default_factory=Tally)  # every paid-call answer the backends consumed (cached or not)
    live: Tally = field(default_factory=Tally)  # this process's real calls
    _cache: dict[str, dict] = field(default_factory=dict)
    _ledger: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        f = self.cache_dir / "responses.jsonl"
        if f.is_file():
            for line in f.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    self._cache[e["key"]] = e
        lf = self.cache_dir / "ledger.json"
        self._ledger = json.loads(lf.read_text()) if lf.is_file() else {"openai_usd": 0.0, "ucc_queries": 0, "calls": 0}

    @property
    def ledger(self) -> dict:
        return dict(self._ledger)

    def transport(self, service: str) -> MeteredTransport:
        return MeteredTransport(self, service)

    def _check(self, kind: str) -> None:
        if kind == "ucc" and self._ledger.get("ucc_queries", 0) >= self.max_ucc:
            raise BudgetExceeded(f"UCC budget of {self.max_ucc} queries reached")
        if kind != "ucc" and self._ledger.get("openai_usd", 0.0) >= self.max_usd:
            raise BudgetExceeded(f"OpenAI budget of ${self.max_usd:.2f} reached")

    def _record_live(self, e: dict) -> None:
        self.live.add(e)
        self.all.add(e)
        self._ledger["openai_usd"] = self._ledger.get("openai_usd", 0.0) + e.get("usd", 0.0)
        self._ledger["ucc_queries"] = self._ledger.get("ucc_queries", 0) + (e["kind"] == "ucc")
        self._ledger["calls"] = self._ledger.get("calls", 0) + 1
        if self.cache_dir is not None:
            (self.cache_dir / "ledger.json").write_text(json.dumps(self._ledger, indent=1))
            if e.get("status") == 200:
                with (self.cache_dir / "responses.jsonl").open("a") as f:
                    f.write(json.dumps(e) + "\n")
        if e.get("status") == 200:
            self._cache[e["key"]] = e


def _kind(service: str, request: httpx.Request) -> str | None:
    path = request.url.path
    if service == "openai" and path.endswith("/chat/completions"):
        return "chat"
    if service == "openai" and path.endswith("/embeddings"):
        return "embed"
    if service == "ucc" and path.endswith("/queries/select"):
        body = json.loads(request.content or b"{}")
        return "ucc" if body.get("columns") else None
    return None


class MeteredTransport(httpx.AsyncBaseTransport):
    def __init__(self, meter: Meter, service: str, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self.meter = meter
        self.service = service
        self.inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        kind = _kind(self.service, request)
        if kind is None:
            return await self.inner.handle_async_request(request)
        key = hashlib.sha256(
            b"\n".join([request.method.encode(), request.url.path.encode(), request.content])
        ).hexdigest()
        hit = self.meter._cache.get(key)
        if hit is not None:
            if self.meter.replay_latency:
                await asyncio.sleep(hit["latency_s"])
            self.meter.all.add(hit)
            return httpx.Response(200, json=hit["body"], request=request)
        self.meter._check(kind)
        t0 = time.perf_counter()
        resp = await self.inner.handle_async_request(request)
        content = await resp.aread()
        e: dict = {"key": key, "kind": kind, "status": resp.status_code, "latency_s": time.perf_counter() - t0}
        if resp.status_code == 200:
            body = json.loads(content)
            e["body"] = body
            if kind != "ucc":
                e["usage"] = body.get("usage") or {}
                e["usd"] = openai_cost(json.loads(request.content).get("model", ""), e["usage"])
        self.meter._record_live(e)
        headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
        }
        return httpx.Response(resp.status_code, headers=headers, content=content, request=request)

    async def aclose(self) -> None:
        await self.inner.aclose()
