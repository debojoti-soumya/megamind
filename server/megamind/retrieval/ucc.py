"""Universal Constructor "UCC Constructor-1" backend (hackathon sponsor), selected with `RETRIEVER_BACKEND=ucc`.

UCC is not a vector index: every uploaded document is read once by a language model and kept, and a *Facet*
question ("does this document show X?") is then answered by that model for every document of a view, returning
p(yes). So each memory token becomes one small text document (caption + OCR + entities + local time) carrying
`token_id / device_id / ts / kind / episode_id` as metadata; a question becomes one `select` whose `where` holds the
time window / device as free metadata filters and whose columns are relevance facets; tokens are ranked by p(yes).

- `UCCClient`: thin async REST client (`/v1/corpora`, `/v1/corpora/{name}/documents`, `/v1/queries/select`, JSON
  answers) over httpx, so neither the private `ucc` SDK nor pyarrow is a dependency and tests use `MockTransport`.
- `UCCIndexer`: periodic sync job (store -> corpus); pushes enriched tokens it has not pushed yet. New documents
  become queryable a few seconds after upload (until then a select just doesn't see them).
- `UCCRetriever`: the `Retriever`. The planner decides routing (`QueryPlan.route`): questions with no content words
  (pure-time / anchor-only / recency: "what was I doing at 11", "what did I do after lunch", "what did I just
  see?") are `route="hybrid"` and never call UCC, since there is nothing for a facet to judge. Everything else goes
  to UCC; a cool-down after an error, timeouts, and an empty window/corpus go to the `HybridRetriever` fallback, so
  an outage never breaks `recall_memory`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from megamind.models import MemoryToken
from megamind.retrieval.base import RetrievalResult, RetrievedToken, RetrieveMode, Retriever
from megamind.retrieval.hybrid import HybridRetriever
from megamind.retrieval.planner import QueryPlan, plan_query
from megamind.store.base import MemoryStore

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.universalconstructor.co/v1"
MAX_BATCH = 1000  # documents per POST /corpora/{name}/documents
MAX_QUESTION_BYTES = 1200
STEERING_PROMPT = (
    "Each document is one first-person snapshot captured by a wearer's AR headset (or a summary of a short episode "
    "of them): when it was taken, a caption of the scene, any text that was legible in it, and the objects seen. "
    "Answer about what the snapshot itself shows."
)

Json = dict[str, Any]


class UCCError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"UCC {status}: {detail}")
        self.status = status
        self.detail = detail


# --------------------------------------------------------------------------- predicates (engine JSON)


def facet(question: str) -> Json:
    q = question.encode()[:MAX_QUESTION_BYTES].decode(errors="ignore")
    return {"type": "facet", "question": q, "threshold": 0.5, "operator": "ge", "reasoning": 0}


def meta(field: str, operator: str, value: Any = None) -> Json:
    return {"type": "metadata", "field": field, "operator": operator, "value": value}


def all_of(*preds: Json | None) -> Json | None:
    kept = [p for p in preds if p is not None]
    if not kept:
        return None
    return kept[0] if len(kept) == 1 else {"type": "and", "children": kept}


# --------------------------------------------------------------------------- client


class UCCClient:
    """Async REST client for one corpus. `transport` lets tests inject `httpx.MockTransport`."""

    def __init__(
        self,
        api_key: str,
        corpus: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 10.0,
        keep_alive_s: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.corpus = corpus
        self.keep_alive_s = keep_alive_s
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "X-UCC-Corpus": corpus},
            timeout=timeout_s,
            transport=transport,
        )
        self._corpus_id: str | None = None

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _call(self, method: str, path: str, *, json: object = None, headers: dict | None = None) -> Json:
        r = await self.http.request(method, path, json=json, headers=headers)
        if r.is_error:
            try:
                body = r.json()
                detail = body.get("detail") or body.get("error") or r.text
            except ValueError:
                detail = r.text
            raise UCCError(r.status_code, str(detail)[:300])
        return r.json()

    async def corpus_info(self) -> Json | None:
        try:
            return await self._call("GET", f"/corpora/{self.corpus}")
        except UCCError as e:
            if e.status == 404:
                return None
            raise

    async def ensure_corpus(self, *, wait_s: float = 60.0) -> str:
        """Create the corpus if needed and wait until it is ready; returns (and caches) its id."""
        if self._corpus_id:
            return self._corpus_id
        info = await self.corpus_info()
        if info is None:
            body: Json = {"name": self.corpus, "steering_prompt": STEERING_PROMPT, "max_document_tokens": 4096}
            if self.keep_alive_s:
                body["keep_alive"] = self.keep_alive_s
            try:
                info = await self._call("POST", "/corpora", json=body)
            except UCCError as e:
                if e.status != 409:  # corpus_exists: created concurrently
                    raise
                info = await self.corpus_info() or {}
        deadline = time.monotonic() + wait_s
        while info.get("status") == "provisioning" and time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            info = await self.corpus_info() or {}
        if info.get("status") != "ready":
            raise UCCError(409, f"corpus {self.corpus} is {info.get('status')}")
        self._corpus_id = info["id"]
        return self._corpus_id

    async def add_documents(self, docs: Sequence[Json]) -> list[Json]:
        """POST up to 1000 `{"name", "text", "metadata"}` documents (all-or-nothing, idempotent per batch)."""
        out: list[Json] = []
        for i in range(0, len(docs), MAX_BATCH):
            batch = list(docs[i : i + MAX_BATCH])
            # scoped to the corpus and the content: the same token ids in another corpus are a different request
            body = "\n".join([self.corpus, *(f"{d['name']}\t{d.get('text', '')}" for d in batch)])
            key = hashlib.sha256(body.encode()).hexdigest()[:64]
            ans = await self._call(
                "POST",
                f"/corpora/{self.corpus}/documents",
                json={"documents": batch},
                headers={"Idempotency-Key": f"megamind-{key}"},
            )
            out.extend(ans.get("data", []))
        return out

    async def select(
        self, *, where: Json | None = None, fields: Sequence[str] = (), columns: dict[str, Json] | None = None
    ) -> list[Json]:
        """Run a select (JSON answer); returns the rows. Raises on an unfinished answer or failed documents."""
        body = {
            "corpus_id": await self.ensure_corpus(),
            "where": where,
            "documents": "all",
            "fields": list(fields),
            "columns": columns or {},
        }
        ans = await self._call("POST", "/queries/select", json=body, headers={"Accept": "application/json"})
        if ans.get("status") not in (None, "finished") or ans.get("failures"):
            raise UCCError(200, f"select {ans.get('status')}: {len(ans.get('failures') or [])} failures")
        return ans.get("rows", [])

    async def token_ids(self) -> set[str]:
        """Every `token_id` already in the corpus (a fields-only select runs no model)."""
        if await self.corpus_info() is None:
            return set()
        return {str(r["token_id"]) for r in await self.select(fields=["token_id"]) if r.get("token_id")}


# --------------------------------------------------------------------------- indexing


def token_document(t: MemoryToken, tz: ZoneInfo | None = None) -> Json:
    when = datetime.fromtimestamp(t.ts, tz or UTC)
    lines = [
        f"{'Episode summary' if t.kind == 'episode' else 'Snapshot'} taken {when:%A %Y-%m-%d %H:%M} ({when.tzname()}).",
        f"Scene: {t.caption.strip()}",
        f"Text visible: {t.ocr_text.strip() or 'none'}",
    ]
    if t.entities:
        lines.append(f"Objects: {', '.join(t.entities)}")
    md: Json = {"token_id": t.id, "device_id": t.device_id, "ts": float(t.ts), "kind": t.kind}
    if t.episode_id:
        md["episode_id"] = t.episode_id
    return {"name": f"{t.id}.txt", "text": "\n".join(lines), "metadata": md}


class UCCIndexer:
    """Store -> corpus sync. `sync_once()` pushes enriched tokens not yet in the corpus; `start()` runs it every
    `interval_s`. Re-scans `lookback_s` before the newest pushed token so late enrichments are not missed.
    Episode summaries are stamped with the episode's *start* (often behind that cursor) when the consolidator writes
    them, so whenever the store has grown, the `kind="episode"` tokens (a few per hour) are diffed against `pushed`.
    While UCC is unreachable the wait doubles per consecutive failure, capped at `max_backoff_s`."""

    def __init__(
        self,
        client: UCCClient,
        store: MemoryStore,
        *,
        tz: str | ZoneInfo | None = None,
        interval_s: float = 30.0,
        lookback_s: float = 900.0,
        scan_limit: int = 50_000,
        max_backoff_s: float = 300.0,
    ) -> None:
        self.client = client
        self.store = store
        self.tz = ZoneInfo(tz) if isinstance(tz, str) else tz
        self.interval_s = interval_s
        self.lookback_s = lookback_s
        self.scan_limit = scan_limit
        self.max_backoff_s = max_backoff_s
        self.failures = 0
        self.pushed: set[str] | None = None
        self.cursor: float | None = None
        self._count: int | None = None
        self._task: asyncio.Task | None = None

    async def sync_once(self) -> int:
        if self.pushed is None:
            self.pushed = await self.client.token_ids()
        since = None if self.cursor is None else self.cursor - self.lookback_s
        found = await self.store.range(since, None, self.scan_limit)
        count = await self.store.count()
        if since is not None and count != self._count:
            found += await self.store.range(None, since, self.scan_limit, kinds=("episode",))
        new = {t.id: t for t in found if t.caption.strip() and t.id not in self.pushed}
        toks = list(new.values())
        if toks:
            await self.client.ensure_corpus()
            await self.client.add_documents([token_document(t, self.tz) for t in toks])
            self.pushed.update(t.id for t in toks)
            self.cursor = max([self.cursor or 0.0, *(t.ts for t in toks)])
            log.info("ucc sync: pushed %d tokens to corpus %s", len(toks), self.client.corpus)
        self._count = count
        return len(toks)

    async def _loop(self) -> None:
        while True:
            try:
                await self.sync_once()
                self.failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep syncing through transient API errors, backing off
                self.failures += 1
                log.warning("ucc sync failed (%d in a row): %s", self.failures, e)
            await asyncio.sleep(self.next_delay())

    def next_delay(self) -> float:
        if not self.failures:
            return self.interval_s
        return min(self.interval_s * 2 ** min(self.failures, 30), max(self.max_backoff_s, self.interval_s))

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


# --------------------------------------------------------------------------- retrieval


@dataclass
class UCCConfig:
    timeout_s: float = 2.0  # whole UCC leg of one retrieve(); the hybrid runs in parallel and answers past it
    cooldown_s: float = 60.0  # after an error, skip UCC for this long
    min_p: float = 0.2  # facet p(yes) below this is not a hit
    candidates: int = 0  # >0: rerank only the hybrid's top-N (cheaper); 0: facet over the whole window


def relevance_questions(plan: QueryPlan) -> dict[str, Json]:
    q = plan.query.strip().rstrip("?")
    cols = {"rel_q": facet(f'The wearer asks: "{q}". Does this snapshot show or mention what they are asking about?')}
    s = plan.search_text.strip()
    if s and s.lower() != q.lower():
        cols["rel_s"] = facet(f"Does this snapshot show or mention {s}?")
    return cols


class UCCRetriever(Retriever):
    def __init__(
        self,
        client: UCCClient,
        fallback: HybridRetriever,
        *,
        tz: str | ZoneInfo | None = None,
        config: UCCConfig | None = None,
    ) -> None:
        self.client = client
        self.fallback = fallback
        self.tz = tz
        self.cfg = config or UCCConfig()
        self._down_until = 0.0
        self.stats = {"ucc": 0, "fallback": 0, "errors": 0}

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 12,
        mode: RetrieveMode = "hybrid",
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        now: datetime | None = None,
    ) -> RetrievalResult:
        now = now or datetime.now(UTC)
        args = dict(k=k, mode=mode, time_from=time_from, time_to=time_to, device_id=device_id, now=now)
        if time.monotonic() < self._down_until:
            self.stats["fallback"] += 1
            return await self.fallback.retrieve(query, **args)
        # race: the hybrid runs alongside UCC, so a UCC miss/error/timeout answers at once instead of starting it then;
        # in candidates mode the same run is also UCC's rerank pool (sized to cover both)
        pool_k = max(k, self.cfg.candidates)
        hybrid = asyncio.create_task(self.fallback.retrieve(query, **{**args, "k": pool_k}))
        hybrid.add_done_callback(lambda t: t.cancelled() or t.exception())  # never "exception was never retrieved"
        try:
            res = await asyncio.wait_for(self._ucc(query, hybrid=hybrid, **args), self.cfg.timeout_s)
            if res is not None:
                self.stats["ucc"] += 1
                hybrid.cancel()
                return res
        except Exception as e:  # timeouts, HTTP errors, malformed answers: never fail the tool call
            self.stats["errors"] += 1
            self._down_until = time.monotonic() + self.cfg.cooldown_s
            log.warning("ucc retrieve failed (%s: %s); hybrid fallback", type(e).__name__, e)
        self.stats["fallback"] += 1
        res = await hybrid
        if pool_k > k:  # the pool ran wider than asked: keep the k best, still chronological
            keep = {r.token.id for r in res.ranked[:k]}
            res = replace(res, tokens=[r for r in res.tokens if r.token.id in keep])
        return res

    def invalidate_caches(self, device_id: str | None = None) -> None:
        self.fallback.invalidate_caches(device_id)

    async def _ucc(
        self,
        query: str,
        *,
        k: int,
        mode: RetrieveMode,
        time_from: float | None,
        time_to: float | None,
        device_id: str | None,
        now: datetime,
        hybrid: asyncio.Future[RetrievalResult] | None = None,
    ) -> RetrievalResult | None:
        """None = defer to the hybrid (planner routed to hybrid, or nothing indexed in the window)."""
        k = max(1, k)
        plan = plan_query(query, now=now, tz=self.tz, time_from=time_from, time_to=time_to)
        if plan.route == "hybrid" or not (plan.search_text or plan.query.strip()):
            return None
        anchor_ts = None
        if plan.anchor_event and not plan.has_time:
            anchor_ts = await self.fallback._resolve_anchor(plan, device_id)
            if anchor_ts is not None:
                plan.time_from, plan.time_to = self.fallback._anchor_window(plan.query, anchor_ts, now.timestamp())

        rows = await self._rank(plan, plan.time_from, plan.time_to, device_id, now, pool=hybrid)
        if rows is None:
            return None
        widened = False
        if not rows and plan.has_time and not plan.explicit_time and anchor_ts is None:
            rows = await self._rank(plan, None, None, device_id, now) or []
            widened = bool(rows)
        if not rows:
            return RetrievalResult(plan=plan, tokens=[], anchor_ts=anchor_ts)

        toks = {t.id: t for t in await self.fallback.adapter.get_many([i for i, _ in rows])}
        ranked: list[RetrievedToken] = []
        for i, p in rows:
            t = toks.get(i)
            if t is None or any(self.fallback._same_frame(t, r.token) for r in ranked):
                continue
            ranked.append(RetrievedToken(token=t, score=p, reasons=["ucc"]))
        lo, hi = (None, None) if widened else (plan.time_from, plan.time_to)
        picked = await self.fallback._expand(ranked, k, lo, hi)
        picked.sort(key=lambda r: r.token.ts)
        return RetrievalResult(plan=plan, tokens=picked, widened=widened, anchor_ts=anchor_ts)

    async def _rank(
        self,
        plan: QueryPlan,
        lo: float | None,
        hi: float | None,
        device_id: str | None,
        now: datetime,
        pool: asyncio.Future[RetrievalResult] | None = None,
    ) -> list[tuple[str, float]] | None:
        """(token_id, p) best first with p >= min_p; None if the corpus has no documents in the view.
        `pool`: the hybrid run already racing over this view, reused as the candidates pool."""
        where = all_of(
            meta("ts", "ge", float(lo)) if lo is not None else None,
            meta("ts", "le", float(hi)) if hi is not None else None,
            meta("device_id", "eq", device_id) if device_id else None,
        )
        if self.cfg.candidates > 0:
            # shield: a UCC timeout cancels this coroutine, but the racing hybrid must survive as the fallback
            res = await asyncio.shield(pool) if pool is not None else None
            if res is None or res.widened:  # no race, or the hybrid's window differs from this view
                res = await self.fallback.retrieve(
                    plan.query, k=self.cfg.candidates, time_from=lo, time_to=hi, device_id=device_id, now=now
                )
            ids = [r.token.id for r in res.ranked[: self.cfg.candidates]]
            if not ids:
                return []
            where = all_of(where, meta("token_id", "in", ids))
        cols = relevance_questions(plan)
        rows = await self.client.select(where=where, fields=["token_id", "ts"], columns=cols)
        if not rows:
            return None
        scored = []
        for r in rows:
            ps = [float(r[c]) for c in cols if r.get(c) is not None]
            if ps:
                scored.append((str(r["token_id"]), max(ps), float(r.get("ts") or 0.0)))
        scored.sort(key=lambda x: (-x[1], -x[2]))  # ties: most recent sighting first
        return [(i, p) for i, p, _ in scored if p >= self.cfg.min_p]


def build_ucc(settings, store: MemoryStore, fallback: HybridRetriever) -> tuple[UCCRetriever, UCCIndexer]:
    client = UCCClient(
        settings.ucc_api_key,
        settings.ucc_corpus,
        base_url=settings.ucc_base_url,
        timeout_s=max(settings.ucc_timeout_s, 10.0),
        keep_alive_s=settings.ucc_keep_alive_s,
    )
    cfg = replace(UCCConfig(), timeout_s=settings.ucc_timeout_s, candidates=settings.ucc_candidates)
    tz = getattr(settings, "wearer_tz", None)
    retr = UCCRetriever(client, fallback, tz=tz, config=cfg)
    indexer = UCCIndexer(client, store, tz=tz, interval_s=settings.ucc_sync_interval_s)
    return retr, indexer
