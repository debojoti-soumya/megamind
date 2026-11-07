"""LanceDB-backed MemoryStore (docs/stream1-memory-engine.pdf §6).

One `tokens` table. Vectors are fixed-size float32 columns with `has_*` flags (missing vectors are
stored as zeros and excluded by prefilter). Native BM25 FTS over `search_text` (caption + OCR +
entities + transcript); unindexed rows are still searched, and `optimize()` folds them into the
index periodically. Flat vector search is exact and fast below ~100k rows per device.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pyarrow as pa

from megamind.models import MemoryHit, MemoryToken, Pose, SearchMode
from megamind.store.base import MemoryStore
from megamind.store.fusion import POOL, FusionWeights, diversify, tokenize, weighted_rrf

_TABLE = "tokens"
_OPTIMIZE_EVERY = 200


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


class LanceDBStore(MemoryStore):
    def __init__(self, path: Path, text_dim: int, image_dim: int | None, weights: FusionWeights | None = None) -> None:
        import lancedb

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.text_dim = text_dim
        self.image_dim = image_dim or 1
        self.weights = weights or FusionWeights()
        self._lock = threading.Lock()
        self._writes = 0
        self._db = lancedb.connect(str(self.path))
        if _TABLE in self._db.table_names():
            self._t = self._db.open_table(_TABLE)
        else:
            self._t = self._db.create_table(_TABLE, schema=self._schema())
            self._t.create_fts_index("search_text", use_tantivy=False)

    def _schema(self) -> pa.Schema:
        s = pa.string()
        return pa.schema(
            [
                ("id", s),
                ("device_id", s),
                ("ts", pa.float64()),
                ("caption", s),
                ("ocr_text", s),
                ("entities", pa.list_(s)),
                ("transcript", s),
                ("search_text", s),
                ("pose", s),
                ("text_vec", pa.list_(pa.float32(), self.text_dim)),
                ("has_text_vec", pa.bool_()),
                ("image_vec", pa.list_(pa.float32(), self.image_dim)),
                ("has_image_vec", pa.bool_()),
                ("thumb_path", s),
                ("frame_hash", s),
                ("episode_id", s),
                ("source", s),
                ("trigger", s),
                ("sharpness", pa.float32()),
                ("enriched", pa.bool_()),
                ("tier", s),
                ("chunk_id", s),
                ("chunk_offset", pa.int32()),
            ]
        )

    # -- row <-> token ---------------------------------------------------------------
    def _row(self, t: MemoryToken) -> dict:
        def vec(v: list[float] | None, dim: int) -> tuple[list[float], bool]:
            if v is None or len(v) != dim:
                return [0.0] * dim, False
            return v, True

        tv, has_tv = vec(t.text_vec, self.text_dim)
        iv, has_iv = vec(t.image_vec, self.image_dim)
        return {
            "id": t.id,
            "device_id": t.device_id,
            "ts": t.ts,
            "caption": t.caption,
            "ocr_text": t.ocr_text,
            "entities": t.entities,
            "transcript": t.transcript,
            "search_text": t.search_text(),
            "pose": t.pose.model_dump_json() if t.pose else None,
            "text_vec": tv,
            "has_text_vec": has_tv,
            "image_vec": iv,
            "has_image_vec": has_iv,
            "thumb_path": t.thumb_path,
            "frame_hash": t.frame_hash,
            "episode_id": t.episode_id,
            "source": t.source,
            "trigger": t.trigger,
            "sharpness": t.sharpness,
            "enriched": t.enriched,
            "tier": t.tier,
            "chunk_id": t.chunk_id,
            "chunk_offset": t.chunk_offset,
        }

    @staticmethod
    def _token(r: dict) -> MemoryToken:
        return MemoryToken(
            id=r["id"],
            device_id=r["device_id"],
            ts=r["ts"],
            caption=r["caption"] or "",
            ocr_text=r["ocr_text"] or "",
            entities=list(r["entities"] or []),
            transcript=r["transcript"] or "",
            pose=Pose(**json.loads(r["pose"])) if r["pose"] else None,
            text_vec=list(r["text_vec"]) if r["has_text_vec"] else None,
            image_vec=list(r["image_vec"]) if r["has_image_vec"] else None,
            thumb_path=r["thumb_path"],
            frame_hash=r["frame_hash"],
            episode_id=r["episode_id"],
            source=r["source"] or "quest",
            trigger=r["trigger"],
            sharpness=r["sharpness"] or 0.0,
            enriched=bool(r["enriched"]),
            tier=r["tier"] or "hot",
            chunk_id=r["chunk_id"],
            chunk_offset=r["chunk_offset"],
        )

    def _rows(self, where: str, limit: int | None = None) -> list[dict]:
        q = self._t.search().where(where)
        return q.limit(limit).to_list() if limit else q.limit(None).to_list()

    # -- MemoryStore -----------------------------------------------------------------
    def _add_sync(self, tokens: list[MemoryToken]) -> None:
        with self._lock:
            ids = ", ".join(_q(t.id) for t in tokens)
            self._t.delete(f"id IN ({ids})")
            self._t.add([self._row(t) for t in tokens])
            self._writes += len(tokens)
            if self._writes >= _OPTIMIZE_EVERY:
                self._writes = 0
                self._t.optimize()

    async def add(self, token: MemoryToken) -> None:
        await asyncio.to_thread(self._add_sync, [token])

    async def add_many(self, tokens: list[MemoryToken]) -> None:
        if tokens:
            await asyncio.to_thread(self._add_sync, tokens)

    async def get(self, token_id: str) -> MemoryToken | None:
        rows = await asyncio.to_thread(self._rows, f"id = {_q(token_id)}", 1)
        return self._token(rows[0]) if rows else None

    async def get_many(self, ids: list[str]) -> dict[str, MemoryToken]:
        if not ids:
            return {}
        rows = await asyncio.to_thread(self._rows, f"id IN ({', '.join(_q(i) for i in ids)})", len(ids))
        return {r["id"]: self._token(r) for r in rows}

    async def count(self, device_id: str | None = None) -> int:
        return await asyncio.to_thread(self._t.count_rows, f"device_id = {_q(device_id)}" if device_id else None)

    async def recent_hashes(self, device_id: str, since_ts: float) -> list[tuple[float, str]]:
        def run() -> list[tuple[float, str]]:
            rows = (
                self._t.search()
                .where(f"device_id = {_q(device_id)} AND ts >= {float(since_ts)}")
                .select(["ts", "frame_hash"])
                .limit(None)
                .to_list()
            )
            return [(r["ts"], r["frame_hash"]) for r in rows]

        return await asyncio.to_thread(run)

    def _search_sync(
        self,
        query_text_vec: list[float] | None,
        query_image_vec: list[float] | None,
        query_text: str,
        k: int,
        mode: SearchMode,
        time_from: float | None,
        time_to: float | None,
        device_id: str | None,
    ) -> list[MemoryHit]:
        conds = []
        if device_id:
            conds.append(f"device_id = {_q(device_id)}")
        if time_from is not None:
            conds.append(f"ts >= {float(time_from)}")
        if time_to is not None:
            conds.append(f"ts <= {float(time_to)}")
        base = " AND ".join(conds)

        def where(extra: str | None = None) -> str | None:
            parts = [p for p in (base, extra) if p]
            return " AND ".join(parts) or None

        def ids(q, w: str | None, score: str) -> list[str]:
            if w:
                q = q.where(w, prefilter=True)
            return [r["id"] for r in q.limit(POOL).select(["id", score]).to_list()]

        w = self.weights
        rankings: list[tuple[float, list[str]]] = []
        if mode in ("semantic", "hybrid"):
            terms = " ".join(tokenize(query_text))
            if terms:
                rankings.append((w.bm25, ids(self._t.search(terms, query_type="fts"), where(), "_score")))
            if query_text_vec is not None and len(query_text_vec) == self.text_dim:
                q = self._t.search(query_text_vec, vector_column_name="text_vec").metric("cosine")
                rankings.append((w.text, ids(q, where("has_text_vec = true"), "_distance")))
        if mode in ("visual", "hybrid") and query_image_vec is not None and len(query_image_vec) == self.image_dim:
            q = self._t.search(query_image_vec, vector_column_name="image_vec").metric("cosine")
            weight = w.image_visual_mode if mode == "visual" else w.image
            rankings.append((weight, ids(q, where("has_image_vec = true"), "_distance")))

        fused = weighted_rrf(rankings)
        if not fused:
            return []
        rows = self._rows(f"id IN ({', '.join(_q(i) for i in fused)})", len(fused))
        tokens = {r["id"]: self._token(r) for r in rows}
        ordered = sorted(
            ((i, s) for i, s in fused.items() if i in tokens), key=lambda kv: (kv[1], tokens[kv[0]].ts), reverse=True
        )
        picked = diversify(ordered, {i: tokens[i].episode_id for i in tokens}, k, w.per_episode_cap)
        return [MemoryHit.from_token(tokens[i], s) for i, s in picked]

    async def search(
        self,
        query_text_vec: list[float] | None,
        query_image_vec: list[float] | None,
        query_text: str,
        *,
        k: int = 12,
        mode: SearchMode = "hybrid",
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
    ) -> list[MemoryHit]:
        return await asyncio.to_thread(
            self._search_sync, query_text_vec, query_image_vec, query_text, k, mode, time_from, time_to, device_id
        )

    async def purge(self, device_id: str | None = None) -> int:
        def run() -> int:
            with self._lock:
                w = f"device_id = {_q(device_id)}" if device_id else "true"
                n = self._t.count_rows(None if device_id is None else w)
                self._t.delete(w)
                return n

        return await asyncio.to_thread(run)
