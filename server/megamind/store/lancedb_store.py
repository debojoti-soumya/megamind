"""LanceDB-backed MemoryStore: persistent, embedded, zero-infra.

Layout (see store/README.md): one table `memory_tokens` under `settings.lancedb_path`.
Columns mirror `MemoryToken` (docs/api.md) plus two derived columns:
  - `search_text`: caption + ocr_text + entities, the BM25 (full-text) index target
  - `kind`: "frame" for captured keyframes, "episode" for consolidation summaries
Vector columns are nullable fixed-size lists; their dimension is taken from the first token
that carries a vector (the mock embedder is 256-d, OpenAI/SigLIP 1024/768-d), or from the
`text_dim` / `image_dim` constructor arguments.

All LanceDB calls are synchronous and run in a worker thread (`asyncio.to_thread`); writes
are serialised with a lock. Rows written after an index was built are still searched
(LanceDB scans unindexed rows), and `optimize()` folds them into the indices periodically.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import lancedb
import pyarrow as pa
from lancedb.index import FTS

from megamind.models import MemoryHit, MemoryToken, Pose, SearchMode
from megamind.store.base import MemoryStore
from megamind.store.rrf import rrf

log = logging.getLogger(__name__)

TABLE_NAME = "memory_tokens"
DEFAULT_TEXT_DIM = 1024
DEFAULT_IMAGE_DIM = 768
VectorField = Literal["text_vec", "image_vec"]

_WORD = re.compile(r"[\w]+", re.UNICODE)
_SCALAR_COLUMNS = [
    "id",
    "device_id",
    "ts",
    "kind",
    "caption",
    "ocr_text",
    "entities",
    "pose",
    "thumb_path",
    "frame_hash",
    "episode_id",
]
_INDEXED_COLUMNS = [*_SCALAR_COLUMNS, "text_vec", "image_vec"]


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _search_text(caption: str, ocr_text: str, entities: Iterable[str]) -> str:
    return "\n".join(part for part in (caption, ocr_text, " ".join(entities)) if part)


class LanceDBStore(MemoryStore):
    def __init__(
        self,
        path: Path | str,
        *,
        text_dim: int | None = None,
        image_dim: int | None = None,
        optimize_every: int = 256,
        vector_index_min_rows: int = 4096,
        fts_index_min_rows: int = 1,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._text_dim = text_dim
        self._image_dim = image_dim
        self._optimize_every = optimize_every
        self._vector_index_min_rows = vector_index_min_rows
        self._fts_index_min_rows = fts_index_min_rows
        self._lock = threading.RLock()
        self._db = lancedb.connect(str(self.path))
        self._table: lancedb.table.Table | None = None
        self._writes_since_optimize = 0
        if TABLE_NAME in self._db.list_tables():
            self._table = self._db.open_table(TABLE_NAME)
            schema = self._table.schema
            self._text_dim = schema.field("text_vec").type.list_size
            self._image_dim = schema.field("image_vec").type.list_size

    # ------------------------------------------------------------------ schema

    def _schema(self) -> pa.Schema:
        pose = pa.struct(
            [
                pa.field("x", pa.float64()),
                pa.field("y", pa.float64()),
                pa.field("z", pa.float64()),
                pa.field("yaw", pa.float64()),
            ]
        )
        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("device_id", pa.string(), nullable=False),
                pa.field("ts", pa.float64(), nullable=False),
                pa.field("kind", pa.string(), nullable=False),
                pa.field("caption", pa.string(), nullable=False),
                pa.field("ocr_text", pa.string(), nullable=False),
                pa.field("entities", pa.list_(pa.string()), nullable=False),
                pa.field("search_text", pa.string(), nullable=False),
                pa.field("pose", pose, nullable=True),
                pa.field("thumb_path", pa.string(), nullable=False),
                pa.field("frame_hash", pa.string(), nullable=False),
                pa.field("episode_id", pa.string(), nullable=True),
                pa.field("text_vec", pa.list_(pa.float32(), self._text_dim or DEFAULT_TEXT_DIM), nullable=True),
                pa.field("image_vec", pa.list_(pa.float32(), self._image_dim or DEFAULT_IMAGE_DIM), nullable=True),
            ]
        )

    def _ensure_table(self, first: MemoryToken) -> lancedb.table.Table:
        if self._table is not None:
            return self._table
        if self._text_dim is None and first.text_vec:
            self._text_dim = len(first.text_vec)
        if self._image_dim is None and first.image_vec:
            self._image_dim = len(first.image_vec)
        self._text_dim = self._text_dim or DEFAULT_TEXT_DIM
        self._image_dim = self._image_dim or DEFAULT_IMAGE_DIM
        self._table = self._db.create_table(TABLE_NAME, schema=self._schema(), exist_ok=True)
        return self._table

    def _check_dims(self, token: MemoryToken) -> None:
        dims = (("text_vec", token.text_vec, self._text_dim), ("image_vec", token.image_vec, self._image_dim))
        for field, vec, dim in dims:
            if vec is not None and len(vec) != dim:
                raise ValueError(
                    f"{field} has {len(vec)} dims but the LanceDB table at {self.path} was created with {dim}; "
                    "purge the store (DELETE /v1/memories or delete the directory) after changing embedders"
                )

    @staticmethod
    def _row(token: MemoryToken) -> dict[str, Any]:
        return {
            "id": token.id,
            "device_id": token.device_id,
            "ts": float(token.ts),
            "kind": getattr(token, "kind", None) or "frame",
            "caption": token.caption,
            "ocr_text": token.ocr_text,
            "entities": list(token.entities),
            "search_text": _search_text(token.caption, token.ocr_text, token.entities),
            "pose": token.pose.model_dump() if token.pose else None,
            "thumb_path": token.thumb_path,
            "frame_hash": token.frame_hash,
            "episode_id": token.episode_id,
            "text_vec": token.text_vec,
            "image_vec": token.image_vec,
        }

    @staticmethod
    def _token(row: dict[str, Any]) -> MemoryToken:
        data = {
            "id": row["id"],
            "device_id": row["device_id"],
            "ts": row["ts"],
            "caption": row.get("caption", ""),
            "ocr_text": row.get("ocr_text", ""),
            "entities": list(row.get("entities") or []),
            "pose": Pose(**row["pose"]) if row.get("pose") else None,
            "text_vec": list(row["text_vec"]) if row.get("text_vec") is not None else None,
            "image_vec": list(row["image_vec"]) if row.get("image_vec") is not None else None,
            "thumb_path": row["thumb_path"],
            "frame_hash": row["frame_hash"],
            "episode_id": row.get("episode_id"),
        }
        if "kind" in MemoryToken.model_fields:
            data["kind"] = row.get("kind") or "frame"
        return MemoryToken(**data)

    # ------------------------------------------------------------------ writes

    def _add_sync(self, tokens: Sequence[MemoryToken]) -> None:
        if not tokens:
            return
        with self._lock:
            table = self._ensure_table(tokens[0])
            for t in tokens:
                self._check_dims(t)
            rows = [self._row(t) for t in tokens]
            (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(pa.Table.from_pylist(rows, schema=table.schema), on_bad_vectors="null")
            )
            self._writes_since_optimize += len(tokens)
            self._maintain_indices(table)

    def _maintain_indices(self, table: lancedb.table.Table) -> None:
        n = table.count_rows()
        existing = {idx.columns[0]: idx for idx in table.list_indices()}
        if "search_text" not in existing and n >= self._fts_index_min_rows:
            table.create_index("search_text", config=FTS(), replace=True)
        for col in ("text_vec", "image_vec"):
            if col not in existing and n >= self._vector_index_min_rows:
                if table.count_rows(f"{col} IS NOT NULL") >= self._vector_index_min_rows:
                    self._create_vector_index(table, col, n)
        if self._writes_since_optimize >= self._optimize_every:
            self._writes_since_optimize = 0
            table.optimize()

    @staticmethod
    def _create_vector_index(table: lancedb.table.Table, col: str, n: int) -> None:
        partitions = max(1, min(256, int(n**0.5)))
        log.info("creating IVF_FLAT index on %s (%d rows, %d partitions)", col, n, partitions)
        table.create_index(metric="cosine", vector_column_name=col, index_type="IVF_FLAT", num_partitions=partitions)

    async def add(self, token: MemoryToken) -> None:
        await asyncio.to_thread(self._add_sync, [token])

    async def add_many(self, tokens: Sequence[MemoryToken]) -> None:
        await asyncio.to_thread(self._add_sync, list(tokens))

    async def optimize(self) -> None:
        if self._table is not None:
            await asyncio.to_thread(self._optimize_sync)

    def _optimize_sync(self) -> None:
        with self._lock:
            assert self._table is not None
            self._writes_since_optimize = 0
            self._table.optimize()

    # ------------------------------------------------------------------ reads

    @staticmethod
    def _where(
        *,
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
        enriched_only: bool = False,
        extra: Sequence[str] = (),
    ) -> str | None:
        parts = list(extra)
        if device_id is not None:
            parts.append(f"device_id = {_sql_str(device_id)}")
        if time_from is not None:
            parts.append(f"ts >= {float(time_from)!r}")
        if time_to is not None:
            parts.append(f"ts <= {float(time_to)!r}")
        if kinds:
            parts.append(f"kind IN ({', '.join(_sql_str(k) for k in kinds)})")
        if enriched_only:
            parts.append("caption != ''")
        return " AND ".join(parts) if parts else None

    def _scan(self, where: str | None, columns: Sequence[str], limit: int | None = None) -> list[dict[str, Any]]:
        if self._table is None:
            return []
        q = self._table.search().select(list(columns)).limit(limit)
        if where:
            q = q.where(where)
        return q.to_list()

    async def get(self, token_id: str) -> MemoryToken | None:
        rows = await asyncio.to_thread(self._scan, f"id = {_sql_str(token_id)}", _INDEXED_COLUMNS, 1)
        return self._token(rows[0]) if rows else None

    async def get_many(self, ids: Sequence[str]) -> list[MemoryToken]:
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        where = f"id IN ({', '.join(_sql_str(i) for i in ids)})"
        rows = await asyncio.to_thread(self._scan, where, _INDEXED_COLUMNS, None)
        by_id = {r["id"]: self._token(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    async def count(self, device_id: str | None = None) -> int:
        if self._table is None:
            return 0
        where = self._where(device_id=device_id)
        return await asyncio.to_thread(self._table.count_rows, where)

    async def recent_hashes(self, device_id: str, since_ts: float) -> list[tuple[float, str]]:
        where = self._where(device_id=device_id, time_from=since_ts, kinds=("frame",))
        rows = await asyncio.to_thread(self._scan, where, ["ts", "frame_hash"], None)
        return [(r["ts"], r["frame_hash"]) for r in rows]

    async def range(
        self,
        time_from: float | None,
        time_to: float | None,
        limit: int = 100,
        *,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[MemoryToken]:
        where = self._where(time_from=time_from, time_to=time_to, device_id=device_id, kinds=kinds)
        rows = await asyncio.to_thread(self._scan, where, ["id", "ts"], None)
        rows.sort(key=lambda r: (r["ts"], r["id"]))
        return await self.get_many([r["id"] for r in rows[:limit]])

    async def neighbors(self, token_id: str, *, before: int = 2, after: int = 2) -> list[MemoryToken]:
        anchor = await self.get(token_id)
        if anchor is None:
            return []
        kinds = (getattr(anchor, "kind", None) or "frame",)
        where = self._where(device_id=anchor.device_id, kinds=kinds)
        rows = await asyncio.to_thread(self._scan, where, ["id", "ts"], None)
        rows.sort(key=lambda r: (r["ts"], r["id"]))
        idx = next((i for i, r in enumerate(rows) if r["id"] == token_id), None)
        if idx is None:
            return [anchor]
        window = rows[max(0, idx - before) : idx + after + 1]
        return await self.get_many([r["id"] for r in window])

    def _vector_search_sync(
        self, vec: list[float], k: int, field: str, where: str | None
    ) -> list[tuple[MemoryToken, float]]:
        if self._table is None or k <= 0:
            return []
        q = self._table.search(vec, vector_column_name=field).metric("cosine").limit(k).select(_INDEXED_COLUMNS)
        if where:
            q = q.where(where, prefilter=True)
        return [(self._token(r), 1.0 - float(r["_distance"])) for r in q.to_list()]

    async def vector_search(
        self,
        vec: list[float],
        k: int,
        *,
        field: VectorField = "text_vec",
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[tuple[MemoryToken, float]]:
        if field not in ("text_vec", "image_vec"):
            raise ValueError(f"unknown vector field {field!r}")
        where = self._where(
            time_from=time_from,
            time_to=time_to,
            device_id=device_id,
            kinds=kinds,
            enriched_only=True,
            extra=(f"{field} IS NOT NULL",),
        )
        return await asyncio.to_thread(self._vector_search_sync, list(vec), k, field, where)

    def _keyword_search_sync(self, query: str, k: int, where: str | None) -> list[tuple[MemoryToken, float]]:
        if self._table is None or k <= 0:
            return []
        q = self._table.search(query, query_type="fts", fts_columns="search_text").limit(k).select(_INDEXED_COLUMNS)
        if where:
            q = q.where(where, prefilter=True)
        return [(self._token(r), float(r["_score"])) for r in q.to_list()]

    async def keyword_search(
        self,
        query: str,
        k: int,
        *,
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[tuple[MemoryToken, float]]:
        terms = _WORD.findall(query.lower())
        if not terms:
            return []
        where = self._where(time_from=time_from, time_to=time_to, device_id=device_id, kinds=kinds, enriched_only=True)
        return await asyncio.to_thread(self._keyword_search_sync, " ".join(terms), k, where)

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
        if self._table is None:
            return []
        filters = {"time_from": time_from, "time_to": time_to, "device_id": device_id}
        n = max(k * 4, 48)
        rankings: list[list[tuple[MemoryToken, float]]] = []
        if mode in ("semantic", "hybrid"):
            if query_text_vec is not None and len(query_text_vec) == self._text_dim:
                rankings.append(await self.vector_search(query_text_vec, n, field="text_vec", **filters))
            rankings.append(await self.keyword_search(query_text, n, **filters))
        if mode in ("visual", "hybrid") and query_image_vec is not None and len(query_image_vec) == self._image_dim:
            rankings.append(await self.vector_search(query_image_vec, n, field="image_vec", **filters))

        tokens: dict[str, MemoryToken] = {}
        for ranking in rankings:
            for t, _ in ranking:
                tokens.setdefault(t.id, t)
        fused = rrf([[t.id for t, _ in ranking] for ranking in rankings])
        ordered = sorted(fused.items(), key=lambda kv: (kv[1], tokens[kv[0]].ts), reverse=True)
        return [MemoryHit.from_token(tokens[i], s) for i, s in ordered[:k]]

    # ------------------------------------------------------------------ deletes

    def _purge_sync(self, where: str | None, delete_thumbs: bool) -> int:
        if self._table is None:
            return 0
        with self._lock:
            rows = self._scan(where, ["id", "thumb_path"], None)
            if not rows:
                return 0
            if delete_thumbs:
                for r in rows:
                    p = Path(r["thumb_path"])
                    if p.is_file():
                        p.unlink(missing_ok=True)
            self._table.delete(where or "id IS NOT NULL")
            return len(rows)

    async def purge(
        self, device_id: str | None = None, *, since_ts: float | None = None, delete_thumbs: bool = True
    ) -> int:
        where = self._where(device_id=device_id, time_from=since_ts)
        return await asyncio.to_thread(self._purge_sync, where, delete_thumbs)
