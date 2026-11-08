"""LanceDB-backed MemoryStore: persistent, embedded, zero-infra.

Layout (see store/README.md): one table `memory_tokens` under `settings.lancedb_path`.
Columns mirror `MemoryToken` (docs/api.md) plus two derived columns:
  - `search_text`: caption + ocr_text + entities, the BM25 (full-text) index target
  - `kind`: "frame" for captured keyframes, "episode" for consolidation summaries
Vector columns are nullable fixed-size lists; their dimension is taken from the first token
that carries a vector (the mock embedder is 256-d, OpenAI/SigLIP 1024/768-d), or from the
`text_dim` / `image_dim` constructor arguments. New tables store vectors as float16
(`vector_dtype`, LANCEDB_VECTOR_DTYPE): half the bytes (1024+768 dims: 7.2 -> 3.6 KB/token) with the
same nearest neighbours (ported from PR #54). An existing table keeps the dtype it was created with.

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
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from lancedb.index import FTS

from megamind.models import DeviceInfo, MemoryHit, MemoryToken, Pose, SearchMode
from megamind.store.base import MemoryStore, Scored, VectorField, device_infos
from megamind.store.episodes import EpisodeTracker
from megamind.store.rrf import MIN_SCORE, calibrated, fuse_hits

log = logging.getLogger(__name__)

TABLE_NAME = "memory_tokens"
DEFAULT_TEXT_DIM = 1024
DEFAULT_IMAGE_DIM = 768
VECTOR_DTYPES = {"float16": pa.float16(), "float32": pa.float32()}

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
    "tier",
    "chunk_id",
    "chunk_offset",
]
# media-tier columns added after the first release: (name, SQL fill for existing rows)
_MIGRATIONS = (
    ("tier", "CAST('hot' AS STRING)"),
    ("chunk_id", "CAST(NULL AS STRING)"),
    ("chunk_offset", "CAST(NULL AS BIGINT)"),
)
_INDEXED_COLUMNS = [*_SCALAR_COLUMNS, "text_vec", "image_vec"]
_TS_ORDER = [("ts", "ascending"), ("id", "ascending")]


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
        episodes: EpisodeTracker | None = None,
        vector_dtype: str = "float16",
    ) -> None:
        if vector_dtype not in VECTOR_DTYPES:
            raise ValueError(f"vector_dtype must be one of {sorted(VECTOR_DTYPES)}, got {vector_dtype!r}")
        self.episodes = episodes
        self.vector_dtype = vector_dtype
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
            self._migrate(self._table)
            schema = self._table.schema
            self._text_dim = schema.field("text_vec").type.list_size
            self._image_dim = schema.field("image_vec").type.list_size
            value_type = schema.field("text_vec").type.value_type
            self.vector_dtype = next((n for n, t in VECTOR_DTYPES.items() if t == value_type), str(value_type))

    # ------------------------------------------------------------------ schema

    @staticmethod
    def _migrate(table: lancedb.table.Table) -> None:
        """Add columns introduced after a table was created (existing rows get the fill value)."""
        names = set(table.schema.names)
        missing = {col: fill for col, fill in _MIGRATIONS if col not in names}
        if missing:
            log.info("migrating %s: adding columns %s", TABLE_NAME, sorted(missing))
            table.add_columns(missing)

    def _schema(self) -> pa.Schema:
        vec_type = VECTOR_DTYPES[self.vector_dtype]
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
                pa.field("tier", pa.string(), nullable=True),
                pa.field("chunk_id", pa.string(), nullable=True),
                pa.field("chunk_offset", pa.int64(), nullable=True),
                pa.field("text_vec", pa.list_(vec_type, self._text_dim or DEFAULT_TEXT_DIM), nullable=True),
                pa.field("image_vec", pa.list_(vec_type, self._image_dim or DEFAULT_IMAGE_DIM), nullable=True),
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
            "kind": token.kind,
            "caption": token.caption,
            "ocr_text": token.ocr_text,
            "entities": list(token.entities),
            "search_text": _search_text(token.caption, token.ocr_text, token.entities),
            "pose": token.pose.model_dump() if token.pose else None,
            "thumb_path": token.thumb_path,
            "frame_hash": token.frame_hash,
            "episode_id": token.episode_id,
            "tier": token.tier,
            "chunk_id": token.chunk_id,
            "chunk_offset": token.chunk_offset,
            "text_vec": token.text_vec,
            "image_vec": token.image_vec,
        }

    @staticmethod
    def _vectors(col: pa.ChunkedArray) -> list[list[float] | None]:
        """Fixed-size-list column -> per-row lists via one NumPy pass (`to_pylist` boxes every float)."""
        arr = col.combine_chunks()
        n = len(arr)
        if arr.null_count == n:
            return [None] * n
        dense = np.asarray(arr.flatten()).reshape(-1, arr.type.list_size).tolist()
        if not arr.null_count:
            return dense
        it = iter(dense)
        return [next(it) if ok else None for ok in arr.is_valid().to_pylist()]

    @classmethod
    def _tokens(cls, tbl: pa.Table) -> list[MemoryToken]:
        """Arrow rows -> MemoryTokens. Rows were validated on write against this table's schema, so
        `model_construct` skips re-validating ~1.8k floats per row."""
        n = tbl.num_rows
        if not n:
            return []
        cols = {c: tbl.column(c).to_pylist() for c in _SCALAR_COLUMNS}
        names = set(tbl.column_names)
        none = [None] * n
        text_vecs = cls._vectors(tbl.column("text_vec")) if "text_vec" in names else none
        image_vecs = cls._vectors(tbl.column("image_vec")) if "image_vec" in names else none
        return [
            MemoryToken.model_construct(
                id=cols["id"][i],
                device_id=cols["device_id"][i],
                ts=cols["ts"][i],
                caption=cols["caption"][i] or "",
                ocr_text=cols["ocr_text"][i] or "",
                entities=list(cols["entities"][i] or []),
                pose=Pose(**cols["pose"][i]) if cols["pose"][i] else None,
                text_vec=text_vecs[i],
                image_vec=image_vecs[i],
                thumb_path=cols["thumb_path"][i],
                frame_hash=cols["frame_hash"][i],
                episode_id=cols["episode_id"][i],
                kind=cols["kind"][i] or "frame",
                tier=cols["tier"][i] or "hot",
                chunk_id=cols["chunk_id"][i],
                chunk_offset=cols["chunk_offset"][i],
            )
            for i in range(n)
        ]

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
        await self._episodize([token])
        await asyncio.to_thread(self._add_sync, [token])

    async def add_many(self, tokens: Sequence[MemoryToken]) -> None:
        tokens = list(tokens)
        await self._episodize(tokens)
        await asyncio.to_thread(self._add_sync, tokens)

    async def rewrite_many(self, tokens: Sequence[MemoryToken]) -> None:
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

    def _scan_arrow(self, where: str | None, columns: Sequence[str], limit: int | None = None) -> pa.Table | None:
        if self._table is None:
            return None
        q = self._table.search().select(list(columns)).limit(limit)
        if where:
            q = q.where(where)
        return q.to_arrow()

    def _scan_tokens(self, where: str | None, limit: int | None = None) -> list[MemoryToken]:
        tbl = self._scan_arrow(where, _INDEXED_COLUMNS, limit)
        return [] if tbl is None else self._tokens(tbl)

    def _range_sync(
        self, where: str | None, limit: int | None, columns: Sequence[str] = _INDEXED_COLUMNS
    ) -> list[MemoryToken]:
        tbl = self._scan_arrow(where, columns)
        if tbl is None or (limit is not None and limit <= 0):
            return []
        order = pc.sort_indices(tbl, sort_keys=_TS_ORDER)
        return self._tokens(tbl.take(order if limit is None else order[:limit]))

    def _window_ids_sync(self, where: str | None, token_id: str, before: int, after: int) -> list[str] | None:
        tbl = self._scan_arrow(where, ["id", "ts"])
        if tbl is None:
            return None
        ids = tbl.take(pc.sort_indices(tbl, sort_keys=_TS_ORDER)).column("id").to_pylist()
        try:
            idx = ids.index(token_id)
        except ValueError:
            return None
        return ids[max(0, idx - before) : idx + after + 1]

    async def get(self, token_id: str) -> MemoryToken | None:
        toks = await asyncio.to_thread(self._scan_tokens, f"id = {_sql_str(token_id)}", 1)
        return toks[0] if toks else None

    async def get_many(self, ids: Sequence[str]) -> list[MemoryToken]:
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        where = f"id IN ({', '.join(_sql_str(i) for i in ids)})"
        by_id = {t.id: t for t in await asyncio.to_thread(self._scan_tokens, where)}
        return [by_id[i] for i in ids if i in by_id]

    async def count(self, device_id: str | None = None) -> int:
        if self._table is None:
            return 0
        where = self._where(device_id=device_id)
        return await asyncio.to_thread(self._table.count_rows, where)

    def _devices_sync(self) -> list[DeviceInfo]:
        tbl = self._scan_arrow(None, ["device_id", "kind", "ts"])
        if tbl is None or tbl.num_rows == 0:
            return []
        g = tbl.group_by(["device_id", "kind"]).aggregate([("ts", "max"), ("ts", "count")])
        groups = list(zip(*(g.column(c).to_pylist() for c in ("device_id", "kind", "ts_max", "ts_count")), strict=True))
        counts = {(dev, kind): n for dev, kind, _, n in groups}
        return device_infos(((dev, kind, ts) for dev, kind, ts, _ in groups), counts)

    async def devices(self) -> list[DeviceInfo]:
        return await asyncio.to_thread(self._devices_sync)

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
        return await asyncio.to_thread(self._range_sync, where, limit)

    async def list_tokens(
        self,
        *,
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
        tier: str | None = None,
        chunk_id: str | None = None,
        vectors: bool = True,
    ) -> list[MemoryToken]:
        extra = []
        if tier is not None:
            extra.append(f"tier = {_sql_str(tier)}")
        if chunk_id is not None:
            extra.append(f"chunk_id = {_sql_str(chunk_id)}")
        where = self._where(time_from=time_from, time_to=time_to, device_id=device_id, kinds=kinds, extra=extra)
        cols = _INDEXED_COLUMNS if vectors else _SCALAR_COLUMNS
        return await asyncio.to_thread(self._range_sync, where, None, cols)

    async def episode_spans(
        self,
        *,
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
    ) -> dict[str, tuple[int, float, float]]:
        where = self._where(time_from=time_from, time_to=time_to, device_id=device_id, kinds=kinds,
                            extra=("episode_id IS NOT NULL", "episode_id != ''"))  # fmt: skip
        return await asyncio.to_thread(self._episode_spans_sync, where)

    def _episode_spans_sync(self, where: str | None) -> dict[str, tuple[int, float, float]]:
        tbl = self._scan_arrow(where, ["episode_id", "ts"])
        if tbl is None or tbl.num_rows == 0:
            return {}
        g = tbl.group_by("episode_id").aggregate([("ts", "count"), ("ts", "min"), ("ts", "max")])
        g = g.take(pc.sort_indices(g, sort_keys=[("ts_min", "ascending"), ("episode_id", "ascending")]))
        cols = (g.column(c).to_pylist() for c in ("episode_id", "ts_count", "ts_min", "ts_max"))
        return {ep: (n, lo, hi) for ep, n, lo, hi in zip(*cols, strict=True)}

    async def tier_counts(self, kinds: Sequence[str] | None = ("frame",)) -> dict[str, int]:
        if self._table is None:
            return {"hot": 0, "cold": 0, "text": 0}
        out = {}
        for tier in ("hot", "cold", "text"):
            where = self._where(kinds=kinds, extra=(f"tier = {_sql_str(tier)}",))
            out[tier] = await asyncio.to_thread(self._table.count_rows, where)
        return out

    async def neighbors(self, token_id: str, *, before: int = 2, after: int = 2) -> list[MemoryToken]:
        anchor = await self.get(token_id)
        if anchor is None:
            return []
        kinds = (anchor.kind,)
        where = self._where(device_id=anchor.device_id, kinds=kinds)
        window = await asyncio.to_thread(self._window_ids_sync, where, token_id, before, after)
        if window is None:
            return [anchor]
        return await self.get_many(window)

    def _vector_search_sync(self, vec: list[float], k: int, field: str, where: str | None) -> list[Scored]:
        if self._table is None or k <= 0:
            return []
        q = (
            self._table.search(vec, vector_column_name=field)
            .metric("cosine")
            .limit(k)
            .select([*_INDEXED_COLUMNS, "_distance"])
        )
        if where:
            q = q.where(where, prefilter=True)
        tbl = q.to_arrow()
        out = [(t, 1.0 - float(d)) for t, d in zip(self._tokens(tbl), tbl.column("_distance").to_pylist())]
        return [(t, s) for t, s in out if s > MIN_SCORE]

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
    ) -> list[Scored]:
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

    def _keyword_search_sync(self, query: str, k: int, where: str | None) -> list[Scored]:
        if self._table is None or k <= 0:
            return []
        q = (
            self._table.search(query, query_type="fts", fts_columns="search_text")
            .limit(k)
            .select([*_INDEXED_COLUMNS, "_score"])
        )
        if where:
            q = q.where(where, prefilter=True)
        tbl = q.to_arrow()
        return [(t, float(sc)) for t, sc in zip(self._tokens(tbl), tbl.column("_score").to_pylist())]

    async def keyword_search(
        self,
        query: str,
        k: int,
        *,
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[Scored]:
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
        image_mult = None
        rankings: list[list[tuple[MemoryToken, float]]] = []
        if mode in ("semantic", "hybrid"):
            if query_text_vec is not None and len(query_text_vec) == self._text_dim:
                rankings.append(await self.vector_search(query_text_vec, n, field="text_vec", **filters))
            rankings.append(await self.keyword_search(query_text, n, **filters))
        if mode in ("visual", "hybrid") and query_image_vec is not None and len(query_image_vec) == self._image_dim:
            rankings.append(await self.vector_search(query_image_vec, n, field="image_vec", **filters))
            if mode == "hybrid":  # a lone image ranking (visual mode) keeps plain RRF scores
                image_mult = calibrated(self.image_calibration)

        if image_mult is None:
            return fuse_hits(rankings, k)
        # the image ranking is always last; its hits are scaled by the SigLIP match probability
        return fuse_hits(rankings, k, multipliers=[None] * (len(rankings) - 1) + [image_mult])

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
