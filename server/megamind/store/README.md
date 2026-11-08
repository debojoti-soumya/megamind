# `megamind.store` — memory token storage

Everything the server remembers lives under `server/data/` (git-ignored). Two backends implement the same
`MemoryStore` interface (`base.py`); `build_store(settings)` picks one from `STORE_BACKEND`.

| `STORE_BACKEND` | class | persistence |
|---|---|---|
| `memory` (default) | `InMemoryStore` | none — process lifetime, for tests / demos |
| `lancedb` | `LanceDBStore` | `LANCEDB_PATH` (default `./data/lancedb`) |

## On-disk layout

```
server/data/
├── lancedb/                      # LanceDBStore (embedded, no daemon)
│   └── memory_tokens.lance/      # one Lance table = all tokens for all devices
│       ├── data/*.lance          # columnar fragments (append + merge_insert on `id`)
│       ├── _versions/            # MVCC manifests; `table.optimize()` compacts every 256 writes
│       ├── _indices/             # FTS (BM25) index on `search_text`; IVF_FLAT cosine index on
│       │                         #   text_vec / image_vec once ≥ 4096 non-null rows (flat scan below that)
│       └── _transactions/
├── thumbs/                       # THUMB_DIR — one JPEG per token: <token_id>.jpg (frames: ≤960 px,
│                                 #   written by ingest; episodes: copy of a representative frame)
└── audio/                        # AUDIO_DIR — stream-3 voice, not managed by the store
```

Thumbnails are referenced by `MemoryToken.thumb_path` and served at `GET /v1/thumbs/<id>.jpg`.
The store never stores raw frames or VLM visual tokens — only JPEG thumbnail + text + two vectors
(see `research/digest/storage.md`).

## Table schema (`memory_tokens`)

Mirrors `MemoryToken` in `docs/api.md` plus one derived column:

| column | type | notes |
|---|---|---|
| `id` | string (PK) | ULID-ish; `merge_insert("id")` makes `add` an upsert |
| `device_id` | string | filter column on every primitive |
| `ts` | float64 | unix seconds; `range`/`neighbors` order by it |
| `kind` | string | `"frame"` \| `"episode"` \| `"transcript"` (heard speech from `POST /v1/audio`; empty `thumb_path`/`frame_hash`) |
| `caption`, `ocr_text` | string | empty string until enriched |
| `entities` | list\<string\> | |
| `search_text` | string | **derived**: `caption + "\n" + ocr_text + "\n" + " ".join(entities)`; BM25 target |
| `pose` | struct{x,y,z,yaw} \| null | |
| `thumb_path` | string | relative path as given by ingest |
| `frame_hash` | string | dHash hex, used by `recent_hashes` for dedupe |
| `episode_id` | string \| null | assigned by `EpisodeTracker` on insert |
| `text_vec` | fixed_size_list\<float16\>[D_t] \| null | D_t inferred from the first token that carries one (1024 for OpenAI via `EMBED_DIM`, 256 for the mock) |
| `image_vec` | fixed_size_list\<float16\>[D_i] \| null | likewise (768 for SigLIP) |

Vectors are float16 in new tables (`LANCEDB_VECTOR_DTYPE=float16`, the default; `float32` to opt out): half the
bytes (1024 + 768 dims: 7.2 → 3.6 KB/token) and the same nearest neighbours (tested against float32 on random
unit vectors, and with an IVF index over a float16 column that contains nulls). The dtype of an existing table is
read from its schema on open and never migrated, so stores created before this change stay float32.

Vector dimensions are frozen once the table exists; switching embedders (e.g. mock 256-d → OpenAI 1024-d, or SigLIP B/16 768-d → B/32 512-d)
requires a purge or a fresh `LANCEDB_PATH`.

## Modules

| file | what |
|---|---|
| `base.py` | `MemoryStore` ABC + primitive contract (`vector_search`, `keyword_search`, `get_many`, `range`, `neighbors`, `search`, `purge`, …) |
| `memory_store.py` / `lancedb_store.py` | the two backends (tests run every store test against both) |
| `rrf.py` | reciprocal-rank fusion; `fuse_hits` drops non-positive scores so unrelated queries return nothing; `weighted_rrf` + `image_conf` for calibrated image fusion (below) |
| `episodes.py` | `EpisodeTracker`: online `episode_id` assignment (gap > 120 s or image cosine < 0.80 → new episode; detours may rejoin one of the last 4) |
| `consolidate.py` | `Consolidator`: writes one `kind="episode"` summary token per closed episode; `python -m megamind.store.consolidate [--loop]` |
| `embeddings.py` | `build_text_embedder` / `build_image_embedder`: OpenAI text-embedding-3-small, local bge-small (sentence-transformers, CPU, query instruction via `embed_query`), open_clip SigLIP (CPU), mocks |
| `purge.py` | retention CLI, see below |
| `bench.py` (planned) | 10k-token latency benchmark of every primitive |

## Retention / privacy

All memories are wearer data; deletion is total (rows **and** thumbnails) and synchronous.

* API: `DELETE /v1/memories?device_id=<id>` — returns `{"deleted": n}` (a `since=` param is proposed in a
  separate contract PR).
* CLI (server stopped — LanceDB must not have two writers):
  ```
  python -m megamind.store.purge                 # wipe everything
  python -m megamind.store.purge --device quest-1 --since 2h --dry-run
  python -m megamind.store.purge --since 2026-11-07T09:00:00 -y
  ```
  `--since` takes ISO-8601 or a relative age (`30m`, `2h`, `1d`); `--keep-thumbs` leaves JPEGs on disk.
* `since` deletes tokens with `ts >= since`, including `kind="episode"` summaries whose episode started
  in that window. A summary whose frames were partially deleted is left as-is (it is already a summary).
* The in-memory backend forgets on restart (the CLI is a no-op for it). Delete `server/data/` for a factory reset.

## Concurrency / durability notes

* LanceDB calls run in `asyncio.to_thread`; writes are serialised by an `RLock` inside the process.
  One server process per `LANCEDB_PATH`.
* Every `add`/`add_many` is durable on return (Lance appends a new version). Crash-safe; no WAL to replay.
* Index maintenance is incremental: FTS index is created on first write and Lance re-indexes new rows on
  `optimize()`; unindexed rows are still searched (brute force) in the meantime.

## Calibrated image fusion

SigLIP is trained with `p(match) = sigmoid(scale * cos + bias)`, so an image–text cosine converts to a match
probability. When the image embedder exposes `calibration` (`OpenCLIPImageEmbedder` with a SigLIP model; CLIP
models and mocks return `None`) and `SEARCH_CALIBRATE_IMAGE=true` (default), `main.py` sets
`MemoryStore.image_calibration = (scale, bias)` and logs it. Each image-vector hit's RRF contribution is then
multiplied by `p(cos)`, in hybrid mode, both in `store.search()` (`/v1/memories`) and in `HybridRetriever._fuse` (the
`recall_memory` path). Visual mode ranks the image list alone and is left as it was. The nearest image of
a question with no visual answer is usually still a poor match, and without calibration it got full rank-0 credit
and could outrank the exact OCR/BM25 hit. Ported from PR #54, which measured sign-text questions going from 42% to
89% right-frame-first. The planner's per-type image weights are unchanged and should be retuned with benchmark data.
