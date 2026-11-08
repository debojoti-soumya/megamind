# Megamind API contract (v1)

Base path: `/v1`. JSON unless noted. Timestamps are Unix seconds (float, UTC) in
bodies and ISO-8601 in query strings. This file is authoritative; `server/megamind/models.py`
and the C# DTOs in `headset/` must match it.

## `GET /health`
```json
{ "status": "ok", "store_count": 1234,
  "models": { "claude": "claude-sonnet-4-6", "reader": "google/gemma-4-31B-it",
              "embed": "mock (openai configured, no key)", "enrich": "vlm:google/gemma-4-31B-it",
              "stt": "mock", "tts": "mock" } }
```
`models` names what is actually serving each role. When the configured provider has no key, or isn't wired up yet,
the value is `mock (<provider> configured, no key | not installed | not implemented)`. Claude without a key reports
`fallback (<model> configured, no key)`, meaning the no-LLM caption path. `enrich` is the keyframe captioner
(`ENRICH_PROVIDER`): `vlm:<model>`, `local` (keyless SigLIP label + Tesseract OCR) or `mock`; a VLM that is failing
over to its fallback reports `vlm:<model> (degraded: last error 503)` until its next successful call.

## `POST /frames`  (headset / feeder -> server)
`multipart/form-data`
- `image`: JPEG bytes (<= 960 px wide recommended)
- `meta`: JSON string
  ```json
  { "device_id": "quest3-abc", "ts": 1731000000.123,
    "pose": { "x": 0.0, "y": 1.6, "z": 0.0, "yaw": 90.0 },
    "transcript": null }
  ```
  `transcript` (optional, string|null): text heard around this frame (subtitles, an external ASR). It is stored as a
  `kind="transcript"` token (same path and merge rule as `POST /audio`, no STT), **never** attached to the frame and
  **never** changes the frame's keep decision: it is stored even when the frame is rejected. `""`/`null` = none.
  Send each piece of text once (a repeat of the last piece within the merge window is ignored).
Response `202 Accepted`:
```json
{ "frame_id": "01J...", "accepted": true, "reason": null }
```
`accepted=false` with `reason` in `{"duplicate","too_soon","dark","blurry","budget"}` means the
frame was dropped by the filter (`blurry` = motion blur / out of focus, `budget` = per-minute cap). Enrichment (caption/embedding) is asynchronous; a token appears
in search a few seconds later.

## `POST /audio`  (headset / feeder -> server)
Ambient speech around the wearer, so "who did I talk to?" and "what did she say about Tuesday?" work.
`multipart/form-data`
- `audio`: a chunk of 16-bit PCM WAV (any rate, mono preferred; 8–16 kHz headset mics are fine) or any
  ffmpeg-decodable audio (needs ffmpeg on the server). Send chunks of ~15–30 s; max 25 MB and 600 s.
- `meta`: JSON string; `ts_start` is the Unix time (seconds, float) of the chunk's first sample
  ```json
  { "device_id": "quest3-abc", "ts_start": 1731000000.0,
    "pose": { "x": 0.0, "y": 1.6, "z": 0.0, "yaw": 90.0 } }
  ```
The server transcribes with timestamps (synchronously) and stores each ≤ ~15 s window of speech as a
`kind="transcript"` memory token (`ts` = `ts_start` + window offset, `caption` = `Speech heard nearby: "<words>"`,
no thumbnail). Response `200 OK`:
```json
{ "audio_id": "01J...", "duration_s": 15.0, "transcript": "hi I'm Priya ...", "token_ids": ["01J..."] }
```
Speech that starts ≤ `SPEECH_MERGE_S` (10 s) after the device's last transcript token ends continues that token
(its caption grows, `ts` stays the token's start) as long as it would span ≤ `SPEECH_MAX_TOKEN_S` (60 s), so an
utterance cut by a chunk boundary stays one token. Only the first window of a chunk can continue the previous chunk;
so `token_ids` may start with an **existing** (merged-into) token id. The merge target is kept per device in server
memory (no merge across a restart) and dropped by `DELETE /memories`.
Silence → `token_ids: []`. Bad `meta` (incl. non-finite `ts_start`) / empty body → 400; > 25 MB or > 600 s → 413;
undecodable audio → 415; STT provider error → 502.

## `GET /memories`  (debug search)
Query: `q` (str), `from`/`to` (ISO-8601, optional; unparseable → 400), `k` (int, default 12),
`mode` (`semantic|visual|hybrid`, default `hybrid`), `device_id` (optional).
```json
{ "results": [ { "id": "...", "ts": 1731000000.1, "caption": "...", "ocr_text": "...",
                 "entities": ["keys","kitchen counter"], "thumb_url": "/v1/thumbs/....jpg",
                 "score": 0.83, "kind": "frame" } ] }
```
`kind` is the token kind (see *Memory token*); `thumb_url` is `""` for tokens without an image (`kind="transcript"`,
or a frame whose pixels aged out, `tier="text"`).

## `GET /memories/recent`  (newest memories, no query)
Query: `k` (int, 1–200, default 24), `device_id` (optional), `kind` (repeatable, default `frame`; e.g.
`&kind=frame&kind=transcript`). Same body as `GET /memories` (`SearchResponse`), newest first, every `score` is `0.0`
(not ranked). Unenriched tokens may appear with an empty `caption`. Used by the Memory Theater (`GET /xr`) for
its cards. `GET /memories` stays a ranked search that drops hits unrelated to `q`, so it can't list "recent".

## `GET /thumbs/{id}.jpg`
Keyframe thumbnail (JPEG) on every media tier: hot frames are served from their JPEG, cold frames are decoded
from their AV1 chunk (recent decodes are cached). 404 for tokens without an image (`tier="text"`, transcripts).

## `POST /chat`
Either `application/json` with `text`, or `multipart/form-data` with `audio` (16 kHz mono WAV).
```json
{ "session_id": "s1", "device_id": "quest3-abc", "text": "where did I leave my keys?" }
```
Response:
```json
{ "transcript": "where did I leave my keys?",
  "answer": "On the kitchen counter next to the coffee machine, about 20 minutes ago.",
  "audio_url": "/v1/audio/01J....wav",
  "cited": [ { "id": "...", "ts": 1731000000.1, "caption": "...", "thumb_url": "/v1/thumbs/....jpg",
               "replay_url": "/v1/replay.mp4?device_id=quest3-abc&from=1731000000.100&to=1731000010.100" } ],
  "tool_calls": [ { "name": "recall_memory", "args": { "query": "keys", "k": 12 }, "confidence": "high" } ] }
```
`replay_url` is optional and `null` when the server holds neither buffered frames nor kept keyframes with pixels
around the citation (pixels aged out, replay disabled, or a different device). It covers `ts - 5 s .. ts + 5 s`.
`cited[].thumb_url` is `""` when the cited token has no image (heard speech, or a frame whose pixels aged out,
`tier="text"`): show the caption, skip the thumbnail.
For questions about the present ("what am I looking at right now", "what did I just see") the evidence also
includes live short-term memory (see `GET /live/{id}.jpg`): a cited raw frame has an id starting `now_`, a
caption starting `LIVE NOW-<n>:` and `thumb_url` `/v1/live/{id}.jpg`. Treat `thumb_url` as an opaque URL.

## `GET /replay.mp4?device_id=...&from=...&to=...`
Moment replay: an H.264 mp4 of **every** frame received from the device in `[from, to]` (not just the
keyframes the filter kept), each frame held until the next, so mp4 offset 0 = the first frame's `ts`.
The server keeps raw frames for a sliding window per device (default 15 min, bounded by count/bytes, at most
`REPLAY_MAX_DEVICES` devices; `REPLAY_*`).
- `from`/`to`: unix seconds or ISO-8601 (naive = server local time). Defaults: `to` = newest buffered frame,
  `from` = `to - 30 s`. `to < from` or span > `REPLAY_MAX_SPAN_S` (300) → 400; unparseable or nan/inf → 400.
- `device_id`: optional; without it the device with the most frames in the window is used.
- `GET` or `HEAD`. Supports `Range: bytes=...` (206 / 416) for seeking; `Accept-Ranges: bytes` on every 200.
- Headers: `X-Replay-Start` (unix ts at offset 0), `X-Replay-Device`, `X-Replay-Frames`.
- Older than the buffer: the clip is rendered from the keyframes the filter kept in the window (hot or cold tier),
  so it is keyframe-rate (one frame every few seconds, each held until the next). `X-Replay-Source` says which:
  `buffer` (every received frame) or `keyframes`.
- 404 when neither buffered nor kept frames exist in the window; 503 when ffmpeg is not available on the server.
Rendered clips are cached; `DELETE /memories` also purges the matching buffered frames and the clip cache.

## `GET /live/{id}.jpg`
Live short-term memory frame (JPEG): a raw frame from the replay buffer that was shown to the reader as a
`NOW-<n>` evidence item because enrichment had not caught up yet. Only `thumb_url`s of `now_*` citations point
here. 404 once the frame leaves the replay buffer (window / `DELETE /memories`) or the small live cache.
Opt out with `LIVE_MEMORY_ENABLED=false` (`LIVE_MEMORY_*` knobs in `server/.env.example`).
## `POST /montage`
Highlight reel: a captioned mp4 of the best few moments for a query and/or a time range.
```json
{ "device_id": "quest3-abc", "query": "lunch with the team", "time_from": "2026-11-07T11:00:00-08:00",
  "time_to": 1731009600.0, "max_clips": 6 }
```
- At least one of `query`, `time_from`, `time_to` (400 otherwise). Time bounds: unix seconds or ISO-8601
  (naive = server local time); `time_to < time_from` → 400. `device_id` optional (default: all devices).
  `max_clips` 0-24 (0/null = default), capped by `MONTAGE_MAX_CLIPS` (8). Empty strings count as unset.
- Candidates: the retriever's top hits for `query` (within the range, if given), else every frame/transcript
  token in the range (long ranges are sampled across the whole span); at most `MONTAGE_MAX_CANDIDATES` (40),
  evenly spaced.
- Clip choice: Claude (`montage_plan` tool, `CLAUDE_MODEL`) when `ANTHROPIC_API_KEY` is set and
  `MONTAGE_PLANNER` is not `offline`; otherwise (or on any API failure) a deterministic plan of evenly spaced,
  non-empty, de-duplicated moments. `planner` says which one ran.
- Each clip shows the moment's local time and a short title burned in, with 0.3 s fades. `source="replay"`: the
  frames buffered for `GET /replay.mp4` around the moment (real frame rate). `source="timelapse"`: the stored
  keyframe thumbnails from that moment on, each held ~1 s (a ~1 fps timelapse; used once the moment is older
  than the replay window, and always when `PRIVACY_BLUR_FACES` is on: buffered frames are not blurred).
Response:
```json
{ "id": "3f2a9c0d1e5b7a8c6d4e", "url": "/v1/montages/3f2a9c0d1e5b7a8c6d4e.mp4", "title": "Your highlights",
  "narration": "Here's your highlight reel: 2 moments from that period.", "planner": "offline",
  "clips": [ { "token_id": "...", "ts": 1731005000.2, "title": "Team lunch at the long table",
               "duration_s": 4.0, "source": "timelapse", "thumb_url": "/v1/thumbs/....jpg" } ] }
```
404 when nothing matches or no chosen moment has frames/thumbnails; 503 without ffmpeg (same lookup as
replay, `REPLAY_FFMPEG`); 404 when `MONTAGE_ENABLED=false`; 409 when `DELETE /memories` ran while it was
being made (nothing is kept; POST again). The same plan always yields the same `id`.

## `GET /montages/{id}.mp4`
The rendered montage (H.264, `MONTAGE_WIDTH`x`MONTAGE_HEIGHT`, default 960x540). `GET` or `HEAD`; supports
`Range: bytes=...` (206 / 416) exactly like `/replay.mp4`. Montages are a bounded cache (newest
`MONTAGE_CACHE_MAX_FILES`, 16): 404 once evicted, so POST again. `DELETE /memories` clears every montage.
Not a Claude tool: chat answers do not include a montage (call this endpoint from a "show me" UI action).

## `GET /audio/{id}.wav`
TTS output for a chat response.

## `DELETE /memories?device_id=...&since=...`
Purge memory tokens (rows **and** thumbnails; frames already aged into an AV1 chunk are cut out of it by
re-encoding the chunk without them, so the chunk's other frames survive). Both params are optional: `device_id` limits the purge to one
device (default: all devices); `since` deletes only tokens with `ts >= since` ("forget the last 2 hours"),
given as ISO-8601 (`2026-11-07T09:00:00Z`; naive = server local time) or unix seconds (`1731000000.0`).
Episode summaries whose `ts` (episode start) falls in the window are deleted too. Invalid, empty/whitespace (`since=`) or non-finite `since` → 400 (omit the param to purge all time).
Returns `{ "deleted": 1234 }`. Offline equivalent: `python -m megamind.store.purge --device --since`.

## `POST /admin/age?hot_older_than_s=...&cold_older_than_s=...`  (ops / demo)
Run one media-aging pass now (the server also runs one every `MEDIA_AGE_INTERVAL_S`, default 3600 s). Frames older
than `cold_older_than_s` (default `COLD_RETENTION_DAYS`, 100 d) drop their pixels (`tier="text"`); hot frames older
than `hot_older_than_s` (default `HOT_RETENTION_H`, 72 h) are re-encoded into one AV1 mp4 per (device, episode)
run (`tier="cold"`) and their JPEGs deleted once the chunk's frame count is verified. Negative / non-finite → 400.
```json
{ "to_cold": 120, "chunks": 1, "jpeg_bytes": 7800000, "chunk_bytes": 820000, "to_text": 0, "aborted": 0,
  "skipped": null }
```
`skipped` explains why nothing went cold (e.g. no ffmpeg with an AV1 encoder: frames then stay hot JPEG).

## `GET /stats`
Storage and ingest counters.
```json
{ "tokens": 1500, "frames": 1400, "tiers": { "hot": 400, "cold": 900, "text": 100 },
  "filter": { "seen": 15000, "kept": 1400, "keep_rate": 0.093, "by_reason": { "novel": 900, "duplicate": 8000 } },
  "worker_queue": 0,
  "disk_bytes": { "hot_jpeg": 26000000, "cold_chunks": 6300000, "index": 9000000 },
  "bytes_per_frame": { "hot": 65000, "cold": 7000, "index": 6000 },
  "projected_100d_gb_at_budget": 8.4,
  "media": { "encoder": "libsvtav1", "hot_retention_h": 72, "cold_retention_days": 100, "cold_decodes": 12 } }
```
`tiers` counts `kind="frame"` tokens. `bytes_per_frame` values are `null` until a tier has frames. The projection
assumes the frame budget (`FILTER_MAX_PER_MIN`) for 16 h/day, `HOT_RETENTION_H` as JPEG and the rest as AV1
(a tier not measured yet is assumed 8x larger / smaller than the other, JPEG vs AV1); `null` while no frame has
pixels.

## Memory token (stored record)
```
id, device_id, ts, caption, ocr_text, entities[], pose{x,y,z,yaw}|null,
text_vec[1024], image_vec[768], thumb_path, frame_hash, episode_id|null,
kind: "frame" | "episode" | "transcript",
tier: "hot" | "cold" | "text", chunk_id|null, chunk_offset|null
```
`tier` is where a frame's pixels live: `"hot"` = the JPEG at `thumb_path` (default), `"cold"` = frame `chunk_offset`
of the AV1 chunk `chunk_id` (`thumb_path` is `""`), `"text"` = no pixels; caption, OCR, entities and vectors are
kept, so the token stays searchable. Only `kind="frame"` tokens are aged; episode tokens keep their own JPEG copy
of a representative frame. `thumb_url` keeps its shape on hot and cold tokens and is `""` on text ones.
`kind` is `"frame"` for captured keyframes (default). The store groups consecutive frames into
`episode_id`s (time gap, visual change against the last few frames, pose movement, and a 30 min length cap) and a consolidation job writes one `kind="episode"` token
per closed episode whose `caption` is a summary of that episode and whose `ts` is the episode start,
so vague queries ("what did I do this morning") hit summaries. Episode tokens are server-internal and
never appear in `FrameAccepted`; they surface through search/chat citations like any other token.
`kind="transcript"` tokens come from `POST /audio` and `FrameMeta.transcript`: `caption` holds the heard words, `thumb_path`/`frame_hash` are
empty and `image_vec` is null; they are never episodized or used for frame dedupe.
Storage note (no wire change): the LanceDB backend stores `text_vec`/`image_vec` as float16 in newly created
tables (`LANCEDB_VECTOR_DTYPE`, default `float16`; existing float32 tables are kept as
they are). Unit-norm vectors read back differ from what was written by < 1e-3 per component; nearest neighbours
are unchanged.

## Internal: `recall_memory` tool (Claude -> server)
```
recall_memory(query, time_from?, time_to?, k=12, mode="hybrid")
  -> { evidence, confidence: high|medium|low|not_found, cited[], all_retrieved[] }
```

## Memory Theater: `GET /xr` (outside `/v1`; not part of the headset contract)
A static WebXR page served by the server (`server/megamind/xr/static/`, plain ES modules, three.js from a pinned
CDN). It only calls the public endpoints above (`/memories/recent`, `/chat`, `/thumbs/*`, `/replay.mp4`).
Assets are under `/xr/static/`. `XR_ENABLED=false` removes both routes. Quest usage: [quest3-guide.md](quest3-guide.md#8-memory-theater-xr-in-the-quest-browser).
