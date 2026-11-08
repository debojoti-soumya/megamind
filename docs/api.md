# Megamind API contract (v1)

Base path: `/v1`. JSON unless noted. Timestamps are Unix seconds (float, UTC) in
bodies and ISO-8601 in query strings. This file is authoritative; `server/megamind/models.py`
and the C# DTOs in `headset/` must match it.

## `GET /health`
```json
{ "status": "ok", "store_count": 1234,
  "models": { "claude": "claude-sonnet-4-6", "reader": "google/gemma-4-31B-it",
              "embed": "mock (openai configured, no key)", "stt": "mock", "tts": "mock" } }
```
`models` names what is actually serving each role. When the configured provider has no key, or isn't wired up yet,
the value is `mock (<provider> configured, no key | not implemented)`. Claude without a key reports
`fallback (<model> configured, no key)`, meaning the no-LLM caption path.

## `POST /frames`  (headset / feeder -> server)
`multipart/form-data`
- `image`: JPEG bytes (<= 960 px wide recommended)
- `meta`: JSON string
  ```json
  { "device_id": "quest3-abc", "ts": 1731000000.123,
    "pose": { "x": 0.0, "y": 1.6, "z": 0.0, "yaw": 90.0 } }
  ```
Response `202 Accepted`:
```json
{ "frame_id": "01J...", "accepted": true, "reason": null }
```
`accepted=false` with `reason` in `{"duplicate","too_soon","dark","blurry","budget"}` means the
frame was dropped by the filter (`blurry` = motion blur / out of focus, `budget` = per-minute cap). Enrichment (caption/embedding) is asynchronous; a token appears
in search a few seconds later.

## `GET /memories`  (debug search)
Query: `q` (str), `from`/`to` (ISO-8601, optional; unparseable → 400), `k` (int, default 12),
`mode` (`semantic|visual|hybrid`, default `hybrid`), `device_id` (optional).
```json
{ "results": [ { "id": "...", "ts": 1731000000.1, "caption": "...", "ocr_text": "...",
                 "entities": ["keys","kitchen counter"], "thumb_url": "/v1/thumbs/....jpg",
                 "score": 0.83 } ] }
```

## `GET /thumbs/{id}.jpg`
Keyframe thumbnail (JPEG).

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
  "cited": [ { "id": "...", "ts": 1731000000.1, "caption": "...", "thumb_url": "/v1/thumbs/....jpg" } ],
  "tool_calls": [ { "name": "recall_memory", "args": { "query": "keys", "k": 12 }, "confidence": "high" } ] }
```

## `GET /audio/{id}.wav`
TTS output for a chat response.

## `DELETE /memories?device_id=...&since=...`
Purge memory tokens (rows **and** thumbnails). Both params are optional: `device_id` limits the purge to one
device (default: all devices); `since` deletes only tokens with `ts >= since` ("forget the last 2 hours"),
given as ISO-8601 (`2026-11-07T09:00:00Z`; naive = server local time) or unix seconds (`1731000000.0`).
Episode summaries whose `ts` (episode start) falls in the window are deleted too. Invalid, empty/whitespace (`since=`) or non-finite `since` → 400 (omit the param to purge all time).
Returns `{ "deleted": 1234 }`. Offline equivalent: `python -m megamind.store.purge --device --since`.

## Memory token (stored record)
```
id, device_id, ts, caption, ocr_text, entities[], pose{x,y,z,yaw}|null,
text_vec[1024], image_vec[768], thumb_path, frame_hash, episode_id|null,
kind: "frame" | "episode"
```
`kind` is `"frame"` for captured keyframes (default). The store groups consecutive frames into
`episode_id`s (time gap + visual change) and a consolidation job writes one `kind="episode"` token
per closed episode whose `caption` is a summary of that episode and whose `ts` is the episode start,
so vague queries ("what did I do this morning") hit summaries. Episode tokens are server-internal and
never appear in `FrameAccepted`; they surface through search/chat citations like any other token.

## Internal: `recall_memory` tool (Claude -> server)
```
recall_memory(query, time_from?, time_to?, k=12, mode="hybrid")
  -> { evidence, confidence: high|medium|low|not_found, cited[], all_retrieved[] }
```
