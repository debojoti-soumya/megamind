# Megamind API contract (v1)

Base path: `/v1`. JSON unless noted. Timestamps are Unix seconds (float, UTC) in
bodies and ISO-8601 in query strings. This file is authoritative; `server/megamind/models.py`
and the C# DTOs in `headset/` must match it.

## `GET /health`
```json
{ "status": "ok", "store_count": 1234, "models": { "claude": "...", "reader": "...", "embed": "..." } }
```

## `POST /frames`  (headset / feeder -> server)
`multipart/form-data`
- `image`: JPEG bytes (<= 960 px wide recommended)
- `meta`: JSON string
  ```json
  { "device_id": "quest3-abc", "ts": 1731000000.123,
    "pose": { "x": 0.0, "y": 1.6, "z": 0.0, "yaw": 90.0 },
    "force": false, "source": "quest" }
  ```
  `force` (optional) bypasses blur/gap/novelty/budget, e.g. at push-to-talk start.
  `source` (optional) labels the sensor (`quest`, `feeder`, `phone`).
Response `202 Accepted`:
```json
{ "frame_id": "01J...", "accepted": true, "reason": null }
```
`accepted=false` with `reason` in `{"duplicate","too_soon","dark","blurry","budget"}` means the frame was
dropped by the filter. Enrichment (caption/embedding) is asynchronous; a token appears
in search a few seconds later.

## `GET /memories`  (debug search)
Query: `q` (str), `from`/`to` (ISO-8601, optional), `k` (int, default 12),
`mode` (`semantic|visual|hybrid`, default `hybrid`), `device_id` (optional).
```json
{ "results": [ { "id": "...", "ts": 1731000000.1, "caption": "...", "ocr_text": "...",
                 "entities": ["keys","kitchen counter"], "thumb_url": "/v1/thumbs/....jpg",
                 "score": 0.83 } ] }
```

## `GET /episodes`  (timeline / activity)
Query: `from`/`to` (ISO-8601, optional), `q` (optional topic), `k` (default 20), `device_id` (optional).
Without `q`: episodes overlapping the range, oldest first. With `q`: ranked by the summed fused
score of their tokens, each with up to 2 `top_hits` (same shape as `/memories` results).
```json
{ "episodes": [ { "id": "ep_...", "start_ts": 1731000000.1, "end_ts": 1731000300.4,
                  "n_tokens": 14, "title": "a kitchen", "summary": "", "entities": ["keys"],
                  "rep_token_id": "...", "rep_thumb_url": "/v1/thumbs/....jpg",
                  "score": null, "top_hits": [] } ] }
```
A new episode starts after a >120 s gap, an image-similarity drop below `EPISODE_SPLIT_SIM`
against the episode centroid, or >3 m of movement; short detours rejoin a recent episode.

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

## `DELETE /memories?device_id=...`
Purge all memory tokens for a device. Returns `{ "deleted": 1234 }`.

## Memory token (stored record)
```
id, device_id, ts, caption, ocr_text, entities[], pose{x,y,z,yaw}|null,
text_vec[d], image_vec[d], thumb_path, frame_hash, episode_id|null,
transcript, source, trigger(novel|pose|heartbeat|forced), sharpness, enriched,
tier(hot|cold|text), chunk_id|null, chunk_offset|null
```
Vector dims follow the configured embedder (e.g. 1024 bge-m3/voyage, 1536 OpenAI 3-small,
768 SigLIP-base). Search hits additionally carry `transcript`, `episode_id`, `tier`.

## Internal: `recall_memory` tool (Claude -> server)
```
recall_memory(query, time_from?, time_to?, k=12, mode="hybrid")
  -> { evidence, confidence: high|medium|low|not_found, cited[], all_retrieved[] }
list_episodes(time_from?, time_to?, query?, k=20) -> EpisodeHit[]   # same shape as GET /episodes
```
