# Megamind API contract (v1)

Base path: `/v1`. JSON unless noted. Timestamps are Unix seconds (float, UTC) in
bodies and ISO-8601 in query strings. This file is authoritative; `server/megamind/models.py`
and the C# DTOs in `headset/` must match it.

## Auth
Off by default. When the server's `API_TOKEN` is set (always on the Modal deploy, [deploy/README.md](../deploy/README.md)),
every route (`/v1/*`, `/ui`, `/xr`, static assets, the SSE stream, thumbnails, replays, audio) needs the token:
- `Authorization: Bearer <token>` (feeder, ask.py, Unity `MegamindConfig.apiToken` / `--es apiToken` launch-intent extra), or
- `?token=<token>` on any URL, for `<img>`/`<video>`/`<audio>` and pasted links. A request authorized this way also
  gets `Set-Cookie: megamind_token=<token>; Path=/; HttpOnly; SameSite=Lax` (`; Secure` over https), so the page it
  opened (`/ui?token=...`, `/xr?token=...`) authenticates its own later fetches and media without code changes.
- The `megamind_token` cookie is accepted on its own.

Missing or wrong token: `401 {"detail": "missing or invalid API token"}` with `WWW-Authenticate: Bearer`
(WebSockets are closed with 1008). CORS preflights (`OPTIONS`) are never blocked. Comparison is constant-time.
Response bodies are unchanged: `thumb_url`/`replay_url`/`audio_url` stay token-free, clients add the header or `?token=`.

## `GET /health`
```json
{ "status": "ok", "store_count": 1234,
  "models": { "claude": "claude-sonnet-5-5", "reader": "google/gemma-4-31B-it",
              "embed": "mock (openai configured, no key)", "enrich": "vlm:google/gemma-4-31B-it",
              "stt": "mock", "tts": "mock", "store": "lancedb" } }
```
`models` names what is actually serving each role. When the configured provider has no key, or isn't wired up yet,
the value is `mock (<provider> configured, no key | not installed | not implemented)`. Claude without a key reports
`fallback (<model> configured, no key)`, meaning the no-LLM caption path. `enrich` is the keyframe captioner
(`ENRICH_PROVIDER`): `vlm:<model>`, `local` (keyless SigLIP label + Tesseract OCR) or `mock`; a VLM that is failing
over to its fallback reports `vlm:<model> (degraded: last error 503)` until its next successful call.
`store` is `STORE_BACKEND` (`memory` | `lancedb`). LanceDB sizes its vector columns from the configured embedders
(`EMBED_PROVIDER`/`IMAGE_EMBED_PROVIDER` widths), whatever kind of memory arrives first. If an existing table was
built with different widths (embedder changed), new memories carrying those vectors can't be stored: the server
logs an error, `status` becomes `"degraded"` and `store` reads `lancedb (degraded: <reason>)` until a full
`DELETE /memories` (an empty mis-sized table is recreated automatically).

## `POST /frames`  (headset / feeder -> server)
`multipart/form-data`
- `image`: JPEG bytes (the headset sends the full 1280x960 camera frame; stored as-is)
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

  Meta limits (shared by `POST /frames`, `/audio` and `/video`): `ts` / `ts_start` must be finite Unix seconds no
  more than 1 day (86400 s) past the server clock; `device_id` must be non-blank and short enough to be a directory
  name once URL-quoted (~230 bytes: plain ASCII ids up to 238 chars). Otherwise `400 {"detail": "bad meta: ..."}`
  and nothing is stored.
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
Silence (incl. a 0-sample or digitally silent 16-bit chunk, which skips STT) → `token_ids: []`, `transcript: ""`. Bad `meta` (same limits as `POST /frames`) / empty body → 400; > 25 MB or > 600 s → 413;
undecodable audio → 415; STT provider error → 502.

## `POST /video`  (headset -> server; optional)
Real video for replay and montage: one short H.264 clip per request (the native Quest app sends one 5 s, 30 fps clip
per chunk next to its `/frames` + `/audio` uploads). Frames still drive memory; video only makes replays and montages
real-frame-rate. Sending no video changes nothing.
`multipart/form-data`
- `video`: an mp4 (or mov) with an H.264 video stream; audio tracks are ignored. Max `VIDEO_MAX_CLIP_BYTES`
  (32 MB) and `VIDEO_MAX_CLIP_S` (60 s); frame size up to 4096 px a side.
- `meta`: JSON string; `ts_start` is the Unix time (seconds, float) of the clip's **first frame**
  ```json
  { "device_id": "quest3-abc", "ts_start": 1731000000.067 }
  ```
The server validates the clip with ffprobe (falls back to `ffmpeg -i` when only ffmpeg is installed) and stores it on
disk (`VIDEO_STORE_DIR`, default `<THUMB_DIR>/../video`). Response `200 OK`:
```json
{ "video_id": "01J...", "accepted": true, "reason": null, "ts_start": 1731000000.067, "duration_s": 4.967,
  "width": 1280, "height": 960 }
```
- `reason="duplicate"` (`accepted: true`): a clip with the same device, start (±50 ms) and length (±0.1 s) is
  already stored (an upload retried after its response was lost); its `video_id` is returned, nothing new is kept.
- `reason="privacy"` (`accepted: false`, `video_id: ""`): `PRIVACY_BLUR_FACES(_VLM)` is on. Real video is never
  blurred, so it is discarded; don't resend.
- Bad `meta` (same limits as `POST /frames`: finite `ts_start` ≤ 1 day ahead, non-blank `device_id` ≤ ~230 bytes) /
  empty body → 400; larger or longer than the caps → 413; not a decodable video, not mp4/mov, not H.264 → 415; `DELETE /memories` covered the clip while it was uploading → 409
  (not kept); `VIDEO_ENABLED=false` → 404 (clients should then stop sending video); no ffmpeg/ffprobe → 503.
Retention: clips are hot media (tiered media, see `POST /admin/age`): one is deleted once its footage is older than
`VIDEO_RETENTION_H` (default `HOT_RETENTION_H`, 72 h, when keyframes move to AV1), and the oldest go first beyond
`VIDEO_MAX_BYTES` (20 GB). `DELETE /memories` deletes every clip holding footage in the purged window (the whole clip,
also when it straddles `since`).

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

## `GET /episodes`  (the day as episodes, no query)
Query: `from`/`to` (ISO-8601, optional; unparseable → 400), `k` (int, 1–500, default 50), `device_id` (optional).
Newest first. Episodes come from the frames' `episode_id` (span and size within the window), plus the consolidator's
`kind="episode"` summary token when one has been written (`summary_id` is `null` and `summary` empty until then).
```json
{ "episodes": [ { "episode_id": "...", "device_id": "quest3-01", "start_ts": 1731000000.1, "end_ts": 1731000420.7,
                  "n_frames": 37, "summary": "Made coffee in the kitchen", "entities": ["mug","kettle"],
                  "thumb_url": "/v1/thumbs/....jpg", "summary_id": "..." } ] }
```
Debug / companion-app route: the headset does not call it, so it has no `Dto.cs` counterpart.

## `GET /thumbs/{id}.jpg`
Keyframe thumbnail (JPEG) on every media tier: hot frames are served from their JPEG, cold frames are decoded
from their AV1 chunk (recent decodes are cached). 404 for tokens without an image (`tier="text"`, transcripts).

## `POST /stt`  (headset -> server; voice turns, step 1)
Transcript only: no agent turn, no TTS, so a spoken **voice command** ("open menu", "pause recording") costs one
STT call (~0.4 s live) instead of a full `POST /chat` (Claude + TTS, several seconds and a paid call).
`multipart/form-data`
- `audio`: the same clip `POST /chat` accepts (16 kHz mono WAV from the headset / browser panel; whatever the
  configured STT provider decodes, e.g. m4a with OpenAI). Max 25 MB.
- `command_hints` (optional): the client's command phrases, as a JSON array of strings (`["open menu","close menu",
  ...]`) or one list separated by `|` / newlines; at most 200. When sent, the server matches the transcript against
  them (below) so every client shares one matcher.
Response `200 OK`:
```json
{ "transcript": "pause recording", "duration_s": 1.9, "model": "openai:gpt-transcribe",
  "command": { "phrase": "pause recording", "score": 1.0 } }
```
- `transcript`: trimmed text; `""` for silence (then the client should do nothing). `duration_s`: clip length when
  the upload is a parseable WAV, else `null`. `model`: what transcribed it (`"mock"` without keys).
- `command`: `null` when no `command_hints` were sent or none matched. Matching normalizes both sides like the
  headset's `VoiceCommands.Normalize` (lower-case, letters/digits only, filler words `please hey ok okay megamind
  can you could would the a my now up` dropped), then an exact match scores `1.0`, else the closest hint by
  `difflib` ratio with `score >= 0.8` ("paws recording" -> `pause recording`, 0.89) among hints with the same
  number of words, each word a likely mishearing of the hint's (same first letter or one clipped leading letter,
  similar spelling), so `start recording` never matches `stop recording` and `unmute` never matches `mute`. A transcript longer than 6
  words after normalization is a question, never a command. `phrase` is the hint as the client sent it.
- Errors: not multipart → 415; missing/empty `audio` or bad `command_hints` → 400; > 25 MB → 413; STT provider
  failure → 502 `{"detail": "STT failed: ..."}`.

Recommended voice flow (the headset, `ChatClient.cs`):
1. `POST /stt` with the clip (+ `command_hints`). Show `transcript` as the user's bubble immediately.
2. `command` set (or a local match on `transcript`) → run the command; done (no chat request).
3. Otherwise `POST /chat/stream` as **JSON** `{"text": transcript, ...}` (same session) and consume the SSE events.
Net effect: commands react in ~0.5 s; questions show their transcript before retrieval starts and never pay for
STT twice. Sending the clip straight to `POST /chat` (multipart) still works and is the fallback on a 404/405
from an older server.

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
  "tool_calls": [ { "name": "recall_memory", "args": { "query": "keys", "k": 12 }, "confidence": "high" } ],
  "device_id_used": "quest3-abc", "warning": null }
```
`device_id` (optional) limits the search to that device's memories; omitted/null = all devices.
`device_id_used` is the device that was actually searched (`null` = all devices). `warning` is `null` normally;
when the request's `device_id` has **no** memory tokens while other devices do, it says so instead of the answer
silently being "no memory", e.g. `"device 'browser-x' has no memories; active devices: quest3-1"` (up to 5
devices, most recent first, see `GET /devices`). Clients should show it.
`CHAT_DEFAULT_DEVICE=latest` (server env, default off; a demo convenience for "type a question on the laptop
while the headset streams"): in that same case the turn searches the most recently active device instead, and
reports it: `"device_id_used": "quest3-1"`, `"warning": "device 'browser-x' has no memories; answered from
'quest3-1' instead"`. A device that has memories is never redirected.
`replay_url` is optional and `null` when the server holds neither buffered frames, kept keyframes with pixels nor
stored real video (`POST /video`) around the citation (pixels aged out, replay disabled, or a different device). It covers `ts - 5 s .. ts + 5 s`.
`cited[].thumb_url` is `""` when the cited token has no image (heard speech, or a frame whose pixels aged out,
`tier="text"`): show the caption, skip the thumbnail.
For questions about the present ("what am I looking at right now", "what did I just see") the evidence also
includes live short-term memory (see `GET /live/{id}.jpg`): a cited raw frame has an id starting `now_`, a
caption starting `LIVE NOW-<n>:` and `thumb_url` `/v1/live/{id}.jpg`. Treat `thumb_url` as an opaque URL.

## `POST /chat/stream`
The same turn as `POST /chat` (same request body, JSON or multipart, same 400/415/422 errors before the stream
starts), answered as Server-Sent Events (`text/event-stream`) so clients can show candidates and answer text
early. Each event is `event: <name>` + one `data:` line of JSON:

| event | data | when |
|---|---|---|
| `retrieval` | `{ "confidence": "high", "candidates": [Citation, ...] }` | after each memory search, before the answer (cited first, then the other retrieved moments; `thumb_url` / `replay_url` as in `cited`) |
| `delta` | `{ "text": "On the kitchen" }` | answer text as Claude generates it; append (the `<cites>` trailer is never sent) |
| `reset` | `{}` | discard the text streamed so far **and any `audio` clips received so far** (that model round turned into another search, or Claude failed and the caption fallback answers) |
| `audio` | `{ "seq": 0, "url": "/v1/audio/01J...-0.wav", "text": "On the kitchen counter next to the coffee machine." }` | TTS of one sentence of the answer, as soon as that sentence is rendered, while Claude is still writing the rest (`TTS_STREAM`, default on). `seq` is 0, 1, 2, ... in sentence order, each sent exactly once and always in that order (a sentence whose TTS failed is skipped: `seq` can then have gaps): queue the clips and play them back to back, starting on `seq` 0. `text` is the sentence the clip speaks. Never sent with the mock TTS provider |
| `final` | exactly the `POST /chat` response body | last event of a successful turn; authoritative (`answer`, `cited`, `tool_calls`, `audio_url`). All `audio` events of the turn precede it; `audio_url` is the whole answer as one file (the clips joined, or one fresh TTS call when a clip failed), for clients that don't play the clips, or `null` with the mock provider |
| `error` | `{ "detail": "RuntimeError: ..." }` | last event when the turn fails after the stream started (no `final`) |

```
event: retrieval
data: {"confidence": "high", "candidates": [{"id": "...", "ts": 1731000000.1, "caption": "...", "thumb_url": "/v1/thumbs/....jpg", "replay_url": null}]}

event: delta
data: {"text": "On the kitchen counter"}

event: audio
data: {"seq": 0, "url": "/v1/audio/01J...-0.wav", "text": "On the kitchen counter next to the coffee machine."}

event: delta
data: {"text": " You put them there about"}

event: audio
data: {"seq": 1, "url": "/v1/audio/01J...-1.wav", "text": "You put them there about 20 minutes ago."}

event: final
data: {"transcript": "where did I leave my keys?", "answer": "On the kitchen counter ...", "audio_url": "/v1/audio/01J....wav", "cited": [...], "tool_calls": [...]}
```
`device_id_used` / `warning` arrive only in `final` (same rules as `POST /chat`).
**Audio**: play each `audio` clip as it arrives (`GET url` with the bearer token, same as `audio_url`), in `seq`
order, one after the other; a client that only knows `audio_url` keeps working unchanged (it just waits for
`final`, exactly as before). Speech starts after the first sentence of the answer instead of after the whole
answer plus one TTS call for all of it. `POST /chat` is unchanged: no clips, it blocks until `audio_url` is ready.
Server env `TTS_STREAM=false` turns the clips off (one TTS call after the turn, `final.audio_url` as before).
Clients can swap it in for `POST /chat` and fall back to `POST /chat` on 404/405 (older servers). The model
requests are identical to `POST /chat` apart from streaming, so answers are the same. With `RECALL_FIRST` a
standalone question's `retrieval` arrives before Claude is called at all.

## `GET /devices`
Every device that has stored memory tokens, most recently active first (by `last_ts`). Use it to pick the
`device_id` for `POST /chat` (the browser panel and `tools/ask.py` do).
```json
{ "devices": [ { "device_id": "quest3-1", "count": 1500, "last_ts": 1731000300.0,
                 "last_frame_ts": 1731000300.0, "last_audio_ts": 1731000290.0 },
               { "device_id": "browser-ab12c", "count": 4, "last_ts": 1730990000.0,
                 "last_frame_ts": 1730990000.0, "last_audio_ts": null } ] }
```
`count` covers every token kind (frames, transcripts, episode summaries). `last_frame_ts` is the newest
`kind="frame"` token and `last_audio_ts` the newest `kind="transcript"` token (heard speech: `POST /audio` or
`FrameMeta.transcript`; a merged transcript token keeps its start `ts`); `null` when the device has none. Only
stored (enriched) tokens count: frames still in the enrichment queue or only in the replay buffer do not.
`{ "devices": [] }` on an empty store.

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
  `buffer` (every received frame), `keyframes`, or `video`.
- Real video (`POST /video`) is preferred when the stored clips cover at least `VIDEO_MIN_COVERAGE` (0.5) of the
  window's footage extent (earliest to latest footage of any kind inside `[from, to]`): the window is cut from the
  clips (frame-exact, re-encoded H.264 at up to `VIDEO_REPLAY_FPS` (30) and `REPLAY_MAX_WIDTH`), gaps between clips
  hold the last frame, and offset 0 is the first clip footage in the window (`X-Replay-Start`). Then
  `X-Replay-Source: video`, `X-Replay-Clips` = clips used, `X-Replay-Frames` = rendered frames. Never while
  `PRIVACY_BLUR_FACES(_VLM)` is on (video is not blurred). Otherwise, or if cutting fails, the frame path above runs
  unchanged. With stored video, the default `to` is the newest footage of either kind (newest buffered frame or end
  of the newest stored clip).
- `PRIVACY_BLUR_FACES(_VLM)` on: every frame in the clip (buffered, hot JPEG or cold AV1 keyframe) gets the same
  bystander face blur as stored thumbnails before rendering (each frame is blurred once and reused across clips,
  `REPLAY_BLUR_CACHE_BYTES`, 64 MiB), and stored video is never used.
- 404 when neither buffered nor kept frames nor stored video exist in the window; 503 when ffmpeg is not available on
  the server.
Rendered clips are cached; `DELETE /memories` also purges the matching buffered frames, stored video and the clip cache.

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
- Each clip shows the moment's local time and a short title burned in, with 0.3 s fades. `source="video"`: cut from
  real headset video (`POST /video`) when stored clips cover at least `VIDEO_MIN_COVERAGE` of the clip (rendered at
  `MONTAGE_FPS`). `source="replay"`: the frames buffered for `GET /replay.mp4` around the moment (real frame rate).
  `source="timelapse"`: the stored keyframe thumbnails from that moment on, each held ~1 s (a ~1 fps timelapse;
  used once the moment is older than the replay window and has no stored video, and always when
  `PRIVACY_BLUR_FACES` is on: real video and buffered frames are not blurred).
Response:
```json
{ "id": "3f2a9c0d1e5b7a8c6d4e", "url": "/v1/montages/3f2a9c0d1e5b7a8c6d4e.mp4", "title": "Your highlights",
  "narration": "Here's your highlight reel: 2 moments from that period.", "planner": "offline",
  "clips": [ { "token_id": "...", "ts": 1731005000.2, "title": "Team lunch at the long table",
               "duration_s": 4.0, "source": "timelapse", "thumb_url": "/v1/thumbs/....jpg" } ] }
```
404 when nothing matches or no chosen moment has video/frames/thumbnails; 503 without ffmpeg (same lookup as
replay, `REPLAY_FFMPEG`); 404 when `MONTAGE_ENABLED=false`; 409 when `DELETE /memories` ran while it was
being made (nothing is kept; POST again). The same plan always yields the same `id`.

### Content-filtered ("smart") montages
"Montage of all shots of the guy in the black coat": every clip must show the query's subject. Additive request
field `mode`: `""`/`"auto"` (default; `MONTAGE_SMART` decides), `"smart"`, `"classic"` (the highlight reel above).
- Subject: the query minus its framing ("make a montage of all shots of") and any time phrase. A query with no
  content words ("my afternoon", "make me a highlight reel") has none, and a request without a query never does:
  those always take the classic path, exactly as before.
- Auto (`MONTAGE_SMART=auto`): smart iff there is a subject AND the query asks for a collection ("all", "every
  time", "whenever", "shots/clips/photos of", "supercut of" ...; not "all day" / "every morning"). Other queries
  with a subject ("what happened at work", "montage of my trip to Paris") keep the classic highlight reel, with or
  without an API key; `mode:"smart"` opts them in. `MONTAGE_SMART=off`: only `mode:"smart"` opts in.
  `mode:"classic"` always opts out.
- Candidates: the retriever's top `2 x MONTAGE_SMART_MAX_CANDIDATES` hits for the subject across all history
  (or `time_from`/`time_to`, or a time phrase in the query), spread so one long visit can't fill the pool, cut to
  `MONTAGE_SMART_MAX_CANDIDATES` (60).
- Verification: Claude `MONTAGE_VERIFY_MODEL` (`claude-haiku-5-5`) looks at each candidate's caption, OCR text and
  thumbnail, `MONTAGE_VERIFY_BATCH` (8) per call, and keeps matches with confidence >= `MONTAGE_VERIFY_MIN_CONFIDENCE`
  (0.5). Without `ANTHROPIC_API_KEY` (or `MONTAGE_VERIFIER=offline`), or when every verifier call failed,
  candidates are kept unverified in retrieval order (`planner:"offline"`). Once any batch was checked, only
  confirmed matches are used; if some calls failed and the rest found nothing: **503** with a speakable `detail`
  ("I couldn't finish checking ...; please try again").
- Clips: matches within `MONTAGE_MOMENT_GAP_S` (90 s) of each other are one moment; one clip per moment (its most
  confident token), chronological, at most `MONTAGE_SMART_MAX_CLIPS` (12, or `max_clips` if smaller), evenly
  spread in time when there are more. Clip sources/render/caching are the same as above.
- No match: **404** with a speakable `detail`, e.g. `"I couldn't find any moments matching the guy in the black
  coat in your memories (checked 37 candidate moments), so there's no montage to make."`

Additive response fields (classic defaults in brackets): `mode` (`"classic"`), `subject` (`""`), `candidates`
(`0`, retriever candidates checked), `matches` (`0`, moments the verifier confirmed). `planner` is `"claude"` when
the verifier checked the clips, `"offline"` when they are unverified retrieval hits.
```json
{ "id": "9b1c...", "url": "/v1/montages/9b1c....mp4", "title": "The guy in the black coat",
  "narration": "Found 4 moments matching the guy in the black coat across 2 days.", "planner": "claude",
  "mode": "smart", "subject": "the guy in the black coat", "candidates": 60, "matches": 4, "clips": [ ... ] }
```

## `GET /montages/{id}.mp4`
The rendered montage (H.264, `MONTAGE_WIDTH`x`MONTAGE_HEIGHT`, default 960x540). `GET` or `HEAD`; supports
`Range: bytes=...` (206 / 416) exactly like `/replay.mp4`. Montages are a bounded cache (newest
`MONTAGE_CACHE_MAX_FILES`, 16): 404 once evicted, so POST again. `DELETE /memories` clears every montage.
Not a Claude tool: chat answers do not include a montage (call this endpoint from a "show me" UI action).

## `GET /audio/{id}.wav`
TTS output for a chat response: `final.audio_url` (the whole answer) or a `/chat/stream` `audio` clip (`{id}-{seq}.wav`, one sentence).

## `DELETE /memories?device_id=...&since=...`
Purge memory tokens (rows **and** thumbnails; frames already aged into an AV1 chunk are cut out of it by
re-encoding the chunk without them, so the chunk's other frames survive). Stored real video (`POST /video`) holding
footage in the window is deleted too (whole clips). Both params are optional: `device_id` limits the purge to one
device (default: all devices); `since` deletes only tokens with `ts >= since` ("forget the last 2 hours"),
given as ISO-8601 (`2026-11-07T09:00:00Z`; naive = server local time) or unix seconds (`1731000000.0`).
Episode summaries whose `ts` (episode start) falls in the window are deleted too. Invalid, empty/whitespace (`since=`) or non-finite `since` → 400 (omit the param to purge all time).
Returns `{ "deleted": 1234 }`. Offline equivalent: `python -m megamind.store.purge --device --since`.

## `POST /admin/age?hot_older_than_s=...&cold_older_than_s=...`  (ops / demo)
Run one media-aging pass now (the server also runs one every `MEDIA_AGE_INTERVAL_S`, default 3600 s). Frames older
than `cold_older_than_s` (default `COLD_RETENTION_DAYS`, 100 d) drop their pixels (`tier="text"`); hot frames older
than `hot_older_than_s` (default `HOT_RETENTION_H`, 72 h) are re-encoded into one AV1 mp4 per (device, episode)
run (`tier="cold"`) and their JPEGs deleted once the chunk's frame count is verified. Stored real video clips
(`POST /video`) older than `hot_older_than_s` (default `VIDEO_RETENTION_H`, i.e. `HOT_RETENTION_H`) are deleted
(`video_deleted`). Negative / non-finite → 400.
```json
{ "to_cold": 120, "chunks": 1, "jpeg_bytes": 7800000, "chunk_bytes": 820000, "to_text": 0, "aborted": 0,
  "skipped": null, "video_deleted": 0 }
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
  "media": { "encoder": "libsvtav1", "hot_retention_h": 72, "cold_retention_days": 100, "cold_decodes": 12 },
  "video": { "clips": 860, "seconds": 4300.0, "bytes": 1650000000, "devices": 1 } }
```
`video` (stored `POST /video` clips) is absent when `VIDEO_ENABLED=false`.
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
Voice questions (mic button / in-headset *Ask by voice*) use the existing multipart form of `POST /chat` (16 kHz mono
WAV `audio`); the theater shows the response's `transcript` as the question, and an empty `transcript` as "didn't catch that".
Assets are under `/xr/static/`. `XR_ENABLED=false` removes both routes. Quest usage: [quest3-guide.md](quest3-guide.md#8-memory-theater-xr-in-the-quest-browser).
