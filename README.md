# Megamind

**An AI assistant on Meta Quest 3 that remembers what you've seen.**

> "Where did I leave my keys?" — *"On the kitchen counter next to the coffee machine, about 20 minutes ago."* (and it shows you the frame)

Megamind continuously samples the Quest 3's passthrough cameras, distils keyframes into
**memory tokens** (caption + OCR + entities + embeddings + thumbnail) and stores them in a
vector database. A **Claude** agent exposes a `recall_memory` tool; when called, relevant
tokens are retrieved and handed to a **Gemma 4 31B reader** sub-agent (a vision LLM on Sail
Research's OpenAI-compatible API) that attends over the concatenated captions *and images* and
returns cited evidence. Claude answers; the wearer
hears it and sees the remembered frame on a world-locked AR panel.

- Design document: [`docs/megamind-design.pdf`](docs/megamind-design.pdf) (read this first)
- API contract (headset ⇄ server): [`docs/api.md`](docs/api.md)
- Prior art & what's different: [`docs/prior-art.md`](docs/prior-art.md)
- Hackathon credits/hardware → model choices: [`docs/resources.md`](docs/resources.md)

## How it works

```
 Quest 3 (Unity)                      Server (FastAPI, Python)
 ┌───────────────────┐   JPEG 1 fps   ┌─────────┐  ┌────────┐  ┌───────────────┐  ┌───────┐  ┌───────────┐
 │ Passthrough camera├───────────────▶│ /frames │─▶│ dedupe │─▶│ caption + OCR │─▶│ embed │─▶│ LanceDB   │
 │ keyframe sampler  │                └─────────┘  └────────┘  │  (Gemma 4)    │  └───────┘  │ memory    │
 ├───────────────────┤                                         └───────────────┘             │ tokens    │
 │ AR chat panel     │   audio/text   ┌─────────┐  ┌─────┐  ┌──────────────┐  recall_memory  └─────┬─────┘
 │ push-to-talk, TTS │◀──────────────▶│  /chat  │─▶│ STT │─▶│ Claude agent │◀────────────────────┐ │ top-k
 └───────────────────┘  answer+thumbs └─────────┘  └─────┘  └──────┬───────┘   evidence + cites  │ ▼
                                                                   │ query, k           ┌────────┴────────┐
                                                                   └───────────────────▶│ Gemma 4 reader  │
                                                                                        │ attention over  │
                                                                                        │ memory tokens   │
                                                                                        └─────────────────┘
```

Three streams of work:

| # | Stream | Where | Status |
|---|---|---|---|
| 1 | **Capture → tokenize → store.** Passthrough Camera API → 1 fps JPEG → dHash dedupe → VLM (Gemma 4 31B) caption/OCR/entities → text + SigLIP embeddings → LanceDB | `headset/.../FrameUploader.cs`, `server/megamind/ingest`, `server/megamind/store` | filter + worker + in-memory store done; VLM enricher, embeddings, LanceDB are `TODO(stream-1)` |
| 2 | **Retrieval tool + reader.** `recall_memory(query, time_from, time_to, k, mode)` → hybrid search (vector + BM25 + time filter, RRF) → Gemma 4 31B reads captions + thumbnails → evidence with citations → Claude | `server/megamind/agent/{tools,reader,claude_agent}.py` | done: Claude tool-use loop, Gemma 4 reader (6 s deadline, then caption-only), `server/tools/eval_agent.py` |
| 3 | **AR chat.** World-locked panel, push-to-talk → STT, spoken reply via TTS, cited thumbnails | `headset/.../{ChatClient,ChatPanel}.cs`, `server/megamind/agent/{stt,tts}.py` | C# stubs written (untested on device); STT/TTS providers are `TODO(stream-3)` |

Key decisions (details and rationale in the design doc §3): one Unity app does both capture
and chat (Horizon OS suspends background apps); a memory token is a *record*, not a learned
vector; Claude orchestrates while an open VLM (Gemma 4 31B) reads; the headset is a thin client; LanceDB for
zero-infra storage; voice-first UI.

## Repository layout

```
docs/       design doc (LaTeX + PDF), api.md, prior-art.md, resources.md
server/     FastAPI backend — ingest, memory store, agent (Python 3.11+)
tools/      feeder.py (webcam/video → /v1/frames), ask.py (CLI chat)
headset/    Unity project for Quest 3 — setup README + C# scripts
.github/    CI: ruff + pytest on server/
```

## Quick start (no headset required)

Needs Python >= 3.11. If `python --version` is older, create the venv with a newer interpreter
instead: `python3.12 -m venv .venv` or `~/.pyenv/versions/3.12.x/bin/python -m venv .venv`.

**Terminal 1: server** (from `server/`)

```bash
git clone https://github.com/debojoti-soumya/megamind.git && cd megamind/server
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,tools]"
cp .env.example .env        # optional: API keys; everything runs on mocks without any
uvicorn megamind.main:app --reload --port 8000
```

**Terminal 2: feed frames, query, ask** (from the repo root; `tools/` lives there, not in `server/`)

```bash
source server/.venv/bin/activate
curl localhost:8000/v1/health

# a video file (--realtime stamps frames with their position in the video), or --source 0 for the webcam
python tools/feeder.py --server http://localhost:8000 --source walk.mp4 --fps 1 --realtime

curl "localhost:8000/v1/memories?q=frame&k=5"
python tools/ask.py --server http://localhost:8000 "what did I just see?"   # prints cited ids + thumbnail URLs
python tools/ask.py --server http://localhost:8000 "frame"                  # mock mode: see below
curl -o t.jpg localhost:8000/v1/thumbs/<id>.jpg
curl -X DELETE "localhost:8000/v1/memories?device_id=feeder"
```

Gotchas:

- **`--realtime` for video files.** Without it, the feeder sends frames as fast as it can decode them and stamps them
  all with the current time, so the keyframe filter (`FILTER_MIN_GAP_S=2`) drops ~90% as `too_soon` (shown in the feeder's `last=` field). The webcam
  (`--source 0`) is paced in real time and doesn't need it.
- **Mock captions.** With no `READER_API_KEY` the enricher writes `mock caption for frame …`, so `q=keys` finds
  nothing, and without `ANTHROPIC_API_KEY` "what did I just see?" answers "I don't have a memory of that". Search
  or ask `frame` to see the loop work, or configure real keys (below) to get searchable captions.
- **Persistence.** The default `STORE_BACKEND=memory` loses everything on restart. Set `STORE_BACKEND=lancedb` in
  `server/.env` to keep memories in `server/data/lancedb` (gitignored).
- **Timestamps in URLs.** Use `Z`, not `+00:00`, e.g. `since=2026-11-07T18:00:00Z`. An unencoded `+` decodes to a
  space and the server returns 400.
- **Episodes.** An episode gets its `kind="episode"` summary token only once it has >= 3 frames, its last frame is
  `EPISODE_GAP_S=120` s in the past, and the consolidator has run (every `CONSOLIDATE_INTERVAL_S=600` s). For a demo,
  set both low in `server/.env` (e.g. `EPISODE_GAP_S=20`, `CONSOLIDATE_INTERVAL_S=30`). With `--realtime`, a
  video's frames are stamped forward from when the feeder started, so wait until the video's length has passed.

With no API keys configured the server runs entirely on deterministic mocks (mock captions,
hashing embeddings, caption-only reader, no STT/TTS), so the whole loop is exercisable on a
laptop. Set `ANTHROPIC_API_KEY`, `READER_API_KEY` + `READER_BASE_URL`, `EMBED_PROVIDER`, `STT_PROVIDER`,
`TTS_PROVIDER` to turn on the real thing — see `server/.env.example` and `docs/resources.md`.

**Reader.** Default: Sail Research (`READER_BASE_URL=https://api.sailresearch.com/v1`, OpenAI-compatible)
serving `READER_MODEL=google/gemma-4-31B-it` with image input; `READER_FALLBACK_MODEL=nvidia/Gemma-4-31B-IT-NVFP4`
takes the single retry after a 429/503/timeout. Calls take ~3 s; a turn with Claude is ~5–6 s. Any
OpenAI-compatible endpoint works by changing those two variables. Plan B if Sail is unavailable:
self-host a VLM (e.g. Qwen2.5-VL-7B) on vLLM on Modal and point `READER_BASE_URL` at it.

### Headset

See [`headset/README.md`](headset/README.md), and the [Quest 3 guide](docs/quest3-guide.md) for first-time
Quest users (connections, sideloading, test runbook, demo-day card). Requirements: Quest 3 / 3S on Horizon OS v74+
with developer mode, Unity 2022.3 LTS or Unity 6, Meta XR Core SDK v74+, and Meta's
[Passthrough Camera API samples](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples).
Build and run Meta's *unmodified* sample first; only then add our scripts.

## Development

```bash
cd server && source .venv/bin/activate
ruff check . && ruff format --check .      # lint
pytest -q                                  # tests (offline, mocks)
```

Rules:
- Headset and server communicate **only** through `docs/api.md`. A contract change is one PR
  that updates `api.md`, `server/megamind/models.py` and `headset/.../Dto.cs` together.
- Everything server-side must be testable with `curl` / the feeder, without hardware.
- Every `TODO(stream-N)` marks work owned by that stream; keep interfaces (`MemoryStore`,
  `Enricher`, `Reader`, `STT`, `TTS`, `TextEmbedder`, `ImageEmbedder`) stable.
- No model training. No on-device inference.

## Demo script (target)

1. Walk around the room wearing the headset for 2–3 minutes; put keys down, glance at a
   whiteboard with text, pick up a coffee mug.
2. "Hey, where did I leave my keys?" → answer + thumbnail of the counter.
3. "What did the whiteboard say?" → OCR'd text read back.
4. "What was I doing about five minutes ago?" → time-filtered recall.

## Privacy

Capture is visible (status HUD), storage is local to the server you run, nothing identifies
people (prompts forbid identity claims), and `DELETE /v1/memories` purges everything.
This is a hackathon prototype; do not wear it where recording isn't welcome.

## License

MIT (to be added).
