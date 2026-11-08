# Megamind

[![server](https://github.com/debojoti-soumya/megamind/actions/workflows/ci.yml/badge.svg)](https://github.com/debojoti-soumya/megamind/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**An AI assistant on Meta Quest 3 that remembers what you've seen.**

> "Where did I leave my keys?" — *"On the kitchen counter next to the coffee machine, about 20 minutes ago."*
> (and it shows you the frame, and pins an arrow on the keys' last known spot)

Megamind samples the Quest 3 passthrough camera (and, optionally, the mic), distils keyframes into
**memory tokens** (caption + OCR + entities + embeddings + thumbnail) and stores them in LanceDB. A **Claude**
agent answers questions with a `recall_memory` tool: hybrid retrieval finds the relevant tokens and Claude reads
the cited images directly. The answer streams back to a world-locked AR panel and is spoken sentence by sentence.

- Design document: [`docs/megamind-design.pdf`](docs/megamind-design.pdf)
- API contract (headset ⇄ server): [`docs/api.md`](docs/api.md)
- Quest 3 setup and demo runbook: [`docs/quest3-guide.md`](docs/quest3-guide.md)
- Prior art: [`docs/prior-art.md`](docs/prior-art.md) · model/credit choices: [`docs/resources.md`](docs/resources.md)

## How it works

```
 Quest 3 (Unity)                      Server (FastAPI, Python)
 ┌────────────────────┐ JPEG ~1 fps  ┌──────────┐  ┌────────┐  ┌───────────────┐  ┌───────┐  ┌───────────┐
 │ Passthrough camera ├─────────────▶│ /frames  │─▶│ filter │─▶│ caption + OCR │─▶│ embed │─▶│ LanceDB   │
 │ + offline spool    │  WAV chunks  │ /audio   │  │ dHash  │  │ (VLM / local) │  │       │  │ memory    │
 ├────────────────────┤              └──────────┘  └────────┘  └───────────────┘  └───────┘  │ tokens    │
 │ AR chat panel      │ voice/text   ┌──────────────┐  ┌───────────────┐  recall_memory      └─────┬─────┘
 │ push-to-talk       ├─────────────▶│ /stt         │  │ Claude agent  │◀─────────────────────┐    │
 │ sentence TTS       │◀─────────────┤ /chat/stream ├─▶│ (Sonnet)      │  hybrid retrieval    │    │ top-k
 │ collage lightbox   │  SSE: text,  └──────────────┘  │ reads cited   │  vector+BM25+time,   └────┘
 │ object markers     │  audio, cites,                 │ images        │  RRF
 └────────────────────┘  collage, markers              └───────┬───────┘
                                                               │ make_collage, locate (marker box)
```

Features on current `main`:

- **Capture → tokenize → store.** `POST /v1/frames` → keyframe filter (dHash dedupe, blur/luma, pose, rate cap) →
  enricher (any OpenAI-compatible VLM, or keyless SigLIP label + Tesseract OCR) → text + SigLIP image embeddings →
  LanceDB (or an in-memory store). Ambient audio (`POST /v1/audio`) becomes transcript tokens; episodes are
  consolidated into summary tokens. Code: `server/megamind/{ingest,store}`.
- **Retrieval.** `server/megamind/retrieval/hybrid.py`: text-vector + BM25 + image-vector lists fused with weighted
  RRF plus a recency prior and time filters. Alternatives behind `RETRIEVER_BACKEND` (`hier`, `ucc`, `combo`).
- **Reader.** With `ANTHROPIC_API_KEY` set (`READER_PROVIDER=auto`) Claude reads the cited snapshots itself
  (`ClaudeVisionReader` in `server/megamind/agent/reader.py`). An OpenAI-compatible VLM reader (e.g. Gemma 4 31B on
  Sail Research, or Qwen-VL on vLLM) is the option without an Anthropic key; without either it is captions-only.
- **Answers.** `POST /v1/chat` and streamed `POST /v1/chat/stream` (SSE; each sentence is spoken via TTS as Claude
  finishes it). `POST /v1/stt` transcribes voice turns first (the bubble shows at once) and handles voice commands without Claude.
- **Photo collage.** Claude's `make_collage` tool (and `POST /v1/collage`) builds a verified collage of matching
  moments; the headset opens it in a Lightbox.
- **Object markers.** For "where is X" questions `server/megamind/agent/locate.py` boxes the object in a cited frame
  (`ChatResponse.markers`); headset `ObjectMarkers.cs` pins an arrow and a pulsing golden halo in the room.
- **Replay and montage.** `GET /v1/replay.mp4`, `POST /v1/montage` (incl. content-filtered "smart" montages).
- **Browser clients.** `/ui` is a browser version of the AR panel (webcam as passthrough, hold Space to talk);
  `/xr` is Memory Theater, a WebXR view of memories.
- **Privacy.** Optional bystander face blur, `DELETE /v1/memories` purge, hot JPEG → cold AV1 → text-only retention.

Without API keys the server runs on deterministic mocks, so the whole loop works on a laptop.

## Quickstart

### Server (local)

Python >= 3.11 (a 3.10 venv makes `pip install` hang).

```bash
git clone https://github.com/debojoti-soumya/megamind.git && cd megamind/server
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,tools]"     # or with uv: uv venv && source .venv/bin/activate && uv pip install -e ".[dev,tools]"
cp .env.example .env          # optional; every key may stay empty (mocks)
uvicorn megamind.main:app --reload --port 8000
```

Main `.env` variables (full list with comments in [`server/.env.example`](server/.env.example)):

| variable | purpose |
|---|---|
| `ANTHROPIC_API_KEY`, `CLAUDE_MODEL` | Claude agent + image reader (default `claude-sonnet-5-5`) |
| `OPENAI_API_KEY` | embeddings, STT, TTS, object locator |
| `READER_API_KEY`, `READER_BASE_URL`, `READER_MODEL` | OpenAI-compatible VLM for captions/OCR (and the reader without Claude) |
| `STORE_BACKEND=lancedb` | persist memories in `server/data/lancedb` (default `memory` is lost on restart) |
| `API_TOKEN` | require `Authorization: Bearer <token>` on every route (public deployments) |

Feed frames and ask without a headset (from the repo root):

```bash
source server/.venv/bin/activate
python tools/feeder.py --server http://localhost:8000 --source walk.mp4 --fps 1 --realtime   # or --source 0 (webcam)
python tools/ask.py --server http://localhost:8000 "what did I just see?"
```

Use `--realtime` for video files, otherwise most frames are dropped as `too_soon`. Or open <http://localhost:8000/ui>.

### Server (Modal, public HTTPS)

```bash
pip install modal && modal setup
MEGAMIND_MIN_CONTAINERS=1 modal deploy deploy/modal_app.py      # from the repo root; keeps one container warm
```

Secrets, volumes, rollback and the keyless `MEGAMIND_PROFILE=mock` smoke test: [`deploy/README.md`](deploy/README.md).

### Headset (Quest 3)

1. Install **Unity 6000.0.66f2** with Android Build Support; open `headset/Megamind/` (Meta XR Core SDK / MRUK
   **85.0.0** resolve from `Packages/manifest.json`).
2. Run the menu items **Megamind › 1. Create Capture Scene**, then **Megamind › 1b. Add Chat Panel To Current
   Scene**, then **Megamind › 3. Build APK** (or **4. Build and Run on headset**).
3. Point the app at your server through the config fields `serverBaseUrl`, `deviceId` and `apiToken` (only when the
   server sets `API_TOKEN`), e.g. `adb shell am start ... --es serverBaseUrl https://... --es deviceId quest3-1 --es apiToken <token>`,
   or from the in-app settings panel.

Details, permissions, the offline frame spool and troubleshooting: [`headset/README.md`](headset/README.md).

## Repository layout

```
server/           FastAPI backend: ingest, memory store (LanceDB), retrieval, Claude agent, STT/TTS, /ui, /xr (Python 3.11+)
headset/          Unity 6 project for Quest 3: capture + offline spool, AR chat panel, lightbox, object markers
headset-android/  optional native Kotlin capture app (frames, ambient audio, 30 fps video; no chat UI)
deploy/           Modal app (modal_app.py) plus latency bench and keep-warm scripts
docs/             design doc (LaTeX + PDF), API contract, Quest 3 guide, prior art, resources
tools/            dev/eval scripts: feeder.py (fake headset), ask.py (CLI chat), example-day builders and evals
.github/          CI: ruff + pytest on server/, FrameSpool C# tests, Unity APK build
```

## Development

```bash
cd server && source .venv/bin/activate
ruff check . && ruff format --check .    # lint
pytest -q                                # tests (offline, mocks)
```

Headset and server talk **only** through [`docs/api.md`](docs/api.md): a contract change updates `api.md`,
`server/megamind/models.py` and `headset/.../Dto.cs` in the same PR.

### Contributing

- Open pull requests against `main`; keep each PR focused.
- CI (ruff, pytest, headset tests) must pass before merge.
- Never commit secrets: keys live in `server/.env` (gitignored) or Modal secrets.

## Privacy

Capture is visible (status HUD), storage stays on the server you run, prompts forbid identity claims about people,
and `DELETE /v1/memories` purges everything (including frames still in the pipeline). `PRIVACY_BLUR_FACES=true`
blurs bystander faces in stored thumbnails (`pip install -e ".[privacy]"`). This is a hackathon prototype; do not
wear it where recording isn't welcome.

## Team

Kartik Ramachandrula ([@kartikramachandrula](https://github.com/kartikramachandrula)),
Debojoti Das Soumya ([@debojoti-soumya](https://github.com/debojoti-soumya)),
Asher ([@asherps](https://github.com/asherps)),
Tim Guan ([@ChefAltoids](https://github.com/ChefAltoids)).
Much of the code was written with [Devin](https://devin.ai) agents.

## License

[MIT](LICENSE) © 2026 Megamind contributors.
