# Megamind

**An AI assistant on Meta Quest 3 that remembers what you've seen.**

> "Where did I leave my keys?" — *"On the kitchen counter next to the coffee machine, about 20 minutes ago."* (and it shows you the frame)

Megamind continuously samples the Quest 3's passthrough cameras, distils keyframes into
**memory tokens** (caption + OCR + entities + embeddings + thumbnail) and stores them in a
vector database. A **Claude** agent exposes a `recall_memory` tool; when called, relevant
tokens are retrieved and handed to a **Qwen-VL reader** sub-agent that attends over the
concatenated captions *and images* and returns cited evidence. Claude answers; the wearer
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
 │ keyframe sampler  │                └─────────┘  └────────┘  │  (Qwen-VL)    │  └───────┘  │ memory    │
 ├───────────────────┤                                         └───────────────┘             │ tokens    │
 │ AR chat panel     │   audio/text   ┌─────────┐  ┌─────┐  ┌──────────────┐  recall_memory  └─────┬─────┘
 │ push-to-talk, TTS │◀──────────────▶│  /chat  │─▶│ STT │─▶│ Claude agent │◀────────────────────┐ │ top-k
 └───────────────────┘  answer+thumbs └─────────┘  └─────┘  └──────┬───────┘   evidence + cites  │ ▼
                                                                   │ query, k           ┌────────┴────────┐
                                                                   └───────────────────▶│ Qwen-VL reader  │
                                                                                        │ attention over  │
                                                                                        │ memory tokens   │
                                                                                        └─────────────────┘
```

Three streams of work:

| # | Stream | Where | Status |
|---|---|---|---|
| 1 | **Capture → tokenize → store.** Passthrough Camera API → 1 fps JPEG → dHash dedupe → Qwen-VL caption/OCR/entities → text + SigLIP embeddings → LanceDB | `headset/.../FrameUploader.cs`, `server/megamind/ingest`, `server/megamind/store` | filter + worker + in-memory store done; VLM enricher, embeddings, LanceDB are `TODO(stream-1)` |
| 2 | **Retrieval tool + reader.** `recall_memory(query, time_from, time_to, k, mode)` → hybrid search (vector + BM25 + time filter, RRF) → Qwen-VL reads captions + thumbnails → evidence with citations → Claude | `server/megamind/agent/{tools,reader,claude_agent}.py` | tool wiring done; Claude tool-use loop and Qwen-VL reader are `TODO(stream-2)` |
| 3 | **AR chat.** World-locked panel, push-to-talk → STT, spoken reply via TTS, cited thumbnails | `headset/.../{ChatClient,ChatPanel}.cs`, `server/megamind/agent/{stt,tts}.py` | C# stubs written (untested on device); STT/TTS providers are `TODO(stream-3)` |

Key decisions (details and rationale in the design doc §3): one Unity app does both capture
and chat (Horizon OS suspends background apps); a memory token is a *record*, not a learned
vector; Claude orchestrates while Qwen-VL reads; the headset is a thin client; LanceDB for
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

```bash
# 1. server
cd server
python3.12 -m venv .venv            # any Python >= 3.11 && source .venv/bin/activate
pip install -e ".[dev,tools]"
cp .env.example .env                       # optional: add API keys; mocks work without any
uvicorn megamind.main:app --reload --port 8000
curl -s localhost:8000/v1/health

# 2. feed frames from your laptop webcam (or a video file) in a second terminal
python tools/feeder.py --server http://localhost:8000 --source 0 --fps 1

# 3. ask
python tools/ask.py --server http://localhost:8000 "what did I just see?"
curl -s "localhost:8000/v1/memories?q=keys&k=5"
```

With no API keys configured the server runs entirely on deterministic mocks (mock captions,
hashing embeddings, caption-only reader, no STT/TTS), so the whole loop is exercisable on a
laptop. Set `ANTHROPIC_API_KEY`, `READER_API_KEY` (Sail Research serving `Qwen/Qwen3.6-35B-A3B`, or any
OpenAI-compatible endpoint: vLLM on Modal, Together, OpenRouter), `EMBED_PROVIDER`, `STT_PROVIDER`,
`TTS_PROVIDER` to turn on the real thing — see `server/.env.example` and `docs/resources.md`.

### Headset

See [`headset/README.md`](headset/README.md). Requirements: Quest 3 / 3S on Horizon OS v74+
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
