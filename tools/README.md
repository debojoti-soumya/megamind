# tools/

| script | what it does |
|---|---|
| `feeder.py` | Acts as the headset: pushes webcam or video frames to `POST /v1/frames` |
| `ask.py` | CLI chat client for `POST /v1/chat`; without `--device-id` it asks the most recently active device from `GET /v1/devices` and prints which one (`--device-id ""` = all devices) |
| `make_example_day.py` | Builds a realistic head-cam day (CC photos + TTS dialogue) and its gold questions |
| `eval_example_day.py` | Asks a running server the example day's questions and scores answers and citations |
| `retrieval_bench.py` | Retrieval benchmark on ~900 real photos plus sign/speech facts: R@1/R@5/MRR per signal, through `HybridRetriever` |
| `egolife_eval.py` | Long real-glasses eval: EgoLife (6.9 h of Meta Aria video per day) and EgoLifeQA, scored with time windows |

Install the server first: `cd server && pip install -e ".[dev,tools]"`. Run every command from the repo root.

## Example day (realistic eval data)

The offline evals (`server/tools/eval_{retrieval,agent}.py`) run on hand-written captions. The example day
runs the real pipeline instead: real photos go through the feeder, filter, captioner and agent.

It is 11 scenes between 07:42 and 19:36. Each scene is a slow, shaky head-cam pan over Creative Commons
photos, with dialogue spoken by macOS `say` voices. The day includes coffee, keys and a wallet on a bedroom
rug, a bus stop, meeting Priya (who leads the design team), a whiteboard plan, ramen with Marcus, a second
coffee, the wallet going into a desk drawer, and a car park "Level 3" sign that nobody mentions. It ends
with keys on the hallway table and a relaxing evening. The people are fictional, and the photo credits are
written to `CREDITS.md` next to the output.

```bash
# 1. generate (macOS only: needs `say`, ffmpeg and internet; ~2 min; ~25 MB, never committed)
python tools/make_example_day.py                 # -> server/data/example_day/ (gitignored)
#    --date YYYY-MM-DD (default today)   --tz Area/City (default: this machine's zone; match the server)

# 2. feed it to a running server (uvicorn megamind.main:app --port 8000)
SERVER=http://localhost:8000 sh server/data/example_day/feed.sh

# 3. score it (live server; real answers need ANTHROPIC_API_KEY / READER_API_KEY on the server)
python tools/eval_example_day.py --gold server/data/example_day/example_day.json
python tools/eval_example_day.py --gold server/data/example_day/example_day.json --modality visual
```

The generator writes `scenes/*.mp4`, `scenes/*.srt` (what was said in each clip), `example_day.json`,
`feed.sh`, `scenes.json` and `CREDITS.md`.

- **Timestamps.** `feed.sh` passes `--start <epoch>` (each scene's clock time) and `--srt` (its dialogue)
  to `feeder.py` when the feeder supports those flags. Otherwise every frame is stamped "now" and the
  dialogue is skipped. Questions that depend on the time of day ("this morning", "this afternoon") need
  real timestamps, a matching `--tz`, and a server clock after 19:36 on `--date`.
- **Feed a fresh server.** The keyframe filter keeps per-device state in memory and drops any frame
  older than the device's last kept frame as `too_soon`. Feeding the day a second time, or after the
  same device id was fed "now" frames, keeps almost nothing, and `DELETE /v1/memories` does not reset
  this. Restart the server before each feed (or edit `--device-id` in `feed.sh` and pass `--device-id` to the scorer).
- **Gold file.** `example_day.json` uses the same schema as `server/tools/data/agent_day.json`. Each
  `stream` entry is one shot, with ground-truth caption, OCR and entities. A few extra keys are ignored
  by the loaders: `at`/`dur_s` (the exact clip window, which the live scorer matches against citation
  timestamps), `said` (the shot's dialogue) and `questions[].modality`:
  - `visual`: answerable from pixels.
  - `speech`: needs transcripts on tokens.
  - `both`: answerable either way.
  - `none`: the agent must decline.
- **Offline.** A copy for 2026-11-08 is committed as `server/tools/data/example_day.json`. The offline
  harnesses run on it as-is:
  `cd server && python tools/eval_agent.py --dataset tools/data/example_day.json` (or `eval_retrieval.py`).
  After editing scenes or questions, regenerate that copy with:
  `python tools/make_example_day.py --gold-only --date 2026-11-08 --tz America/Los_Angeles --gold server/tools/data/example_day.json`.
  `server/tests/test_example_day.py` checks that the copy matches the generator.

## Retrieval benchmarks (ported from PR #54)

The three evals measure different things:

- **The example day** checks end-to-end answers on a small scripted day.
- **`retrieval_bench.py`** gives per-signal ablations on real photos. Use it to tune fusion weights,
  calibration and reranking.
- **`egolife_eval.py`** checks whether retrieval reaches the right moment over hours of real footage.

Neither commits any data. Datasets download at runtime into `~/.cache/megamind/` (or `--data` / `--out`).
COCO, Flickr30k and EgoLife each have their own licence, and EgoLife is gated and research-only.
CI only checks the loaders and the scoring math (`server/tests/test_bench_tools.py`).

### retrieval_bench.py

```bash
pip install -e "server[local-embeddings]"           # + CPU torch; bge-small text, SigLIP image
python tools/retrieval_bench.py --prepare           # COCO val2017 (500 photos) + Flickr 1k-test (300), ~330 MB
python tools/retrieval_bench.py --out run.bench.json
python tools/retrieval_bench.py --weights w.json    # planner weight overrides: {"activity": {"image": 0.5}}
python tools/retrieval_bench.py --text mock --image mock --store memory --limit 40   # plumbing only, no torch
```

The corpus is about 900 tokens in one store, so every question competes with every token:

- COCO photos (object counts, "where did I leave my X")
- Flickr photos (scene descriptions)
- synthetic signs (exact, paraphrased and OCR-noisy text, each with hard negatives)
- speech facts stored as `kind="transcript"` tokens, as `/v1/audio` does
- parking signs on different days, for time-window questions

There are two configs:

- `keyless`: photos carry the `MockEnricher` caption, which is what main stores today without an enricher.
- `captioned`: one human caption is stored as if the VLM had written it, and the question uses a different one.

The signals are:

- `bm25`, `text_vec`, `image_vec`: the store primitives alone.
- `hybrid`: `HybridRetriever.retrieve`, the `recall_memory` path, with neighbour expansion off so the ranking
  itself is scored.
- `hybrid_nocal`: the same with the image embedder's SigLIP calibration hidden. It only appears once
  calibrated image fusion (port PR 1) is merged.
- `hybrid_fp16`: the same over a float16 LanceDB table. It only appears when `LanceDBStore` takes
  `vector_dtype` (port PR 1).
- `store_search`: `MemoryStore.search`, the `/v1/memories` path.

`--reader vlm` adds a reading stage with main's `VLMReader`, which needs `READER_API_KEY`. It compares counts and
"where" answers on the retrieved top 3 against the right frame alone.

The JSON report records:

- the embedders and SigLIP calibration
- the planner weights per query type
- the query types the planner chose per category
- the git revision

Results are not comparable to the numbers posted on PR #54, because the filter, enricher, fusion and weights
differ. Record a baseline on main before tuning anything.

### egolife_eval.py

```bash
export HF_TOKEN=...   # accept the EgoLife terms on huggingface.co first
python tools/egolife_eval.py fetch --person A1_JAKE --day DAY1      # prints where clips / SRTs / QA landed
python tools/egolife_eval.py extract --clips <clip dir>              # 1 fps JPEGs (ffmpeg)
# fresh server (the keyframe filter is per-device and in memory), STORE_BACKEND=lancedb:
python tools/egolife_eval.py ingest --srt-dir <Transcript_HH.srt dir> --out ingest.bench.json
python tools/egolife_eval.py ingest --audio --clips <clip dir>      # or: speech via POST /v1/audio
python tools/egolife_eval.py eval --qa <EgoLifeQA_A1_JAKE.json> --lancedb server/data/lancedb --out egolife.bench.json
# server stopped: in-process HybridRetriever, A/B the port PRs
python tools/egolife_eval.py eval --qa ... --offline --lancedb server/data/lancedb --calibrate off
python tools/egolife_eval.py eval --qa ... --offline --lancedb server/data/lancedb --speech-link 0
python tools/egolife_eval.py reepisode --lancedb server/data/lancedb --dry-run   # after changing EPISODE_*
```

**How a question is scored.** A question is a hit at ±W s when a returned memory lies within W seconds of
its target time, searching only memories from before `query_time`. The default windows are 30, 120 and 600 s.

**Baselines.** With `--lancedb` there are two baselines that need no retrieval:

- `recent`: the k newest memories.
- `chance`: k random earlier memories.

**Speech with `--srt-dir`.** Each subtitle cue goes out once, as `FrameMeta.transcript`. Servers without the
speech port PR ignore the field, so there `--audio` is the way to get speech in.

**Offline eval.** `--offline` builds main's retriever from the server's env with `RETRIEVER_BACKEND=hybrid` and
the planner clock in UTC (the eval calendar maps EgoLife wall-clock times onto UTC). `--calibrate default` follows
`SEARCH_CALIBRATE_IMAGE` like the server does; `on`/`off` force it.

**Waiting for enrichment.** Ingest waits on `/v1/stats.worker_queue` when the server has it (tiered-media port
PR). Otherwise it waits until `/v1/health.store_count` stops changing.

**The `fetch` defaults.** The `allow_patterns` are a best guess at the dataset layout. Files are found by name
afterwards. If nothing matches, pass `--pattern` (repeatable).

### feeder.py --start / --srt

```bash
python tools/feeder.py --source walk.mp4 --start 1767600000 --srt walk.srt
```

- **`--start`** stamps the video's first frame with that unix time and implies `--realtime`.
- **`--srt`** sends each cue once, as `FrameMeta.transcript`, on the first sampled frame at or after the cue
  starts.

The example day's `feed.sh` detects both flags automatically.
