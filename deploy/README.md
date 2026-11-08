# Megamind on Modal (public HTTPS)

`modal_app.py` serves the FastAPI server (`megamind.main:create_app`) from one CPU container behind a public
`https://<workspace>--megamind-web.modal.run` URL. The headset, the browser panel (`/ui`) and Memory Theater (`/xr`)
talk to it from any network: no hotspot, Tailscale or firewall rules. Quest setup: [quest3-guide.md](../docs/quest3-guide.md) option F.

| | |
|---|---|
| Container | CPU only (1 vCPU, 2 GiB), `min_containers=0`, `max_containers=1`, `scaledown_window=600` (10 min idle, then 0 cost) |
| Why one container | LanceDB and the in-process workers (enrichment, consolidator, UCC sync, media aging) are single-writer. `@modal.concurrent(max_inputs=64)` lets that one container serve the headset uploads, the panel and SSE streams at once. |
| State | Modal Volume mounted at `/data`: `lancedb/`, `thumbs/` (+ sibling `chunks/`, replay/montage caches), `audio/`, `hier_notes.json` (recall notes, live profile). Memories survive `modal app stop`, redeploys and scale-to-zero. |
| Persistence | Modal commits Volume writes in the background and on container exit; the app also calls `volume.commit()` after the workers stop in the lifespan shutdown. Verified: memories + thumbnails were still there after `modal app stop` + redeploy. |
| ffmpeg | Debian bookworm apt `ffmpeg` 5.1.x (verified `5.1.9` in the container; `-fps_mode` needs ≥ 5.1, #114) |
| Auth | `API_TOKEN` from the `megamind-auth` secret. Required: the URL is public. See [api.md#auth](../docs/api.md#auth). |
| Profiles | `MEGAMIND_PROFILE=live` (default): OpenAI embeddings/STT/TTS, Sail reader, keys from `megamind-providers`. `MEGAMIND_PROFILE=mock`: every provider mocked (recall = `hybrid`), no paid calls, no provider secret needed. Each profile has its own Volume (`megamind-data` / `megamind-data-mock`) because LanceDB fixes the embedding dimension per table. |

## One-time setup

```sh
pip install modal && modal token new          # or export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET

# shared secret for the public URL (keep a copy: it goes into megamind.json and the panel's settings)
modal secret create megamind-auth API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')" --force

# provider keys (live profile only) - placeholders, paste your own; omit any you don't have
modal secret create megamind-providers \
  ANTHROPIC_API_KEY=<anthropic-key> \
  OPENAI_API_KEY=<openai-key> \
  READER_API_KEY=<sail-key> \
  UCC_API_KEY=<ucc-key> --force
```

Modal only shows secret values once; to read the token back later, recreate it (`--force`) and update the clients.
The Volumes are created on first deploy (`create_if_missing=True`).

### Recall backend

The live profile recalls with `RETRIEVER_BACKEND=hier` (chosen by the hybrid / UCC / hier bake-off and model sweep
in #152, `server/tools/eval_retrieval.py --backend all` / `--hier`): the local hybrid (vector + BM25 + OCR +
temporal) proposes 30 candidates, then two OpenAI calls run: `HIER_MODEL=gpt-4.1-mini` picks the right 10-minute
blocks from per-block notes, and `HIER_RANK_MODEL=gpt-4.1` ranks the moments inside them. ~1.9 s p50 / 2.3 s p95 per
recall, ~$0.006 per question; the notes cost ~$0.04 per recorded day (`HIER_NOTE_MODEL=gpt-4.1-mini`).

| Setting | Value | Why |
|---|---|---|
| `RETRIEVER_BACKEND` | `hier` (live), `hybrid` (mock) | mock must make zero paid calls |
| `HIER_MODEL` / `HIER_RANK_MODEL` / `HIER_NOTE_MODEL` | `gpt-4.1-mini` / `gpt-4.1` / `gpt-4.1-mini` | best R@1 under 2 s p50 in the sweep; 2.7x cheaper than gpt-4.1 for both stages |
| `HIER_CACHE_PATH` | `/data/hier_notes.json` | notes survive scale-to-zero and redeploys; a cold start does not rewrite them |
| `HIER_REFRESH_S` | `300` | background note refresh while the container is up; a recall writes up to 3 missing notes inline and schedules a bigger backlog, so `min_containers=0` still works |
| key | `OPENAI_API_KEY` in `megamind-providers` (already needed for embeddings / STT) | `HIER_API_KEY` + `HIER_BASE_URL` override it for another OpenAI-compatible endpoint |
| `HIER_GATE_MARGIN` | unset (0) | optional: skip the LLM when the hybrid's top-1 leads top-2 by this relative margin (0.25 saved 21% of LLM calls for -4 pts R@1) |

Fallbacks: with no OpenAI key `build_retriever` logs a warning and uses the hybrid; any LLM error / timeout in a
recall returns the hybrid's ranking for that question. To go back to the plain hybrid, set
`"RETRIEVER_BACKEND": "hybrid"` in the live `PROFILE_ENV` and redeploy (the notes file can stay). UCC
(`RETRIEVER_BACKEND=ucc`, `UCC_API_KEY`) remains available but scored below hier on the bake-off.

## Deploy / stop

```sh
modal deploy deploy/modal_app.py                          # live keys (from the repo root)
MEGAMIND_PROFILE=mock modal deploy deploy/modal_app.py    # keyless smoke test
modal app stop megamind -y                                # stop billing; the Volume (memories) is kept
```

The deploy prints the URL. Smoke it:

```sh
export MEGAMIND_API_TOKEN=<token>  U=https://<workspace>--megamind-web.modal.run
curl -H "Authorization: Bearer $MEGAMIND_API_TOKEN" $U/v1/health
python tools/feeder.py --server $U --source clip.mp4 --realtime     # reads MEGAMIND_API_TOKEN (or --token)
python tools/ask.py --server $U "what did I see?"
open "$U/ui?token=$MEGAMIND_API_TOKEN"                                # sets an HttpOnly cookie, then plain /ui works
```

To change provider settings beyond keys (models, `READER_BASE_URL`, `ENRICH_PROVIDER`, ...) edit `PROFILE_ENV` in
`modal_app.py` then redeploy.
To wipe memories: `DELETE /v1/memories`, or `modal volume rm megamind-data -r /` (deletes everything).

## Cost and latency

- Idle: 0 (`min_containers=0`); after the last request the container stays warm 10 min. For a demo window set
  `min_containers=1` (no cold starts, ~1 vCPU-hour per hour) and set it back afterwards. hier does not need it
  (notes are on the Volume and caught up at query time), but a warm container keeps them fresher during a demo.
- Cold start ~10-15 s (image is cached; time is container boot + Python import + LanceDB open). Frames the headset sends
  during a cold start may time out and be dropped (verify on device).
- Warm round trip measured from a US cloud VM: `GET /v1/health` ~0.37 s, mock `POST /v1/chat` ~0.5 s (local:
  1.5 ms / 11-28 ms). Real turns are dominated by Claude + the reader (~5 s), so the extra hop is small.
