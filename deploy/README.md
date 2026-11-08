# Megamind on Modal (public HTTPS)

`modal_app.py` serves the FastAPI server (`megamind.main:create_app`) from one CPU container behind a public
`https://<workspace>--megamind-web.modal.run` URL. The headset, the browser panel (`/ui`) and Memory Theater (`/xr`)
talk to it from any network: no hotspot, Tailscale or firewall rules. Quest setup: [quest3-guide.md](../docs/quest3-guide.md) option F.

| | |
|---|---|
| Container | CPU only (1 vCPU, 2 GiB), `min_containers=0` (`MEGAMIND_MIN_CONTAINERS=1` at deploy time keeps one warm), `max_containers=1`, `scaledown_window=600` (10 min idle, then 0 cost) |
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

Both profiles recall with the plain hybrid (`RETRIEVER_BACKEND=hybrid`, the config default; vector + BM25 + OCR +
temporal, no paid calls beyond the query embedding). `hier` and `combo` are opt-in: add `"RETRIEVER_BACKEND": "hier"`
(or `"combo"`) to the live `PROFILE_ENV` and redeploy.

- **On one 540-memory day** (#152, `server/tools/eval_retrieval.py`), hier wins on recall: R@1 0.90 vs combo 0.80 vs
  hybrid 0.60. It costs about 1.9 s p50 per recall and ~$0.006 per question.
- **On a 9.2k-memory week** (#149, `tools/eval_agent.py` on `long_week`), the hybrid answers 90 % at a 2.4 s turn. hier
  answers 93 % but at 16.5 s, and combo answers 90 % at 5.8 s p50 / 68 s p95. That is why the default stays hybrid.

Recommended values if you opt into hier (also the config defaults for the `HIER_*` settings, used by combo's hier
route too):

| Setting | Value | Why |
|---|---|---|
| `HIER_MODEL` / `HIER_RANK_MODEL` / `HIER_NOTE_MODEL` | `gpt-4.1-mini` / `gpt-4.1` / `gpt-4.1-mini` | best R@1 under 2 s p50 in the #152 sweep; 2.7x cheaper than gpt-4.1 for both stages |
| `HIER_TIMEOUT_S` | `8` | wall bound on the LLM stages of one recall; past it the hybrid's ranking answers |
| `HIER_CALL_TIMEOUT_S` / `HIER_RETRIES` | `4` / `2` | recall-time chat calls fail fast (note writing keeps 120 s / 6 tries) |
| `HIER_CACHE_PATH` | `/data/hier_notes.json` | notes survive scale-to-zero and redeploys; a cold start does not rewrite them |
| `HIER_REFRESH_S` | `300` | background note refresh while the container is up; a recall writes up to 3 missing notes inline and schedules a bigger backlog |
| key | `OPENAI_API_KEY` in `megamind-providers` | `HIER_API_KEY` + `HIER_BASE_URL` override it for another OpenAI-compatible endpoint |
| `HIER_GATE_MARGIN` | unset (0) | optional: skip the LLM when the hybrid's top-1 leads top-2 by this relative margin (0.25 saved 21% of LLM calls for -4 pts R@1) |

Fallbacks: hier and combo fall back to the hybrid in two cases:
- With no OpenAI / UCC key, `build_retriever` logs a warning and uses the hybrid.
- Any LLM error, or a recall that runs past `HIER_TIMEOUT_S`, returns the hybrid's ranking for that question.

The mock profile pins `hybrid` and makes zero paid calls.

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

## Runbook (live deploy)

Deploy what `main` says: provider/model/retriever defaults come from `server/megamind/config.py`; `modal_app.py`
only pins storage paths, the provider profile and `IMAGE_EMBED_PROVIDER`. Run from a clean checkout so
`modal app history megamind` records the commit (a `*` there means uncommitted changes were deployed).

```sh
git fetch origin && git checkout --detach origin/main
MEGAMIND_MIN_CONTAINERS=1 modal deploy deploy/modal_app.py      # demo window: no cold starts
modal app history megamind                                      # version -> commit
T=<API_TOKEN>; U=https://kartikramachandrula--megamind-web.us-east.modal.run
curl -s -H "Authorization: Bearer $T" $U/v1/health              # models: no "mock" / "fallback"
MEGAMIND_API_TOKEN=$T python tools/ask.py --server $U "what did I just see?"
```

- **Rollback:** `git checkout --detach <previous commit from modal app history> && MEGAMIND_MIN_CONTAINERS=1 modal deploy deploy/modal_app.py`
  (or `modal app rollback megamind v<N>` to the previous version without rebuilding). Memories live on the Volume
  and are untouched by either.
- **After the demo:** `modal deploy deploy/modal_app.py` (back to `min_containers=0`) or `modal app stop megamind -y`.
- **Headset:** URL above, a unique `deviceId` per headset, `apiToken` = the `megamind-auth` secret's `API_TOKEN`
  (Unity: launch-intent extras / `megamind.json`; native: `--es api_token`).

## Cost and latency

- Idle: 0 (`min_containers=0`); after the last request the container stays warm 10 min. For a demo window set
  `min_containers=1` (no cold starts, ~1 vCPU-hour per hour) and set it back afterwards. hier does not need it
  (notes are on the Volume and caught up at query time), but a warm container keeps them fresher during a demo.
- Cold start ~10-15 s (image is cached; time is container boot + Python import + LanceDB open). Frames the headset sends
  during a cold start may time out and be dropped (verify on device).
- Warm round trip measured from a US cloud VM: `GET /v1/health` ~0.37 s, mock `POST /v1/chat` ~0.5 s (local:
  1.5 ms / 11-28 ms). Real turns are dominated by Claude + the reader (~5 s), so the extra hop is small.
