# Megamind on Modal (public HTTPS)

`modal_app.py` serves the FastAPI server (`megamind.main:create_app`) from one CPU container behind a public
`https://<workspace>--megamind-web.modal.run` URL. The headset, the browser panel (`/ui`) and Memory Theater (`/xr`)
talk to it from any network: no hotspot, Tailscale or firewall rules. Quest setup: [quest3-guide.md](../docs/quest3-guide.md) option F.

| | |
|---|---|
| Container | CPU only (1 vCPU, 2 GiB), `min_containers=0`, `max_containers=1`, `scaledown_window=600` (10 min idle, then 0 cost) |
| Why one container | LanceDB and the in-process workers (enrichment, consolidator, UCC sync, media aging) are single-writer. `@modal.concurrent(max_inputs=64)` lets that one container serve the headset uploads, the panel and SSE streams at once. |
| State | Modal Volume mounted at `/data`: `lancedb/`, `thumbs/` (+ sibling `chunks/`, replay/montage caches), `audio/`. Memories survive `modal app stop`, redeploys and scale-to-zero. |
| Persistence | Modal commits Volume writes in the background and on container exit; the app also calls `volume.commit()` after the workers stop in the lifespan shutdown. Verified: memories + thumbnails were still there after `modal app stop` + redeploy. |
| ffmpeg | Debian bookworm apt `ffmpeg` 5.1.x (verified `5.1.9` in the container; `-fps_mode` needs ≥ 5.1, #114) |
| Auth | `API_TOKEN` from the `megamind-auth` secret. Required: the URL is public. See [api.md#auth](../docs/api.md#auth). |
| Profiles | `MEGAMIND_PROFILE=live` (default): OpenAI embeddings/STT/TTS, Sail reader, keys from `megamind-providers`. `MEGAMIND_PROFILE=mock`: every provider mocked, no paid calls, no provider secret needed. Each profile has its own Volume (`megamind-data` / `megamind-data-mock`) because LanceDB fixes the embedding dimension per table. |

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
  `min_containers=1` (no cold starts, ~1 vCPU-hour per hour) and set it back afterwards.
- Cold start ~10-15 s (image is cached; time is container boot + Python import + LanceDB open). Frames the headset sends
  during a cold start may time out and be dropped (verify on device).
- Warm round trip measured from a US cloud VM: `GET /v1/health` ~0.37 s, mock `POST /v1/chat` ~0.5 s (local:
  1.5 ms / 11-28 ms). Real turns are dominated by Claude + the reader (~5 s), so the extra hop is small.
