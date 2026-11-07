# research/ — the research message bus

Agents in this repo cannot message each other. This directory is how research flows to the
build streams and how build streams ask for research. **Everything here is committed directly
to `main`** (`git pull --rebase` first, `--author="kartikramachandrula <k_rama@mit.edu>"`);
anything under `docs/` still goes through a PR.

```
research/
  README.md          this protocol (owner: agent 6, research integrator)
  requests.md        questions for the research bot, one bullet each, prioritised P0/P1/P2 (anyone appends; agent 6 curates)
  reports/           raw findings written by the research bot (agent 5). One file per topic, markdown, cite sources.
  digest/<stream>.md curated, actionable to-do lists per build stream (agent 6 writes; build agents read)
```

## Streams and who reads which digest

| digest | agent | owns |
|---|---|---|
| `digest/capture.md`   | 1 | `headset/.../FrameUploader.cs`, `server/megamind/ingest/{filter,worker,router}.py` |
| `digest/storage.md`   | 2 | `server/megamind/store/`, `ingest/enrich.py` (enricher + embedders), `models.py` token fields |
| `digest/retrieval.md` | 3 | `server/megamind/retrieval/`, `agent/tools.py` above `reader.read(...)` (ranked token list), fusion |
| `digest/harness.md`   | 4 | `tools/`, `server/tests/`, eval + latency harness, `agent/{claude_agent,reader,router}.py`, `agent/tools.py` from `reader.read(...)` down (coordinator decision, #5) |
| `digest/ui.md`        | 7 | `headset/.../{ChatClient,ChatPanel}.cs`, `server/megamind/ui/`, `agent/{stt,tts}.py` |

(The design doc numbers streams 1/2/3 = capture→store / retrieval / AR chat. Agent numbers above
are the ones used in branch names `stream-<N>/...`.)

## Protocol

**Build agents (1, 2, 3, 4, 7)**
1. At the start of every iteration: `git pull`, read `research/digest/<your-stream>.md`.
2. Items are tagged `NEW` (not yet acted on), `APPLIED` (seen in a merged PR), `REJECTED` (decided against; reason given).
   Apply what is useful; if you deliberately skip an item, say so in your PR description so it can be flipped to `REJECTED`.
3. When a question comes up that literature or prior art could answer, append one line to `research/requests.md`:
   `- [stream-N] <question> — <why it matters / what decision it unblocks>` and commit it to `main`.
   Don't wait for the answer; keep building with your best guess and note the guess in code.

**Research bot (5)**
1. Read `research/requests.md`, take the highest-priority unanswered item (P0 before P1 before P2).
2. Write findings to `research/reports/<slug>.md`: TL;DR first, then numbers/quotes with links, then
   "what this means for Megamind" in concrete terms (thresholds, prompts, library names). Mark estimates as estimates.
3. Append `→ answered in reports/<slug>.md` to the request bullet.

**Research integrator (6)**
1. Every ~20 min: `git pull`, read new reports and recently merged/open PRs.
2. Rewrite each `digest/<stream>.md` as a short prioritised list: *do X in file Y because Z (source)*. One screen max; stale items removed.
3. Flip `NEW` → `APPLIED`/`REJECTED` by checking merged PRs.
4. Dedupe/prioritise `requests.md`; add requests inferred from TODOs and "not sure if" comments in PRs.
5. Findings that change the design doc or the API contract go through a PR against `docs/` (contract changes follow the
   three-file rule: `docs/api.md` + `server/megamind/models.py` + `headset/.../Dto.cs`).

## Rules of thumb
- Digests say what to change and where. They never paste a report.
- A digest item that needs a contract change is marked `CONTRACT` and is not to be applied without coordinator approval.
- Numbers in digests carry their provenance: *measured* (on what) or *estimate*.

## GitHub Issues channel (added 2026-11-07)
Direct asks between agents go through Issues on the repo, labelled `to:capture` (1), `to:storage` (2), `to:retrieval` (3),
`to:harness` (4), `to:research` (5), `to:integrator` (6), `to:coordinator` (keys/hardware/merges).
`gh issue create --title "[from:<stream>] <ask>" --label to:<stream> --body "<need, why, proposed interface, deadline>"`.
Check `gh issue list --label to:<your-stream> --state open` at the start of every iteration and after every PR.
Interface agreements between two streams are settled in an issue before either PR; paste the final signature in the PR body.
Research questions: prefer an issue labelled `to:research`; agent 6 mirrors them into `research/requests.md`.
