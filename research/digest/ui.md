# Digest — ui (agent 7)
_Updated 2026-11-07 15:15 PT by agent 6. Tags: NEW / APPLIED / REJECTED / CONTRACT. Sources: [reports/01](../reports/01-egocentric-memory-systems.md), [PR #1 INTEGRATION.md](https://github.com/debojoti-soumya/megamind/blob/devin/1794085388-research-integration/docs/research/INTEGRATION.md), design §Stream 3, [PR #2](https://github.com/debojoti-soumya/megamind/pull/2)._

**Heads-up:** [PR #2](https://github.com/debojoti-soumya/megamind/pull/2) (open) already has a browser panel at `/ui`, real STT/TTS providers, Unity `ChatClient/ChatPanel/PanelAnchor/SettingsPanel`. Start from it.

1. **NEW (P0)** show the cited thumbnail(s) **with "seen 21 min ago"** next to the answer and the confidence label. None of EgoRAG/Vinci show the evidence frame; Ego4D VQ asks for exactly the where/when of the last sighting — this is the differentiator. [reports/01](../reports/01-egocentric-memory-systems.md) rec 10, [PR #1 INTEGRATION.md](https://github.com/debojoti-soumya/megamind/blob/devin/1794085388-research-integration/docs/research/INTEGRATION.md).
2. **NEW (P1)** demo script = Ego4D NLQ templates ("Where did I put my keys?", "What did the whiteboard say?", "Who did I talk to in the kitchen?") — rehearse these; harness's gold set uses the same list. [reports/01](../reports/01-egocentric-memory-systems.md) rec 9.
3. **NEW (P1)** latency masking: show Claude text as soon as it arrives, TTS after; "remembering…" state while `recall_memory` runs (~7 s budget). [PR #1 INTEGRATION.md](https://github.com/debojoti-soumya/megamind/blob/devin/1794085388-research-integration/docs/research/INTEGRATION.md).
4. **NEW (P1)** push-to-talk with barge-in (PR #2 has it; keep it in the prefab). Visible capture indicator (dot + frames-accepted count) — privacy promise. design §Risks.
5. **NEW (P2)** relative times everywhere in the panel, never raw timestamps.
