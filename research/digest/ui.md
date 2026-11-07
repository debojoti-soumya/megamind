# Digest — ui (agent 7)
_Updated 2026-11-07 15:10 PT by agent 6. Tags: NEW / APPLIED / REJECTED / CONTRACT. Sources: [PR #1 INTEGRATION.md](https://github.com/debojoti-soumya/megamind/blob/devin/1794085388-research-integration/docs/research/INTEGRATION.md), design doc §Stream 3 / latency, [PR #2](https://github.com/debojoti-soumya/megamind/pull/2)._

**Heads-up:** [PR #2](https://github.com/debojoti-soumya/megamind/pull/2) (open) already has a browser panel at `/ui`, real STT/TTS providers, and Unity `ChatClient/ChatPanel/PanelAnchor/SettingsPanel`. Start from it.

1. **NEW (P0)** show the cited thumbnail(s) next to the answer and the confidence label — grounding is the differentiator (prior-art §5); captions hallucinate, the frame is the evidence.
2. **NEW (P1)** latency masking: display Claude's text as soon as it arrives and play TTS after; show a "remembering…" state while `recall_memory` runs (budget is ~7 s; silence feels broken).
3. **NEW (P1)** push-to-talk with barge-in (stop TTS when recording starts) — PR #2 has it; keep it when porting to the prefab.
4. **NEW (P1)** visible capture indicator (recording dot + frames-accepted count) on the panel — privacy promise in README and design §Risks.
5. **NEW (P2)** if the answer cites an episode/time range, show "n min ago" relative times, not timestamps (prototype UX note).
