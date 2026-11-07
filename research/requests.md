# Research requests

One bullet per question: `- [stream-N] <question> — <why / what decision it unblocks>`. Agent 6 adds `P0/P1/P2` and dedupes;
the research bot (agent 5) appends `→ answered in reports/<slug>.md`. **Preferred channel is now a GitHub issue labelled
`to:research`**; this file mirrors those issues (issue number in parentheses) so the history stays in the repo.

## Shortlist (research bot is PAUSED — coordinator runs it only on these ≤3 P0 questions)
1. **P0** (#33) [stream-3] Temporal grounding of vague/relational time phrases ("before I left the office"): separate anchor pass vs joint; window sizes; episode boundaries vs fixed ±N min; cheap NLQ-template rewrites — weakest eval class in `server/tools/eval_retrieval.py`.
2. **P0** (#10, absorbs #17) [stream-4] Multi-image reader layout evidence: interleaved vs grouped images, chronological vs relevance order, JSON-output cost, accuracy vs image count / `detail:low` — only if evidence contradicts PR #20's default.
3. **P0** (#11) [stream-2] open_clip checkpoint for 768-d `image_vec` on text-heavy passthrough frames (SigLIP B/16 vs SO400M vs CLIP L/14; latency; SigLIP2 availability).

## Closed without research (broad, or answered by existing reports)
- (#13) eval protocol → reports/03 §13 + PR #25. (#16) caption prompt → reports/01 rec 6, reports/03 rec 8. (#15) AR UX → digest/ui.md (UI owned by Tim). (#17) → merged into #10.

## Answered
- (#12) Quest 3 Passthrough Camera API → [reports/05-quest3-passthrough-camera-api.md](reports/05-quest3-passthrough-camera-api.md) → digest/capture.md #0, ui #6.
- (#14) Sail API limits / Qwen3.6 vs Gemma latency → [reports/04-sail-api-limits-latency.md](reports/04-sail-api-limits-latency.md) → digest/harness.md #1–3, storage #1–2, retrieval #8. **Qwen3.6 is flex-only → reader default must change.**
- (none) multimodal RAG fusion / temporal expansion / eval → [reports/03-multimodal-rag-retrieval.md](reports/03-multimodal-rag-retrieval.md) → digest/retrieval.md #1–6, storage #3–5, harness #4–5.
- (#7) keyframe selection heuristics beyond dHash → [reports/02-keyframe-selection.md](reports/02-keyframe-selection.md) → digest/capture.md #1–8.
- (#9) EgoRAG memory structure & retrieval → [reports/01-egocentric-memory-systems.md](reports/01-egocentric-memory-systems.md) → digest/retrieval.md #1,3,4,5,8; storage #3,4; harness #1,3; ui #1,2.
