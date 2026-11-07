# Research requests

One bullet per question: `- [stream-N] <question> — <why / what decision it unblocks>`. Agent 6 adds `P0/P1/P2` and dedupes;
the research bot (agent 5) appends `→ answered in reports/<slug>.md`. **Preferred channel is now a GitHub issue labelled
`to:research`**; this file mirrors those issues (issue number in parentheses) so the history stays in the repo.

## Open
- **P0** [stream-4 harness] (#17, overlaps #10) Best multi-image episodic-QA prompt / context layout for the reader (image count, `[id]` interleaving, evidence-card JSON reliability on Qwen3.6 / Qwen2.5-VL) — EgoButler half answered in reports/01; prompting half still open.
- **P0** [stream-4 harness] (#10, was P1; reader PR #20 is live) Multi-image prompting on Qwen3.6-35B-A3B / Qwen2.5-VL via OpenAI-compatible APIs: image count vs accuracy, resolution/token cost per image, JSON-output reliability, whether interleaving `[id]` text before each image improves citation accuracy — shapes the reader prompt and k.
- **P1** [stream-2 storage] (#11; reports/03 §10 suggests ViT-B-16-SigLIP / so400m, 768-d) SigLIP vs CLIP for `image_vec` on text-heavy egocentric frames and its text-tower quality for `mode=visual` queries — picks the image embedder.
- **P1** [stream-1 capture] (#12) Quest 3 Passthrough Camera API gotchas (Horizon OS v74+, `WebCamTexture` black/16 px frames, permission flow, frame rate limits, QuestCameraKit reuse) — the #1 risk in the design doc.
- **P1** [stream-4 harness] (#13; partly answered in reports/03 §13 — hit@k/MRR on gold_ts) Egocentric episodic-memory QA evaluation protocol (Ego4D NLQ / EgoSchema metrics) that can be shrunk to a 20–30 question gold set with hit@k — defines `tools/eval.py`.
- **P2** [stream-7 ui] (#15) Voice-first AR assistant UX findings (latency masking, earcons, showing evidence frames) from Astra / Ray-Ban Meta / academic AR studies — panel design.
- **P2** [stream-2 storage] (#16) Caption prompt design for retrievable egocentric captions (object-location phrasing, OCR verbatim, no identity) — evidence from EgoLife/Vinci captioners vs our `ENRICH_PROMPT`.

## Answered
- (#14) Sail API limits / Qwen3.6 vs Gemma latency → [reports/04-sail-api-limits-latency.md](reports/04-sail-api-limits-latency.md) → digest/harness.md #1–3, storage #1–2, retrieval #8. **Qwen3.6 is flex-only → reader default must change.**
- (none) multimodal RAG fusion / temporal expansion / eval → [reports/03-multimodal-rag-retrieval.md](reports/03-multimodal-rag-retrieval.md) → digest/retrieval.md #1–6, storage #3–5, harness #4–5.
- (#7) keyframe selection heuristics beyond dHash → [reports/02-keyframe-selection.md](reports/02-keyframe-selection.md) → digest/capture.md #1–8.
- (#9) EgoRAG memory structure & retrieval → [reports/01-egocentric-memory-systems.md](reports/01-egocentric-memory-systems.md) → digest/retrieval.md #1,3,4,5,8; storage #3,4; harness #1,3; ui #1,2.
