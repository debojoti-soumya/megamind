# Research requests

One bullet per question: `- [stream-N] <question> — <why / what decision it unblocks>`. Agent 6 adds `P0/P1/P2` and dedupes;
the research bot (agent 5) appends `→ answered in reports/<slug>.md`. **Preferred channel is now a GitHub issue labelled
`to:research`**; this file mirrors those issues (issue number in parentheses) so the history stays in the repo.

## Open
- **P0** [stream-1 capture] Keyframe selection heuristics for egocentric video beyond pixel dHash (Ego4D / EgoLife / Vinci practice: motion, pose, scene-change, speech triggers) and typical keep rates at 1 fps — decides filter v2 thresholds; the design's 10–30% keep rate is unverified.
- **P0** [stream-3 retrieval] (#9) How EgoLife's EgoRAG structures memory (hierarchical hour/day summaries, retrieval order) and what it reports for recall vs flat RAG — decides whether episodes must be in the MVP and how `recall_memory` should route "activity" questions.
- **P1** [stream-3 retrieval] (#10) Multi-image prompting on Qwen3.6-35B-A3B / Qwen2.5-VL via OpenAI-compatible APIs: image count vs accuracy, resolution/token cost per image, JSON-output reliability, whether interleaving `[id]` text before each image improves citation accuracy — shapes the reader prompt and k.
- **P1** [stream-2 storage] (#11) SigLIP vs CLIP for `image_vec` on text-heavy egocentric frames; which open_clip checkpoint (dims 768?) and its text-tower quality for `mode=visual` queries — picks the image embedder.
- **P1** [stream-1 capture] (#12) Quest 3 Passthrough Camera API gotchas (Horizon OS v74+, `WebCamTexture` black/16 px frames, permission flow, frame rate limits, QuestCameraKit reuse) — the #1 risk in the design doc.
- **P1** [stream-4 harness] (#13) Egocentric episodic-memory QA evaluation protocol (Ego4D NLQ / EgoSchema metrics) that can be shrunk to a 20–30 question gold set with hit@k — defines `tools/eval.py`.
- **P1** [stream-2 storage] (#14) Sail Research API limits (rate limits, concurrency, 20 images/request, max base64 payload, flex-window latency numbers for Qwen3.6-35B-A3B vs Gemma 4 31B) — decides enricher concurrency and whether the reader needs the Gemma fallback.
- **P2** [stream-7 ui] (#15) Voice-first AR assistant UX findings (latency masking, earcons, showing evidence frames) from Astra / Ray-Ban Meta / academic AR studies — panel design.
- **P2** [stream-2 storage] (#16) Caption prompt design for retrievable egocentric captions (object-location phrasing, OCR verbatim, no identity) — evidence from EgoLife/Vinci captioners vs our `ENRICH_PROMPT`.

## Answered
(none yet)
