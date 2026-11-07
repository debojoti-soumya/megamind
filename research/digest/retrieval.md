# Digest — retrieval (agent 3)
_Updated 2026-11-07 15:10 PT by agent 6. Tags: NEW / APPLIED / REJECTED / CONTRACT. Source unless noted: [PR #1 INTEGRATION.md](https://github.com/debojoti-soumya/megamind/blob/devin/1794085388-research-integration/docs/research/INTEGRATION.md)._

**Heads-up:** `origin/devin/1794085849-stream1-plan` (prior wave, no PR) has `store/fusion.py` (weighted RRF + per-episode diversity) and a `list_episodes` tool. Reuse the fusion code; the new tool is a contract change (coordinator).

1. **NEW (P0)** fusion (`store/memory_store.py::_rrf`, and LanceDB equivalent): three-signal weighted RRF — BM25 on caption+OCR (1.0), text-vector (1.0), image-vector via SigLIP text tower (0.6). Image weight below text because CLIP-family models are weak on dense text, strong on objects/scenes/colours. Fused hit@5 was 100% on both gold sets (measured).
2. **NEW (P0)** `agent/reader.py` QwenVLReader prompt: captions are a *ranking* signal, never evidence — the reader must look at the thumbnail; verbatim OCR is the only reliable text evidence. Format each token as `[id] [local time] [episode title if any] caption / ocr` then the image; demand JSON `{evidence, confidence, cited_ids}`; refuse to cite ids not in the list.
3. **NEW (P1)** diversity: cap 2–3 tokens per `episode_id` in top-k (once storage fills it), else 12 near-identical desk frames fill the reader context. Until then, cap tokens whose dHash hamming < 6 to each other.
4. **NEW (P1)** BM25 query building: strip stop-words from the NL question; AND the terms first, fall back to OR if empty.
5. **NEW (P1)** `agent/claude_agent.py` system prompt: inject current local time + timezone; convert relative times ("this morning", "10 min ago") to ISO `time_from/time_to`; on `not_found` answer "I don't have a memory of that" — never guess. Keep Claude blind to thumbnails (D3) unless confidence is low.
6. **NEW (P1)** route by question type: `fact` (object/text → top tokens, `mode=hybrid`), `visual` (colour/shape → `mode=visual`), `activity` ("what was I doing after X" → time-range first, ≥ k=16). Cheap regexes worked in the prototype; `mode` already carries this through the tool schema.
7. **NEW (P2)** for text questions ("what did the whiteboard say") send a high-res crop around the OCR region instead of the whole 960 px thumbnail (needs OCR boxes from enrichment — ask storage via an issue if you want it).
8. **REJECTED for MVP** ColQwen2/ColPali late-interaction over keyframes — needs a GPU and isn't needed for the demo.
