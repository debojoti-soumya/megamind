# Prior art and differentiation

Has this been done before? **The idea has — the open, agent-callable version on a consumer
headset has not.** Summary of the landscape (checked 2026-11-07) and what Megamind does differently.

## Products

| Who | What | Gap vs. Megamind |
|---|---|---|
| **Google Project Astra / Gemini Live** (I/O 2024 →) | The canonical demo: "where did I leave my glasses?" answered from recent camera stream; now in Gemini Live and Android XR glasses prototypes. | Closed; memory is a short rolling window of the current session, not a persistent, searchable store; not available to third-party agents or on Quest. |
| **Meta AI on Ray-Ban Meta / Ray-Ban Display** | "Remember where I parked"-style *explicit* reminders; Live AI continuous video sessions. | Memory is user-triggered, not passive; no retrieval API; no app on Quest 3 exposes it. |
| **Limitless / Rewind pendant, Bee, Omi** | Always-on *audio* lifelogging with transcript search and summaries. | Audio-only — no visual memory, so "where did I put X" is impossible. |
| **Humane AI Pin** | Camera + voice assistant (discontinued 2025). | Live scene QA, no durable visual memory. |

## Research

| Work | Relevance |
|---|---|
| **Ego4D Episodic Memory** (Grauman et al., 2022) — NLQ / Visual Queries / Moments | Defines exactly our task ("where did I last see X", "what did I do after Y") as a benchmark on egocentric video. Offline, dataset-scale, no live assistant. |
| **EgoLife: Towards Egocentric Life Assistant** (CVPR 2025) — EgoButler = EgoGPT + EgoRAG | Closest academic analog: week-long egocentric capture, RAG over a hierarchical text memory, QA benchmark. Retrieval is caption/text-based; captured with Meta Aria glasses, not a consumer headset; no interactive AR UI. |
| **Vinci** (OpenGVLab, 2024) | Real-time egocentric VLM assistant for smart glasses with a memory module; focuses on live streaming understanding and fine-tuned EgoVideo-VL. |
| Egocentric memory / long-video QA work (EgoSchema, MemoryBank, ReKV, etc.) | Model-side contributions; none ship an end-to-end wearable + agent system. |

## Open-source Quest 3 projects

| Repo | What it does | Gap |
|---|---|---|
| [xrdevrob/QuestCameraKit](https://github.com/xrdevrob/QuestCameraKit) (~580★) | Reference samples for the Passthrough Camera API (object detection, OCR, OpenAI vision). | Toolkit, not a memory system. We should reuse its camera plumbing. |
| [sumitaich1998/jarvisvr](https://github.com/sumitaich1998/jarvisvr) | Multi-agent MR "J.A.R.V.I.S." with live passthrough vision, audio and gaze. | Live perception only; its README lists *"spatial + episodic memory ('where did I leave my keys?')"* as **future work**. |
| hyunaseo/OpenAI-for-Meta-Quest-3, Vtestcode/quest-mr-assistant, Pryce22/Document-Your-Reality | Voice + egocentric frame → GPT-4o, answer about the *current* scene. | No persistence, no retrieval, no memory tool. |

## What is actually new in Megamind

1. **Persistent, passive episodic visual memory on a consumer headset.** Continuous keyframe capture via the Quest 3 Passthrough Camera API into a searchable store — the piece every existing Quest project stops short of.
2. **Memory as a tool, not a pipeline.** `recall_memory` is an agent-callable function with a published schema. Any agent or sub-agent (Claude today, anything with tool-use tomorrow) can pull memories into its context — the memory layer is decoupled from the assistant.
3. **Two-model read path with real attention over frames.** Retrieval returns tokens (caption + OCR + thumbnail); a Qwen-VL reader *re-looks at the actual frames* and returns cited evidence, instead of caption-only RAG (EgoRAG) or stuffing images into the orchestrator's context.
4. **Temporal + semantic + visual hybrid retrieval.** Time-range pre-filtering plus text vectors, BM25 and SigLIP image vectors fused with RRF — episodic questions are usually "when" as much as "what".
5. **Grounded answers in AR.** The remembered frame is shown next to the answer in the headset, so the wearer can verify.
6. **Device-agnostic and open.** The headset is a thin client behind `docs/api.md`; a laptop webcam or phone can feed the same memory. Everything is open source with pluggable models and a local-first store with purge.

What we are *not* claiming: novelty of the use case (Astra demoed it in 2024; Ego4D defined it in 2022) or any model-level contribution. The contribution is a working, open, agent-integrated system on hardware people own.
