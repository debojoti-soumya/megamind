"""Live latency profile with the team's provider keys (Modal secret `megamind-providers`), off the live deploy.

    modal run deploy/bench_modal.py                    # every config, 20 agent-eval questions each, + speech/embeds
    modal run deploy/bench_modal.py --limit 5 --only claude_sonnet,claude_haiku

Each config runs `tools/eval_agent.py --live` in its own container (in parallel) with env overrides; `speech`
times the per-turn provider legs outside the agent loop: STT (question WAV), TTS (typical answer), query embed.
Costs real money (cents per config).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import modal

SERVER_DIR = Path(__file__).resolve().parent.parent / "server"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "fonts-dejavu-core")
    .add_local_dir(
        SERVER_DIR,
        "/app/server",
        copy=True,
        ignore=[".venv", ".env", "data", "**/__pycache__", "*.egg-info", ".pytest_cache"],
    )
    .add_local_dir(Path.home() / ".cache/coco/img", str(Path.home() / ".cache/coco/img"), copy=True)
    .run_commands(
        "pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu",
        "pip install --no-cache-dir /app/server open-clip-torch>=2.24 transformers sentencepiece protobuf",
        "python -c \"import open_clip; open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')\"",
    )
)
app = modal.App("megamind-bench")
secrets = [modal.Secret.from_name("megamind-providers")]

SAIL = {"READER_BASE_URL": "https://api.sailresearch.com/v1", "READER_PROVIDER": "vlm"}
CONFIGS: dict[str, dict[str, str]] = {
    "defaults": {},
    "coco_off": {
        **{"_ARGS": "--dataset tools/data/coco_day.json", "EMBED_PROVIDER": "openai"},
        "IMAGE_EMBED_PROVIDER": "none",
    },
    "coco_on": {
        **{"_ARGS": "--dataset tools/data/coco_day.json", "EMBED_PROVIDER": "openai"},
        "IMAGE_EMBED_PROVIDER": "openclip",
    },
    "visual_off": {
        **{"_ARGS": "--dataset tools/data/visual_day.json", "EMBED_PROVIDER": "openai"},
        "IMAGE_EMBED_PROVIDER": "none",
    },
    "visual_on": {
        **{"_ARGS": "--dataset tools/data/visual_day.json", "EMBED_PROVIDER": "openai"},
        "IMAGE_EMBED_PROVIDER": "openclip",
    },
    "long_hybrid": {
        "_ARGS": "--dataset tools/data/long_week.json",
        "EMBED_PROVIDER": "openai",
        "EVAL_UCC_WAIT_S": "900",
    },
    "long_combo": {
        **{"_ARGS": "--dataset tools/data/long_week.json", "EMBED_PROVIDER": "openai", "EVAL_UCC_WAIT_S": "900"},
        "RETRIEVER_BACKEND": "combo",
    },
    "long_hier": {
        **{"_ARGS": "--dataset tools/data/long_week.json", "EMBED_PROVIDER": "openai", "EVAL_UCC_WAIT_S": "900"},
        "RETRIEVER_BACKEND": "hier",
    },
    "haiku55": {"CLAUDE_MODEL": "claude-haiku-5-5"},
    "sonnet55": {"CLAUDE_MODEL": "claude-sonnet-5-5"},
    "opus55": {"CLAUDE_MODEL": "claude-opus-5-5"},
    "sonnet55_8img": {"CLAUDE_MODEL": "claude-sonnet-5-5", "READER_MAX_IMAGES": "8"},
    "combo": {
        "RETRIEVER_BACKEND": "combo"
    },  # hybrid / UCC / hier routed per question  # whatever config.py ships (CLAUDE_MODEL, READER_PROVIDER=auto, ...)
    "sail512_sonnet": {**SAIL, "READER_THUMB_MAX_PX": "512"},
    "sail960_sonnet": {**SAIL, "READER_THUMB_MAX_PX": "960"},
    "claude_sonnet": {"READER_PROVIDER": "claude", "READER_THUMB_MAX_PX": "960"},
    "claude_haiku": {"READER_PROVIDER": "claude", "READER_THUMB_MAX_PX": "960", "CLAUDE_MODEL": "claude-haiku-4-5"},
}


@app.function(image=image, secrets=secrets, timeout=3600, cpu=2.0, memory=4096)
def eval_config(name: str, env: dict[str, str], limit: int) -> tuple[str, str, dict | None]:
    out = Path(f"/tmp/{name}.json")
    env = dict(env)
    extra = env.pop("_ARGS", "").split()
    cmd = ["python", "tools/eval_agent.py", "--live", "--limit", str(limit), "--thumb-width", "960", *extra]
    r = subprocess.run(
        [*cmd, "--answers", "--json", str(out)],
        cwd="/app/server",
        env={**os.environ, "RECALL_FIRST": "true", "RECALL_PREFETCH": "true", **env},
        capture_output=True,
        text=True,
        timeout=3500,
    )
    return name, r.stdout[-8000:] + r.stderr[-2000:], json.loads(out.read_text()) if out.exists() else None


def _ms(fn, reps: int = 3) -> dict:
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000)
    return {"p50": round(statistics.median(xs)), "min": round(min(xs)), "max": round(max(xs))}


async def _render_ms(tts, text: str, reps: int = 3) -> dict:
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        await tts._render(text)
        xs.append((time.perf_counter() - t0) * 1000)
    await tts.client.close()
    return {"p50": round(statistics.median(xs)), "min": round(min(xs)), "max": round(max(xs))}


@app.function(image=image, secrets=secrets, timeout=600)
def speech() -> dict:
    from openai import OpenAI

    c = OpenAI()
    answer = "Your keys are on the kitchen counter next to the coffee machine, about twenty minutes ago."
    res: dict[str, dict] = {}
    for model in ("tts-1", "gpt-4o-mini-tts"):
        res[f"tts {model} wav (full)"] = _ms(
            lambda m=model: c.audio.speech.create(model=m, voice="alloy", input=answer, response_format="wav").read()
        )

        def first_byte(m=model):
            with c.audio.speech.with_streaming_response.create(
                model=m, voice="alloy", input=answer, response_format="pcm"
            ) as r:
                next(r.iter_bytes(1024))

        res[f"tts {model} pcm first byte"] = _ms(first_byte)
    import asyncio

    from megamind.agent.tts import OpenAITTS
    from megamind.config import Settings

    two = answer + " You walked in from the car, put them down and went straight to the fridge."
    for model in ("tts-1", "gpt-4o-mini-tts"):
        cfg = Settings(_env_file=None, openai_api_key=os.environ["OPENAI_API_KEY"], tts_model=model)
        res[f"tts {model} 2 sentences, one call"] = _ms(
            lambda m=model: c.audio.speech.create(model=m, voice="alloy", input=two, response_format="pcm").read()
        )
        res[f"tts {model} 2 sentences, OpenAITTS (parallel)"] = asyncio.run(_render_ms(OpenAITTS(cfg), two))
    wav = c.audio.speech.create(
        model="tts-1", voice="alloy", input="Where did I leave my keys this morning?", response_format="wav"
    ).read()
    for model in ("whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe", "gpt-live-transcribe"):
        try:
            c.audio.transcriptions.create(model=model, file=("q.wav", wav, "audio/wav"))
        except Exception as e:  # noqa: BLE001
            res[f"stt {model}"] = {"error": str(e)[:120]}
            continue
        res[f"stt {model}"] = _ms(
            lambda m=model: c.audio.transcriptions.create(model=m, file=("q.wav", wav, "audio/wav"))
        )
    res["embed text-embedding-3-small (1 query)"] = _ms(
        lambda: c.embeddings.create(model="text-embedding-3-small", input="keys kitchen", dimensions=1024)
    )
    return res


@app.local_entrypoint()
def main(limit: int = 20, only: str = "", skip_speech: bool = False):
    names = [n for n in CONFIGS if not only or n in only.split(",")]
    sp = None if skip_speech else speech.spawn()
    summary = {}
    for name, text, data in eval_config.starmap([(n, CONFIGS[n], limit) for n in names]):
        print(f"\n===== {name}\n{text}")
        summary[name] = data and data["summary"]
    if sp is not None:
        print("\n===== speech / embeddings (ms)")
        for k, v in sp.get().items():
            print(f"  {k:42s} {v}")
    print("\n===== summaries")
    print(json.dumps(summary, indent=1))
