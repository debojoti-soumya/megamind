"""First question after a cold start, with and without the startup warmup (megamind/warmup.py).

Each run starts a fresh `uvicorn megamind.main:app` process (a cold start) on a LanceDB store seeded once, waits
for the first `GET /v1/health` (what the headset's 4 s poll does when the app opens), waits `--delay` seconds (the
wearer pressing the trigger), then times the first `POST /v1/chat` and a second, warm one.

    uv run python tools/bench_cold_start.py --mock               # keyless: every provider mocked
    uv run python tools/bench_cold_start.py --runs 5             # whatever server/.env configures (bills: one
                                                                 # Claude turn + STT-free text chat per run)

With --mock the difference is the local first-use costs only (LanceDB opening its indices, the reader's image
path, lazy imports). The provider-side part (SDK first use + TLS to Anthropic/OpenAI, SigLIP's first pass) needs
the team keys / the live Modal image: run without --mock there.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from PIL import Image

SERVER_DIR = Path(__file__).resolve().parents[1]
QUESTIONS = ["what was on the whiteboard?", "where did I leave my keys?"]
WORDS = "keys mug whiteboard laptop desk kitchen counter standup demo notebook charger window plant door".split()
MOCK_ENV = {
    "EMBED_PROVIDER": "mock",
    "IMAGE_EMBED_PROVIDER": "none",
    "READER_PROVIDER": "mock",
    "ENRICH_PROVIDER": "mock",
    "STT_PROVIDER": "mock",
    "TTS_PROVIDER": "mock",
    "AUDIO_STT_PROVIDER": "mock",
    "SUMMARIZER_PROVIDER": "mock",
    "ANTHROPIC_API_KEY": "",
    "OPENAI_API_KEY": "",
    "READER_API_KEY": "",
}


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _jpeg(seed: int) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", (160, 120))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(160 * 120)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class Server:
    def __init__(self, env: dict[str, str]) -> None:
        self.port = _port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = tempfile.TemporaryFile(mode="w+")  # a file, not a pipe: INFO logs must never block the server
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "megamind.main:app", "--port", str(self.port), "--log-level", "warning"],
            cwd=SERVER_DIR,
            env=os.environ | env,
            stdout=subprocess.DEVNULL,
            stderr=self.log,
            text=True,
        )

    def logs(self) -> str:
        self.log.seek(0)
        return self.log.read()

    def wait_up(self, c: httpx.Client, limit_s: float = 120) -> tuple[float, dict]:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < limit_s:
            if self.proc.poll() is not None:
                raise SystemExit(f"server exited: {self.logs()[-2000:]}")
            try:
                r = c.get(f"{self.url}/v1/health", timeout=2)
                if r.status_code == 200:
                    return time.perf_counter() - t0, r.json()
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise SystemExit("server did not come up")

    def stop(self) -> str:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        out = self.logs()
        self.log.close()
        return out


def seed(env: dict[str, str], n: int) -> None:
    srv = Server(env | {"WARMUP": "false"})
    try:
        with httpx.Client(timeout=30) as c:
            srv.wait_up(c)
            rnd = random.Random(0)
            t0 = 1_700_000_000.0
            for i in range(n):
                meta = {"device_id": "bench", "ts": t0 + 5 * i, "transcript": " ".join(rnd.sample(WORDS, 5))}
                r = c.post(
                    f"{srv.url}/v1/frames",
                    files={"image": ("f.jpg", _jpeg(i), "image/jpeg")},
                    data={"meta": json.dumps(meta)},
                )
                r.raise_for_status()
            last, stable = -1, 0
            while stable < 6:  # enrichment is async: wait until the count stops moving
                time.sleep(0.5)
                cnt = c.get(f"{srv.url}/v1/health").json()["store_count"]
                stable = stable + 1 if cnt == last else 0
                last = cnt
            print(f"seeded {last} memories")
    finally:
        srv.stop()


def one_run(env: dict[str, str], warm: bool, delay: float) -> dict[str, float | str]:
    srv = Server(env | {"WARMUP": "true" if warm else "false"})
    try:
        with httpx.Client(timeout=120) as c:
            up_s, _ = srv.wait_up(c)
            time.sleep(delay)
            state = c.get(f"{srv.url}/v1/health").json().get("warmup", "?")
            out: dict[str, float | str] = {"up_s": up_s, "warmup_at_ask": state}
            for i, q in enumerate(QUESTIONS):
                t0 = time.perf_counter()
                r = c.post(f"{srv.url}/v1/chat", json={"session_id": f"b{i}", "text": q, "device_id": "bench"})
                r.raise_for_status()
                out[f"q{i + 1}_s"] = time.perf_counter() - t0
            return out
    finally:
        done = [ln for ln in srv.stop().splitlines() if "warmup done" in ln]
        if warm and done:
            print("  " + done[-1].strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="force every provider to mock (no keys, no bills)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=600, help="memories to seed (frames + transcripts)")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between the first health and the question")
    ap.add_argument("--data", default="", help="reuse a LanceDB dir (skips seeding if it exists)")
    a = ap.parse_args()

    data = Path(a.data or tempfile.mkdtemp(prefix="mm-cold-"))
    env = {
        "STORE_BACKEND": "lancedb",
        "LANCEDB_PATH": str(data / "lancedb"),
        "THUMB_DIR": str(data / "thumbs"),
        "AUDIO_DIR": str(data / "audio"),
        "CONSOLIDATE_ENABLED": "false",
        "FILTER_MIN_GAP_S": "0",
    } | (MOCK_ENV if a.mock else {})
    if not (data / "lancedb").exists():
        seed(env, a.seed)

    res: dict[bool, list[dict]] = {False: [], True: []}
    for i in range(a.runs):
        for warm in (False, True):  # interleaved, so drift hits both
            r = one_run(env, warm, a.delay)
            res[warm].append(r)
            cols = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items())
            print(f"run {i + 1} warmup={'on ' if warm else 'off'} {cols}")
    which = "mock providers" if a.mock else "configured providers"
    print(f"\nmedian over {a.runs} cold starts ({which}, delay {a.delay:g} s):")
    for warm in (False, True):
        med = {k: statistics.median(r[k] for r in res[warm]) for k in ("up_s", "q1_s", "q2_s")}
        print(
            f"  warmup {'on ' if warm else 'off'}: startup {med['up_s']:.2f} s"
            f" | first question {med['q1_s'] * 1000:.0f} ms | second {med['q2_s'] * 1000:.0f} ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
