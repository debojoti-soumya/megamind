from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bench_reader_offline_runs_without_keys():
    out = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "bench_reader.py"), "--limit", "3", "--thumb-px", "512,256"],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "", "READER_API_KEY": ""},
        cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert "| 512 |" in out.stdout and "| 256 |" in out.stdout
