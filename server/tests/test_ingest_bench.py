import json

from PIL import Image

from megamind.ingest.bench import load_frames, main, run, synth_clip


def test_synth_clip_is_deterministic_and_1fps():
    a, b = synth_clip(8, width=160), synth_clip(8, width=160)
    assert [j for j, _ in a] == [j for j, _ in b]
    assert [m.ts - a[0][1].ts for _, m in a] == [float(i) for i in range(8)]
    assert all(m.pose is not None for _, m in a)


async def test_run_covers_every_filter_outcome():
    out = await run(60, backends=["memory", "lancedb"], width=320)
    assert set(out["stages"]) == {"decode", "dhash", "mean_luma", "sharpness", "decide"}
    for backend in ("memory", "lancedb"):
        res = out["e2e"][backend]
        assert res["n"] == 60 and res["p95_ms"] >= res["p50_ms"] > 0
        assert {"accepted", "duplicate", "blurry", "dark"} <= set(res["reasons"])
    assert out["e2e"]["memory"]["reasons"] == out["e2e"]["lancedb"]["reasons"]


def test_cli_frames_dir(tmp_path, capsys):
    for i, (jpeg, meta) in enumerate(synth_clip(12, width=160)):
        (tmp_path / f"{meta.ts:.6f}_{i:02d}.jpg").write_bytes(jpeg)
    Image.new("RGB", (8, 8)).save(tmp_path / "frame.jpg")  # no ts prefix -> synthetic 1 fps ts
    assert len(load_frames(tmp_path)) == 13
    dest = tmp_path / "out.json"
    assert main(["--frames-dir", str(tmp_path), "--backends", "memory", "--json", str(dest)]) == 0
    assert "sharpness percentiles" in capsys.readouterr().out
    out = json.loads(dest.read_text())
    assert out["frames"] == 13 and set(out["metrics"]) == {"sharpness", "mean_luma"}
