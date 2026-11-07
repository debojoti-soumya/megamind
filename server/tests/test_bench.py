import json

from megamind.store.bench import main, run, synth_tokens

PRIMITIVES = {
    "vector_search text",
    "vector_search image",
    "keyword_search",
    "range 1h",
    "get_many 12",
    "neighbors ±2",
    "search hybrid",
}


def test_synth_tokens_shape():
    toks = synth_tokens(40, text_dim=16, image_dim=8)
    assert len({t.id for t in toks}) == 40
    assert all(len(t.text_vec) == 16 and len(t.image_vec) == 8 for t in toks)
    assert {t.device_id for t in toks} == {"quest-0", "quest-1"}
    assert [t.ts for t in toks[::2]] == sorted(t.ts for t in toks[::2])


async def test_run_both_backends(tmp_path):
    out = await run(300, backends=["memory", "lancedb"], queries=2, text_dim=16, image_dim=8)
    for backend in ("memory", "lancedb"):
        res = out["results"][backend]
        assert PRIMITIVES <= set(res)
        assert res["add_many (total ingest)"]["n"] == 300
        assert all(res[p]["n"] == 2 and res[p]["p95_ms"] >= res[p]["p50_ms"] >= 0 for p in PRIMITIVES)


def test_cli_json(tmp_path, capsys):
    dest = tmp_path / "bench.json"
    assert main(["--n", "60", "--queries", "1", "--text-dim", "8", "--image-dim", "4", "--json", str(dest)]) == 0
    assert "vector_search text" in capsys.readouterr().out
    assert set(json.loads(dest.read_text())["results"]) == {"memory", "lancedb"}
