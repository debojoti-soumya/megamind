import asyncio
import os
import time
from datetime import UTC, datetime

import pytest

from megamind.config import Settings
from megamind.models import MemoryToken
from megamind.store.lancedb_store import LanceDBStore
from megamind.store.purge import main, parse_since, purge

NOW = 1_800_000_000.0


def test_parse_since_empty():
    assert parse_since(None) is None
    assert parse_since("") is None


@pytest.mark.parametrize(
    ("value", "age_s"),
    [("30s", 30), ("90m", 5400), ("2h", 7200), ("1d", 86400), ("1.5h", 5400), (" 2h ", 7200)],
)
def test_parse_since_relative(value, age_s):
    assert parse_since(value, now=NOW) == NOW - age_s


def test_parse_since_relative_defaults_to_wall_clock():
    before = time.time()
    got = parse_since("1h")
    assert before - 3600 <= got <= time.time() - 3600


def test_parse_since_iso():
    assert parse_since("2026-11-07T09:00:00+00:00") == datetime(2026, 11, 7, 9, tzinfo=UTC).timestamp()
    assert parse_since("2026-11-07T09:00:00Z") == datetime(2026, 11, 7, 9, tzinfo=UTC).timestamp()
    assert parse_since("2026-11-07T09:00:00") == datetime(2026, 11, 7, 9).timestamp()  # naive = local


@pytest.mark.parametrize("value", ["yesterday", "2x", "-5m", "h"])
def test_parse_since_invalid(value):
    with pytest.raises(ValueError):
        parse_since(value)


def _tok(i: int, ts: float, tmp_path, *, device="dev1", kind="frame") -> MemoryToken:
    thumb = tmp_path / f"t{i}.jpg"
    thumb.write_bytes(b"jpeg")
    return MemoryToken(
        id=f"tok{i:03d}",
        device_id=device,
        ts=ts,
        caption=f"caption {i}",
        thumb_path=str(thumb),
        frame_hash=f"{i:016x}",
        kind=kind,
    )


@pytest.fixture
async def populated(tmp_path):
    """LanceDB at tmp_path/lancedb: dev1 frames 3h/90m/30m/10m old, a 20m-old dev1 episode, a 10m-old dev2 frame."""
    now = time.time()
    tokens = [
        _tok(0, now - 3 * 3600, tmp_path),
        _tok(1, now - 90 * 60, tmp_path),
        _tok(2, now - 30 * 60, tmp_path),
        _tok(3, now - 10 * 60, tmp_path),
        _tok(4, now - 20 * 60, tmp_path, kind="episode"),
        _tok(5, now - 10 * 60, tmp_path, device="dev2"),
    ]
    await LanceDBStore(tmp_path / "lancedb").add_many(tokens)
    settings = Settings(store_backend="lancedb", lancedb_path=tmp_path / "lancedb", _env_file=None)
    return settings, tokens


async def _remaining(settings) -> set[str]:
    rows = await LanceDBStore(settings.lancedb_path).range(None, None, limit=100)
    return {t.id for t in rows}


async def test_purge_dry_run_counts_without_deleting(populated):
    settings, tokens = populated
    assert await purge(settings, since="1h", dry_run=True) == 4
    assert await purge(settings, device_id="dev1", since="1h", dry_run=True) == 3
    assert await purge(settings, dry_run=True) == 6
    assert await _remaining(settings) == {t.id for t in tokens}
    assert all(open(t.thumb_path, "rb").read() == b"jpeg" for t in tokens)


async def test_purge_since_and_device(populated):
    settings, tokens = populated
    assert await purge(settings, device_id="dev1", since="1h") == 3  # tok002, tok003 + episode tok004
    assert await _remaining(settings) == {"tok000", "tok001", "tok005"}
    gone = {"tok002", "tok003", "tok004"}
    for t in tokens:
        assert (not os.path.exists(t.thumb_path)) == (t.id in gone)


async def test_purge_iso_since_and_keep_thumbs(populated):
    settings, tokens = populated
    cutoff = datetime.fromtimestamp(tokens[1].ts - 1, tz=UTC).isoformat()
    assert await purge(settings, since=cutoff, delete_thumbs=False) == 5
    assert await _remaining(settings) == {"tok000"}
    assert all(open(t.thumb_path, "rb").read() == b"jpeg" for t in tokens)


async def test_purge_everything(populated):
    settings, tokens = populated
    assert await purge(settings) == 6
    assert await _remaining(settings) == set()
    assert await purge(settings) == 0


async def test_purge_empty_store(tmp_path):
    settings = Settings(store_backend="lancedb", lancedb_path=tmp_path / "fresh", _env_file=None)
    assert await purge(settings, dry_run=True) == 0
    assert await purge(settings) == 0


def _cli_env(monkeypatch, settings):
    monkeypatch.setenv("STORE_BACKEND", "lancedb")
    monkeypatch.setenv("LANCEDB_PATH", str(settings.lancedb_path))


async def test_cli_dry_run_yes_and_abort(populated, monkeypatch, capsys):
    # main() runs its own event loop, so drive it from a worker thread.
    settings, _ = populated
    _cli_env(monkeypatch, settings)

    assert await asyncio.to_thread(main, ["--since", "1h", "--dry-run"]) == 0
    assert "would delete 4 token(s)" in capsys.readouterr().out

    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert await asyncio.to_thread(main, ["--device", "dev2"]) == 1
    assert "aborted" in capsys.readouterr().out
    assert len(await _remaining(settings)) == 6

    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    assert await asyncio.to_thread(main, ["--device", "dev2"]) == 0
    assert "deleted 1 token(s)" in capsys.readouterr().out

    assert await asyncio.to_thread(main, ["--since", "2h", "-y", "--keep-thumbs"]) == 0
    assert "deleted 4 token(s)" in capsys.readouterr().out
    assert await _remaining(settings) == {"tok000"}


def test_cli_rejects_bad_since(monkeypatch, tmp_path):
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "x"))
    with pytest.raises(SystemExit) as e:
        main(["--since", "yesterday", "--dry-run"])
    assert e.value.code == 2


def test_cli_memory_backend_notes_noop(monkeypatch, capsys):
    monkeypatch.setenv("STORE_BACKEND", "memory")
    assert main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "keeps nothing on disk" in out and "would delete 0" in out
