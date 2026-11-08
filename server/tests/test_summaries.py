from megamind.config import Settings
from megamind.ingest.summaries import SummaryBuilder
from megamind.models import MemoryToken
from megamind.store.memory_store import InMemoryStore


class _Emb:
    dim = 2

    async def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def _tok(i: int, ts: float, **kw) -> MemoryToken:
    return MemoryToken(id=f"t{i}", device_id="d", ts=ts, thumb_path="x", frame_hash="0", **kw)


async def test_sliding_windows_cover_each_token_twice():
    store = InMemoryStore()
    b = SummaryBuilder(Settings(summary_window_s=120, summary_step_s=60), store, _Emb())
    assert b.window_starts(130) == [120, 60]
    await b.add(_tok(1, 65, caption="kitchen", transcript="let's make coffee"))
    await b.add(_tok(2, 130, description="Jake pours coffee into a cup"))
    sums = {s.start_ts: s for s in await store.list_summaries()}
    assert set(sums) == {0, 60, 120}
    assert sums[60].token_ids == ["t1", "t2"]
    assert "let's make coffee" in sums[60].text and "pours coffee" in sums[60].text
    assert sums[120].token_ids == ["t2"]


async def test_rebuild_after_transcript_merge():
    store = InMemoryStore()
    b = SummaryBuilder(Settings(summary_window_s=120, summary_step_s=60), store, _Emb())
    t = _tok(1, 65, caption="desk")
    await b.add(t)
    t.transcript = "the password is falcon"
    await b.add(t)
    assert all("falcon" in s.text for s in await store.list_summaries())
