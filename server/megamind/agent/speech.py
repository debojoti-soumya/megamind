"""Sentence-by-sentence TTS for `POST /v1/chat/stream` (`TTS_STREAM`, default on).

`StreamingSpeaker` watches the answer deltas, starts a TTS call (`TTS.synthesize`, one file per sentence) as soon
as a sentence is complete while Claude is still writing the rest, and emits `audio` events `{seq, url, text}` in
sentence order as the clips finish. `finish()` speaks the tail, waits for every clip and concatenates them into
the one file `final.audio_url` points at, so clients that only know `audio_url` keep working.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path

from ulid import ULID

from megamind.agent.tts import SENTENCE_END, TTS

log = logging.getLogger(__name__)

Emit = Callable[[str, dict], Awaitable[None]]
GAP_S = 0.15  # silence between sentences in the concatenated file (as OpenAITTS._render)


class SentenceBuffer:
    """Incremental `tts.split_sentences`: feed text deltas, get back complete sentences to speak."""

    def __init__(self, min_chars: int = 25) -> None:
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        pos = 0
        while m := SENTENCE_END.search(self.buf, pos):
            head = self.buf[: m.start()].strip()
            if len(head) < self.min_chars:  # too short to speak alone: it joins the next sentence
                pos = m.end()
                continue
            out.append(head)
            self.buf, pos = self.buf[m.end() :], 0
        return out

    def flush(self) -> str:
        rest, self.buf = self.buf.strip(), ""
        return rest


def _words(text: str) -> str:
    return " ".join(text.split())


def concat_wavs(paths: list[Path], out: Path, gap_s: float = GAP_S) -> bool:
    """Join 16-bit PCM WAV clips (same rate/channels) with `gap_s` of silence; False when they don't match."""
    frames: list[bytes] = []
    params = None
    for p in paths:
        with wave.open(str(p)) as w:
            cur = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            if params is None:
                params = cur
            elif cur != params:
                return False
            frames.append(w.readframes(w.getnframes()))
    if params is None:
        return False
    channels, width, rate = params
    gap = bytes(int(rate * gap_s) * channels * width)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(gap.join(frames))
    return True


class StreamingSpeaker:
    def __init__(self, tts: TTS, audio_dir: Path, emit: Emit, *, min_chars: int = 25) -> None:
        self.tts, self.audio_dir, self.emit = tts, audio_dir, emit
        self.min_chars = min_chars
        self.t0 = time.perf_counter()
        self.first_audio_ms: float | None = None
        self._new_generation()

    def _new_generation(self) -> None:
        self.audio_id = str(ULID())
        self.buf = SentenceBuffer(self.min_chars)
        self.texts: list[str] = []
        self.started_ms: list[float] = []  # when each clip's TTS call began (ms since the turn started)
        self.parts: list[asyncio.Task[str | None]] = []
        self.sent = 0  # clips whose `audio` event went out (the sender's cursor; it survives sender restarts)
        self.sender: asyncio.Task[None] | None = None

    async def feed(self, text: str) -> None:
        for sentence in self.buf.feed(text):
            self._start(sentence)

    def _start(self, sentence: str) -> None:
        seq = len(self.texts)
        self.texts.append(sentence)
        self.started_ms.append((time.perf_counter() - self.t0) * 1000)
        self.parts.append(asyncio.create_task(self.tts.synthesize(sentence, f"{self.audio_id}-{seq}")))
        if self.sender is None:
            self.sender = asyncio.create_task(self._send())

    async def _send(self) -> None:
        """Emit `audio` events strictly in sentence order (a later clip waits for the earlier ones), each once."""
        while self.sent < len(self.parts):
            i = self.sent
            url = await self.parts[i]
            if url:
                if self.first_audio_ms is None:
                    self.first_audio_ms = (time.perf_counter() - self.t0) * 1000
                await self.emit("audio", {"seq": i, "url": url, "text": self.texts[i]})
            self.sent = i + 1
        self.sender = None

    async def reset(self) -> None:
        """The streamed text was withdrawn (`reset` event): drop its clips; the client discards them too."""
        await self.cancel()
        self._new_generation()

    async def cancel(self) -> None:
        tasks = [t for t in [*self.parts, self.sender] if t is not None and not t.done()]
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

    async def finish(self, answer: str) -> str | None:
        """Speak the tail, wait for every clip (all `audio` events are emitted before this returns) and return
        the URL of the whole answer: the clips concatenated, or one fresh TTS call when the streamed sentences
        are not the final answer (nothing streamed, or the agent replaced the text)."""
        if rest := self.buf.flush():
            self._start(rest)
        if self.sender is not None:
            await self.sender
        urls = [await p for p in self.parts]
        if not any(urls):
            return None  # TTS disabled (mock) or every clip failed, as `tts.synthesize` would have returned
        if _words(" ".join(self.texts)) != _words(answer) or not all(urls):
            log.info("tts stream: streamed text != answer, or a clip failed; speaking the answer whole")
            return await self.tts.synthesize(answer, self.audio_id)
        paths = [self.audio_dir / Path(u).name for u in urls]
        out = self.audio_dir / f"{self.audio_id}.wav"
        try:
            ok = concat_wavs(paths, out)
        except (wave.Error, OSError):
            log.exception("tts stream: concat failed")
            ok = False
        if not ok:
            return await self.tts.synthesize(answer, self.audio_id)
        log.info(
            "tts stream: %d clips started at %s ms, first audio at %s ms, done at %.0f ms",
            len(paths),
            [round(t) for t in self.started_ms],
            f"{self.first_audio_ms:.0f}" if self.first_audio_ms is not None else "-",
            (time.perf_counter() - self.t0) * 1000,
        )
        return f"/v1/audio/{self.audio_id}.wav"
