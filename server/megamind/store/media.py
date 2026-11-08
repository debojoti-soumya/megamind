"""Tiered frame media: hot JPEG -> cold AV1 chunk per episode -> text only.

Ported from PR #54 (Devin, "Stream 1 memory engine", `store/media.py`), adapted to main's store and
privacy rules. A token keeps its `id` on every tier, so `GET /v1/thumbs/{id}.jpg` works throughout:

- **hot**: the JPEG at `thumb_path` (the stored thumbnail, already face-blurred when PRIVACY_BLUR_FACES is on).
- **cold**: frame `chunk_offset` of `<chunk_dir>/<chunk_id>.mp4`, an AV1 mp4 at 1 fps holding one
  (device, episode) run of frames, decoded on demand with an LRU of recent decodes.
- **text**: no pixels; caption, OCR, entities and both vectors stay searchable. `thumb_url` is `""`.

Only `kind="frame"` tokens are aged. Episode summary tokens own a JPEG copy of their representative frame
and are never aged, so every episode keeps one image; transcript tokens have no pixels.

Concurrency: token rewrites take `lock`, the enrichment worker's add/purge lock, so an aging pass can never
re-insert a token that a purge removed. `purge()` expects the caller to hold that lock (the worker does).
Purging a frame that lives in a chunk re-encodes the chunk without it (or, without an encoder, drops the
surviving frames to `tier="text"`): deleting the whole chunk would destroy un-purged frames, keeping it
would keep forgotten pixels.

ffmpeg with an AV1 encoder (libsvtav1, else libaom-av1) is optional. Without one, aging to cold is skipped
(reported in the result and in `GET /v1/stats`), JPEGs stay hot, and hot -> text still works.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import subprocess
import tempfile
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from ulid import ULID

from megamind.models import MemoryToken
from megamind.store.base import MemoryStore

log = logging.getLogger(__name__)

_MAX_CHUNK = 120  # frames per chunk: bounds decode-from-keyframe latency and rewrite cost on purge
_GOP = 30  # keyframe every 30 frames (30 s at 1 fps), so a cold read decodes at most 30 frames
_CACHE = 256  # decoded cold frames kept in memory
_DECODE_CONCURRENCY = 4
_TEXT_BATCH = 256  # loose frames dropped to text per locked store rewrite
_UNSET: Any = object()


def _find_ffmpeg() -> str | None:
    from megamind.replay.render import find_ffmpeg  # PATH, then the optional imageio-ffmpeg binary

    return find_ffmpeg()


def _pick_encoder(ffmpeg: str | None) -> list[str] | None:
    """ffmpeg args for the best available AV1 encoder, or None."""
    if not ffmpeg:
        return None
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if "libsvtav1" in out:
        return ["-c:v", "libsvtav1", "-preset", "8"]
    if "libaom-av1" in out:
        return ["-c:v", "libaom-av1", "-cpu-used", "8", "-row-mt", "1", "-b:v", "0"]
    return None


def _du(p: Path | None) -> int:
    if p is None or not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


class MediaStore:
    def __init__(self, settings: Any, store: MemoryStore, *, lock: asyncio.Lock | None = None) -> None:
        self.store = store
        self.thumb_dir = Path(settings.thumb_dir)
        self.chunk_dir = Path(settings.chunk_dir) if settings.chunk_dir else self.thumb_dir.parent / "chunks"
        self.index_dir = Path(settings.lancedb_path) if settings.store_backend == "lancedb" else None
        self.hot_s = settings.hot_retention_h * 3600.0
        self.cold_s = settings.cold_retention_days * 86400.0
        self.cold_width = settings.cold_width
        self.cold_crf = settings.cold_crf
        self.lock = lock or asyncio.Lock()  # shared with EnrichmentWorker: store.add vs purge vs aging
        self._age_lock = asyncio.Lock()  # one aging pass at a time (background loop vs POST /admin/age)
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._decode_sem = asyncio.Semaphore(_DECODE_CONCURRENCY)
        self._ffmpeg: str | None = _UNSET
        self._encoder: list[str] | None = _UNSET
        self.decodes = 0  # cold decodes that missed the cache (tests, stats)

    # ------------------------------------------------------------------ tools

    @property
    def ffmpeg(self) -> str | None:
        if self._ffmpeg is _UNSET:
            self._ffmpeg = _find_ffmpeg()
        return self._ffmpeg

    @property
    def encoder(self) -> list[str] | None:
        """Lazily probed so app startup never shells out."""
        if self._encoder is _UNSET:
            self._encoder = _pick_encoder(self.ffmpeg)
        return self._encoder

    async def probe_encoder(self) -> list[str] | None:
        """`encoder`, probed off the event loop (the first probe runs `ffmpeg -encoders`)."""
        if self._encoder is _UNSET:
            await asyncio.to_thread(lambda: self.encoder)
        return self.encoder

    def chunk_path(self, chunk_id: str) -> Path:
        return self.chunk_dir / f"{chunk_id}.mp4"

    # ------------------------------------------------------------------ read

    async def read(self, token: MemoryToken) -> bytes | None:
        """JPEG bytes for any tier, or None (no image / unreadable). A stale token (aged or moved to a new
        chunk since it was fetched) is re-read once from the store."""
        data = await self._read(token)
        if data is None and token.kind == "frame":
            fresh = await self.store.get(token.id)
            if fresh is not None and _media_key(fresh) != _media_key(token):
                data = await self._read(fresh)
        return data

    async def _read(self, t: MemoryToken) -> bytes | None:
        if not t.has_image:
            return None
        if t.tier == "cold":
            return await self._read_cold(t.chunk_id or "", int(t.chunk_offset or 0))
        try:
            return Path(t.thumb_path).read_bytes()
        except OSError:
            return None

    async def _read_cold(self, chunk_id: str, offset: int) -> bytes | None:
        key = f"{chunk_id}#{offset}"
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self.chunk_path(chunk_id)
        if not path.is_file() or not self.ffmpeg:
            return None
        async with self._decode_sem:
            data = await self._decode_one(path, offset)
        if data and path.is_file():  # not cached when a purge/aging removed the chunk mid-decode
            self.decodes += 1
            self._cache[key] = data
            while len(self._cache) > _CACHE:
                self._cache.popitem(last=False)
        return data

    async def _decode_one(self, path: Path, offset: int) -> bytes | None:
        # chunks are 1 fps, so frame N starts at N s: seek half a frame early (accurate seek decodes from the
        # previous keyframe and drops everything before it) and take the first frame out.
        seek = ["-ss", f"{offset - 0.5:.3f}"] if offset > 0 else []
        proc = await asyncio.create_subprocess_exec(
            self.ffmpeg, "-v", "error", *seek, "-i", str(path), "-frames:v", "1",
            "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "3", "-",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )  # fmt: skip
        out, err = await proc.communicate()
        if proc.returncode or not out:
            log.warning("decode %s#%d failed: %s", path.name, offset, err.decode(errors="replace")[-200:])
            return None
        return out

    def _decode_all(self, path: Path) -> list[bytes]:
        """Every frame of a chunk as JPEG, in order (purge rewrites)."""
        assert self.ffmpeg
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".decode-", dir=self.chunk_dir) as td:
            cmd = [self.ffmpeg, "-v", "error", "-i", str(path), "-fps_mode", "passthrough", "-q:v", "2",
                   str(Path(td) / "%06d.jpg")]  # fmt: skip
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            return [p.read_bytes() for p in sorted(Path(td).glob("*.jpg"))]

    def _drop_cache(self, chunk_id: str) -> None:
        for k in [k for k in self._cache if k.startswith(f"{chunk_id}#")]:
            del self._cache[k]

    # ------------------------------------------------------------------ encode

    def _encode(self, jpegs: Sequence[bytes], out: Path) -> int:
        """Encode JPEGs (resized to the first frame's size, <= cold_width) to an AV1 mp4 at 1 fps.
        Returns the frame count read back from the file; `out` only appears once complete."""
        assert self.ffmpeg and self.encoder
        with Image.open(io.BytesIO(jpegs[0])) as first:
            w0, h0 = first.size
        w = max(2, min(self.cold_width, w0) // 2 * 2)
        h = max(2, round(h0 * w / w0) // 2 * 2)
        buf = io.BytesIO()
        for j in jpegs:
            with Image.open(io.BytesIO(j)) as im:
                im = im.convert("RGB")
                if im.size != (w, h):
                    im = im.resize((w, h), Image.Resampling.LANCZOS)
                im.save(buf, format="JPEG", quality=95)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.tmp.mp4")
        try:
            cmd = [self.ffmpeg, "-v", "error", "-y", "-f", "image2pipe", "-framerate", "1", "-c:v", "mjpeg",
                   "-i", "-", *self.encoder, "-crf", str(self.cold_crf), "-g", str(_GOP), "-pix_fmt", "yuv420p",
                   "-an", str(tmp)]  # fmt: skip
            subprocess.run(cmd, input=buf.getvalue(), check=True, capture_output=True, timeout=600)
            n = self._count_frames(tmp)
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
        return n

    def _count_frames(self, path: Path) -> int:
        """Decode the file and count frames (verification before any JPEG is deleted)."""
        proc = subprocess.run(
            [self.ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-",
             "-progress", "pipe:1", "-nostats"],
            capture_output=True, text=True, timeout=300,
        )  # fmt: skip
        frames = [line.split("=", 1)[1] for line in proc.stdout.splitlines() if line.startswith("frame=")]
        return int(frames[-1]) if proc.returncode == 0 and frames else -1

    # ------------------------------------------------------------------ age

    async def age(self, now: float | None = None, hot_s: float | None = None, cold_s: float | None = None) -> dict:
        """One aging pass: frames older than `cold_s` lose their pixels (tier text), then hot frames older
        than `hot_s` move into AV1 chunks. Defaults: HOT_RETENTION_H / COLD_RETENTION_DAYS."""
        now = time.time() if now is None else now
        hot_s = self.hot_s if hot_s is None else hot_s
        cold_s = self.cold_s if cold_s is None else cold_s
        stats: dict[str, Any] = {
            "to_cold": 0, "chunks": 0, "jpeg_bytes": 0, "chunk_bytes": 0, "to_text": 0, "aborted": 0, "skipped": None,
        }  # fmt: skip
        async with self._age_lock:
            await self._to_text(now - cold_s, stats)
            if await self.probe_encoder() is None:
                stats["skipped"] = "no AV1 encoder (ffmpeg with libsvtav1 or libaom-av1)"
            else:
                await self._to_cold(now - hot_s, stats)
        if stats["to_cold"] or stats["to_text"]:
            log.info("media aging: %s", stats)
        return stats

    async def _to_cold(self, cutoff: float, stats: dict) -> None:
        # planning scan without vectors (can be every hot frame on a first pass); _encode_group re-reads
        # each <= _MAX_CHUNK part in full before writing it back
        groups: dict[tuple[str, str], list[MemoryToken]] = {}
        for t in await self.store.list_tokens(time_to=cutoff, kinds=("frame",), tier="hot", vectors=False):
            if t.thumb_path and Path(t.thumb_path).is_file():
                groups.setdefault((t.device_id, t.episode_id or ""), []).append(t)
        for toks in groups.values():
            for i in range(0, len(toks), _MAX_CHUNK):
                await self._encode_group(toks[i : i + _MAX_CHUNK], stats)

    async def _encode_group(self, part: list[MemoryToken], stats: dict) -> None:
        try:
            jpegs = [Path(t.thumb_path).read_bytes() for t in part]  # the stored (blurred) thumbnails
        except OSError:
            return  # purged meanwhile; the next pass regroups
        cid = f"ck_{ULID()}"
        out = self.chunk_path(cid)
        try:
            n = await asyncio.to_thread(self._encode, jpegs, out)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            err = getattr(exc, "stderr", b"") or b""
            log.warning("AV1 encode failed (%s): %s", exc, err.decode(errors="replace")[-300:])
            stats["aborted"] += 1
            return
        if n != len(part):
            log.warning("chunk %s decodes to %d frames, expected %d; keeping JPEGs", cid, n, len(part))
            out.unlink(missing_ok=True)
            stats["aborted"] += 1
            return
        async with self.lock:
            fresh = {t.id: t for t in await self.store.get_many([t.id for t in part])}
            if any(_media_key(fresh.get(t.id)) != _media_key(t) for t in part):
                # purged or changed while encoding: the chunk may hold pixels the wearer asked to forget
                out.unlink(missing_ok=True)
                stats["aborted"] += 1
                return
            moved = [
                fresh[t.id].model_copy(update={"tier": "cold", "chunk_id": cid, "chunk_offset": off, "thumb_path": ""})
                for off, t in enumerate(part)
            ]
            await self.store.rewrite_many(moved)
            for t in part:
                Path(t.thumb_path).unlink(missing_ok=True)
        stats["jpeg_bytes"] += sum(len(j) for j in jpegs)
        stats["chunk_bytes"] += out.stat().st_size if out.exists() else 0
        stats["to_cold"] += len(part)
        stats["chunks"] += 1

    async def _to_text(self, cutoff: float, stats: dict) -> None:
        cold = await self.store.list_tokens(time_to=cutoff, kinds=("frame",), tier="cold", vectors=False)
        loose = await self.store.list_tokens(time_to=cutoff, kinds=("frame",), tier="hot", vectors=False)
        loose += [t for t in cold if not t.chunk_id]
        for cid in dict.fromkeys(t.chunk_id for t in cold if t.chunk_id):
            async with self.lock:
                members = await self.store.list_tokens(tier="cold", chunk_id=cid)
                if not members or any(m.ts > cutoff for m in members):
                    continue  # the chunk still holds younger frames
                await self.store.rewrite_many([_as_text(m) for m in members])
                self.chunk_path(cid).unlink(missing_ok=True)
                self._drop_cache(cid)
                stats["to_text"] += len(members)
        for i in range(0, len(loose), _TEXT_BATCH):  # re-read in full (with vectors) batch by batch
            planned = {t.id: _media_key(t) for t in loose[i : i + _TEXT_BATCH]}
            async with self.lock:
                fresh = [t for t in await self.store.get_many(list(planned)) if _media_key(t) == planned[t.id]]
                await self.store.rewrite_many([_as_text(t) for t in fresh])
                for t in fresh:
                    if t.thumb_path:
                        Path(t.thumb_path).unlink(missing_ok=True)
                stats["to_text"] += len(fresh)

    # ------------------------------------------------------------------ purge

    async def purge(self, tokens: Sequence[MemoryToken]) -> dict:
        """Remove the pixels of `tokens` that live in chunks, keeping every other frame of those chunks.
        Call with `lock` held, right before the rows are deleted (EnrichmentWorker.purge does)."""
        victims = {t.id for t in tokens}
        stats = {"chunks_rewritten": 0, "chunks_deleted": 0, "demoted_to_text": 0}
        for cid in dict.fromkeys(t.chunk_id for t in tokens if t.tier == "cold" and t.chunk_id):
            path = self.chunk_path(cid)
            members = await self.store.list_tokens(tier="cold", chunk_id=cid)
            survivors = sorted((m for m in members if m.id not in victims), key=lambda m: m.chunk_offset or 0)
            self._drop_cache(cid)
            if survivors:
                if await self._rewrite_chunk(path, survivors):
                    stats["chunks_rewritten"] += 1
                else:
                    stats["demoted_to_text"] += len(survivors)
            else:
                stats["chunks_deleted"] += 1
            path.unlink(missing_ok=True)
            self._drop_cache(cid)  # again: a decode that finished during the rewrite may have re-cached it
        return stats

    async def purge_window(self, device_id: str | None = None, since_ts: float | None = None) -> dict:
        """`purge()` for the tokens a `store.purge(device_id, since_ts=...)` is about to delete."""
        tokens = await self.store.list_tokens(
            time_from=since_ts, device_id=device_id, kinds=("frame",), tier="cold", vectors=False
        )  # purge() only needs ids and chunk ids; it re-reads each affected chunk's members in full
        return await self.purge(tokens)

    async def _rewrite_chunk(self, old: Path, survivors: list[MemoryToken]) -> bool:
        """Re-encode `survivors` into a new chunk. False (and survivors demoted to text) when impossible."""
        moved: list[MemoryToken] | None = None
        if await self.probe_encoder() is not None and old.is_file():
            cid = f"ck_{ULID()}"
            out = self.chunk_path(cid)
            try:
                frames = await asyncio.to_thread(self._decode_all, old)
                jpegs = [frames[int(s.chunk_offset or 0)] for s in survivors]
                if await asyncio.to_thread(self._encode, jpegs, out) == len(jpegs):
                    moved = [s.model_copy(update={"chunk_id": cid, "chunk_offset": i}) for i, s in enumerate(survivors)]
                else:
                    out.unlink(missing_ok=True)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
                log.warning("chunk rewrite of %s failed (%s); demoting survivors to text", old.name, exc)
                out.unlink(missing_ok=True)
        if moved is None:
            await self.store.rewrite_many([_as_text(s) for s in survivors])
            return False
        await self.store.rewrite_many(moved)
        return True

    # ------------------------------------------------------------------ stats

    def disk_bytes(self) -> dict[str, int]:
        return {"hot_jpeg": _du(self.thumb_dir), "cold_chunks": _du(self.chunk_dir), "index": _du(self.index_dir)}


def _media_key(t: MemoryToken | None) -> tuple | None:
    return None if t is None else (t.tier, t.chunk_id, t.chunk_offset, t.thumb_path)


def _as_text(t: MemoryToken) -> MemoryToken:
    return t.model_copy(update={"tier": "text", "chunk_id": None, "chunk_offset": None, "thumb_path": ""})


async def run_aging(state: Any, **kwargs: Any) -> dict:
    """`MediaStore.age` for the server: also drops real video clips (POST /v1/video) older than the hot
    retention (`hot_s` when given, else VIDEO_RETENTION_H / HOT_RETENTION_H) and rendered replay clips once
    pixels were dropped."""
    stats = await state.media.age(**kwargs)
    video = getattr(state, "video", None)
    if video is not None:
        stats["video_deleted"] = video.age(now=kwargs.get("now"), retention_s=kwargs.get("hot_s"))
    cache = getattr(state, "replay_cache", None)
    if (stats["to_text"] or stats.get("video_deleted")) and cache is not None:
        cache.clear()
    return stats


async def aging_loop(state: Any, interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            await run_aging(state)
        except Exception:  # noqa: BLE001
            log.exception("media aging pass failed")
