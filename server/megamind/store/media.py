"""Pixel tiers (docs/stream1-memory-engine.pdf §5): hot JPEG -> cold AV1 chunk per episode -> text only.

Tokens keep `id`, so `/v1/thumbs/{id}.jpg` works on every tier: hot reads the JPEG, cold decodes
frame `chunk_offset` from `chunk_id`, text returns 404 except episode representatives, which stay
as JPEG forever. Text, OCR, transcripts and vectors are never aged.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

from PIL import Image
from ulid import ULID

from megamind.config import Settings
from megamind.models import MemoryToken
from megamind.store.base import MemoryStore

log = logging.getLogger(__name__)
_MAX_CHUNK = 600
_CACHE = 256


def _pick_encoder() -> list[str] | None:
    if not shutil.which("ffmpeg"):
        return None
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "libsvtav1" in out:
        return ["-c:v", "libsvtav1", "-preset", "8"]
    if "libaom-av1" in out:
        return ["-c:v", "libaom-av1", "-cpu-used", "8", "-row-mt", "1", "-b:v", "0"]
    return None


class MediaStore:
    def __init__(self, settings: Settings, store: MemoryStore) -> None:
        self.s = settings
        self.store = store
        self.chunk_dir = settings.chunk_dir
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = _pick_encoder()
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._lock = asyncio.Lock()

    # -- read --------------------------------------------------------------------------
    async def read(self, token: MemoryToken) -> bytes | None:
        if token.thumb_path and Path(token.thumb_path).exists():
            return Path(token.thumb_path).read_bytes()
        if token.tier != "cold" or not token.chunk_id or token.chunk_offset is None:
            return None
        if token.id in self._cache:
            self._cache.move_to_end(token.id)
            return self._cache[token.id]
        data = await self._decode(self.chunk_dir / f"{token.chunk_id}.mp4", token.chunk_offset)
        if data:
            self._cache[token.id] = data
            if len(self._cache) > _CACHE:
                self._cache.popitem(last=False)
        return data

    @staticmethod
    async def _decode(path: Path, offset: int) -> bytes | None:
        if not path.exists():
            return None
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"select=eq(n\\,{offset})",
            "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "3", "-",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )  # fmt: skip
        out, err = await proc.communicate()
        if proc.returncode or not out:
            log.warning("decode %s#%d failed: %s", path.name, offset, err.decode()[-200:])
            return None
        return out

    # -- age ---------------------------------------------------------------------------
    def _encode(self, jpegs: list[bytes], out: Path) -> int:
        """Encode JPEGs (resized to one size) to AV1 mp4 at 1 fps; returns the decoded frame count."""
        first = Image.open(io.BytesIO(jpegs[0]))
        w = min(self.s.cold_width, first.width) // 2 * 2
        h = int(first.height * w / first.width) // 2 * 2
        buf = io.BytesIO()
        for j in jpegs:
            Image.open(io.BytesIO(j)).convert("RGB").resize((w, h)).save(buf, format="JPEG", quality=95)
        tmp = out.with_suffix(".tmp.mp4")
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", "1", "-c:v", "mjpeg",
               "-i", "-", *self.encoder, "-crf", str(self.s.cold_crf), "-g", "60", "-pix_fmt", "yuv420p",
               str(tmp)]  # fmt: skip
        subprocess.run(cmd, input=buf.getvalue(), check=True, capture_output=True)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(tmp)],
            capture_output=True, text=True,
        )  # fmt: skip
        n = int(probe.stdout.strip() or 0)
        tmp.rename(out)
        return n

    async def age(self, now: float | None = None, hot_s: float | None = None, cold_s: float | None = None) -> dict:
        """Move hot tokens older than `hot_s` into AV1 chunks and drop chunks older than `cold_s`."""
        now = time.time() if now is None else now
        hot_s = self.s.hot_retention_h * 3600 if hot_s is None else hot_s
        cold_s = self.s.cold_retention_days * 86400 if cold_s is None else cold_s
        async with self._lock:
            reps = {e.rep_token_id for e in await self.store.list_episodes(limit=10**9)}
            stats = {"to_cold": 0, "chunks": 0, "jpeg_bytes": 0, "chunk_bytes": 0, "to_text": 0, "skipped": None}
            if self.encoder is None:
                stats["skipped"] = "no AV1 encoder (ffmpeg with libsvtav1/libaom-av1)"
            else:
                await self._to_cold(now - hot_s, reps, stats)
            await self._to_text(now - cold_s, reps, stats)
            return stats

    async def _to_cold(self, cutoff: float, reps: set[str], stats: dict) -> None:
        groups: dict[tuple[str, str], list[MemoryToken]] = {}
        for t in await self.store.list_tokens(time_to=cutoff, tier="hot"):
            if t.id in reps or not t.thumb_path or not Path(t.thumb_path).exists():
                continue
            groups.setdefault((t.device_id, t.episode_id or ""), []).append(t)
        for toks in groups.values():
            for i in range(0, len(toks), _MAX_CHUNK):
                part = toks[i : i + _MAX_CHUNK]
                jpegs = [Path(t.thumb_path).read_bytes() for t in part]
                cid = f"ck_{ULID()}"
                out = self.chunk_dir / f"{cid}.mp4"
                try:
                    n = await asyncio.to_thread(self._encode, jpegs, out)
                except subprocess.CalledProcessError as exc:
                    log.warning("AV1 encode failed: %s", exc.stderr.decode()[-300:])
                    continue
                if n != len(part):
                    log.warning("chunk %s has %d frames, expected %d; keeping JPEGs", cid, n, len(part))
                    out.unlink(missing_ok=True)
                    continue
                paths = [Path(t.thumb_path) for t in part]
                for off, t in enumerate(part):
                    t.tier, t.chunk_id, t.chunk_offset, t.thumb_path = "cold", cid, off, ""
                await self.store.add_many(part)
                stats["jpeg_bytes"] += sum(len(j) for j in jpegs)
                stats["chunk_bytes"] += out.stat().st_size
                for p in paths:
                    p.unlink(missing_ok=True)
                stats["to_cold"] += len(part)
                stats["chunks"] += 1

    async def _to_text(self, cutoff: float, reps: set[str], stats: dict) -> None:
        by_chunk: dict[str, list[MemoryToken]] = {}
        for t in await self.store.list_tokens(time_to=cutoff, tier="cold"):
            by_chunk.setdefault(t.chunk_id or "", []).append(t)
        for t in await self.store.list_tokens(time_to=cutoff, tier="hot"):
            if t.id not in reps:
                by_chunk.setdefault("", []).append(t)
        for cid, toks in by_chunk.items():
            if cid and len(toks) != len(await self._chunk_members(cid)):
                continue  # chunk still has younger frames
            for t in toks:
                if t.thumb_path:
                    Path(t.thumb_path).unlink(missing_ok=True)
                t.tier, t.chunk_id, t.chunk_offset, t.thumb_path = "text", None, None, ""
            await self.store.add_many(toks)
            if cid:
                (self.chunk_dir / f"{cid}.mp4").unlink(missing_ok=True)
            stats["to_text"] += len(toks)

    async def _chunk_members(self, cid: str) -> list[MemoryToken]:
        return [t for t in await self.store.list_tokens(tier="cold") if t.chunk_id == cid]

    async def delete_files(self, tokens: list[MemoryToken]) -> None:
        for t in tokens:
            if t.thumb_path:
                Path(t.thumb_path).unlink(missing_ok=True)
            if t.chunk_id:
                (self.chunk_dir / f"{t.chunk_id}.mp4").unlink(missing_ok=True)
            self._cache.pop(t.id, None)

    def disk_bytes(self) -> dict[str, int]:
        def du(p: Path) -> int:
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

        return {
            "hot_jpeg": du(self.s.thumb_dir),
            "cold_chunks": du(self.chunk_dir),
            "index": du(self.s.lancedb_path) if self.s.store_backend == "lancedb" else 0,
        }
