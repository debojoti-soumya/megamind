"""Render buffered frames to an H.264 mp4 with ffmpeg, plus a small on-disk cache.

ffmpeg is an optional system dependency: `find_ffmpeg()` uses `REPLAY_FFMPEG`, then `ffmpeg` on PATH,
then the binary bundled by the optional `imageio-ffmpeg` package (`pip install -e ".[replay]"`).
"""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image

from megamind.ingest.privacy import FaceBlur, blur_faces
from megamind.replay.buffer import BufferedFrame


class FfmpegMissing(RuntimeError):
    pass


FFMPEG_HINT = (
    "ffmpeg not found: install it (brew install ffmpeg / apt install ffmpeg), "
    'pip install -e ".[replay]" for a bundled binary, or set REPLAY_FFMPEG=/path/to/ffmpeg'
)


def find_ffmpeg(explicit: str = "") -> str | None:
    if explicit:
        return shutil.which(explicit) or (explicit if Path(explicit).is_file() else None)
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # optional extra

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001  (not installed, or no binary for this platform)
        return None


def render_mp4(
    frames: Sequence[tuple[float, bytes]],
    out: Path,
    *,
    ffmpeg: str,
    fps: int = 10,
    max_width: int = 960,
    tail_s: float = 1.0,
    timeout_s: float = 120.0,
    extra_vf: str = "",
) -> Path:
    """Each frame is held until the next one (like Recall's `frames_to_video`), so offsets in the mp4
    equal wall-clock offsets from the first frame. The last frame is held for `tail_s`.
    `extra_vf` is appended to the filter chain (the montage uses it for captions and fades)."""
    frames = sorted(frames, key=lambda f: f[0])
    if not frames:
        raise ValueError("no frames")
    with Image.open(io.BytesIO(frames[0][1])) as im:
        w, h = im.size
    if w > max_width:
        w, h = max_width, round(h * max_width / w)
    w, h = max(2, w - w % 2), max(2, h - h % 2)

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".render-") as td:
        tmp = Path(td)
        lines = ["ffconcat version 1.0"]
        for i, (ts, jpeg) in enumerate(frames):
            name = f"{i:06d}.jpg"
            (tmp / name).write_bytes(jpeg)
            nxt = frames[i + 1][0] if i + 1 < len(frames) else ts + tail_s
            lines += [f"file '{name}'", f"duration {max(nxt - ts, 0.01):.3f}"]
        lines.append(f"file '{len(frames) - 1:06d}.jpg'")  # concat demuxer quirk: repeat the last frame
        (tmp / "list.ffconcat").write_text("\n".join(lines) + "\n")
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps},format=yuv420p"
        ) + (f",{extra_vf}" if extra_vf else "")
        tmp_out = tmp / "out.mp4"
        cmd = [
            ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.ffconcat"),
            "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-movflags", "+faststart",
            str(tmp_out),
        ]  # fmt: skip
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr.strip()[-500:]}") from e
        os.replace(tmp_out, out)
    return out


def _read_frames(frames: Sequence[BufferedFrame]) -> list[tuple[float, bytes]]:
    """Snapshot JPEG bytes; spooled frames evicted since the request was planned are skipped."""
    out = []
    for f in frames:
        try:
            out.append((f.ts, f.read()))
        except FileNotFoundError:
            continue
    return out


class BlurredFrames:
    """Face-blurred copies of replay source frames (PRIVACY_BLUR_FACES), bounded LRU by bytes.

    Each source frame is blurred once and reused by every clip that covers it, so a rolling replay window
    (new frames change the clip cache key) only runs face detection on the frames it hasn't seen."""

    def __init__(self, max_bytes: int = 64 << 20) -> None:
        self.max_bytes = max_bytes
        self._items: OrderedDict[tuple, bytes] = OrderedDict()
        self._nbytes = 0
        self._lock = threading.Lock()  # blur() runs in worker threads, possibly two renders at once

    def blur(self, face_blur: FaceBlur, tag: str, frames: Sequence[BufferedFrame]) -> list[BufferedFrame]:
        """Blurred in-memory copies of `frames` (call off the event loop); evicted spool frames are skipped."""
        out = []
        for f in frames:
            key = (tag, f.ts, f.size)
            with self._lock:
                data = self._items.get(key)
                if data is not None:
                    self._items.move_to_end(key)
            if data is None:
                try:
                    raw = f.read()
                except FileNotFoundError:
                    continue
                data, _ = blur_faces(raw, face_blur.detector)
                self._put(key, data)
            out.append(BufferedFrame(ts=f.ts, size=len(data), data=data))
        return out

    def _put(self, key: tuple, data: bytes) -> None:
        with self._lock:
            if key in self._items:
                return
            self._items[key] = data
            self._nbytes += len(data)
            while self._nbytes > self.max_bytes and len(self._items) > 1:
                _, old = self._items.popitem(last=False)
                self._nbytes -= len(old)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._nbytes = 0

    def __len__(self) -> int:
        return len(self._items)


class ReplayCache:
    """Rendered mp4s keyed by content; keeps the newest `max_files`, one render per key at a time,
    and at most `max_concurrent` ffmpeg processes overall."""

    def __init__(self, cache_dir: Path, max_files: int = 32, max_concurrent: int = 2) -> None:
        self.dir = Path(cache_dir)
        self.max_files = max_files
        self._locks: dict[str, asyncio.Lock] = {}
        self._renders = asyncio.Semaphore(max(1, max_concurrent))

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.mp4"

    async def get_or_render(
        self,
        key: str,
        frames: Sequence[BufferedFrame],
        *,
        ffmpeg: str,
        fps: int,
        max_width: int,
        render: Callable[[Path], object] | None = None,
    ) -> Path:
        """`render(path)`, when given, replaces the frame render (e.g. real video cut by `megamind.video`)."""
        p = self.path(key)
        lock = self._locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                if not p.exists():
                    async with self._renders:
                        if render is not None:
                            await asyncio.to_thread(render, p)
                        else:
                            await asyncio.to_thread(self._render, frames, p, ffmpeg, fps, max_width)
                    self._prune(keep=p)
        finally:
            if not lock.locked():
                self._locks.pop(key, None)
        return p

    @staticmethod
    def _render(frames: Sequence[BufferedFrame], p: Path, ffmpeg: str, fps: int, max_width: int) -> None:
        payload = _read_frames(frames)  # off the event loop (spool mode reads from disk)
        if not payload:
            raise FileNotFoundError("buffered frames were evicted before rendering")
        render_mp4(payload, p, ffmpeg=ffmpeg, fps=fps, max_width=max_width)

    def clear(self) -> None:
        if self.dir.is_dir():
            for p in self.dir.glob("*.mp4"):
                p.unlink(missing_ok=True)

    def _prune(self, keep: Path) -> None:
        def mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except FileNotFoundError:
                return 0.0

        files = sorted(self.dir.glob("*.mp4"), key=mtime, reverse=True)
        for p in files[self.max_files :]:
            if p != keep:
                p.unlink(missing_ok=True)
