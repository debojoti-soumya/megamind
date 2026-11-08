"""On-disk store of real H.264 clips uploaded by the headset (`POST /v1/video`).

Layout: `<video_dir>/d_<quoted device_id>/<ULID>.mp4` plus a `.json` sidecar with what ffprobe
measured (`ts_start`, `duration`, `width`, `height`, `fps`, `bytes`). The sidecar is written last, so a clip
without one is an interrupted upload and is deleted on startup. The per-device index (sorted by `ts_start`)
lives in memory and is rebuilt from the sidecars when the server starts.

Retention (tiered media #108): a clip is hot media, so it is dropped once it is older than `retention_s`
(default `HOT_RETENTION_H`, when keyframes move to AV1), and the oldest clips go first once the store
exceeds `max_bytes`. Purge (#64): `purge(device_id, since_ts)` deletes every clip that holds footage at or
after `since_ts` (the whole clip, so a clip spanning `since_ts` goes too).
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, unquote

log = logging.getLogger(__name__)

_DIR_PREFIX = "d_"  # keeps device ids like "." / ".." inside the video dir
PART_SUFFIX = ".part"


@dataclass(frozen=True)
class VideoClip:
    id: str
    device_id: str
    ts_start: float
    duration: float
    path: Path
    width: int = 0
    height: int = 0
    fps: float = 0.0
    bytes: int = 0

    @property
    def ts_end(self) -> float:
        return self.ts_start + self.duration

    def sidecar(self) -> Path:
        return self.path.with_suffix(".json")


@dataclass(frozen=True)
class Segment:
    """`[start, end]` (unix seconds) of `clip`, i.e. clip offsets `start - clip.ts_start` .. `end - clip.ts_start`."""

    clip: VideoClip
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def offset(self) -> float:
        return self.start - self.clip.ts_start


def privacy_blocks_video(settings) -> bool:
    """Real video is stored and served unblurred, so it is off while face blurring is on."""
    return bool(getattr(settings, "privacy_blur_faces", False) or getattr(settings, "privacy_blur_faces_vlm", False))


def covered_s(segments: list[Segment]) -> float:
    return sum(s.duration for s in segments)


@dataclass
class Probe:
    format_name: str
    codec: str
    width: int
    height: int
    duration: float
    fps: float


def find_ffprobe(ffmpeg: str | None) -> str | None:
    """`ffprobe` next to `ffmpeg`, then on PATH. imageio-ffmpeg ships no ffprobe: `probe` then parses `ffmpeg -i`."""
    if ffmpeg:
        sib = Path(ffmpeg).with_name("ffprobe" + Path(ffmpeg).suffix)
        if sib.is_file():
            return str(sib)
    return shutil.which("ffprobe")


def _rate(s: str | None) -> float:
    try:
        num, _, den = (s or "0/1").partition("/")
        v = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def probe(path: Path, *, ffmpeg: str | None, timeout_s: float = 20.0) -> Probe:
    """Container, codec, size, duration and frame rate of the first video stream. Raises ValueError when the
    file is not a decodable video, FileNotFoundError when neither ffprobe nor ffmpeg is available."""
    ffprobe = find_ffprobe(ffmpeg)
    if ffprobe:
        # `-f mov` pins the demuxer: an upload is never content-sniffed as a playlist (hls/concat) that would make
        # ffprobe open other files or URLs.
        cmd = [ffprobe, "-v", "error", "-f", "mov", "-print_format", "json", "-show_format", "-show_streams", str(path)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0:
            err = r.stderr.replace(str(path), "upload").strip()  # no server paths in the 415 detail
            raise ValueError(f"ffprobe: {err[-300:] or 'not a media file'}")
        try:
            info = json.loads(r.stdout or "{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"ffprobe output: {e}") from None
        vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        if vs is None:
            raise ValueError("no video stream")
        fmt = info.get("format", {})
        try:
            dur = float(fmt.get("duration") or vs.get("duration") or 0.0)
        except ValueError:
            dur = 0.0
        return Probe(
            format_name=str(fmt.get("format_name", "")),
            codec=str(vs.get("codec_name", "")),
            width=int(vs.get("width") or 0),
            height=int(vs.get("height") or 0),
            duration=dur if math.isfinite(dur) else 0.0,
            fps=_rate(vs.get("avg_frame_rate")) or _rate(vs.get("r_frame_rate")),
        )
    if not ffmpeg:
        raise FileNotFoundError("ffprobe/ffmpeg not found")
    return _probe_with_ffmpeg(path, ffmpeg, timeout_s)


def _probe_with_ffmpeg(path: Path, ffmpeg: str, timeout_s: float) -> Probe:
    """Fallback for ffmpeg-only installs: parse the stream summary `ffmpeg -i` prints, then count the duration."""
    import re

    cmd = [ffmpeg, "-hide_banner", "-f", "mov", "-i", str(path)]  # demuxer pinned, as in `probe`
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    err = r.stderr
    m_in = re.search(r"Input #0, ([^,\n]+(?:,[^,\n ]+)*), from", err)
    m_v = re.search(r"Stream #0:\d+[^\n]*: Video: (\w+)[^\n]*?, (\d{2,5})x(\d{2,5})", err)
    if m_in is None or m_v is None:
        raise ValueError("no video stream")
    m_d = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    dur = int(m_d.group(1)) * 3600 + int(m_d.group(2)) * 60 + float(m_d.group(3)) if m_d else 0.0
    m_f = re.search(r"(\d+(?:\.\d+)?) fps", err)
    return Probe(
        m_in.group(1), m_v.group(1), int(m_v.group(2)), int(m_v.group(3)), dur, float(m_f.group(1)) if m_f else 0.0
    )


def _device_dir(device_id: str) -> str:
    return _DIR_PREFIX + quote(device_id, safe="")


@dataclass
class _Purge:
    seq: int
    device_id: str | None
    since_ts: float | None


@dataclass
class VideoStore:
    root: Path
    retention_s: float = 72 * 3600.0
    max_bytes: int = 20 << 30
    _clips: dict[str, list[VideoClip]] = field(default_factory=dict)  # per device, sorted by ts_start
    _bytes: int = 0
    _seq: int = 0  # bumped by every purge; uploads that started before one re-check against it
    _purges: list[_Purge] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.root.is_dir():
            self._rehydrate()

    # ---- writes ----

    def staging_path(self, device_id: str, name: str) -> Path:
        d = self.root / _device_dir(device_id)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}{PART_SUFFIX}"

    @property
    def seq(self) -> int:
        return self._seq

    def purged_since(self, seq: int, device_id: str, ts_end: float) -> bool:
        """True when a purge that ran after `seq` covers footage of a clip of `device_id` ending at `ts_end`."""
        return any(
            p.seq > seq
            and (p.device_id is None or p.device_id == device_id)
            and (p.since_ts is None or ts_end > p.since_ts)
            for p in self._purges
        )

    def find_duplicate(self, device_id: str, ts_start: float, duration: float) -> VideoClip | None:
        """A stored clip with the same start and length (an upload retried after its response was lost)."""
        for c in self._clips.get(device_id, ()):
            if abs(c.ts_start - ts_start) <= 0.05 and abs(c.duration - duration) <= 0.1:
                return c
        return None

    def commit(self, staged: Path, clip_id: str, device_id: str, ts_start: float, p: Probe) -> VideoClip:
        """Move a validated staged upload into place (mp4 first, sidecar last) and index it."""
        final = staged.with_name(f"{clip_id}.mp4")  # ts_start lives in the sidecar (any finite float is valid)
        clip = VideoClip(
            id=clip_id, device_id=device_id, ts_start=ts_start, duration=p.duration, path=final,
            width=p.width, height=p.height, fps=round(p.fps, 3), bytes=staged.stat().st_size,
        )  # fmt: skip
        staged.replace(final)
        meta = {
            k: getattr(clip, k) for k in ("id", "device_id", "ts_start", "duration", "width", "height", "fps", "bytes")
        }
        clip.sidecar().write_text(json.dumps(meta))
        self._insert(clip)
        self.enforce_cap()
        return clip

    def purge(self, device_id: str | None = None, since_ts: float | None = None) -> int:
        """Delete every clip of `device_id` (all devices by default) holding footage at/after `since_ts`."""
        self._seq += 1
        self._purges = [*self._purges[-63:], _Purge(self._seq, device_id, since_ts)]
        n = 0
        for dev in [device_id] if device_id is not None else list(self._clips):
            for c in list(self._clips.get(dev, ())):
                if since_ts is None or c.ts_end > since_ts:
                    self._delete(c)
                    n += 1
        return n  # in-flight uploads re-check `purged_since` before they commit

    def age(self, now: float | None = None, retention_s: float | None = None) -> int:
        """Drop clips whose footage ended more than `retention_s` (default: the store's) before `now`."""
        now = time.time() if now is None else now
        cutoff = now - (self.retention_s if retention_s is None else retention_s)
        old = [c for clips in self._clips.values() for c in clips if c.ts_end <= cutoff]
        for c in old:
            self._delete(c)
        return len(old) + self.enforce_cap()

    def enforce_cap(self) -> int:
        n = 0
        while self._bytes > self.max_bytes > 0:
            oldest = min((cs[0] for cs in self._clips.values() if cs), key=lambda c: c.ts_start, default=None)
            if oldest is None:
                break
            self._delete(oldest)
            n += 1
        if n:
            log.info("video store over %d bytes: dropped %d oldest clip(s)", self.max_bytes, n)
        return n

    # ---- reads ----

    def clips(self, device_id: str) -> list[VideoClip]:
        return list(self._clips.get(device_id, ()))

    def segments(self, device_id: str, t0: float, t1: float) -> list[Segment]:
        """Non-overlapping, time-ordered pieces of stored clips covering parts of `[t0, t1]`."""
        out: list[Segment] = []
        cursor = t0
        for c in self._clips.get(device_id, ()):
            if c.ts_start >= t1:
                break
            a, b = max(c.ts_start, cursor), min(c.ts_end, t1)
            if b - a > 1e-3:
                out.append(Segment(c, a, b))
                cursor = b
        return out

    def covered(self, device_id: str, t0: float, t1: float) -> float:
        return covered_s(self.segments(device_id, t0, t1))

    def resolve_device(self, t0: float, t1: float, device_id: str | None = None) -> str | None:
        """`device_id` if it has video in `[t0, t1]`; with no device, the one with the most video there."""
        if device_id is not None:
            return device_id if self.covered(device_id, t0, t1) > 0 else None
        best, best_s = None, 0.0
        for dev in self._clips:
            s = self.covered(dev, t0, t1)
            if s > best_s:
                best, best_s = dev, s
        return best

    def newest_ts(self, device_id: str | None = None) -> float | None:
        devs = [device_id] if device_id is not None else list(self._clips)
        ends = [self._clips[d][-1].ts_end for d in devs if self._clips.get(d)]
        return max(ends) if ends else None

    def stats(self) -> dict:
        n = sum(len(cs) for cs in self._clips.values())
        secs = sum(c.duration for cs in self._clips.values() for c in cs)
        return {
            "clips": n,
            "seconds": round(secs, 1),
            "bytes": self._bytes,
            "devices": len([1 for cs in self._clips.values() if cs]),
        }

    # ---- internals ----

    def _insert(self, clip: VideoClip) -> None:
        cs = self._clips.setdefault(clip.device_id, [])
        bisect.insort(cs, clip, key=lambda c: c.ts_start)
        self._bytes += clip.bytes

    def _delete(self, clip: VideoClip) -> None:
        cs = self._clips.get(clip.device_id, [])
        if clip in cs:
            cs.remove(clip)
            self._bytes -= clip.bytes
        if not cs:
            self._clips.pop(clip.device_id, None)
        clip.path.unlink(missing_ok=True)
        clip.sidecar().unlink(missing_ok=True)

    def _rehydrate(self) -> None:
        n = 0
        for d in self.root.iterdir():
            if not d.is_dir() or not d.name.startswith(_DIR_PREFIX):
                continue
            device_id = unquote(d.name[len(_DIR_PREFIX) :])
            for p in d.iterdir():
                if p.name.endswith(PART_SUFFIX) or (p.suffix == ".mp4" and not p.with_suffix(".json").is_file()):
                    p.unlink(missing_ok=True)  # interrupted upload
                    continue
                if p.suffix != ".json":
                    continue
                mp4 = p.with_suffix(".mp4")
                try:
                    m = json.loads(p.read_text())
                    clip = VideoClip(
                        id=str(m["id"]), device_id=device_id, ts_start=float(m["ts_start"]),
                        duration=float(m["duration"]), path=mp4, width=int(m.get("width", 0)),
                        height=int(m.get("height", 0)), fps=float(m.get("fps", 0.0)), bytes=mp4.stat().st_size,
                    )  # fmt: skip
                except (OSError, ValueError, KeyError, TypeError):
                    p.unlink(missing_ok=True)
                    mp4.unlink(missing_ok=True)
                    continue
                self._insert(clip)
                n += 1
        if n:
            log.info("video store: %d clip(s) (%d bytes) in %s", n, self._bytes, self.root)
        self.enforce_cap()
