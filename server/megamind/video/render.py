"""Cut a wall-clock window out of stored real-video clips into one H.264 mp4.

Each `Segment` becomes one ffmpeg input (`-ss offset -t duration`, an accurate seek: ffmpeg decodes from the
keyframe before `offset` and drops the frames before it, so cuts are frame-exact, not keyframe-aligned). Every
input is normalised to one size and frame rate; a gap before the next segment (the 1-3 frames the headset drops
at a chunk boundary, or a missing chunk) is filled by holding the segment's last frame, so mp4 offsets equal
wall-clock offsets from the first segment's start, the same rule as the frame-based replay. The pieces are joined
with the concat filter and re-encoded with the same libx264 settings as `replay.render.render_mp4`, so montage
parts from either source concatenate without re-encoding.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from megamind.video.store import Segment


def output_size(segments: Sequence[Segment], max_width: int) -> tuple[int, int]:
    first = segments[0].clip
    w, h = first.width or max_width, first.height or round((first.width or max_width) * 3 / 4)
    if w > max_width:
        w, h = max_width, round(h * max_width / w)
    return max(2, w - w % 2), max(2, h - h % 2)


def timeline_s(segments: Sequence[Segment]) -> float:
    """Length of the rendered mp4: first segment start to last segment end (gaps held)."""
    return segments[-1].end - segments[0].start


def render_video(
    segments: Sequence[Segment],
    out: Path,
    *,
    ffmpeg: str,
    fps: int = 30,
    max_width: int = 960,
    extra_vf: str = "",
    timeout_s: float = 300.0,
) -> Path:
    segs = sorted(segments, key=lambda s: s.start)
    if not segs:
        raise ValueError("no video segments")
    w, h = output_size(segs, max_width)
    cmd = [ffmpeg, "-y", "-v", "error"]
    chains = []
    for i, s in enumerate(segs):
        cmd += ["-ss", f"{max(s.offset, 0.0):.3f}", "-t", f"{s.duration:.3f}", "-f", "mov", "-i", str(s.clip.path)]
        span = (segs[i + 1].start - s.start) if i + 1 < len(segs) else s.duration
        hold = max(span - s.duration, 0.0) + 0.5  # pad past the target, then trim to it exactly
        chains.append(
            f"[{i}:v]setpts=PTS-STARTPTS,fps={fps},"
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"format=yuv420p,tpad=stop_mode=clone:stop_duration={hold:.3f},trim=duration={span:.3f},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
    joined = "".join(f"[v{i}]" for i in range(len(segs)))
    graph = ";".join(chains) + f";{joined}concat=n={len(segs)}:v=1:a=0" + (f",{extra_vf}" if extra_vf else "") + "[out]"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".video-") as td:
        tmp_out = Path(td) / "out.mp4"
        cmd += ["-filter_complex", graph, "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "28", "-movflags", "+faststart", str(tmp_out)]  # fmt: skip
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr.strip()[-500:]}") from e
        os.replace(tmp_out, out)
    return out
