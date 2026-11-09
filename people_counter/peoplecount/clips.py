"""Evidence clips: `pre_s` seconds before the violation started until `post_s` seconds after the alert was raised.

Overlapping / back-to-back alerts are merged into ONE clip instead of producing many near-duplicates:
  * annotated frames of the last few seconds are kept in a RAM ring buffer (JPEG-compressed, ~10-15 MB for 8 s of 1080p);
  * a clip starts when an alert fires (frames are taken from the buffer, so the pre-roll is real footage);
  * a later alert whose pre-roll window overlaps (or is within `merge_gap_s` of) the open clip just EXTENDS it;
  * the clip is finalised only when no later alert could still merge into it (or the stream ends).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


def _fmt(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


class ClipRecorder:
    def __init__(self, out_dir, pre_s: float = 3.0, post_s: float = 3.0, hold_s: float = 2.0,
                 merge_gap_s: float = 1.0, jpeg_quality: int = 90,
                 on_clip: Optional[Callable[[dict], None]] = None):
        self.dir, self.pre, self.post, self.gap = Path(out_dir), pre_s, post_s, merge_gap_s
        self.hold, self.q, self.on_clip = hold_s, jpeg_quality, on_clip
        # an alert that could still merge into the open clip fires at most pre+hold+gap after the clip's end
        self.settle = pre_s + hold_s + merge_gap_s
        self.horizon = self.settle + 2.0
        self.ring: deque = deque()          # (t, jpeg bytes)
        self.clips: list[dict] = []         # finished clips: path, start_t, end_t, episodes
        self._w: Optional[cv2.VideoWriter] = None
        self._cur: Optional[dict] = None
        self._end = 0.0                     # stream time up to which the open clip wants frames
        self._last_t = -1.0                 # stream time of the last frame written to the open clip

    # -- per frame (pass the ANNOTATED frame) ---------------------------------------------------------------------
    def add_frame(self, t: float, frame: np.ndarray):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        if ok:
            self.ring.append((t, buf))
        while self.ring and t - self.ring[0][0] > self.horizon:
            self.ring.popleft()
        if self._w is not None:
            if t <= self._end:
                self._w.write(frame)
                self._last_t = t
            elif t > self._end + self.settle:
                self._finalize()

    # -- when an ALERT fires (call AFTER add_frame for the same frame) ---------------------------------------------
    def on_alert(self, start_t: float, alert_t: float, episode: int) -> Optional[str]:
        want_start, new_end = start_t - self.pre, alert_t + self.post
        if self._w is not None and want_start <= self._end + self.gap:
            for t, buf in self.ring:                    # fill the gap between the old end and now
                if t > self._last_t:
                    self._w.write(cv2.imdecode(buf, cv2.IMREAD_COLOR))
                    self._last_t = t
            self._end = max(self._end, new_end)
            self._cur["episodes"].append(episode)
            self._cur["merged"] = True
        else:
            self._finalize()
            self._open(want_start, new_end, episode)
        return self._cur["path"] if self._cur else None

    def close(self):
        self._finalize()

    # -- internals -------------------------------------------------------------------------------------------------
    def _open(self, want_start: float, new_end: float, episode: int):
        frames = [(t, b) for t, b in self.ring if t >= want_start] or list(self.ring)[-1:]
        if not frames:
            return
        ts = [t for t, _ in (self.ring if len(frames) < 3 else frames)]
        dts = np.diff(ts)
        fps = float(np.clip(1.0 / np.median(dts), 1.0, 120.0)) if len(dts) and np.median(dts) > 0 else 25.0
        first = cv2.imdecode(frames[0][1], cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        self.dir.mkdir(parents=True, exist_ok=True)
        path = str(self.dir / f"clip{len(self.clips) + 1:03d}_{_fmt(frames[0][0])}_alert{episode}.mp4")
        self._w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for t, b in frames:
            self._w.write(cv2.imdecode(b, cv2.IMREAD_COLOR))
            self._last_t = t
        self._end = new_end
        self._cur = {"path": path, "start_t": round(frames[0][0], 3), "end_t": None, "fps": round(fps, 2),
                     "episodes": [episode], "merged": False}

    def _finalize(self):
        if self._w is None:
            return
        self._w.release()
        self._w = None
        self._cur["end_t"] = round(self._last_t, 3)
        self._cur["duration_s"] = round(self._cur["end_t"] - self._cur["start_t"], 3)
        self.clips.append(self._cur)
        if self.on_clip:
            self.on_clip(self._cur)
        self._cur = None
