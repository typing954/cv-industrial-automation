"""Detector + ByteTrack + count smoothing. One `step()` per frame."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .models import Detector


class _Dets:
    """Minimal Boxes-like view the Ultralytics trackers accept (conf, cls, xywh, boolean indexing)."""

    def __init__(self, d: np.ndarray):
        self.xyxy, self.conf = d[:, :4].astype(np.float32), d[:, 4].astype(np.float32)
        self.cls = np.zeros(len(d), np.float32)
        self.xywh = np.c_[(self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2, self.xyxy[:, 2:] - self.xyxy[:, :2]]
        self.data = d

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, i):
        o = object.__new__(_Dets)
        o.xyxy, o.conf, o.cls, o.xywh, o.data = (x[i] for x in (self.xyxy, self.conf, self.cls, self.xywh, self.data))
        return o


def _make_tracker(kind: str, conf: float, buffer: int):
    """kind: bytetrack | botsort  (both ship with Ultralytics; no ReID model needed)."""
    from pathlib import Path
    from types import SimpleNamespace

    import ultralytics
    from ultralytics.trackers import BOTSORT, BYTETracker
    from ultralytics.utils import YAML
    cfg = YAML.load(Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{kind}.yaml")
    cfg.update(track_buffer=buffer, track_high_thresh=conf, new_track_thresh=conf)
    cfg["with_reid"] = False
    return {"bytetrack": BYTETracker, "botsort": BOTSORT}[kind](SimpleNamespace(**cfg))


@dataclass
class FrameResult:
    raw: int                      # detections above conf threshold
    tracked: int                  # confirmed tracks visible this frame (de-duplicated, flicker-suppressed)
    smooth: int                   # rolling median of `tracked` (used for alerting)
    boxes: np.ndarray             # (M,4) tracked boxes
    ids: np.ndarray               # (M,)
    t_detect: float = 0.0
    t_track: float = 0.0
    unique_ids_seen: int = 0


class PeopleCounter:
    def __init__(self, detector: Detector, fps: float = 30.0, smooth_window: int = 9,
                 track: bool = True, tracker: str = "bytetrack", lost_buffer_s: float = 1.5):
        self.det, self.track_enabled = detector, track
        self.tracker = _make_tracker(tracker, detector.conf, max(5, int(lost_buffer_s * fps))) if track else None
        self.hist: deque[int] = deque(maxlen=smooth_window)
        self.seen: set[int] = set()

    def step(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        d = self.det.detect(frame)
        t1 = time.perf_counter()
        if self.track_enabled:
            rows = self.tracker.update(_Dets(d), frame)  # [x1,y1,x2,y2,id,score,cls,idx]
            rows = np.asarray(rows).reshape(-1, 8)
            boxes, ids = rows[:, :4], rows[:, 4].astype(int)
        else:
            boxes, ids = d[:, :4], np.arange(len(d))
        t2 = time.perf_counter()
        n = len(ids)
        self.seen.update(int(i) for i in ids)
        self.hist.append(n)
        return FrameResult(len(d), n, int(np.median(self.hist)), boxes, ids, t1 - t0, t2 - t1, len(self.seen))
