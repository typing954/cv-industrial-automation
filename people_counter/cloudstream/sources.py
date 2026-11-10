"""Frame sources. The main one is LoopingVideoSource: plays a video file over and over like a live camera."""
from __future__ import annotations

import time
from typing import Iterator, NamedTuple

import cv2
import numpy as np


class Frame(NamedTuple):
    seq: int            # 0,1,2... every frame the source produced
    t: float            # stream time in seconds (seq / rate); continues across loops
    frame: np.ndarray
    loop: int           # how many times the file has wrapped around so far


class LoopingVideoSource:
    """Plays `path` in a loop.

    rate        playback speed in frames per second of the file (None = the file's own fps). With pace=True frames are
                released in real time at this speed (like a camera); a higher rate than the file's fps plays it faster.
    target_fps  detect at this many frames per second: frames are picked evenly in stream time, the rest are skipped
                (None = every frame). Can't exceed `rate`. `self.rate` is the resulting detection rate.
    pace        False = push as fast as the consumer accepts (throughput testing); stream time is still frame/rate.
    loops       stop after this many full passes over the file (0 = no limit)
    duration_s  stop after this much stream time (0 = no limit)
    """

    def __init__(self, path: str, rate: float | None = None, pace: bool = True, loops: int = 0, duration_s: float = 0.0,
                 target_fps: float | None = None):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.native_fps = native if 1 <= native <= 240 else 30.0
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frames_per_loop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self.path, self.play_rate = path, float(rate or self.native_fps)
        self.rate = min(float(target_fps), self.play_rate) if target_fps else self.play_rate   # detection rate
        self.pace, self.loops, self.duration_s = pace, loops, duration_s
        self.name = path
        self._stop = False

    def close(self):
        self._stop = True

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.path)
        n, loop, t0, next_t, period = -1, 0, time.perf_counter(), 0.0, 1.0 / self.rate
        try:
            while not self._stop:
                if not cap.grab():                           # end of file -> wrap around
                    loop += 1
                    if self.loops and loop >= self.loops:
                        return
                    cap.release()
                    cap = cv2.VideoCapture(self.path)        # reopen: seeking to frame 0 is unreliable for webm/ogv
                    if not cap.grab():
                        return
                n += 1                                       # source frame counter, continues across loops
                t = n / self.play_rate
                if self.duration_s and t >= self.duration_s:
                    return
                if t + 1e-9 < next_t:                        # not one of the evenly spaced detection slots
                    continue
                next_t += period
                if self.pace:
                    wait = t0 + t - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                yield Frame(n, t, frame, loop)
        finally:
            cap.release()


class LiveSource:
    """Webcam index / RTSP URL via the existing peoplecount reader (latest-frame reader, no reconnect)."""

    def __init__(self, src: str, duration_s: float = 0.0, target_fps: float | None = None):
        from peoplecount.video import VideoSource
        self._v = VideoSource(src, target_fps=target_fps)
        self.rate, self.name, self.duration_s = float(self._v.eff_fps), str(src), duration_s

    def close(self):
        self._v.close()

    def __iter__(self) -> Iterator[Frame]:
        for idx, t, frame in self._v:
            if self.duration_s and t >= self.duration_s:
                return
            yield Frame(idx, t, frame, 0)


def open_source(src: str, rate: float | None = None, pace: bool = True, loops: int = 0, duration_s: float = 0.0,
                target_fps: float | None = None):
    from peoplecount.video import is_live
    if is_live(src):
        return LiveSource(src, duration_s, target_fps)
    return LoopingVideoSource(src, rate, pace, loops, duration_s, target_fps)
