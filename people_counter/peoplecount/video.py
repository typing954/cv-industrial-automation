"""Video sources.

target_fps: the rate at which frames are handed to the detector. The video is still read at its own speed, but only frames
sampled evenly in TIME to this rate are decoded fully and returned (None = every frame). It can't exceed the source's own FPS.
Files: sequential, or paced in real time with `realtime=True` (frames we're too slow to process are dropped, not queued).
Live (RTSP/webcam): a threaded latest-frame reader so a slow model drops stale frames instead of building lag.
"""
from __future__ import annotations

import threading
import time

import cv2


def _open(src):
    s = int(src) if isinstance(src, str) and src.isdigit() else src
    cap = cv2.VideoCapture(s)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source: {src}")
    return cap


def is_live(src) -> bool:
    s = str(src)
    return s.isdigit() or s.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))


class VideoSource:
    def __init__(self, src, realtime: bool = False, target_fps: float | None = None, max_frames: int | None = None):
        self.src, self.live = src, is_live(src)
        self.cap = _open(src)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps > 240 or self.fps < 1:
            self.fps = 30.0
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self.target_fps = float(target_fps) if target_fps else None
        # the rate frames are actually handed to the detector (never above what the source provides)
        self.eff_fps = min(self.target_fps, self.fps) if self.target_fps else self.fps
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.live else -1
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.realtime, self.max_frames = realtime, max_frames
        self.dropped = 0          # frames dropped because the detector was too slow (not counting frames skipped by target_fps)
        self._latest, self._lock, self._stop = None, threading.Lock(), False
        if self.live:
            threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        n = 0
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.5)
                continue
            n += 1
            with self._lock:
                self._latest = (n, f)

    def __iter__(self):
        """yields (frame_index, stream_time_s, frame)"""
        return self._iter_live() if self.live else self._iter_file()

    def _iter_live(self):
        last, t0, yielded, last_yield = 0, time.perf_counter(), 0, -1e18
        min_dt = 1.0 / self.target_fps if self.target_fps else 0.0
        while not self._stop:
            now = time.perf_counter()
            if min_dt and now - last_yield < min_dt:
                time.sleep(min(0.002, min_dt - (now - last_yield)))
                continue
            with self._lock:
                item = self._latest
            if item is None or item[0] == last:
                time.sleep(0.002)
                continue
            if not min_dt:
                self.dropped += item[0] - last - 1 if last else 0
            last, last_yield = item[0], now
            yield last, now - t0, item[1]
            yielded += 1
            if self.max_frames and yielded >= self.max_frames:
                return

    def _iter_file(self):
        cap, period = self.cap, 1.0 / self.eff_fps
        idx, yielded, next_t, t0 = -1, 0, 0.0, time.perf_counter()
        while True:
            if not cap.grab():                       # grab() decodes without the costly copy/convert of read()
                return
            idx += 1
            t = idx / self.fps
            if t + 1e-9 < next_t:                    # not one of the evenly spaced detection slots
                continue
            next_t += period
            if self.realtime:
                lag = (time.perf_counter() - t0) - t
                if lag > period:                     # more than one slot behind real time: drop it instead of queueing
                    self.dropped += 1
                    continue
                if lag < 0:                          # ahead of real time: wait for this frame's moment
                    time.sleep(-lag)
            ok, frame = cap.retrieve()
            if not ok:
                return
            yield idx, t, frame
            yielded += 1
            if self.max_frames and yielded >= self.max_frames:
                return

    def close(self):
        self._stop = True
        self.cap.release()
