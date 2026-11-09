"""Video sources. Files: sequential (or paced real-time with frame dropping). Live (RTSP/webcam): threaded
latest-frame reader so slow models drop stale frames instead of building lag."""
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
    def __init__(self, src, realtime: bool = False, stride: int = 1, max_frames: int | None = None):
        self.src, self.live = src, is_live(src)
        self.cap = _open(src)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps > 240 or self.fps < 1:
            self.fps = 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.live else -1
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.realtime, self.stride, self.max_frames = realtime, max(1, stride), max_frames
        self.dropped = 0
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
        if self.live:
            last, t0, yielded = 0, time.perf_counter(), 0
            while not self._stop:
                with self._lock:
                    item = self._latest
                if item is None or item[0] == last:
                    time.sleep(0.002)
                    continue
                self.dropped += item[0] - last - 1 if last else 0
                last = item[0]
                yield last, time.perf_counter() - t0, item[1]
                yielded += 1
                if self.max_frames and yielded >= self.max_frames:
                    return
            return
        idx, yielded, t0 = -1, 0, time.perf_counter()
        while True:
            if self.realtime:  # skip frames we're too slow to process
                target = int((time.perf_counter() - t0) * self.fps)
                while idx + 1 < target:
                    if not self.cap.grab():
                        return
                    idx += 1
                    self.dropped += 1
            ok, f = self.cap.read()
            if not ok:
                return
            idx += 1
            if idx % self.stride:
                continue
            yield idx, idx / self.fps, f
            yielded += 1
            if self.max_frames and yielded >= self.max_frames:
                return
            if self.realtime:  # don't run ahead of wall-clock
                wait = (idx + 1) / self.fps - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)

    def close(self):
        self._stop = True
        self.cap.release()
