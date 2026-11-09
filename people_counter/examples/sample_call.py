"""Minimal programmatic use of the repo: count people in a video and print alerts.

Run from the repo root:
    python examples/sample_call.py videos/crowd_roma_pride_720p.webm --model yolo11n --threshold 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so it works without installing the package

from peoplecount.alert import AlertManager, ConsoleSink
from peoplecount.counter import PeopleCounter
from peoplecount.models import Detector, resolve
from peoplecount.video import VideoSource

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--model", default="yolo11n")
ap.add_argument("--threshold", type=int, default=10)
ap.add_argument("--stride", type=int, default=5, help="process every Nth frame")
ap.add_argument("--max-frames", type=int, default=40)
a = ap.parse_args()

detector = Detector(resolve(a.model), device="auto", conf=0.3, threads=4)         # 1. pick a model
src = VideoSource(a.video, stride=a.stride, max_frames=a.max_frames)               # 2. file / webcam index / rtsp:// URL
counter = PeopleCounter(detector, fps=src.fps / a.stride)                          # 3. detector + ByteTrack + smoothing
alerts = AlertManager(a.threshold, hold_s=1.0, clear_s=2.0, sinks=[ConsoleSink()]) # 4. alert rule (add WebhookSink etc.)

for frame_idx, t, frame in src:
    r = counter.step(frame)                  # r.smooth = people count, r.boxes / r.ids = tracked people
    alerts.update(r.smooth, t)
    print(f"frame {frame_idx:4d} t={t:5.1f}s  people={r.smooth:2d} (raw {r.raw})  detect={r.t_detect*1000:.0f}ms")
src.close()
