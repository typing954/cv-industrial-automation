"""Real-time run: overlay + alerting on a file / webcam / RTSP stream."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .alert import AlertManager, ConsoleSink, JsonlSink, WebhookSink
from .counter import PeopleCounter
from .models import Detector, resolve
from .video import VideoSource


def output_paths(args, spec):
    """samples/<video-stem>/<YYYYmmdd-HHMMSS>_<uuid8>_<model>_<video-stem>.mp4 (+ .alerts.jsonl alongside)."""
    import re
    import time
    import uuid
    from pathlib import Path
    src = str(args.source)
    stem = f"camera{src}" if src.isdigit() else (re.sub(r"\W+", "_", src.split("://", 1)[1]).strip("_")[:40]
                                              if "://" in src else Path(src).stem)
    model = re.sub(r"\W+", "_", Path(spec.name).name.rstrip("/")).strip("_")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}_{model}_{stem}"
    d = Path(args.out_dir) / stem
    d.mkdir(parents=True, exist_ok=True)
    video = None if args.no_save else (args.save or str(d / f"{run_id}.mp4"))
    alerts = args.alert_log or str(d / f"{run_id}.alerts.jsonl")
    return video, alerts


def draw(frame, res, fps, threshold, alerting, model):
    for (x1, y1, x2, y2), i in zip(res.boxes.astype(int), res.ids):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
        cv2.putText(frame, f"#{i}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1)
    col = (0, 0, 255) if alerting else (40, 40, 40)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), col, -1)
    cv2.putText(frame, f"{model} | people: {res.smooth} (limit {threshold}) | {fps:4.1f} FPS" +
                ("  !! OVER CAPACITY !!" if alerting else ""), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def run(args):
    spec = resolve(args.model)
    args.save, args.alert_log = output_paths(args, spec)
    det = Detector(spec, args.device, args.conf, args.imgsz, threads=args.threads)
    src = VideoSource(args.source, realtime=args.realtime, stride=args.stride)
    det.warmup(next(iter(VideoSource(args.source, max_frames=1)))[2], 3)
    pc = PeopleCounter(det, fps=src.fps / args.stride, tracker=args.tracker)
    sinks = [ConsoleSink()] + ([JsonlSink(args.alert_log)] if args.alert_log else []) + \
            ([WebhookSink(args.webhook)] if args.webhook else [])
    am = AlertManager(args.threshold, args.hold, args.clear, cooldown_s=args.cooldown, sinks=sinks)
    writer, ema, n, t_start = None, 0.0, 0, time.perf_counter()
    print(f"video out: {args.save or '(disabled)'}\nalert log: {args.alert_log}", flush=True)
    print(f"model={spec.name} device={det.device} source={args.source} {src.size} @ {src.fps:.1f}fps "
          f"{'(live)' if src.live else ''}", flush=True)
    for idx, t, frame in src:
        t0 = time.perf_counter()
        res = pc.step(frame)
        am.update(res.smooth, t)
        dt = time.perf_counter() - t0
        ema = dt if not ema else 0.9 * ema + 0.1 * dt
        n += 1
        if args.show or args.save:
            out = draw(frame, res, 1 / ema, args.threshold, am.active, spec.name)
            if args.save and writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src.fps / args.stride, src.size)
            if writer:
                writer.write(out)
            if args.show:
                cv2.imshow("people-counter", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if n % 100 == 0:
            print(f"  frame {idx}: count={res.smooth} raw={res.raw} fps={1/ema:.1f} dropped={src.dropped}", flush=True)
    if writer:
        writer.release()
    src.close()
    wall = time.perf_counter() - t_start
    print(f"done: {n} frames in {wall:.1f}s ({n/wall:.1f} FPS), dropped={src.dropped}, "
          f"alerts={sum(e.kind=='ALERT' for e in am.events)}, unique people tracked={pc.seen and len(pc.seen)}")
