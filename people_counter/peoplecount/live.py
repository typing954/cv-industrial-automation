"""Real-time run: overlay + alerting on a file / webcam / RTSP stream."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .alert import AlertManager, ConsoleSink, JsonlSink, WebhookSink, timecode
from .clips import ClipRecorder
from .counter import PeopleCounter
from .models import Detector, resolve
from .video import VideoSource


def output_paths(args, spec):
    """samples/<video-stem>/<YYYYmmdd-HHMMSS>_<uuid8>_<model>_<video-stem>{.mp4,.alerts.jsonl,.episodes.json,_clips/}"""
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
    return video, alerts, str(d / f"{run_id}.episodes.json"), str(d / f"{run_id}_clips")


def draw(frame, res, fps, threshold, alerting, model, t=None):
    for (x1, y1, x2, y2), i in zip(res.boxes.astype(int), res.ids):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
        cv2.putText(frame, f"#{i}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1)
    col = (0, 0, 255) if alerting else (40, 40, 40)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), col, -1)
    cv2.putText(frame, f"{model} | people: {res.smooth} (limit {threshold}) | {fps:4.1f} FPS" +
                ("  !! OVER CAPACITY !!" if alerting else ""), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if t is not None:
        cv2.putText(frame, timecode(t)[:-4], (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def run(args):
    spec = resolve(args.model)
    args.save, args.alert_log, episodes_path, clips_dir = output_paths(args, spec)
    det = Detector(spec, args.device, args.conf, args.imgsz, threads=args.threads)
    src = VideoSource(args.source, realtime=args.realtime, stride=args.stride)
    det.warmup(next(iter(VideoSource(args.source, max_frames=1)))[2], 3)
    pc = PeopleCounter(det, fps=src.fps / args.stride, tracker=args.tracker)
    sinks = [ConsoleSink()] + ([JsonlSink(args.alert_log)] if args.alert_log else []) + \
            ([WebhookSink(args.webhook)] if args.webhook else [])
    am = AlertManager(args.threshold, args.hold, args.clear, cooldown_s=args.cooldown, sinks=sinks)
    clips = None if args.no_clips else ClipRecorder(
        clips_dir, args.clip_pre, args.clip_post, hold_s=args.hold, merge_gap_s=args.merge_gap,
        on_clip=lambda c: print(f"🎞  clip saved: {c['path']} ({c['duration_s']:.1f}s, alerts {c['episodes']})", flush=True))
    writer, ema, n, t_start = None, 0.0, 0, time.perf_counter()
    print(f"video out: {args.save or '(disabled)'}\nalert log: {args.alert_log}\nepisodes : {episodes_path}\n"
          f"clips dir: {'(disabled)' if clips is None else clips_dir}", flush=True)
    print(f"model={spec.name} device={det.device} source={args.source} {src.size} @ {src.fps:.1f}fps "
          f"{'(live)' if src.live else ''}", flush=True)
    t_last, last_count = 0.0, 0
    for idx, t, frame in src:
        t0 = time.perf_counter()
        res = pc.step(frame)
        ev = am.update(res.smooth, t)
        dt = time.perf_counter() - t0
        ema = dt if not ema else 0.9 * ema + 0.1 * dt
        n += 1
        t_last, last_count = t, res.smooth
        if args.show or args.save or clips:
            out = draw(frame, res, 1 / ema, args.threshold, am.active, spec.name, t)
            if clips:
                clips.add_frame(t, out)
                if ev and ev.kind == "ALERT":
                    am.episodes[-1]["clip"] = clips.on_alert(ev.start_t, ev.t, ev.episode)
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
    am.finish(t_last, last_count)
    if clips:
        clips.close()
        by_ep = {e: c["path"] for c in clips.clips for e in c["episodes"]}
        for ep in am.episodes:
            ep["clip"] = by_ep.get(ep["episode"], ep["clip"])
    import json
    with open(episodes_path, "w") as f:
        json.dump({"source": str(args.source), "model": spec.name, "threshold": args.threshold, "hold_s": args.hold,
                   "clear_s": args.clear, "episodes": am.episodes, "clips": clips.clips if clips else []}, f, indent=1)
    src.close()
    wall = time.perf_counter() - t_start
    print(f"done: {n} frames in {wall:.1f}s ({n/wall:.1f} FPS), dropped={src.dropped}, "
          f"alert episodes={len(am.episodes)}, clips={len(clips.clips) if clips else 0}, unique people tracked={len(pc.seen)}")
    for ep in am.episodes:
        print(f"  #{ep['episode']}: exceeded at {ep['start_timecode']} | alert {ep['alert_timecode']} | over limit for "
              f"{ep['duration_s']:.1f}s | peak {ep['peak_count']} | clip: {ep['clip'] and ep['clip'].split('/')[-1]}")
