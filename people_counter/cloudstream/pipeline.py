"""Streaming pipeline: source -> JPEG -> batched, pipelined requests to a hosted YOLO endpoint -> (in order) tracker ->
the SAME alert system as the local pipeline (peoplecount.alert.AlertManager) -> the SAME clip recorder
(peoplecount.clips.ClipRecorder) -> the SAME output files as `python -m peoplecount run`.

Threads:  capture (caller) -> queue -> batcher -> worker pool (network calls, `max_inflight` at once)
          -> ordered queue -> consumer (tracking, alerts, clips, files; strictly in frame order)
Real-time behaviour: if the endpoint can't keep up, `on_backlog="drop"` drops the newest frames instead of letting delay
build up (reported as `dropped`); `"block"` applies back-pressure to the source (use it for throughput tests).
"""
from __future__ import annotations

import csv
import json
import queue
import re
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np

from peoplecount.alert import AlertManager, ConsoleSink, JsonlSink, WebhookSink, timecode
from peoplecount.clips import ClipRecorder
from peoplecount.counter import _Dets, _make_tracker
from peoplecount.live import draw

from . import codec

_TIMEOUT = object()


@dataclass
class Item:
    seq: int
    t: float
    loop: int
    frame: np.ndarray       # the (resized) frame that was sent; overlays/clips use it
    jpeg: bytes
    t_cap: float


@dataclass
class Job:
    id: int
    items: list


@dataclass
class JobResult:
    job: Job
    boxes: Optional[list]
    rtt: float
    server_ms: Optional[float]
    error: Optional[str]


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.t0 = time.monotonic()
        self.captured = self.dropped = self.sent = self.done = self.errors = self.jobs = 0
        self.lat, self.rtt, self.srv = deque(maxlen=5000), deque(maxlen=2000), deque(maxlen=2000)
        self.done_ts: deque = deque()

    def add(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def add_done(self, n_frames: int, lat_ms: list, rtt: float, srv_ms: Optional[float]):
        now = time.monotonic()
        with self.lock:
            self.done += n_frames
            self.jobs += 1
            self.lat.extend(lat_ms)
            self.rtt.append(rtt * 1e3)
            if srv_ms is not None:
                self.srv.append(srv_ms / max(n_frames, 1))
            self.done_ts.extend([now] * n_frames)
            while self.done_ts and now - self.done_ts[0] > 5.0:
                self.done_ts.popleft()

    def fps_recent(self) -> float:
        with self.lock:
            now = time.monotonic()
            n = sum(1 for x in self.done_ts if now - x <= 5.0)
            return n / min(5.0, max(now - self.t0, 1e-6))

    @staticmethod
    def _p(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    def line(self) -> str:
        with self.lock:
            el = time.monotonic() - self.t0
            return (f"  [{el:5.0f}s] in {self.captured / max(el, 1e-6):5.1f} fps | processed {self.done / max(el, 1e-6):5.1f} fps "
                    f"| dropped {self.dropped} | errors {self.errors} | latency p50 {self._p(self.lat, 50):5.0f} ms "
                    f"p95 {self._p(self.lat, 95):5.0f} ms | endpoint rtt p50 {self._p(self.rtt, 50):4.0f} ms "
                    f"| server {self._p(self.srv, 50):4.1f} ms/frame")

    def summary(self) -> dict:
        with self.lock:
            el = time.monotonic() - self.t0
            return {"elapsed_s": round(el, 2), "frames_captured": self.captured, "frames_processed": self.done,
                    "frames_dropped": self.dropped, "frame_errors": self.errors, "requests": self.jobs,
                    "input_fps": round(self.captured / max(el, 1e-6), 2),
                    "processed_fps": round(self.done / max(el, 1e-6), 2),
                    "latency_ms": {"p50": self._p(self.lat, 50), "p95": self._p(self.lat, 95), "p99": self._p(self.lat, 99),
                                   "max": float(max(self.lat)) if self.lat else float("nan")},
                    "endpoint_rtt_ms": {"p50": self._p(self.rtt, 50), "p95": self._p(self.rtt, 95)},
                    "server_ms_per_frame_p50": self._p(self.srv, 50)}


class StreamPipeline:
    def __init__(self, transport_factory: Callable, *, label: str = "endpoint", send_max_side: int = 640,
                 jpeg_quality: int = 80, batch_size: int = 4, max_inflight: int = 4, max_wait_ms: float = 50.0,
                 on_backlog: str = "drop", conf: float = 0.3, tracker: str = "bytetrack",
                 smooth_window: Optional[int] = None, retries: int = 2,
                 # --- same alert system / defaults as `peoplecount run` ---
                 threshold: int = 10, hold_s: float = 2.0, clear_s: float = 3.0, cooldown_s: float = 0.0,
                 webhook: Optional[str] = None, clips: bool = True, clip_pre_s: float = 3.0, clip_post_s: float = 3.0,
                 merge_gap_s: float = 1.0, out_dir: str = "samples", save_video: bool = False,
                 stats_every: float = 5.0):
        self.factory, self.label = transport_factory, label
        self.send_max_side, self.jpeg_quality = send_max_side, jpeg_quality
        self.batch_size, self.max_inflight, self.max_wait = batch_size, max_inflight, max_wait_ms / 1000.0
        self.on_backlog, self.conf, self.tracker_kind, self.retries = on_backlog, conf, tracker, retries
        self.smooth_window = smooth_window
        self.threshold, self.hold_s, self.clear_s, self.cooldown_s = threshold, hold_s, clear_s, cooldown_s
        self.webhook, self.use_clips, self.clip_pre, self.clip_post, self.merge_gap = webhook, clips, clip_pre_s, clip_post_s, merge_gap_s
        self.out_dir, self.save_video, self.stats_every = out_dir, save_video, stats_every
        self.stats = Stats()
        self._tl = threading.local()

    # ------------------------------------------------------------------------------------------------------ outputs
    def _setup_outputs(self, source):
        name = str(source.name)
        stem = f"camera{name}" if name.isdigit() else (re.sub(r"\W+", "_", name.split("://", 1)[1]).strip("_")[:40]
                                                       if "://" in name else Path(name).stem)
        label = re.sub(r"\W+", "_", self.label)
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}_{label}_{stem}"
        d = Path(self.out_dir) / stem
        d.mkdir(parents=True, exist_ok=True)
        self.paths = SimpleNamespace(alerts=str(d / f"{run_id}.alerts.jsonl"), episodes=str(d / f"{run_id}.episodes.json"),
                                     counts=str(d / f"{run_id}.counts.csv"), stats=str(d / f"{run_id}.stats.json"),
                                     clips=str(d / f"{run_id}_clips"), video=str(d / f"{run_id}.mp4"))

    # ------------------------------------------------------------------------------------------------------ network
    def _transport(self):
        if not hasattr(self._tl, "t"):
            self._tl.t = self.factory()
        return self._tl.t

    def _invoke(self, job: Job) -> JobResult:
        body = codec.make_request([it.jpeg for it in job.items], conf=self.conf)
        t0, err = time.monotonic(), None
        for attempt in range(self.retries + 1):
            try:
                boxes, meta = codec.parse_response(self._transport().invoke(body))
                if len(boxes) != len(job.items):
                    raise RuntimeError(f"endpoint returned {len(boxes)} results for {len(job.items)} frames")
                return JobResult(job, boxes, time.monotonic() - t0, meta.get("server_ms"), None)
            except Exception as e:  # noqa: BLE001  network/throttling errors are retried, then the batch is skipped
                err = f"{type(e).__name__}: {e}"
                self._tl.__dict__.pop("t", None)
                time.sleep(0.05 * 2 ** attempt)
        return JobResult(job, None, time.monotonic() - t0, None, err)

    # ------------------------------------------------------------------------------------------------------ batcher
    def _batcher(self):
        pending: list[Item] = []
        deadline = 0.0
        while True:
            timeout = None if not pending else max(0.0, deadline - time.monotonic())
            try:
                item = self.q.get(timeout=timeout)
            except queue.Empty:
                item = _TIMEOUT
            if item is None:
                if pending:
                    self._submit(pending, final=True)
                self.ordered.put(None)
                return
            if item is not _TIMEOUT:
                if not pending:
                    deadline = time.monotonic() + self.max_wait
                pending.append(item)
            if pending and (len(pending) >= self.batch_size or item is _TIMEOUT or time.monotonic() >= deadline):
                self._submit(pending)
                pending = []

    def _submit(self, items: list, final: bool = False):
        if self.on_backlog == "drop" and not final:
            if not self.sem.acquire(blocking=False):
                self.stats.add(dropped=len(items))
                return
        else:
            self.sem.acquire()
        self._job_id += 1
        job = Job(self._job_id, items)
        fut = self.pool.submit(self._invoke, job)
        fut.add_done_callback(lambda _f: self.sem.release())
        self.stats.add(sent=len(items))
        self.ordered.put(fut)

    # ------------------------------------------------------------------------------------------------------ consumer
    def _reset_tracking(self):
        self.tracker = _make_tracker(self.tracker_kind, self.conf, self.track_buffer)
        self.hist.clear()

    def _consume(self):
        while True:
            fut = self.ordered.get()
            if fut is None:
                return
            res: JobResult = fut.result()
            n = len(res.job.items)
            if res.error:
                self.stats.add(errors=n)
                print(f"  ⚠ batch failed ({n} frames): {res.error}", flush=True)
            for i, it in enumerate(res.job.items):
                self._process(it, res.boxes[i] if res.boxes else None)
            if not res.error:
                now = time.monotonic()
                self.stats.add_done(n, [(now - it.t_cap) * 1e3 for it in res.job.items], res.rtt, res.server_ms)

    def _process(self, it: Item, boxes: Optional[np.ndarray]):
        if it.loop != self._loop:                         # the looped video wrapped around: new scene, fresh tracks
            self._loop = it.loop
            self._reset_tracking()
        if boxes is None:                                 # failed batch: keep the video/clip continuous, no new detections
            res = SimpleNamespace(boxes=np.zeros((0, 4)), ids=np.zeros(0, int), smooth=self._last_smooth, raw=0)
            ev = None
        else:
            rows = np.asarray(self.tracker.update(_Dets(boxes), it.frame)).reshape(-1, 8)
            ids = rows[:, 4].astype(int)
            self.hist.append(len(ids))
            smooth = int(np.median(self.hist))
            self._last_smooth = smooth
            self.seen.update(int(x) for x in ids)
            res = SimpleNamespace(boxes=rows[:, :4], ids=ids, smooth=smooth, raw=len(boxes))
            ev = self.am.update(smooth, it.t)
            self._series.append(smooth)
            self._csv.writerow([it.seq, round(it.t, 3), timecode(it.t), smooth, len(boxes)])
            self._rows += 1
            if self._rows % 50 == 0:
                self._csv_f.flush()
        self._t_last = it.t
        if self.clips or self._writer_wanted:
            out = draw(it.frame, res, self.stats.fps_recent(), self.threshold, self.am.active, self.label, it.t)
            if self.clips:
                self.clips.add_frame(it.t, out)
                if ev and ev.kind == "ALERT":
                    self.am.episodes[-1]["clip"] = self.clips.on_alert(ev.start_t, ev.t, ev.episode)
            if self._writer_wanted:
                import cv2
                if self._writer is None:
                    self._writer = cv2.VideoWriter(self.paths.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                                   max(self.rate, 1.0), (out.shape[1], out.shape[0]))
                self._writer.write(out)
        if ev:
            self._write_episodes()

    def _write_episodes(self, final: bool = False):
        smooth = self._series or [0]
        summary = {"frames": len(self._series), "duration_s": round(self._t_last, 3), "video_fps": self.rate,
                   "max_people": int(max(smooth)), "avg_people": round(sum(smooth) / len(smooth), 2),
                   "distinct_tracks": len(self.seen)}
        clips = self.clips.clips if self.clips else []
        if final and self.clips:
            by_ep = {e: c["path"] for c in clips for e in c["episodes"]}
            for ep in self.am.episodes:
                ep["clip"] = by_ep.get(ep["episode"], ep["clip"])
        with open(self.paths.episodes, "w") as f:       # same schema as `peoplecount run`
            json.dump({"source": self.source_name, "model": self.label, "threshold": self.threshold, "hold_s": self.hold_s,
                       "clear_s": self.clear_s, "summary": summary, "counts_csv": self.paths.counts,
                       "episodes": self.am.episodes, "clips": clips}, f, indent=1)

    # ------------------------------------------------------------------------------------------------------ run
    def run(self, source) -> dict:
        self._setup_outputs(source)
        self.source_name, self.rate = str(source.name), float(source.rate)
        self.track_buffer = max(5, int(1.5 * self.rate))
        window = self.smooth_window or max(3, min(15, int(round(self.rate * 0.4)) | 1))
        self.hist: deque = deque(maxlen=window)
        self.seen: set = set()
        self._series: list = []
        self._loop, self._last_smooth, self._t_last, self._rows = 0, 0, 0.0, 0
        self._writer, self._writer_wanted = None, self.save_video
        self._reset_tracking()

        sinks = [ConsoleSink(), JsonlSink(self.paths.alerts)] + ([WebhookSink(self.webhook)] if self.webhook else [])
        self.am = AlertManager(self.threshold, self.hold_s, self.clear_s, cooldown_s=self.cooldown_s, sinks=sinks)
        self.clips = ClipRecorder(self.paths.clips, self.clip_pre, self.clip_post, hold_s=self.hold_s,
                                  merge_gap_s=self.merge_gap, on_clip=lambda c: print(
                                      f"🎞  clip saved: {c['path']} ({c['duration_s']:.1f}s, alerts {c['episodes']})",
                                      flush=True)) if self.use_clips else None
        self._csv_f = open(self.paths.counts, "w", newline="")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(["frame", "t", "timecode", "people", "raw"])

        self.q: queue.Queue = queue.Queue(maxsize=max(8, self.batch_size * self.max_inflight * 2))
        self.ordered: queue.Queue = queue.Queue()
        self.sem = threading.Semaphore(self.max_inflight)
        self.pool = ThreadPoolExecutor(max_workers=self.max_inflight, thread_name_prefix="invoke")
        self._job_id = 0
        batcher = threading.Thread(target=self._batcher, daemon=True)
        consumer = threading.Thread(target=self._consume, daemon=True)
        batcher.start(); consumer.start()
        print(f"stream: {self.source_name} @ {self.rate:g} fps → {self.label} | batch {self.batch_size}, in-flight {self.max_inflight}, "
              f"send size {self.send_max_side}px, on-backlog={self.on_backlog}\n"
              f"outputs: {Path(self.paths.episodes).parent}/ (run id {Path(self.paths.episodes).name.split('.')[0]})", flush=True)

        next_print = time.monotonic() + self.stats_every
        try:
            for fr in source:
                small, jpg = codec.encode_frame(fr.frame, self.send_max_side, self.jpeg_quality)
                item = Item(fr.seq, fr.t, fr.loop, small, jpg, time.monotonic())
                self.stats.add(captured=1)
                if self.on_backlog == "drop":
                    try:
                        self.q.put_nowait(item)
                    except queue.Full:
                        self.stats.add(dropped=1)
                else:
                    self.q.put(item)
                if self.stats_every and time.monotonic() >= next_print:
                    print(self.stats.line(), flush=True)
                    next_print += self.stats_every
        except KeyboardInterrupt:
            print("\nstopping: finishing in-flight requests and saving outputs…", flush=True)
        finally:
            source.close()
            self.q.put(None)
            batcher.join()
            consumer.join()
            self.pool.shutdown(wait=True)
            self.am.finish(self._t_last, self._last_smooth)
            if self.clips:
                self.clips.close()
            if self._writer:
                self._writer.release()
            self._csv_f.close()
            self._write_episodes(final=True)
            summary = self.stats.summary()
            summary["alert_episodes"] = len(self.am.episodes)
            summary["clips"] = len(self.clips.clips) if self.clips else 0
            summary["config"] = {"label": self.label, "batch_size": self.batch_size, "max_inflight": self.max_inflight,
                                 "send_max_side": self.send_max_side, "jpeg_quality": self.jpeg_quality,
                                 "on_backlog": self.on_backlog, "source_rate_fps": self.rate}
            Path(self.paths.stats).write_text(json.dumps(summary, indent=1))
        print(self.stats.line())
        print(f"done: {summary['frames_processed']} frames processed ({summary['processed_fps']} fps), "
              f"{summary['frames_dropped']} dropped, {summary['frame_errors']} errors | alert episodes: "
              f"{summary['alert_episodes']}, clips: {summary['clips']}")
        for ep in self.am.episodes:
            print(f"  #{ep['episode']}: exceeded at {ep['start_timecode']} | alert {ep['alert_timecode']} | over limit for "
                  f"{ep['duration_s']:.1f}s | peak {ep['peak_count']} | clip: {ep['clip'] and ep['clip'].split('/')[-1]}")
        return summary
