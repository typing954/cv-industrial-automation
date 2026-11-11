"""Dashboard backend: ONE run at a time, configured with the same parameters as `python -m peoplecount run`.

  python -m webapp.server            (from people_counter/)  ->  http://localhost:8000

POST /api/run starts a run (any previous run is stopped). The run reuses peoplecount.{models,counter,alert,clips,video}
and the CLI's overlay (`live.draw`) and output files (annotated mp4, alerts.jsonl, episodes.json, counts.csv, clips).
"""
from __future__ import annotations

import csv
import json
import os
import re
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from peoplecount.alert import AlertManager, ConsoleSink, JsonlSink, WebhookSink, timecode
from peoplecount.clips import ClipRecorder
from peoplecount.counter import PeopleCounter
from peoplecount.live import draw, output_paths
from peoplecount.models import DEFAULT_SET, REGISTRY, Detector, resolve
from peoplecount.video import VideoSource, is_live

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "videos"
SAMPLES = ROOT / "samples"                      # git-ignored; same place the CLI writes to
HISTORY = SAMPLES / "webapp" / "history.json"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ogv", ".m4v"}
PUBLISH_WIDTH = 960                             # frames sent to the browser are downscaled to this width


class RunParams(BaseModel):
    """Mirrors `python -m peoplecount run` (see peoplecount/cli.py)."""
    source: str                                             # file name in videos/, webcam index, or rtsp:// URL
    model: str = "yolo11n"                                  # --model
    device: str = "auto"                                    # --device
    conf: float = Field(0.3, ge=0.01, le=0.99)              # --conf
    threads: Optional[int] = Field(None, ge=1, le=64)       # --threads
    imgsz: Optional[int] = Field(None, ge=160, le=2048)     # --imgsz
    threshold: int = Field(10, ge=0, le=10000)              # --threshold
    hold: float = Field(2.0, ge=0, le=600)                  # --hold
    clear: float = Field(3.0, ge=0, le=600)                 # --clear
    cooldown: float = Field(0.0, ge=0, le=3600)             # --cooldown
    target_fps: Optional[float] = Field(None, gt=0, le=120) # --target-fps
    tracker: Literal["bytetrack", "botsort"] = "bytetrack"  # --tracker
    realtime: bool = True                                   # --realtime
    loop: bool = False                                      # web only: restart a video file when it ends
    no_clips: bool = False                                  # --no-clips
    clip_pre: float = Field(3.0, ge=0, le=60)               # --clip-pre
    clip_post: float = Field(3.0, ge=0, le=60)              # --clip-post
    merge_gap: float = Field(1.0, ge=0, le=60)              # --merge-gap
    save_video: bool = False                                # inverse of --no-save (off by default here)
    webhook: Optional[str] = None                           # --webhook


def resolve_source(s: str) -> str:
    """Only files inside videos/, webcam indexes and stream URLs are accepted (no arbitrary server paths)."""
    s = s.strip()
    if s.isdigit() or re.match(r"^(rtsp|rtmp|http|https|udp)://", s):
        return s
    p = (VIDEOS / s).resolve()
    if p.parent != VIDEOS.resolve() or not p.is_file():
        raise HTTPException(400, f"unknown video '{s}' (put files in videos/, or use a webcam index / rtsp:// URL)")
    return str(p)


def source_label(s: str) -> str:
    return f"camera{s}" if s.isdigit() else (re.sub(r"\W+", "_", s.split("://", 1)[1]).strip("_")[:40] if "://" in s else Path(s).stem)


# ----------------------------------------------------------------------------------------------------------------
class History:
    """Every alert episode of every run, persisted so it survives restarts."""

    def __init__(self):
        self.lock = threading.Lock()
        try:
            self.rows = json.loads(HISTORY.read_text())
            for r in self.rows:
                if r["status"] == "active":
                    r["status"] = "interrupted"
        except (OSError, ValueError):
            self.rows = []

    def save(self):                                          # caller holds the lock
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        tmp = HISTORY.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rows))
        tmp.replace(HISTORY)


history = History()


class Runner:
    def __init__(self):
        self.thread: threading.Thread | None = None
        self.stop_ev = threading.Event()
        self._det, self._det_key = None, None
        self.cond = threading.Condition()
        self.jpeg, self.seq = None, 0
        self.reset(None)

    def reset(self, params):
        self.state = "idle"             # idle | loading | running | finished | stopped | error
        self.error = ""
        self.params = params
        self.run_id = None
        self.count = self.frames = self.dropped = 0
        self.alerting = False
        self.fps = self.stream_t = 0.0
        self.outputs: dict = {}
        self.started_at = None
        self.max_people = 0

    # -- control -------------------------------------------------------------------------------------------------
    def start(self, p: RunParams):
        self.stop()
        self.reset(p)
        self.state = "loading"
        self.jpeg = None
        self.stop_ev = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(p, self.stop_ev), daemon=True, name="run")
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_ev.set()
            self.thread.join(timeout=30)

    def status(self) -> dict:
        return {"state": self.state, "error": self.error, "params": self.params.model_dump() if self.params else None,
                "run_id": self.run_id, "count": self.count, "threshold": self.params.threshold if self.params else None,
                "alerting": self.alerting, "fps": round(self.fps, 1), "frames": self.frames, "dropped": self.dropped,
                "elapsed_s": round(self.stream_t, 1), "max_people": self.max_people, "outputs": self.outputs,
                "device": self._det.device if self._det else None}

    def _publish(self, f):
        h, w = f.shape[:2]
        if w > PUBLISH_WIDTH:
            f = cv2.resize(f, (PUBLISH_WIDTH, int(h * PUBLISH_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.cond:
                self.jpeg, self.seq = buf.tobytes(), self.seq + 1
                self.cond.notify_all()
            return f

    # -- the run (same steps as peoplecount.live.run) ------------------------------------------------------------
    def _run(self, p: RunParams, stop: threading.Event):
        src = pc = clips = am = writer = None
        series, t_last, last_count, n = [], 0.0, 0, 0
        paths = None
        try:
            spec = resolve(p.model)
            key = (spec.name, p.device, p.conf, p.imgsz, p.threads)
            if key != self._det_key:                          # reload the model only when it changed
                self._det, self._det_key = None, None
                self._det = Detector(spec, p.device, p.conf, p.imgsz, threads=p.threads)
                self._det_key = key
                self._det.warmup(np.zeros((640, 640, 3), np.uint8), 2)
            det = self._det
            source = resolve_source(p.source)
            live = is_live(source)
            label = source_label(source)
            video, alert_log, episodes_path, clips_dir, counts_path = output_paths(
                Namespace(source=source, out_dir=str(SAMPLES), no_save=not p.save_video, save=None, alert_log=None), spec)
            self.run_id = Path(alert_log).name.removesuffix(".alerts.jsonl")
            self.started_at = time.time()
            self.outputs = {"video": video, "alerts": alert_log, "episodes": episodes_path, "counts": counts_path,
                            "clips_dir": None if p.no_clips else clips_dir}
            snap_dir = SAMPLES / "webapp" / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            last_out = {"f": None}
            rows: dict[int, dict] = {}                        # episode number -> history row of this run

            def on_event(e):
                with history.lock:
                    if e.kind == "ALERT":
                        snap = f"{self.run_id}_{e.episode}.jpg"
                        if last_out["f"] is not None:
                            cv2.imwrite(str(snap_dir / snap), last_out["f"], [cv2.IMWRITE_JPEG_QUALITY, 80])
                        row = {"id": f"{self.run_id}#{e.episode}", "run_id": self.run_id, "source": label,
                               "model": spec.name, "status": "active", "threshold": e.threshold, "peak_count": e.peak_count,
                               "started_at": e.wall_time - (e.t - e.start_t), "alerted_at": e.wall_time, "duration_s": None,
                               "snapshot": f"webapp/snapshots/{snap}", "clip": None}
                        rows[e.episode] = row
                        history.rows.append(row)
                    elif e.episode in rows:
                        rows[e.episode].update(status="cleared", peak_count=e.peak_count, duration_s=e.duration_s,
                                               reason=e.reason)
                    history.save()

            sinks = [ConsoleSink(), JsonlSink(alert_log), on_event] + ([WebhookSink(p.webhook)] if p.webhook else [])
            am = AlertManager(p.threshold, p.hold, p.clear, cooldown_s=p.cooldown, sinks=sinks)
            if not p.no_clips:
                clips = ClipRecorder(clips_dir, p.clip_pre, p.clip_post, hold_s=p.hold, merge_gap_s=p.merge_gap)
            ema, offset, loops = 0.0, 0.0, 0
            self.state = "running"
            while not stop.is_set():
                src = VideoSource(source, realtime=p.realtime, target_fps=p.target_fps)
                pc = PeopleCounter(det, fps=src.eff_fps, tracker=p.tracker)   # new tracker per pass: the scene jumps
                for idx, t, frame in src:
                    if stop.is_set():
                        break
                    t += offset
                    t0 = time.perf_counter()
                    res = pc.step(frame)
                    ev = am.update(res.smooth, t)
                    series.append((n, round(t, 3), timecode(t), res.smooth, res.raw))
                    dt = time.perf_counter() - t0
                    ema = dt if not ema else 0.9 * ema + 0.1 * dt
                    n += 1
                    t_last, last_count = t, res.smooth
                    out = draw(frame, res, 1 / ema, p.threshold, am.active, spec.name, t)
                    last_out["f"] = self._publish(out)
                    if clips:
                        clips.add_frame(t, out)
                        if ev and ev.kind == "ALERT":
                            path = clips.on_alert(ev.start_t, ev.t, ev.episode)
                            am.episodes[-1]["clip"] = path
                            if ev.episode in rows and path:
                                with history.lock:
                                    rows[ev.episode]["clip"] = str(Path(path).relative_to(SAMPLES))
                                    history.save()
                    if video:
                        if writer is None:
                            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), src.eff_fps, src.size)
                        writer.write(out)
                    self.count, self.alerting, self.fps = res.smooth, am.active, 1 / ema
                    self.frames, self.dropped, self.stream_t = n, src.dropped, t
                    self.max_people = max(self.max_people, res.smooth)
                if stop.is_set() or live or not p.loop:
                    src.close()
                    break
                offset = t_last + 1.0 / src.eff_fps
                src.close()
                loops += 1
            self.state = "stopped" if stop.is_set() else "finished"
        except Exception as ex:  # noqa: BLE001
            self.state, self.error = "error", f"{type(ex).__name__}: {ex}"
        finally:
            self.alerting = False
            try:
                if writer:
                    writer.release()
                if am:
                    am.finish(t_last, last_count)
                if clips:
                    clips.close()
                    by_ep = {e: c["path"] for c in clips.clips for e in c["episodes"]}
                    for ep in am.episodes:
                        ep["clip"] = by_ep.get(ep["episode"], ep["clip"])
                    with history.lock:
                        for r in history.rows:
                            if r["run_id"] == self.run_id and r["id"].split("#")[1].isdigit():
                                c = by_ep.get(int(r["id"].split("#")[1]))
                                if c:
                                    r["clip"] = str(Path(c).relative_to(SAMPLES))
                        history.save()
                if paths is None and self.outputs:
                    with open(self.outputs["counts"], "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["frame", "t", "timecode", "people", "raw"])
                        w.writerows(series)
                    smooth = [r[3] for r in series] or [0]
                    with open(self.outputs["episodes"], "w") as f:
                        json.dump({"source": p.source, "model": p.model, "threshold": p.threshold, "hold_s": p.hold,
                                   "clear_s": p.clear, "summary": {"frames": n, "duration_s": round(t_last, 3),
                                                                  "max_people": int(max(smooth))},
                                   "episodes": am.episodes if am else [], "clips": clips.clips if clips else []}, f, indent=1)
            except Exception as ex:  # noqa: BLE001
                print(f"[run] cleanup failed: {ex}", flush=True)


runner = Runner()
app = FastAPI(title="people-counter dashboard")


@app.get("/api/options")
def options():
    devices = ["auto", "cpu"]
    try:
        import torch
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
    except ImportError:
        pass
    videos = sorted(p.name for p in VIDEOS.iterdir() if p.suffix.lower() in VIDEO_EXT) if VIDEOS.is_dir() else []
    return {"videos": videos, "devices": devices, "trackers": ["bytetrack", "botsort"],
            "models": [{"name": s.name, "imgsz": s.imgsz, "note": s.note, "default": s.name in DEFAULT_SET}
                       for s in REGISTRY.values()],
            "defaults": RunParams(source="").model_dump()}


@app.post("/api/run")
def start_run(p: RunParams):
    resolve_source(p.source)                                  # fail fast with a 400 before spawning the run
    runner.start(p)
    return runner.status()


@app.post("/api/stop")
def stop_run():
    runner.stop()
    return runner.status()


@app.get("/api/status")
def status():
    return runner.status()


@app.get("/api/frame")
def frame():
    if runner.jpeg is None:
        raise HTTPException(503, "no frame yet")
    return Response(runner.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/alerts")
def alerts(source: str | None = None, limit: int = 1000):
    with history.lock:
        rows = [r for r in history.rows if source in (None, r["source"])]
    rows.sort(key=lambda r: r["started_at"], reverse=True)
    sources: dict[str, int] = {}
    with history.lock:
        for r in history.rows:
            sources[r["source"]] = sources.get(r["source"], 0) + 1
    return {"alerts": rows[:limit], "sources": sources}


@app.get("/api/files/{path:path}")
def files(path: str):
    """Snapshots and evidence clips under samples/."""
    p = (SAMPLES / path).resolve()
    if SAMPLES.resolve() not in p.parents or not p.is_file() or p.suffix.lower() not in {".jpg", ".mp4"}:
        raise HTTPException(404)
    return FileResponse(p)


DIST = Path(__file__).parent / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
