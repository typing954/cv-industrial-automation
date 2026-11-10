"""Run N models over the same video; store per-model JSON (latency stats, FPS, resources, per-frame counts)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import hardware
from .alert import AlertManager
from .counter import PeopleCounter
from .models import Detector, DEFAULT_SET, resolve, safe_name
from .video import VideoSource


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def bench_one(name, video, args, hw, tag):
    spec = resolve(name)
    if args.threads and args.device == "cpu":
        import torch
        torch.set_num_threads(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    t_load = time.perf_counter()
    det = Detector(spec, args.device, args.conf, args.imgsz, threads=args.threads)
    probe = VideoSource(video, max_frames=1)
    first = next(iter(probe))[2]
    probe.close()
    det.warmup(first, args.warmup)
    load_s = time.perf_counter() - t_load

    src = VideoSource(video, target_fps=args.target_fps, max_frames=args.max_frames)
    pc = PeopleCounter(det, fps=src.eff_fps, track=not args.no_track, tracker=args.tracker)
    am = AlertManager(args.threshold, hold_s=args.hold, clear_s=args.clear, cooldown_s=args.cooldown)
    sampler = hardware.ResourceSampler().start()
    lat, det_t, trk_t, idxs, raw, tracked, smooth, flags = [], [], [], [], [], [], [], []
    t_wall = time.perf_counter()
    t_dec = 0.0
    it = iter(src)
    while True:
        t0 = time.perf_counter()
        try:
            idx, t, frame = next(it)
        except StopIteration:
            break
        t_dec += time.perf_counter() - t0
        t1 = time.perf_counter()
        r = pc.step(frame)
        lat.append(time.perf_counter() - t1)
        det_t.append(r.t_detect); trk_t.append(r.t_track)
        am.update(r.smooth, t)
        idxs.append(idx); raw.append(r.raw); tracked.append(r.tracked); smooth.append(r.smooth); flags.append(int(am.active))
    wall = time.perf_counter() - t_wall
    res = sampler.stop()
    src.close()
    n = len(lat)
    ms = np.array(lat) * 1e3
    out = {
        "model": spec.name, "backend": spec.backend, "weights": spec.weights, "imgsz": det.imgsz, "device": det.device,
        "half": det.half, "tag": tag, "hardware": hw, "video": os.path.basename(video), "video_fps": src.fps,
        "video_size": src.size, "target_fps": args.target_fps, "detect_fps": src.eff_fps, "frames": n, "conf": args.conf, "threshold": args.threshold,
        "load_warmup_s": round(load_s, 2),
        "latency_ms": {"mean": ms.mean(), "p50": pct(ms, 50), "p95": pct(ms, 95), "p99": pct(ms, 99),
                       "max": ms.max(), "detect_mean": np.mean(det_t) * 1e3, "track_mean": np.mean(trk_t) * 1e3,
                       "decode_mean": t_dec / max(n, 1) * 1e3},
        "fps_inference": 1000 / ms.mean(),               # model+tracker only
        "fps_end_to_end": n / wall,                      # incl. video decode
        "fps_p95_worst": 1000 / pct(ms, 95),              # sustained-ish worst case
        "realtime_factor": (n / wall) / src.eff_fps,  # >=1 => keeps up with the detection rate
        "resources": res,
        "alerts": sum(e.kind == "ALERT" for e in am.events), "alert_events": [vars(e) for e in am.events],
        "series": {"frame_idx": idxs, "raw": raw, "tracked": tracked, "smooth": smooth, "alert_active": flags},
        "unique_people": len(pc.seen), "max_count": int(max(smooth)) if smooth else 0,
    }
    return out


def run(args):
    hw = hardware.collect(not args.no_cloud_detect)
    dev = Detector._pick_device(args.device)
    args.device = dev
    tag = args.tag or hardware.auto_tag(hw, dev)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    models = args.models or DEFAULT_SET
    print(f"hardware tag: {tag}\n{json.dumps({k: v for k, v in hw.items()}, indent=1)}", flush=True)
    written, summary = [], []
    bar = args.target_fps or 25.0   # real-time bar: the detection rate you asked for (25 FPS if you asked for every frame)
    for video in args.video:
        for m in models:
            print(f"\n=== {m} on {tag} | {os.path.basename(video)}", flush=True)
            try:
                r = bench_one(m, video, args, hw, tag)
            except Exception as e:  # noqa: BLE001  one bad model must not kill the sweep
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            f = outdir / f"{tag}__{safe_name(m)}__{Path(video).stem}.json"
            f.write_text(json.dumps(r, indent=1, default=float))
            written.append(f); summary.append(r)
            print(f"  {r['fps_end_to_end']:.1f} FPS e2e | p95 {r['latency_ms']['p95']:.1f} ms | "
                  f"max count {r['max_count']} | alerts {r['alerts']} | rt-factor {r['realtime_factor']:.2f} -> {f}", flush=True)

    if summary:
        print_summary(summary, bar)
    if summary and not args.no_report:
        from .report import build
        stamp = time.strftime("%Y%m%d-%H%M%S")
        rep = Path(args.reports) / f"report_{tag}_{stamp}.html"
        rep.parent.mkdir(parents=True, exist_ok=True)
        build(outdir, rep, args.gt, bar, files=written)
        print(f"\nReport for this run: {rep}  (+ {rep.with_suffix('.csv')})")
        print("All runs combined:   python -m peoplecount report")


def print_summary(rs, target_fps):
    """Console table at the end of every run: one row per (video, model)."""
    hdr = f"{'video':22s} {'model':20s} {'E2E FPS':>8s} {'model FPS':>9s} {'p50 ms':>7s} {'p95 ms':>7s} {'p99 ms':>7s} " \
          f"{'RT x':>5s} {'avg/max':>8s} {'alerts':>6s} {'CPU%':>6s} {'RAM MB':>7s} {'GPU%':>5s} {'VRAM MB':>8s} {'RT?':>4s}"
    print("\n" + "=" * len(hdr) + f"\nRESULTS on {rs[0]['tag']}  (real-time = >= {target_fps:g} FPS and p95 <= {1000/target_fps:.0f} ms)\n" + hdr)
    f = lambda v, fmt: "-" if v is None else format(v, fmt)  # noqa: E731
    for r in sorted(rs, key=lambda r: (r["video"], -r["fps_end_to_end"])):
        l, u = r["latency_ms"], r["resources"]
        ok = r["fps_end_to_end"] >= target_fps and l["p95"] <= 1000 / target_fps
        avg = sum(r["series"]["smooth"]) / max(len(r["series"]["smooth"]), 1)
        print(f"{r['video'][:22]:22s} {r['model'][:20]:20s} {r['fps_end_to_end']:8.1f} {r['fps_inference']:9.1f} {l['p50']:7.1f} "
              f"{l['p95']:7.1f} {l['p99']:7.1f} {r['realtime_factor']:5.2f} {avg:4.1f}/{r['max_count']:<3d} {r['alerts']:6d} "
              f"{f(u['cpu_pct_avg'], '6.0f'):>6s} {u['rss_peak_mb']:7d} {f(u['gpu_util_avg'], '5.0f'):>5s} "
              f"{f(u['gpu_mem_peak_mb'], '8.0f'):>8s} {'yes' if ok else 'NO':>4s}")
    print("=" * len(hdr))
