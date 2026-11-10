"""Endpoint throughput test: how many frames/s can this endpoint sustain, and at what latency?
Pre-encodes frames from your (looped) video, then hammers the endpoint with N concurrent requests of B frames each,
for every (batch size, concurrency) combination. No tracking/alerts, just the endpoint."""
from __future__ import annotations

import itertools
import json
import threading
import time
from pathlib import Path

import numpy as np

from . import codec
from .sources import open_source


def _grab_frames(video: str, n: int, send_max_side: int, quality: int) -> list[bytes]:
    src = open_source(video, pace=False, loops=0, duration_s=0)
    out = []
    for i, fr in enumerate(src):
        if i % 3 == 0:                                   # spread over the clip
            out.append(codec.encode_frame(fr.frame, send_max_side, quality)[1])
        if len(out) >= n:
            break
    src.close()
    if not out:
        raise SystemExit("no frames could be read from the video")
    return out


def run_combo(factory, jpegs, batch, conc, seconds, conf):
    lock = threading.Lock()
    rtts, srv, frames, errors = [], [], [0], [0]
    stop = time.monotonic() + seconds

    def worker(k):
        tr = factory()
        i = k * batch
        while time.monotonic() < stop:
            sel = [jpegs[(i + j) % len(jpegs)] for j in range(batch)]
            i += batch
            body = codec.make_request(sel, conf=conf)
            t0 = time.monotonic()
            try:
                boxes, meta = codec.parse_response(tr.invoke(body))
                dt = time.monotonic() - t0
                with lock:
                    frames[0] += len(boxes)
                    rtts.append(dt * 1e3)
                    if meta.get("server_ms") is not None:
                        srv.append(meta["server_ms"] / len(boxes))
            except Exception:  # noqa: BLE001
                with lock:
                    errors[0] += 1
                time.sleep(0.2)

    ths = [threading.Thread(target=worker, args=(k,)) for k in range(conc)]
    t0 = time.monotonic()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    el = time.monotonic() - t0
    return {"batch": batch, "concurrency": conc, "fps": frames[0] / el, "rtt_p50_ms": float(np.percentile(rtts, 50)) if rtts else None,
            "rtt_p95_ms": float(np.percentile(rtts, 95)) if rtts else None,
            "server_ms_per_frame": float(np.mean(srv)) if srv else None, "errors": errors[0]}


def bench(args, factory):
    jpegs = _grab_frames(args.video, 24, args.send_size, args.jpeg_quality)
    kb = np.mean([len(j) for j in jpegs]) / 1024
    print(f"benchmarking endpoint | {len(jpegs)} sample frames, avg {kb:.0f} KB each @ ≤{args.send_size}px | {args.seconds}s per setting")
    print("warm-up…", flush=True)
    run_combo(factory, jpegs, 1, 1, 3, args.conf)
    rows = []
    print(f"{'batch':>5} {'conc':>5} {'FPS':>8} {'rtt p50':>9} {'rtt p95':>9} {'server ms/frame':>16} {'errors':>7}")
    for b, c in itertools.product(args.batch_sizes, args.concurrency):
        r = run_combo(factory, jpegs, b, c, args.seconds, args.conf)
        rows.append(r)
        f = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
        srv = f"{r['server_ms_per_frame']:.1f}" if r["server_ms_per_frame"] else "-"
        print(f"{b:5d} {c:5d} {r['fps']:8.1f} {f(r['rtt_p50_ms']):>8}ms {f(r['rtt_p95_ms']):>8}ms "
              f"{srv:>16} {r['errors']:7d}", flush=True)
    good = [r for r in rows if r["errors"] == 0] or rows
    best = max(good, key=lambda r: r["fps"])
    print(f"\nbest: batch {best['batch']} × {best['concurrency']} concurrent requests → {best['fps']:.1f} frames/s "
          f"(p95 round-trip {best['rtt_p95_ms']:.0f} ms)")
    print(f"=> one endpoint instance sustains about {best['fps']:.0f} frames/s; cameras ≈ that ÷ (frames/s you send per camera)")
    out = Path(args.out_dir) / "endpoint_bench"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.endpoint_type}.json"
    f.write_text(json.dumps({"args": {k: v for k, v in vars(args).items() if k not in ("func",)}, "rows": rows, "best": best}, indent=1, default=str))
    print("saved", f)
    return rows
