"""Merge result JSONs (from any number of machines) into one HTML + CSV comparison report."""
from __future__ import annotations

import base64
import csv
import html
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PALETTE = ["#2a6fdb", "#e8743b", "#19a979", "#9b59d0", "#d1478c", "#7f8c8d", "#c9a227", "#16a2b8", "#8c564b", "#4c5fd5"]


def load(results_dir, files=None):
    paths = [Path(f) for f in files] if files else sorted(Path(results_dir).glob("*.json"))
    rs = [json.loads(p.read_text()) for p in paths]
    return [r for r in rs if "series" in r]


def load_gt(path):
    gt = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            gt[int(row["frame"])] = int(row["count"])
    return gt


def _mae(series, ref):
    common = [i for i in series if i in ref]
    if not common:
        return None, None
    e = np.array([series[i] - ref[i] for i in common], float)
    return float(np.abs(e).mean()), float((np.abs(e) <= 0).mean() * 100)  # MAE, exact-match %


def _png(fig):
    import matplotlib.pyplot as plt
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return '<img src="data:image/png;base64,%s">' % base64.b64encode(b.getvalue()).decode()


def build(results_dir, out_html, gt_path=None, target_fps=25.0, files=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = load(results_dir, files)
    if not rs:
        raise SystemExit(f"no results in {results_dir}")
    gt = load_gt(gt_path) if gt_path else None
    by_video = defaultdict(list)
    for r in rs:
        by_video[r["video"]].append(r)

    # reference series per video: GT if given, else per-frame median of every model's smoothed count
    refs, ref_kind = {}, {}
    for v, rr in by_video.items():
        per_model = {}
        for r in rr:
            per_model.setdefault(r["model"], dict(zip(r["series"]["frame_idx"], r["series"]["smooth"])))
        if gt:
            refs[v], ref_kind[v] = gt, "ground truth"
        elif len(per_model) < 2:  # a consensus of one model is meaningless
            refs[v], ref_kind[v] = {}, "n/a (need >=2 models or --gt for an error metric)"
        else:
            frames = set.intersection(*[set(s) for s in per_model.values()])
            refs[v] = {f: float(np.median([s[f] for s in per_model.values()])) for f in frames}
            ref_kind[v] = "consensus (median of all models) — agreement, NOT true accuracy"

    tags = sorted({r["tag"] for r in rs})
    models = sorted({r["model"] for r in rs}, key=lambda m: -np.mean([r["fps_end_to_end"] for r in rs if r["model"] == m]))
    col = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    rows = []
    for r in rs:
        s = dict(zip(r["series"]["frame_idx"], r["series"]["smooth"]))
        mae, exact = _mae(s, refs[r["video"]])
        res, lat = r["resources"], r["latency_ms"]
        rows.append({"video": r["video"], "hardware": r["tag"], "model": r["model"], "device": r["device"],
                     "fps_e2e": r["fps_end_to_end"], "fps_infer": r["fps_inference"], "p50_ms": lat["p50"],
                     "p95_ms": lat["p95"], "p99_ms": lat["p99"], "detect_ms": lat["detect_mean"], "track_ms": lat["track_mean"],
                     "rt_factor": r["realtime_factor"], "mae": mae, "exact_pct": exact, "max_count": r["max_count"], "avg_count": float(np.mean(r["series"]["smooth"])), "unique_people": r["unique_people"],
                     "alerts": r["alerts"], "cpu_pct": res["cpu_pct_avg"], "rss_mb": res["rss_peak_mb"],
                     "gpu_util": res["gpu_util_avg"], "gpu_mem_mb": res["gpu_mem_peak_mb"], "gpu_w": res["gpu_power_avg_w"],
                     "meets_target": r["fps_end_to_end"] >= target_fps and lat["p95"] <= 1000 / target_fps})
    out = Path(out_html)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    figs = []
    # 1. FPS grouped by model x hardware
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.1), 4.2))
    wdt = 0.8 / len(tags)
    for j, t in enumerate(tags):
        ys = [np.mean([x["fps_e2e"] for x in rows if x["model"] == m and x["hardware"] == t] or [0]) for m in models]
        ax.bar(np.arange(len(models)) + j * wdt, ys, wdt, label=t, color=PALETTE[j % len(PALETTE)])
    ax.axhline(target_fps, color="k", ls="--", lw=1); ax.text(0, target_fps, f" real-time target {target_fps:g} FPS", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(models)) + wdt * (len(tags) - 1) / 2, models, rotation=30, ha="right")
    ax.set_ylabel("end-to-end FPS (higher is better)"); ax.legend(fontsize=7); ax.set_title("Throughput by model and hardware")
    figs.append(("Throughput", _png(fig)))
    # 2. p95 latency
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.1), 4.2))
    for j, t in enumerate(tags):
        ys = [np.mean([x["p95_ms"] for x in rows if x["model"] == m and x["hardware"] == t] or [0]) for m in models]
        ax.bar(np.arange(len(models)) + j * wdt, ys, wdt, label=t, color=PALETTE[j % len(PALETTE)])
    ax.axhline(1000 / target_fps, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(len(models)) + wdt * (len(tags) - 1) / 2, models, rotation=30, ha="right")
    ax.set_ylabel("p95 per-frame latency, ms (lower is better)"); ax.legend(fontsize=7); ax.set_title("Tail latency")
    figs.append(("Latency", _png(fig)))
    # 3. speed vs agreement
    if any(x["mae"] is not None for x in rows):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for x in rows:
            if x["mae"] is None:
                continue
            ax.scatter(x["fps_e2e"], x["mae"], s=60, color=col[x["model"]], marker="os^Dv<>"[tags.index(x["hardware"]) % 8])
            ax.annotate(x["model"], (x["fps_e2e"], x["mae"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("end-to-end FPS (log)"); ax.set_ylabel("MAE of count vs reference (lower is better)")
        ax.set_title("Speed vs counting error (marker shape = hardware)"); ax.grid(alpha=.3)
        figs.append(("Speed vs error", _png(fig)))
    # 4. count over time per video (one hardware: the first)
    for v, rr in by_video.items():
        fig, ax = plt.subplots(figsize=(10, 4))
        seen = set()
        for r in sorted(rr, key=lambda r: r["model"]):
            if r["model"] in seen:
                continue
            seen.add(r["model"])
            ax.plot(np.array(r["series"]["frame_idx"]) / r["video_fps"], r["series"]["smooth"], label=r["model"], color=col[r["model"]], lw=1.3)
        if gt:
            ax.plot(np.array(sorted(gt)) / rr[0]["video_fps"], [gt[k] for k in sorted(gt)], "k--", label="ground truth", lw=1.5)
        ax.axhline(rr[0]["threshold"], color="red", ls=":", lw=1.2, label=f"alert threshold ({rr[0]['threshold']})")
        ax.set_xlabel("time (s)"); ax.set_ylabel("people (smoothed)"); ax.legend(fontsize=7, ncol=2); ax.set_title(f"Count over time — {v}")
        figs.append((f"Counts: {v}", _png(fig)))

    cols = [("video", "Video"), ("hardware", "Hardware"), ("model", "Model"), ("fps_e2e", "E2E FPS"), ("p50_ms", "p50 ms"),
            ("p95_ms", "p95 ms"), ("fps_infer", "model FPS"), ("p99_ms", "p99 ms"), ("detect_ms", "detect ms"), ("track_ms", "track ms"), ("rt_factor", "RT factor"),
            ("mae", "MAE"), ("exact_pct", "exact %"), ("avg_count", "avg cnt"), ("max_count", "max cnt"), ("unique_people", "unique IDs"), ("alerts", "alerts"), ("cpu_pct", "CPU %"),
            ("rss_mb", "RSS MB"), ("gpu_util", "GPU %"), ("gpu_mem_mb", "GPU MB"), ("gpu_w", "GPU W"), ("meets_target", f"≥{target_fps:g}FPS RT")]

    def cell(k, v):
        if v is None:
            return "–"
        if isinstance(v, bool):
            return "✅" if v else "❌"
        return f"{v:.1f}" if isinstance(v, float) else html.escape(str(v))

    table = "<table><tr>" + "".join(f"<th>{h}</th>" for _, h in cols) + "</tr>"
    for x in sorted(rows, key=lambda x: (x["video"], x["hardware"], -x["fps_e2e"])):
        table += "<tr>" + "".join(f"<td>{cell(k, x[k])}</td>" for k, _ in cols) + "</tr>"
    table += "</table>"
    hwrows = "".join(f"<tr><td>{html.escape(t)}</td><td>{html.escape(json.dumps({k: v for k, v in next(r for r in rs if r['tag']==t)['hardware'].items() if k in ('cpu','cpu_cores_logical','ram_gb','gpu','gpu_mem_gb','cloud','torch','cuda')}))}</td></tr>" for t in tags)
    refnote = "".join(f"<li><b>{html.escape(v)}</b>: error measured against {k}</li>" for v, k in ref_kind.items())
    page = f"""<!doctype html><meta charset=utf-8><title>People counting model comparison</title>
<style>body{{font:14px system-ui;max-width:1300px;margin:2em auto;padding:0 1em}}table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccc;padding:3px 7px;text-align:right}}td:nth-child(-n+3),th:nth-child(-n+3){{text-align:left}}th{{background:#f2f2f2}}
img{{max-width:100%}}.w{{overflow-x:auto}}</style>
<h1>People counting — model × hardware comparison</h1>
<p>Real-time target: <b>{target_fps:g} FPS</b> end-to-end and p95 latency ≤ {1000/target_fps:.0f} ms. RT factor ≥ 1 means the pipeline keeps up with the source frame rate.
Counts are tracker-confirmed (ByteTrack) and rolling-median smoothed.</p><ul>{refnote}</ul>
<h2>Hardware</h2><table>{hwrows}</table>
<h2>Results</h2><div class=w>{table}</div>
{''.join(f'<h2>{t}</h2>{i}' for t, i in figs)}"""
    out.write_text(page)
    return out
