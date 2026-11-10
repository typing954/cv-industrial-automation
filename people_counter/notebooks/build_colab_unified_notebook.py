"""Generates notebooks/people_counter_colab_unified.ipynb (Part A benchmark + Part B alerts/clips, per video).
Run:  python notebooks/build_colab_unified_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("people_counter_colab_unified.ipynb")
files = {f"peoplecount/{p.name}": p.read_text() for p in sorted((ROOT / "peoplecount").glob("*.py"))}
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s, form=False, title=None):
    s = s.strip("\n")
    if title:
        s = f"#@title {title}\n" + s
    cells.append({"cell_type": "code", "metadata": {"cellView": "form"} if form else {}, "execution_count": None,
                  "outputs": [], "source": s.splitlines(True)})


md(r"""
# People counter: unified run (benchmark + alerts + clips, per video)

Give it **one video** or a **whole folder**. For **each video**, one after another, it runs:

* **Part A, model benchmark**: every model in `MODELS` → FPS, latency (p50/p95/p99), real-time pass/fail, GPU/CPU/RAM use, counts, and an HTML report.
* **Part B, alerts + clips**: counts people frame by frame, records **every** time the limit is exceeded (when it was first exceeded, alert time, duration, peak) and cuts clips from **3 s before the violation to 3 s after the alert** (overlapping alerts merge into one clip).

…then shows that video's **report, results table, alert table and playable clips** before moving to the next video. A cross-video summary comes at the end.

**How to use**
1. `Runtime ▸ Change runtime type ▸ GPU`.
2. Put videos under `videos/` (file browser ▸ drag & drop, the optional Drive-unzip cell, or the upload prompt). Set `VIDEO_PATH` in step 4 to **a single file** (e.g. `videos/lobby.mp4`) or **a folder** (`videos`).
3. `Runtime ▸ Run all`.
""")

code(r'''
import subprocess, torch
print(subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout or "nvidia-smi not found")
GPU = torch.cuda.is_available()
print("CUDA available:", GPU, "|", torch.cuda.get_device_name(0) if GPU else "")
if not GPU:
    print("⚠️  No GPU found. Runtime ▸ Change runtime type ▸ pick a GPU, then run all again. (It will still run on CPU, slowly.)")
''', title="1. Check the GPU")

code(r'''
!pip -q install ultralytics lap nvidia-ml-py "opencv-python<5"
''', title="2. Install dependencies")

code("FILES = " + json.dumps(files, ensure_ascii=False, indent=0) + r'''

import pathlib
for name, src in FILES.items():
    p = pathlib.Path(name); p.parent.mkdir(exist_ok=True); p.write_text(src)
print("wrote", len(FILES), "files to", pathlib.Path("peoplecount").resolve())
''', form=True, title="3. Write the peoplecount package (generated from the repo, no need to edit)")

code(r'''
import os, time
T0 = time.time()

#@markdown ### What to run
VIDEO_PATH = "videos"          #@param {type:"string"}
NAME_FILTER = ""               #@param {type:"string"}
#@markdown `VIDEO_PATH` = one video file **or** a folder (searched recursively). `NAME_FILTER` = only videos whose file name contains this text.
RUN_BENCHMARK = True           #@param {type:"boolean"}
RUN_ALERTS_AND_CLIPS = True    #@param {type:"boolean"}

#@markdown ### Part A: benchmark
MODELS = "yolov8n,yolo11n,yolo11s,yolo11m,yolo11l,rtdetr-l,ssdlite_mobilenet,fasterrcnn_mobilenet"  #@param {type:"string"}
MAX_FRAMES = 0                 #@param {type:"integer"}

#@markdown ### Part B: alerts + clips
ALERT_MODEL = "yolo11s"        #@param {type:"string"}
THRESHOLD = 10                 #@param {type:"integer"}
HOLD_S = 2.0                   #@param {type:"number"}
CLEAR_S = 3.0                  #@param {type:"number"}
CLIP_PRE_S = 3.0               #@param {type:"number"}
CLIP_POST_S = 3.0              #@param {type:"number"}
SAVE_FULL_ANNOTATED_VIDEO = True  #@param {type:"boolean"}

#@markdown ### Detection (both parts)
CONF = 0.3                     #@param {type:"number"}
IMGSZ = 0                      #@param {type:"integer"}
TARGET_FPS = 25               #@param {type:"number"}
#@markdown **TARGET_FPS** = how many frames per second to run detection on (0 = every frame). A 30 fps video with 5 here is analysed 5 times per second, and played in real time. It also sets the real-time bar in Part A (25 when 0).
TRACKER = "bytetrack"          #@param ["bytetrack", "botsort"]

#@markdown ### Display
SHOW_FULL_REPORT = True        #@param {type:"boolean"}
MAX_PREVIEWS_PER_VIDEO = 4     #@param {type:"integer"}

DEVICE = "cuda:0" if GPU else "cpu"
MODEL_LIST = [m.strip() for m in MODELS.split(",") if m.strip()]
print("device:", DEVICE, "| models:", MODEL_LIST, "| alert model:", ALERT_MODEL, "| alert when people >", THRESHOLD)
''', form=True, title="4. Settings")

code(r'''
#@markdown Optional: unzip videos from a zip in your Google Drive (e.g. `MyDrive/videos.zip`). Leave empty to skip.
DRIVE_ZIP = ""  #@param {type:"string"}
if DRIVE_ZIP:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs("videos", exist_ok=True)
    !unzip -qo "/content/drive/MyDrive/{DRIVE_ZIP}" -d videos
    print("unzipped into videos/")
''', form=True, title="(optional) Load videos from a Google Drive zip")

code(r'''
import glob, shutil, cv2
EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".ogv", ".m4v", ".mpg", ".mpeg", ".wmv")

def find_videos():
    if os.path.isfile(VIDEO_PATH):
        found = [VIDEO_PATH]
    else:
        found = sorted(p for p in glob.glob(os.path.join(VIDEO_PATH, "**", "*"), recursive=True) if p.lower().endswith(EXTS))
    return [p for p in found if NAME_FILTER.lower() in os.path.basename(p).lower()]

VIDEOS = find_videos()
if not VIDEOS and not os.path.isfile(VIDEO_PATH):
    try:
        from google.colab import files
        os.makedirs(VIDEO_PATH, exist_ok=True)
        print(f"No videos found in {VIDEO_PATH}/. Choose files to upload (or drag them into the file browser and re-run this cell):")
        for name in files.upload():
            shutil.move(name, os.path.join(VIDEO_PATH, name))
        VIDEOS = find_videos()
    except ImportError:
        pass
assert VIDEOS, f"No videos found at {os.path.abspath(VIDEO_PATH)} (filter: {NAME_FILTER!r}). Put videos there or fix VIDEO_PATH."

print(f"{len(VIDEOS)} video(s) will be processed, one at a time:")
for v in VIDEOS:
    c = cv2.VideoCapture(v); fps = c.get(5) or 0
    print(f"  {v:60s} {int(c.get(3))}x{int(c.get(4))}  {fps:5.1f} fps  {c.get(7)/fps if fps else 0:6.1f}s")
''', title="5. Find the video(s)")

code(r'''
import sys, json, html as _html, traceback
import pandas as pd
from pathlib import Path
from IPython.display import HTML, Markdown, Video, display

def sh(args):
    """Run a command and stream its output live into the cell."""
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if "it/s" in line or "━" in line or "Downloading" in line: continue   # skip progress-bar noise
        print(line, end="")
    return p.wait()

COMMON = ["--device", DEVICE, "--conf", str(CONF), "--threshold", str(THRESHOLD), "--hold", str(HOLD_S),
          "--clear", str(CLEAR_S), *(["--target-fps", str(TARGET_FPS)] if TARGET_FPS else []), "--tracker", TRACKER]
if IMGSZ: COMMON += ["--imgsz", str(IMGSZ)]

BENCH_COLS = ["model", "fps_e2e", "fps_infer", "p50_ms", "p95_ms", "p99_ms", "detect_ms", "track_ms", "rt_factor", "mae",
              "avg_count", "max_count", "unique_people", "alerts", "gpu_util", "gpu_mem_mb", "cpu_pct", "rss_mb", "meets_target"]

def bench_video(v):
    """Part A for one video -> DataFrame (one row per model) and report path."""
    t = time.time()
    cmd = [sys.executable, "-m", "peoplecount", "benchmark", v, "--models", *MODEL_LIST, *COMMON,
           "--out", "results", "--reports", "reports"]
    if MAX_FRAMES: cmd += ["--max-frames", str(MAX_FRAMES)]
    sh(cmd)
    reps = [r for r in sorted(glob.glob("reports/report_*.csv"), key=os.path.getmtime) if os.path.getmtime(r) >= t]
    if not reps:
        return None, None
    return pd.read_csv(reps[-1]), reps[-1].replace(".csv", ".html")

def alerts_video(v):
    """Part B for one video -> (episodes DataFrame, list of clip paths)."""
    t = time.time()
    cmd = [sys.executable, "-m", "peoplecount", "run", v, "--model", ALERT_MODEL, *COMMON,
           "--clip-pre", str(CLIP_PRE_S), "--clip-post", str(CLIP_POST_S), "--out-dir", "samples"]
    if not SAVE_FULL_ANNOTATED_VIDEO: cmd.append("--no-save")
    sh(cmd)
    stem = Path(v).stem
    rows, clips = [], []
    for f in sorted(glob.glob(f"samples/{stem}/*.episodes.json"), key=os.path.getmtime):
        if os.path.getmtime(f) < t: continue
        d = json.load(open(f))
        for ep in d["episodes"]:
            rows.append({"video": os.path.basename(d["source"]), "model": d["model"], "alert #": ep["episode"],
                         "exceeded at": ep["start_timecode"], "alert raised": ep["alert_timecode"],
                         "over limit (s)": ep["duration_s"], "peak": ep["peak_count"], "limit": ep["threshold"],
                         "clip": os.path.basename(ep["clip"]) if ep["clip"] else None, "_clip_path": ep["clip"]})
        clips += [c["path"] for c in d.get("clips", [])]
    return pd.DataFrame(rows), clips

def preview_clips(clip_paths, episodes_df):
    for p in clip_paths[:MAX_PREVIEWS_PER_VIDEO]:
        out = p.replace(".mp4", "_preview.mp4")      # browsers can't play the raw mp4v clips -> re-encode to H.264
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", p, "-vf", "scale=640:-2", "-c:v", "libx264",
                            "-crf", "28", "-pix_fmt", "yuv420p", out])
        if r.returncode == 0:
            eps = episodes_df.loc[episodes_df["_clip_path"] == p, "alert #"].tolist() if len(episodes_df) else []
            display(Markdown(f"`{os.path.basename(p)}` covers alert(s) {eps}"))
            display(Video(out, embed=True, width=520))
''', title="Helpers")

code(r'''
BENCH_ALL, ALERTS_ALL, OVERVIEW = [], [], []

for k, v in enumerate(VIDEOS, 1):
    c = cv2.VideoCapture(v); fps = c.get(5) or 0
    display(Markdown(f"---\n# 🎬 Video {k}/{len(VIDEOS)}: `{os.path.basename(v)}`\n"
                     f"{int(c.get(3))}×{int(c.get(4))} · {fps:.1f} fps · {c.get(7)/fps if fps else 0:.1f} s"))
    row = {"video": os.path.basename(v)}
    try:
        # ---------------- Part A ----------------
        if RUN_BENCHMARK:
            display(Markdown("## Part A: model benchmark"))
            df, rep = bench_video(v)
            if df is not None:
                cols = [x for x in BENCH_COLS if x in df]
                display(df[cols].round(1).sort_values("fps_e2e", ascending=False).reset_index(drop=True))
                best = df.sort_values("fps_e2e", ascending=False).iloc[0]
                ok = df[df["meets_target"] == True]["model"].tolist()
                display(Markdown(f"**Fastest:** `{best['model']}` at {best['fps_e2e']:.1f} FPS · "
                                 f"**real-time (≥{(TARGET_FPS or 25):g} FPS):** {', '.join(ok) if ok else 'none'}"))
                row.update(fastest=best["model"], fastest_fps=round(best["fps_e2e"], 1))
                BENCH_ALL.append(df)
                if SHOW_FULL_REPORT:
                    display(Markdown(f"Full report with charts (`{rep}`):"))
                    display(HTML(f'<iframe srcdoc="{_html.escape(open(rep).read())}" width="100%" height="1000" '
                                 f'style="border:1px solid #ccc"></iframe>'))
        # ---------------- Part B ----------------
        if RUN_ALERTS_AND_CLIPS:
            display(Markdown(f"## Part B: alerts + clips (model `{ALERT_MODEL}`, limit {THRESHOLD})"))
            eps, clips = alerts_video(v)
            if len(eps):
                display(eps.drop(columns="_clip_path"))
                display(Markdown(f"**{len(eps)} violation(s)** · peak {int(eps['peak'].max())} people · "
                                 f"over the limit for {eps['over limit (s)'].sum():.1f} s in total · {len(clips)} clip(s)"))
                preview_clips(clips, eps)
                ALERTS_ALL.append(eps)
                row.update(violations=len(eps), peak=int(eps["peak"].max()), over_limit_s=round(eps["over limit (s)"].sum(), 1),
                           clips=len(clips))
            else:
                display(Markdown(f"✅ No violations: the count never stayed above {THRESHOLD} for {HOLD_S:g} s."))
                row.update(violations=0, peak=0, over_limit_s=0.0, clips=0)
    except Exception:
        display(Markdown("⚠️ **This video failed** (the rest will continue):"))
        traceback.print_exc()
        row["error"] = True
    OVERVIEW.append(row)
''', title="6. Run every video: Part A + Part B, with results shown per video")

code(r'''
display(Markdown("# 📊 Summary across all videos"))
if OVERVIEW:
    display(pd.DataFrame(OVERVIEW))
if BENCH_ALL:
    allb = pd.concat(BENCH_ALL)
    display(Markdown("**End-to-end FPS by model and video**"))
    display(allb.pivot_table(index="model", columns="video", values="fps_e2e").round(1))
    display(Markdown("**Which models were real-time (≥ target) on how many videos**"))
    display(allb.groupby("model").agg(videos=("video", "nunique"), mean_fps=("fps_e2e", "mean"),
                                      p95_ms=("p95_ms", "mean"), realtime_on=("meets_target", "sum")).round(1)
            .sort_values("mean_fps", ascending=False))
if ALERTS_ALL:
    display(Markdown("**All violations**"))
    display(pd.concat(ALERTS_ALL).drop(columns="_clip_path").reset_index(drop=True))
''', title="7. Cross-video summary")

code(r'''
!zip -rq people_counter_outputs.zip samples results reports 2>/dev/null; ls -lh people_counter_outputs.zip
try:
    from google.colab import files
    files.download("people_counter_outputs.zip")
except ImportError:
    pass
''', title="8. Download everything (annotated videos, alert logs, clips, reports)")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(cells):
    c["id"] = f"cell{i:02d}"
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", OUT, f"({OUT.stat().st_size/1e3:.0f} KB, {len(cells)} cells)")
