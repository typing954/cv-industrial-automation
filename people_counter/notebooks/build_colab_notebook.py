"""Generates notebooks/people_counter_colab.ipynb with the current peoplecount/ package embedded.
Run from anywhere:  python notebooks/build_colab_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("people_counter_colab.ipynb")

files = {f"peoplecount/{p.name}": p.read_text() for p in sorted((ROOT / "peoplecount").glob("*.py"))}

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s, form=False, title=None):
    meta = {"cellView": "form"} if form else {}
    s = s.strip("\n")
    if title:
        s = f"#@title {title}\n" + s
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": s.splitlines(True)})


md("""
# People counter: model × GPU benchmark, alerts and evidence clips

Runs the `people_counter` codebase on a **Colab GPU** over **every video in `videos/`**:

1. **Benchmark**: compares detectors (YOLOv8/11/10, RT-DETR, torchvision) with ByteTrack/BoT-SORT tracking → FPS, latency (p50/p95/p99), GPU/CPU/RAM use, counts, a combined HTML report.
2. **Alerts + clips**: counts people frame by frame, records **every** time the limit is exceeded (start time, alert time, duration, peak) and cuts a clip from **3 s before the violation to 3 s after the alert** (overlapping alerts merge into one clip).

**How to use**
1. `Runtime ▸ Change runtime type ▸ GPU` (T4 is fine; A100/L4 are faster).
2. Put your videos in the Colab working directory under `videos/` (left file browser ▸ drag & drop, or run the upload cell below).
3. `Runtime ▸ Run all`. Adjust the settings in **step 4** first if you like.

Outputs: `results/` (raw JSON), `reports/` (HTML/CSV), `samples/<video>/` (annotated video, alerts, episodes, clips); the last cell zips and downloads them.
""")

code('''
import subprocess, torch
print(subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout or "nvidia-smi not found")
GPU = torch.cuda.is_available()
print("CUDA available:", GPU, "|", torch.cuda.get_device_name(0) if GPU else "")
if not GPU:
    print("⚠️  No GPU found. Runtime ▸ Change runtime type ▸ pick a GPU, then run all again. (It will still run on CPU, slowly.)")
''', title="1. Check the GPU")

code('''
!pip -q install ultralytics lap nvidia-ml-py "opencv-python<5"
''', title="2. Install dependencies")

code("FILES = " + json.dumps(files, ensure_ascii=False, indent=0) + '''

import pathlib
for name, src in FILES.items():
    p = pathlib.Path(name); p.parent.mkdir(exist_ok=True); p.write_text(src)
print("wrote", len(FILES), "files to", pathlib.Path("peoplecount").resolve())
''', form=True, title="3. Write the peoplecount package (generated from the repo, no need to edit)")

code('''
import os, time
T0 = time.time()                                  # used to pick up only this session's outputs

VIDEO_DIR = "videos"                              #@param {type:"string"}
MODELS = "yolov8n,yolo11n,yolo11s,yolo11m,yolo11l,rtdetr-l,ssdlite_mobilenet,fasterrcnn_mobilenet"  #@param {type:"string"}
THRESHOLD = 10                                    #@param {type:"integer"}
HOLD_S = 2.0                                      #@param {type:"number"}
CLEAR_S = 3.0                                     #@param {type:"number"}
CONF = 0.3                                        #@param {type:"number"}
IMGSZ = 0                                         #@param {type:"integer"}
STRIDE = 1                                        #@param {type:"integer"}
TRACKER = "bytetrack"                             #@param ["bytetrack", "botsort"]
MAX_FRAMES = 0                                    #@param {type:"integer"}
TARGET_FPS = 25                                   #@param {type:"number"}
RUN_BENCHMARK = True                              #@param {type:"boolean"}
RUN_ALERTS_AND_CLIPS = True                       #@param {type:"boolean"}
ALERT_MODEL = "yolo11s"                           #@param {type:"string"}
CLIP_PRE_S = 3.0                                  #@param {type:"number"}
CLIP_POST_S = 3.0                                 #@param {type:"number"}
SAVE_FULL_ANNOTATED_VIDEO = True                  #@param {type:"boolean"}
TRY_TENSORRT = False                              #@param {type:"boolean"}

DEVICE = "cuda:0" if GPU else "cpu"
MODEL_LIST = [m.strip() for m in MODELS.split(",") if m.strip()]
print("device:", DEVICE, "| models:", MODEL_LIST, "| alert when people >", THRESHOLD)
''', form=True, title="4. Settings (IMGSZ/MAX_FRAMES 0 = model default / whole video; try IMGSZ=1280 for dense crowds)")

code('''
import glob
os.makedirs(VIDEO_DIR, exist_ok=True)
EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".ogv", ".m4v", ".mpg", ".mpeg", ".wmv")
def find_videos():
    return sorted(p for p in glob.glob(os.path.join(VIDEO_DIR, "**", "*"), recursive=True) if p.lower().endswith(EXTS))

VIDEOS = find_videos()
if not VIDEOS:
    try:
        from google.colab import files
        print(f"No videos in {VIDEO_DIR}/ yet. Choose files to upload (or drag them into the file browser and re-run this cell):")
        import shutil
        for name in files.upload():
            shutil.move(name, os.path.join(VIDEO_DIR, name))
        VIDEOS = find_videos()
    except ImportError:
        pass
assert VIDEOS, f"Put at least one video in {os.path.abspath(VIDEO_DIR)}/"

import cv2
print(f"{len(VIDEOS)} video(s):")
for v in VIDEOS:
    c = cv2.VideoCapture(v); fps = c.get(5) or 0
    print(f"  {v:60s} {int(c.get(3))}x{int(c.get(4))}  {fps:5.1f} fps  {c.get(7)/fps if fps else 0:6.1f}s")
''', title="5. Find the videos in videos/ (offers an upload if the folder is empty)")

code('''
import subprocess, sys

def sh(args):
    """Run a command and stream its output live into the cell."""
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if "it/s" in line or "━" in line or "Downloading" in line: continue   # skip progress-bar noise
        print(line, end="")
    return p.wait()

COMMON = ["--device", DEVICE, "--conf", str(CONF), "--threshold", str(THRESHOLD), "--hold", str(HOLD_S),
          "--clear", str(CLEAR_S), "--stride", str(STRIDE), "--tracker", TRACKER]
if IMGSZ: COMMON += ["--imgsz", str(IMGSZ)]
''', title="Helpers")

md("## Part A: Benchmark: every model on every video\nEnds with a table (FPS, latency, real-time yes/no, CPU/GPU/RAM) and an HTML report.")

code('''
import html as _html, pandas as pd
from IPython.display import HTML, display

def show_latest_report(since):
    reps = sorted(glob.glob("reports/report_*.html"), key=os.path.getmtime)
    reps = [r for r in reps if os.path.getmtime(r) >= since]
    if not reps:
        return
    rep = reps[-1]
    df = pd.read_csv(rep.replace(".html", ".csv"))
    cols = [c for c in ["video","hardware","model","fps_e2e","fps_infer","p50_ms","p95_ms","detect_ms","track_ms","rt_factor",
                        "mae","avg_count","max_count","alerts","gpu_util","gpu_mem_mb","cpu_pct","rss_mb","meets_target"] if c in df]
    display(df[cols].round(1).sort_values(["video", "fps_e2e"], ascending=[True, False]).reset_index(drop=True))
    print("report:", rep)
    page = open(rep).read()
    display(HTML(f'<iframe srcdoc="{_html.escape(page)}" width="100%" height="1100" style="border:1px solid #ccc"></iframe>'))

if RUN_BENCHMARK:
    t_bench = time.time()
    cmd = [sys.executable, "-m", "peoplecount", "benchmark", *VIDEOS, "--models", *MODEL_LIST, *COMMON,
           "--target-fps", str(TARGET_FPS), "--out", "results", "--reports", "reports"]
    if MAX_FRAMES: cmd += ["--max-frames", str(MAX_FRAMES)]
    sh(cmd)
    show_latest_report(t_bench)
''', title="6. Run the benchmark")

md("## Part B: Alerts + evidence clips: every video\nFor each video: counts people, records **every** time the limit is exceeded and cuts clips (3 s before the violation → 3 s after the alert; overlapping alerts merge into one clip).")

code('''
import json
if RUN_ALERTS_AND_CLIPS:
    t_alerts = time.time()
    for v in VIDEOS:
        print("\\n" + "=" * 100 + f"\\n{v}\\n" + "=" * 100)
        cmd = [sys.executable, "-m", "peoplecount", "run", v, "--model", ALERT_MODEL, *COMMON,
               "--clip-pre", str(CLIP_PRE_S), "--clip-post", str(CLIP_POST_S), "--out-dir", "samples"]
        if not SAVE_FULL_ANNOTATED_VIDEO: cmd.append("--no-save")
        sh(cmd)

    rows = []
    for f in sorted(glob.glob("samples/*/*.episodes.json"), key=os.path.getmtime):
        if os.path.getmtime(f) < t_alerts: continue
        d = json.load(open(f))
        for ep in d["episodes"]:
            rows.append({"video": os.path.basename(d["source"]), "model": d["model"], "alert #": ep["episode"],
                         "exceeded at": ep["start_timecode"], "alert raised": ep["alert_timecode"],
                         "over limit (s)": ep["duration_s"], "peak": ep["peak_count"], "limit": ep["threshold"],
                         "clip": os.path.basename(ep["clip"]) if ep["clip"] else None, "_clip_path": ep["clip"]})
    ALERT_TABLE = pd.DataFrame(rows)
    if len(ALERT_TABLE):
        display(ALERT_TABLE.drop(columns="_clip_path"))
    else:
        print("No alert episodes: nobody exceeded the limit of", THRESHOLD, "(try a lower THRESHOLD).")
''', title="7. Run alerts + clips on every video")

code('''
#@markdown Browser-playable previews of the evidence clips (re-encoded to H.264, since the raw clips use a codec browsers can't play).
MAX_PREVIEWS = 6  #@param {type:"integer"}
from IPython.display import Video
if RUN_ALERTS_AND_CLIPS and len(ALERT_TABLE):
    seen = []
    for p in ALERT_TABLE["_clip_path"].dropna().unique()[:MAX_PREVIEWS]:
        out = p.replace(".mp4", "_preview.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", p, "-vf", "scale=640:-2", "-c:v", "libx264",
                            "-crf", "28", "-pix_fmt", "yuv420p", out])
        if r.returncode == 0:
            eps = ALERT_TABLE.loc[ALERT_TABLE["_clip_path"] == p, "alert #"].tolist()
            print(f"{os.path.basename(p)}  (covers alert(s) {eps})")
            display(Video(out, embed=True, width=520))
''', form=True, title="8. Preview the clips")

code('''
#@markdown Optional: export to **TensorRT** (NVIDIA only) and benchmark it against the PyTorch version. Takes a few minutes per model.
if TRY_TENSORRT and GPU:
    TRT_MODELS = ["yolo11n", "yolo11s"]
    t_trt = time.time()
    for m in TRT_MODELS:
        sh([sys.executable, "-m", "peoplecount", "export", m, "--format", "engine", "--half"])
    cmd = [sys.executable, "-m", "peoplecount", "benchmark", *VIDEOS, "--models",
           *[x for m in TRT_MODELS for x in (m, f"{m}.engine")], *COMMON, "--out", "results", "--reports", "reports"]
    if MAX_FRAMES: cmd += ["--max-frames", str(MAX_FRAMES)]
    sh(cmd)
    show_latest_report(t_trt)
else:
    print("Skipped (set TRY_TENSORRT = True in step 4 and use a GPU runtime).")
''', form=True, title="9. (Optional) TensorRT")

code('''
!zip -rq people_counter_outputs.zip samples results reports 2>/dev/null; ls -lh people_counter_outputs.zip
try:
    from google.colab import files
    files.download("people_counter_outputs.zip")
except ImportError:
    pass
''', title="10. Download everything (annotated videos, alert logs, clips, reports)")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(cells):
    c["id"] = f"cell{i:02d}"
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", OUT, f"({OUT.stat().st_size/1e3:.0f} KB, {len(cells)} cells)")
