"""Generates notebooks/people_counter_colab_simple.ipynb: detect people, show when the count goes over the limit.
Run:  python notebooks/build_colab_simple_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("people_counter_colab_simple.ipynb")
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
# People over the limit: where in the video?

Detects **people** in your videos (it only detects people) and shows **the parts of each video where more people are in view than your limit**:
a chart of people over time, the exact time ranges, a snapshot of the busiest moment with the detected people boxed, and a short clip of each.

1. `Runtime ▸ Change runtime type ▸ GPU`
2. Put your videos in `videos/` (drag them into the file browser) and set the limit in **step 2**
3. `Runtime ▸ Run all`
""")

code(r'''
!pip -q install ultralytics lap nvidia-ml-py "opencv-python<5"
import torch, pathlib
GPU = torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0) if GPU else "none ⚠️  (Runtime ▸ Change runtime type ▸ GPU; it will still run, just slowly)")

FILES = ''' + json.dumps(files, ensure_ascii=False, indent=0) + r'''
for name, src in FILES.items():
    p = pathlib.Path(name); p.parent.mkdir(exist_ok=True); p.write_text(src)
''', form=True, title="1. Setup (run once)")

code(r'''
VIDEO_PATH = "videos"        #@param {type:"string"}
THRESHOLD = 10               #@param {type:"integer"}
MODEL = "yolo11s"            #@param ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolov8s", "rtdetr-l"] {allow-input: true}
MIN_SECONDS = 2.0            #@param {type:"number"}
TARGET_FPS = 5               #@param {type:"number"}
#@markdown **VIDEO_PATH**: one video file or a folder of videos. **THRESHOLD**: flag moments with *more than* this many people.
#@markdown **MIN_SECONDS**: it must stay over the limit this long to count (ignores one-second blips).
#@markdown **TARGET_FPS**: how many frames per second to run detection on (0 = every frame). 5 is plenty for counting people, and much faster than analysing every frame.
''', form=True, title="2. Settings")

code(r'''
import os, sys, glob, json, time, shutil, subprocess, traceback
import cv2, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Markdown, Video, display

DEVICE = "cuda:0" if GPU else "cpu"
EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".ogv", ".m4v", ".mpg", ".mpeg", ".wmv")

def find_videos():
    if os.path.isfile(VIDEO_PATH):
        return [VIDEO_PATH]
    return sorted(p for p in glob.glob(os.path.join(VIDEO_PATH, "**", "*"), recursive=True) if p.lower().endswith(EXTS))

VIDEOS = find_videos()
if not VIDEOS:
    try:
        from google.colab import files
        os.makedirs(VIDEO_PATH, exist_ok=True)
        print(f"No videos in {VIDEO_PATH}/. Choose files to upload:")
        for name in files.upload():
            shutil.move(name, os.path.join(VIDEO_PATH, name))
        VIDEOS = find_videos()
    except ImportError:
        pass
assert VIDEOS, f"No videos found at {os.path.abspath(VIDEO_PATH)}"
print(f"{len(VIDEOS)} video(s):", *[os.path.basename(v) for v in VIDEOS], sep="\n  ")
''', title="3. Find the videos")

code(r'''
def analyse(video):
    """Detect + track people, find every stretch over the limit, cut a clip around each (3 s before / 3 s after)."""
    t0 = time.time()
    cmd = [sys.executable, "-m", "peoplecount", "run", video, "--model", MODEL, "--device", DEVICE,
           "--threshold", str(THRESHOLD), "--hold", str(MIN_SECONDS), *(["--target-fps", str(TARGET_FPS)] if TARGET_FPS else []),
           "--no-save", "--out-dir", "samples"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if line.startswith("  frame"): print("  ", line.strip())        # progress only
    p.wait()
    found = [f for f in glob.glob(f"samples/{Path(video).stem}/*.episodes.json") if os.path.getmtime(f) >= t0]
    return json.load(open(max(found, key=os.path.getmtime))) if found else None

def plot_people(d):
    df = pd.read_csv(d["counts_csv"])
    fig, ax = plt.subplots(figsize=(11, 3.3))
    ax.plot(df["t"], df["people"], lw=1.3, color="#2a6fdb", label="people in view")
    ax.axhline(d["threshold"], color="red", ls="--", lw=1.2, label=f"limit ({d['threshold']})")
    for i, ep in enumerate(d["episodes"]):
        ax.axvspan(ep["start_t"], ep["end_t"], color="red", alpha=0.18, label="over the limit" if i == 0 else None)
    ax.set_xlabel("time in video (s)"); ax.set_ylabel("people"); ax.set_ylim(bottom=0); ax.grid(alpha=.25)
    ax.legend(loc="upper right"); plt.tight_layout(); plt.show()

def snapshot(clip_path, clip_start, t_peak):
    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(t_peak - clip_start, 0) * 1000)
    ok, frame = cap.read(); cap.release()
    return frame if ok else None

def playable(path):
    out = path.replace(".mp4", "_preview.mp4")            # browsers can't play the raw clip codec -> H.264
    ok = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vf", "scale=640:-2", "-c:v", "libx264",
                         "-crf", "28", "-pix_fmt", "yuv420p", out]).returncode == 0
    return out if ok else None

def show_video_results(video, d):
    s, eps = d["summary"], d["episodes"]
    display(Markdown(f"**People detected:** up to **{s['max_people']}** at once (average {s['avg_people']}); "
                     f"about {s['distinct_tracks']} different people tracked over {s['duration_s']:.0f} s."))
    plot_people(d)
    if not eps:
        display(Markdown(f"✅ **Never more than {d['threshold']} people for {d['hold_s']:g}+ seconds.**"))
        return
    over = sum(e["duration_s"] for e in eps)
    display(Markdown(f"### 🚨 {len(eps)} time range(s) with more than {d['threshold']} people (total {over:.0f} s)"))
    display(pd.DataFrame([{"#": e["episode"], "from": e["start_timecode"][:-2], "to": e["end_timecode"][:-2],
                           "lasted (s)": round(e["duration_s"], 1), "peak people": e["peak_count"],
                           "peak at": e["start_timecode"][:-2] if e["peak_t"] == e["start_t"] else f"{e['peak_t']:.1f} s"}
                          for e in eps]).set_index("#"))
    clips = {c["path"]: c for c in d["clips"]}
    shown = set()
    for e in eps:
        c = clips.get(e["clip"])
        if not c: continue
        frame = snapshot(c["path"], c["start_t"], e["peak_t"])
        if frame is not None:
            plt.figure(figsize=(8, 4.5)); plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); plt.axis("off")
            plt.title(f"#{e['episode']}: {e['peak_count']} people at the busiest moment ({e['peak_t']:.1f} s)"); plt.show()
        if c["path"] not in shown:
            shown.add(c["path"])
            covers = [x["episode"] for x in eps if clips.get(x["clip"]) is c]
            pv = playable(c["path"])
            if pv:
                display(Markdown(f"🎞 Clip for range(s) {', '.join('#%d' % i for i in covers)} "
                                 f"(from 3 s before to 3 s after the alert)"))
                display(Video(pv, embed=True, width=560))

RESULTS = {}
for k, v in enumerate(VIDEOS, 1):
    c = cv2.VideoCapture(v); fps = c.get(5) or 0
    display(Markdown(f"---\n# 🎬 {k}/{len(VIDEOS)}: `{os.path.basename(v)}`  \n"
                     f"{int(c.get(3))}×{int(c.get(4))} · {fps:.0f} fps · {c.get(7)/fps if fps else 0:.0f} s"))
    try:
        d = analyse(v)
        if d is None:
            raise RuntimeError("analysis produced no output")
        RESULTS[os.path.basename(v)] = d
        show_video_results(v, d)
    except Exception:
        display(Markdown("⚠️ **This video failed** (continuing with the next one):"))
        traceback.print_exc()
''', title="4. Analyse every video")

code(r'''
rows = [{"video": name, "peak people": d["summary"]["max_people"], "ranges over limit": len(d["episodes"]),
         "total time over (s)": round(sum(e["duration_s"] for e in d["episodes"]), 1),
         "first time over": d["episodes"][0]["start_timecode"][:-2] if d["episodes"] else "-"}
        for name, d in RESULTS.items()]
display(Markdown(f"# 📊 Summary (limit: more than {THRESHOLD} people)"))
display(pd.DataFrame(rows).set_index("video"))
''', title="5. Summary of all videos")

code(r'''
!zip -rq people_over_limit.zip samples 2>/dev/null; ls -lh people_over_limit.zip
try:
    from google.colab import files
    files.download("people_over_limit.zip")
except ImportError:
    pass
''', title="6. Download the clips, counts and summaries")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(cells):
    c["id"] = f"cell{i:02d}"
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", OUT, f"({OUT.stat().st_size/1e3:.0f} KB, {len(cells)} cells)")
