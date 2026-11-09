# People counter: model × hardware comparison

Counts people in a video feed with **object detection + multi-object tracking**, raises an **alert** when the
count exceeds a limit, and benchmarks many Python detectors on your video, on any machine (laptop, edge box, AWS, GCP).

Pipeline: `video → detector (person class) → ByteTrack/BoT-SORT → rolling-median count → AlertManager`.
Every detector gets the *same* tracker and alert logic, so differences come from the model and hardware.

## Models (`python -m peoplecount list-models`)
YOLOv8 n/s/m, YOLO11 n/s/m/l, YOLOv10 n/s, RT-DETR-L (Ultralytics) and SSDLite-MobileNetV3, Faster R-CNN (MobileNetV3 / R50),
RetinaNet-R50 (torchvision). Any Ultralytics weight, ONNX, OpenVINO or TensorRT path can also be passed to `--models`.

## Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
# CPU-only box:   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# NVIDIA box:     pip install torch torchvision
pip install -r requirements.txt
```

## Use
(On a laptop CPU add `--threads 4`; on a hybrid P/E-core Intel it was ~3x faster than the default.)
```bash
# 1. Live/real-time with alert (file, webcam index, or rtsp://...). Overlay + mp4 output optional.
python -m peoplecount run videos/room.mp4 --model yolo11s --threshold 10 --hold 2 --realtime --show --save out.mp4 \
       --alert-log alerts.jsonl --webhook https://hooks.slack.com/...      # webhook optional

# (outputs are saved automatically, see "Where outputs go" below)

# 2. Compare models on YOUR video on THIS machine (writes results/<hardware-tag>__<model>__<video>.json)
python -m peoplecount benchmark videos/room.mp4 --threshold 10                # default shortlist
python -m peoplecount benchmark videos/room.mp4 --models yolo11n yolo11m rtdetr-l fasterrcnn_r50 --stride 2

# (every benchmark run ends with a console table + reports/report_<hardware>_<time>.html/.csv for THAT run)

# 3. Merge ALL results (from every machine) into one report: report.html + report.csv
python -m peoplecount report --target-fps 25                                  # add --gt gt.csv if you have ground truth
```
`gt.csv` format: `frame,count` (sparse is fine, e.g. every 50th frame hand-counted). **Without it, the report's
"MAE" is agreement with the median of all models, not true accuracy**; hand-count ~20 frames to get real numbers.

### Where outputs go
`run` saves everything by default, one folder per input video (`<id>` = `<YYYYmmdd-HHMMSS>_<uuid8>_<model>_<video-name>`):
```
samples/<video-name>/<id>.mp4              annotated video (boxes, IDs, count, alert banner, timecode)
samples/<video-name>/<id>.alerts.jsonl     one JSON line per ALERT / CLEARED event
samples/<video-name>/<id>.episodes.json    summary of EVERY time the limit was exceeded (+ which clip covers it)
samples/<video-name>/<id>_clips/clip001_00h00m12s_alert1.mp4 ...   evidence clips
```
Webcam sources use `samples/camera0/`, RTSP sources a folder named after the stream. Options: `--no-save` (no full video),
`--no-clips`, `--out-dir DIR`, `--save PATH` / `--alert-log PATH`.
`benchmark` writes `results/*.json` (raw data) and `reports/report_<hardware>_<time>.html|csv` (per-run report).

### Sample call from Python
`examples/sample_call.py` shows the building blocks (model -> video source -> counter -> alert rule):
```bash
python examples/sample_call.py videos/crowd_roma_pride_720p.webm --model yolo11n --threshold 10
```
```python
from peoplecount.models import Detector, resolve
from peoplecount.counter import PeopleCounter
from peoplecount.alert import AlertManager, ConsoleSink
from peoplecount.video import VideoSource

src = VideoSource("videos/classroom.mp4", stride=3)          # or 0 (webcam) / "rtsp://..."
counter = PeopleCounter(Detector(resolve("yolo11n"), threads=4), fps=src.fps / 3)
alerts = AlertManager(threshold=10, hold_s=2, sinks=[ConsoleSink()])
for idx, t, frame in src:
    r = counter.step(frame)          # r.smooth = people count; r.boxes, r.ids = tracked people
    alerts.update(r.smooth, t)
```

### Alerts: every violation is recorded
An *episode* is one continuous period with the (smoothed) count above `--threshold` (alert fires at limit+1 or more).
* **Time of violation**: `start_t` / `start_timecode` = the first moment the limit was exceeded (not when the alert was raised).
* **ALERT** is raised once the count stays above the limit for `--hold` seconds (default 2; use `--hold 0` to alert on any breach).
* **CLEARED** after `--clear` seconds (default 3) at/below the limit. Short dips don't split an episode. Episode end = last moment over.
* **Every episode is recorded** (`--cooldown` defaults to 0; set it only if you want to throttle notifications).
* Each event/episode carries: episode number, `start_timecode`, alert time, `end_timecode`, `duration_s`, `peak_count`/`peak_t`,
  wall-clock time (`wall_time_iso`) and the clip file. A video that ends mid-episode is closed with `reason: "stream_end"`.
* Sinks: console, JSONL file, webhook (add your own: a callable taking an `AlertEvent`).

### Evidence clips (overlaps merged)
Each alert produces a clip from `--clip-pre` (3 s) **before the violation started** to `--clip-post` (3 s) **after the alert was
raised**, annotated like the main video. To avoid many near-duplicate clips, alerts whose clip windows overlap (or are
within `--merge-gap` seconds, default 1) are merged into ONE clip that is simply extended; episodes list the clip that covers them.
Example from a real run: three alert episodes in one video -> a single 15.7 s clip.
Footage before the violation comes from a small in-RAM buffer of recent annotated frames (JPEG, ~10-15 MB for 1080p), so no pre-recording to disk.

### Real-time behaviour
`run --realtime` paces files at native FPS and **drops frames when the model is too slow** (reported as `dropped`);
RTSP/webcam sources always use a latest-frame reader so latency never accumulates.
Benchmark metrics: end-to-end FPS, p50/p95/p99 latency, detect vs track ms, **RT factor** (≥1 = keeps up with source),
CPU %, RAM, GPU util/mem/power (NVML), number of alerts, per-frame counts.

## Trial on different hardware
Each run is tagged with CPU/GPU or cloud instance type (auto-detected, override with `--tag`). Copy the JSONs into
one `results/` folder and re-run `report`.
```bash
# AWS (g4dn.xlarge=T4, g5.xlarge=A10G, g6.xlarge=L4, c7i.2xlarge=CPU, c7g.2xlarge=ARM CPU)
export AWS_SG=sg-xxxx; cloud/aws_trial.sh g4dn.xlarge my-key videos/room.mp4 --threshold 10
# GCP (n1-standard-8 + nvidia-tesla-t4, g2-standard-8=L4 pass 'none', c3-standard-8 CPU)
cloud/gcp_trial.sh n1-standard-8 nvidia-tesla-t4 videos/room.mp4 --threshold 10
```
Both scripts create one VM, run the benchmark, download results and **delete the VM on exit** (they cost money; check quota,
region and image names for your account first). Docker: see `Dockerfile`.

CPU thread scaling: `benchmark --device cpu --threads 4` (do several runs with different `--threads` and `--tag cpu-4t`, `cpu-8t`).

## Making it fast on edge hardware
Plain PyTorch on CPU is the slowest path. Export, then benchmark the exported artefact (same command):
```bash
python -m peoplecount export yolo11s --format openvino          # Intel CPU / iGPU  -> yolo11s_openvino_model/
python -m peoplecount export yolo11s --format engine --half     # NVIDIA / Jetson TensorRT (run on the target device!)
python -m peoplecount export yolo11s --format onnx
python -m peoplecount benchmark videos/room.mp4 --models yolo11s.pt yolo11s_openvino_model yolo11s.engine --tag edge-box
```
Other levers: `--imgsz 480/416` (large speed gain, check small/far people), `--stride 2-3` (counting rarely needs full FPS),
smaller model, `--conf` tuning.

## Notes
* Ultralytics weights are AGPL-3.0 (commercial use needs a licence); torchvision models are BSD. Check before deploying.
* Counting caveats: heavy occlusion/crowds under-count; use a higher `--imgsz`, an overhead-friendly fine-tuned model, or a crowd-density model for >30 people.
* `pytest tests` covers alert logic.

## Testing with 100+ people
Freely licensed video with hundreds of people per frame is rare. Good sources: your own footage; the MOT20 dataset
(motchallenge.net, 100–250 people/frame, non-commercial licence, large download); FDST / Mall / UCSD crowd datasets.
Expect COCO-pretrained detectors to under-count heavily at that density, so hand-count a few frames (`--gt gt.csv`) and consider a crowd-density model.
