import argparse
import json

from .models import DEFAULT_SET, REGISTRY


def main(argv=None):
    p = argparse.ArgumentParser(prog="peoplecount", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--device", default="auto", help="auto|cpu|cuda:0|mps")
        s.add_argument("--conf", type=float, default=0.3)
        s.add_argument("--threads", type=int, help="CPU threads (try 4; on hybrid P/E-core laptops fewer is often faster)")
        s.add_argument("--imgsz", type=int, default=None, help="override model input size")
        s.add_argument("--threshold", type=int, default=10, help="alert when people count exceeds this")
        s.add_argument("--hold", type=float, default=2.0, help="seconds over threshold before ALERT")
        s.add_argument("--clear", type=float, default=3.0, help="seconds under threshold before CLEARED")
        s.add_argument("--cooldown", type=float, default=0.0, help="min seconds between alerts (0 = every episode is alerted)")
        s.add_argument("--target-fps", type=float, default=None, help="run detection at this many frames per second "
                       "(default: every frame). Frames are picked evenly in time; can't exceed the video's own FPS")
        s.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])

    s = sub.add_parser("list-models"); s.set_defaults(f="list")
    s = sub.add_parser("hardware"); s.set_defaults(f="hw")

    s = sub.add_parser("run", help="real-time count + alert on a file/webcam/RTSP")
    s.add_argument("source"); s.add_argument("--model", default="yolo11n"); common(s)
    s.add_argument("--realtime", action="store_true", help="pace a file at its native FPS and drop frames when slow")
    s.add_argument("--show", action="store_true")
    s.add_argument("--out-dir", default="samples", help="outputs go to <out-dir>/<video-name>/<time>_<uuid>_<model>_<video-name>.*")
    s.add_argument("--save", help="override the annotated-mp4 path")
    s.add_argument("--no-clips", action="store_true", help="do not cut evidence clips")
    s.add_argument("--clip-pre", type=float, default=3.0, help="seconds of footage kept BEFORE the violation started")
    s.add_argument("--clip-post", type=float, default=3.0, help="seconds kept AFTER the alert was raised")
    s.add_argument("--merge-gap", type=float, default=1.0, help="alerts whose clips are closer than this are merged into one clip")
    s.add_argument("--no-save", action="store_true", help="do not write a video")
    s.add_argument("--alert-log", help="override the alert JSONL path"); s.add_argument("--webhook"); s.set_defaults(f="run")

    s = sub.add_parser("benchmark", help="compare models on your video(s) on THIS machine")
    s.add_argument("video", nargs="+"); s.add_argument("--models", nargs="*", help=f"default: {' '.join(DEFAULT_SET)}")
    common(s)
    s.add_argument("--out", default="results"); s.add_argument("--tag"); s.add_argument("--max-frames", type=int)
    s.add_argument("--warmup", type=int, default=10)
    s.add_argument("--no-track", action="store_true"); s.add_argument("--no-cloud-detect", action="store_true")
    s.add_argument("--reports", default="reports", help="folder for the per-run HTML/CSV report")
    s.add_argument("--no-report", action="store_true"); s.add_argument("--gt", help="ground-truth CSV (frame,count)")
    s.set_defaults(f="bench")

    s = sub.add_parser("report", help="HTML/CSV comparison across everything in results/")
    s.add_argument("--results", default="results"); s.add_argument("--out", default="report.html")
    s.add_argument("--gt", help="CSV with columns frame,count (optional ground truth)")
    s.add_argument("--target-fps", type=float, default=25.0, help="real-time bar: FPS a model must sustain to pass"); s.set_defaults(f="report")

    s = sub.add_parser("export", help="export to onnx/openvino/engine(TensorRT)/ncnn...")
    s.add_argument("model"); s.add_argument("--format", required=True); s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--half", action="store_true"); s.add_argument("--int8", action="store_true"); s.set_defaults(f="export")

    a = p.parse_args(argv)
    if a.f == "list":
        for m in REGISTRY.values():
            print(f"{m.name:22s} {m.backend:12s} {m.weights:42s} {m.note}")
    elif a.f == "hw":
        from . import hardware
        print(json.dumps(hardware.collect(), indent=1))
    elif a.f == "run":
        from .live import run; run(a)
    elif a.f == "bench":
        from .benchmark import run; run(a)
    elif a.f == "report":
        from .report import build; print("wrote", build(a.results, a.out, a.gt, a.target_fps))
    elif a.f == "export":
        from .export import export; export(a.model, a.format, a.imgsz, a.half, a.int8)
