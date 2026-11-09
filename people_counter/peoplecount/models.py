"""Model registry: every entry is a *person detector*. Tracking is applied uniformly on top
(ByteTrack via Ultralytics) so that the comparison between detectors is fair."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str  # ultralytics | rtdetr | torchvision
    weights: str  # weight file / torchvision constructor name
    imgsz: int = 640
    note: str = ""


REGISTRY: dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("yolov8n", "ultralytics", "yolov8n.pt", note="nano, edge-friendly"),
    ModelSpec("yolov8s", "ultralytics", "yolov8s.pt"),
    ModelSpec("yolov8m", "ultralytics", "yolov8m.pt"),
    ModelSpec("yolo11n", "ultralytics", "yolo11n.pt", note="nano, edge-friendly"),
    ModelSpec("yolo11s", "ultralytics", "yolo11s.pt"),
    ModelSpec("yolo11m", "ultralytics", "yolo11m.pt"),
    ModelSpec("yolo11l", "ultralytics", "yolo11l.pt", note="accuracy reference"),
    ModelSpec("yolov10n", "ultralytics", "yolov10n.pt", note="NMS-free"),
    ModelSpec("yolov10s", "ultralytics", "yolov10s.pt", note="NMS-free"),
    ModelSpec("rtdetr-l", "rtdetr", "rtdetr-l.pt", note="transformer, GPU-oriented"),
    ModelSpec("ssdlite_mobilenet", "torchvision", "ssdlite320_mobilenet_v3_large", 320, "lowest-cost torchvision"),
    ModelSpec("fasterrcnn_mobilenet", "torchvision", "fasterrcnn_mobilenet_v3_large_320_fpn", 320),
    ModelSpec("retinanet_r50", "torchvision", "retinanet_resnet50_fpn_v2", 800, "heavy"),
    ModelSpec("fasterrcnn_r50", "torchvision", "fasterrcnn_resnet50_fpn_v2", 800, "heavy, accurate"),
]}

# Sensible default shortlist for a first run.
DEFAULT_SET = ["yolov8n", "yolo11n", "yolo11s", "yolov10n", "ssdlite_mobilenet", "fasterrcnn_mobilenet", "rtdetr-l"]


def resolve(name: str) -> ModelSpec:
    """Registry name, or an arbitrary Ultralytics-loadable path (.pt/.onnx/.engine/openvino dir)."""
    if name in REGISTRY:
        return REGISTRY[name]
    return ModelSpec(name, "rtdetr" if "rtdetr" in name.lower() else "ultralytics", name,
                     note="custom/exported")


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


class Detector:
    """detect(frame_bgr) -> float32 array (N,5): x1,y1,x2,y2,conf  (persons only)."""

    def __init__(self, spec: ModelSpec, device: str = "auto", conf: float = 0.3,
                 imgsz: int | None = None, half: bool = True, threads: int | None = None):
        self.spec, self.conf = spec, conf
        self.imgsz = imgsz or spec.imgsz
        self.device = self._pick_device(device)
        self.half = half and self.device.startswith("cuda")
        if threads and self.device == "cpu":
            import torch
            torch.set_num_threads(threads)
        self._load()

    @staticmethod
    def _pick_device(device: str) -> str:
        if device != "auto":
            return device
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self):
        s = self.spec
        if s.backend in ("ultralytics", "rtdetr"):
            from ultralytics import RTDETR, YOLO
            self.model = (RTDETR if s.backend == "rtdetr" else YOLO)(s.weights)
            self._fp16 = self.half and s.weights.endswith(".pt")
        elif s.backend == "torchvision":
            import torch
            import torchvision.models.detection as tvd
            self.model = getattr(tvd, s.weights)(weights="DEFAULT").to(self.device).eval()
            self._torch = torch
        else:
            raise ValueError(s.backend)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        b = self.spec.backend
        if b in ("ultralytics", "rtdetr"):
            kw = {"half": True} if self._fp16 else {}
            r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=[0],
                                   device=self.device, verbose=False, **kw)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return np.zeros((0, 5), np.float32)
            return np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]]).astype(np.float32)
        if b == "torchvision":
            torch = self._torch
            import cv2
            h, w = frame.shape[:2]
            scale = self.imgsz / min(h, w)  # torchvision transform resizes again; pre-shrink to save H2D bandwidth
            if scale < 1:
                frame_in = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                frame_in, scale = frame, 1.0
            t = torch.from_numpy(frame_in[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).to(self.device)
            with torch.inference_mode():
                out = self.model([t])[0]
            keep = (out["labels"] == 1) & (out["scores"] >= self.conf)  # COCO person = 1 in torchvision
            boxes = out["boxes"][keep].cpu().numpy() / scale
            scores = out["scores"][keep].cpu().numpy()[:, None]
            return np.hstack([boxes, scores]).astype(np.float32) if len(scores) else np.zeros((0, 5), np.float32)
        raise ValueError(b)

    def warmup(self, frame: np.ndarray, n: int = 5):
        for _ in range(n):
            self.detect(frame)
