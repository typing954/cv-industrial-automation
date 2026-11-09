"""Export Ultralytics models to edge runtimes. The result path can be passed straight to `benchmark --models`."""
from __future__ import annotations


def export(model: str, fmt: str, imgsz: int = 640, half: bool = False, int8: bool = False):
    """fmt: onnx | openvino (Intel CPU/iGPU/NPU) | engine (TensorRT, needs NVIDIA GPU) | ncnn | coreml | tflite"""
    from ultralytics import YOLO
    from .models import resolve
    w = resolve(model).weights
    out = YOLO(w).export(format=fmt, imgsz=imgsz, half=half, int8=int8, dynamic=False)
    print("exported:", out)
    return out
