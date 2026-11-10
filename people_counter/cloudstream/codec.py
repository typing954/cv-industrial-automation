"""Wire format shared by client and server (see README "Protocol").

request  (JSON): {"frames": ["<base64 jpeg>", ...], "conf": 0.3, "imgsz": 640}
                 or Vertex style {"instances": [{"image": "<b64>"}...], "parameters": {...}}
response (JSON): {"results": [{"boxes": [[x1,y1,x2,y2,conf], ...]}, ...], "server_ms": 12.3, "model": "yolo11s.pt"}
                 or Vertex style {"predictions": [...same per-frame objects...], ...}
Boxes are in the pixel coordinates of the frame that was sent (the client sends the already-resized frame).
"""
from __future__ import annotations

import base64
import json
from typing import Optional

import cv2
import numpy as np


def encode_frame(frame: np.ndarray, max_side: int = 640, quality: int = 80):
    """Resize so the longest side <= max_side (never upscales) and JPEG-encode. Returns (sent_frame, jpeg_bytes)."""
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return frame, buf.tobytes()


def make_request(jpegs: list[bytes], conf: Optional[float] = None, imgsz: Optional[int] = None) -> bytes:
    body: dict = {"frames": [base64.b64encode(j).decode("ascii") for j in jpegs]}
    if conf is not None:
        body["conf"] = conf
    if imgsz is not None:
        body["imgsz"] = imgsz
    return json.dumps(body, separators=(",", ":")).encode()


def parse_response(raw: bytes) -> tuple[list[np.ndarray], dict]:
    """-> (per-frame (N,5) float32 arrays, meta dict with server_ms / model)"""
    d = json.loads(raw)
    if "error" in d and "results" not in d and "predictions" not in d:
        raise RuntimeError(f"endpoint error: {d['error']}")
    items = d.get("results", d.get("predictions"))
    if items is None:
        raise RuntimeError(f"unexpected response: {str(d)[:200]}")
    out = []
    for it in items:
        b = np.asarray(it["boxes"], dtype=np.float32).reshape(-1, 5)
        out.append(b)
    return out, {k: v for k, v in d.items() if k not in ("results", "predictions")}
