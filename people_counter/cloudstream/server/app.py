"""YOLO inference server for SageMaker and Vertex AI custom containers (also runs locally).

Standalone on purpose: this single file is all the container needs.
  SageMaker  : GET /ping, POST /invocations on port 8080 (started with `docker run <image> serve`)
  Vertex AI  : health/predict routes + port come from AIP_HEALTH_ROUTE / AIP_PREDICT_ROUTE / AIP_HTTP_PORT
  Both routes are always registered (/ping, /health, /invocations, /predict), so any of them works.

Protocol (JSON), see cloudstream/codec.py:
  {"frames": [b64 jpeg, ...], "conf": 0.3, "imgsz": 640}   or Vertex {"instances": [{"image": b64}], "parameters": {...}}
  -> {"results": [{"boxes": [[x1,y1,x2,y2,conf], ...]}, ...], "server_ms": .., "model": ..}   (Vertex: "predictions")

Config via env: MODEL (default yolo11s.pt), CONF (0.3), IMGSZ (640), CLASSES ("0" = person), DEVICE (auto), HALF (1).
Custom weights: put a *.pt in /opt/ml/model (SageMaker model artifact) and they are used automatically.
"""
from __future__ import annotations

import base64
import glob
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool


def load_ultralytics_predict():
    """-> predict(frames: list[np.ndarray BGR], conf, imgsz) -> list[(N,5) float32 arrays]"""
    import torch
    from ultralytics import YOLO
    weights = os.environ.get("MODEL", "yolo11s.pt")
    custom = sorted(glob.glob("/opt/ml/model/*.pt"))
    if custom:
        weights = custom[0]
    device = os.environ.get("DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    half = device.startswith("cuda") and os.environ.get("HALF", "1") == "1"
    classes = [int(c) for c in os.environ.get("CLASSES", "0").split(",") if c.strip() != ""]
    model, lock = YOLO(weights), threading.Lock()
    kw = {"half": True} if half else {}

    def predict(frames, conf, imgsz):
        with lock:                                   # one GPU call at a time; requests queue up in the thread pool
            res = model.predict(frames, imgsz=imgsz, conf=conf, classes=classes, device=device, verbose=False, **kw)
        out = []
        for r in res:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(np.zeros((0, 5), np.float32))
            else:
                out.append(np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]]).astype(np.float32))
        return out

    predict.info = {"model": os.path.basename(weights), "device": device, "half": half, "classes": classes}
    return predict


def create_app(loader=load_ultralytics_predict) -> FastAPI:
    state: dict = {"requests": 0, "frames": 0}
    default_conf = float(os.environ.get("CONF", "0.3"))
    default_imgsz = int(os.environ.get("IMGSZ", "640"))

    @asynccontextmanager
    async def lifespan(app):
        state["predict"] = loader()                  # model is loaded BEFORE the port starts answering /ping
        try:
            state["predict"]([np.zeros((default_imgsz, default_imgsz, 3), np.uint8)], default_conf, default_imgsz)  # warm-up
        except Exception as e:  # noqa: BLE001
            print("warm-up failed:", e, flush=True)
        yield

    app = FastAPI(lifespan=lifespan)

    def handle(body: dict) -> dict:
        vertex = "instances" in body
        raw = body["instances"] if vertex else body.get("frames")
        if not raw:
            raise ValueError("no frames in request")
        params = body.get("parameters") or {}
        conf = float(body.get("conf", params.get("conf", default_conf)))
        imgsz = int(body.get("imgsz", params.get("imgsz", default_imgsz)))
        frames = []
        for it in raw:
            b64 = it["image"] if isinstance(it, dict) else it
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("could not decode a frame as an image")
            frames.append(img)
        t0 = time.perf_counter()
        boxes = state["predict"](frames, conf, imgsz)
        ms = (time.perf_counter() - t0) * 1e3
        state["requests"] += 1
        state["frames"] += len(frames)
        results = [{"boxes": np.round(b, 2).tolist()} for b in boxes]
        meta = {"server_ms": round(ms, 2), "model": state["predict"].info["model"] if hasattr(state["predict"], "info") else "stub"}
        return {"predictions" if vertex else "results": results, **meta}

    async def invoke(request: Request):
        try:
            body = await request.json()
            return JSONResponse(await run_in_threadpool(handle, body))
        except (ValueError, KeyError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def health():
        return JSONResponse({"status": "ok"})

    async def info():
        return JSONResponse({**getattr(state.get("predict"), "info", {}), "requests": state["requests"], "frames": state["frames"]})

    health_routes = {"/ping", "/health", os.environ.get("AIP_HEALTH_ROUTE", "/health")}
    predict_routes = {"/invocations", "/predict", os.environ.get("AIP_PREDICT_ROUTE", "/predict")}
    for r in health_routes:
        app.add_api_route(r, health, methods=["GET"])
    for r in predict_routes:
        app.add_api_route(r, invoke, methods=["POST"])
    app.add_api_route("/info", info, methods=["GET"])
    return app


def main():
    import uvicorn
    port = int(os.environ.get("AIP_HTTP_PORT") or os.environ.get("SAGEMAKER_BIND_TO_PORT") or os.environ.get("PORT") or 8080)
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_level="info")   # extra argv ("serve") is ignored on purpose


if __name__ == "__main__":
    main()
