# Inference container for SageMaker and Vertex AI.  Build context: cloudstream/server/
#   docker build -t people-yolo-endpoint cloudstream/server
#   docker run --gpus all -p 8080:8080 people-yolo-endpoint          # local test; then: python -m cloudstream stream ... --endpoint-type http
# Pick the model at build time:  --build-arg MODEL=yolo11m.pt   (weights are baked in, so start-up needs no internet)
ARG BASE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ARG MODEL=yolo11s.pt
ENV MODEL=${MODEL} CONF=0.3 IMGSZ=640
RUN python -c "import os; from ultralytics import YOLO; YOLO(os.environ['MODEL'])"
COPY app.py .
EXPOSE 8080
# SageMaker starts the container as `docker run <image> serve`; the entrypoint ignores that argument.
ENTRYPOINT ["python", "/app/app.py"]
