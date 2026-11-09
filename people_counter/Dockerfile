# On-prem edge box with NVIDIA GPU:  docker build -t peoplecount . && docker run --gpus all -v $PWD/videos:/videos -v $PWD/results:/app/results peoplecount benchmark /videos/room.mp4
# CPU-only: build with --build-arg BASE=python:3.11-slim
ARG BASE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
# (headless servers: pip install opencv-python-headless<5 after, and uninstall opencv-python)
FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY peoplecount peoplecount
ENTRYPOINT ["python","-m","peoplecount"]
