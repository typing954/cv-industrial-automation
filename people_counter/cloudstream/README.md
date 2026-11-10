# cloudstream: stream video to a hosted YOLO endpoint (Amazon SageMaker / Vertex AI)

**New and separate** from the local pipeline in `peoplecount/`. The local code is unchanged; this package only *imports* from it,
and `peoplecount` knows nothing about `cloudstream`.

| | `peoplecount/` (existing) | `cloudstream/` (new) |
|---|---|---|
| Where the model runs | in the same process (your CPU/GPU) | on a SageMaker / Vertex AI endpoint (or any HTTP server) |
| Input | video file, webcam, RTSP | a video **played in a loop** (like a live camera), webcam, RTSP |
| Alerts, episodes, clips, files | ✔ | **identical**: it reuses `AlertManager` and `ClipRecorder` and writes the same files |

## How it works
```
video (looped) ──► resize + JPEG ──► micro-batches of frames ──► N requests in flight ──► endpoint (YOLO, GPU, FP16)
                                                                                              │ boxes
        alerts / episodes / clips / counts  ◄── tracker (ByteTrack) ◄── results, re-ordered ◄──┘
```
* **Endpoints are request/response, not video streams.** "High FPS" comes from three things this client does for you:
  (1) **batching** several frames per request, (2) **pipelining**: several requests in flight at once, so network round-trip time
  stops limiting throughput, (3) **dropping frames instead of lagging** when the endpoint can't keep up (`--on-backlog drop`, the
  default for paced streams), so latency stays bounded.
* **The endpoint is stateless** (detection only). **Tracking, smoothing and alerts run in the client**, in strict frame order, so
  you can scale the endpoint horizontally (more instances) without breaking tracking.
* **Looping:** when the file ends it restarts. Stream time keeps counting up across loops (so alert times keep increasing and an
  episode can span the wrap), and the tracker is reset at each wrap because the scene jumps.
* **Same alert system:** a violation = count above `--threshold` for `--hold` s (default 2); cleared after `--clear` s (default 3);
  **every** episode is recorded with the time it was first exceeded, alert time, end, duration and peak; clips run from `--clip-pre`
  (3 s) before the violation to `--clip-post` (3 s) after the alert, and overlapping alerts merge into one clip. Outputs go to
  `samples/<video>/<run-id>.{alerts.jsonl,episodes.json,counts.csv,stats.json}` + `_clips/` (+ `.mp4` with `--save-video`).
  The run id uses the endpoint label where the local pipeline uses the model name. The notebooks work on these files unchanged.

## Quick start
```bash
pip install -r requirements.txt -r cloudstream/requirements.txt

# 0) Try it locally first (no cloud): run the exact server that goes into the container, then stream to it
python -m cloudstream serve &                                  # MODEL=yolo11s.pt by default; serves on :8080
python -m cloudstream stream videos/classroom.mp4 --endpoint-type http --url http://localhost:8080 \
       --fps 15 --duration 60 --threshold 10

# 1a) Amazon SageMaker
cloudstream/deploy/build_push_ecr.sh us-east-1 yolo11s.pt                       # prints IMAGE_URI
python cloudstream/deploy/sagemaker_deploy.py deploy --name people-yolo --image <IMAGE_URI> --role <ROLE_ARN> \
       --instance-type ml.g4dn.xlarge --region us-east-1
python -m cloudstream stream videos/classroom.mp4 --endpoint-type sagemaker --endpoint-name people-yolo --region us-east-1 \
       --fps 30 --batch-size 8 --max-inflight 6 --duration 120 --threshold 10
python cloudstream/deploy/sagemaker_deploy.py delete --name people-yolo                # STOP THE BILLING

# 1b) Vertex AI
cloudstream/deploy/build_push_gar.sh <project> us-central1 yolo11s.pt                # prints IMAGE_URI
python cloudstream/deploy/vertex_deploy.py deploy --project <project> --image <IMAGE_URI> \
       --machine-type n1-standard-8 --accelerator NVIDIA_TESLA_T4
python -m cloudstream stream videos/classroom.mp4 --endpoint-type vertex --project <project> --endpoint-id <ID> --fps 30
python cloudstream/deploy/vertex_deploy.py delete --project <project> --endpoint-id <ID>
```
**Endpoints bill per hour while they exist.** Always delete them when you finish testing.

## Find the maximum FPS an endpoint can sustain
```bash
python -m cloudstream bench videos/classroom.mp4 --endpoint-type sagemaker --endpoint-name people-yolo \
       --batch-sizes 1 4 8 16 --concurrency 1 2 4 8 --seconds 10
```
It prints frames/s, round-trip and server time for every batch × concurrency pair and the best one. Use that setting
(`--batch-size`, `--max-inflight`) for `stream`. Then: cameras per endpoint instance ≈ sustained FPS ÷ FPS you send per camera.

## Tuning knobs
| Option | Effect |
|---|---|
| `--target-fps N` | **detect on N frames per second** of video (default: every frame). Frames are picked evenly in time and the video still plays in real time, so a 30 fps file with `--target-fps 5` sends 5 frames/s |
| `--fps N` | playback speed of the video. Above the file's own fps it plays faster, a way to simulate a high-FPS stream. To choose the detection rate use `--target-fps` |
| `--max-speed` | no pacing, push as fast as the endpoint accepts (throughput test; blocks instead of dropping) |
| `--duration S` / `--loops N` | how long to keep looping (`--duration 0` = until Ctrl-C) |
| `--batch-size`, `--max-inflight`, `--max-wait-ms` | frames per request, concurrent requests, how long a partial batch may wait |
| `--send-size 640`, `--jpeg-quality 80` | frame size/quality sent: smaller = faster and cheaper but can miss small or distant people |
| `--on-backlog drop\|block` | drop frames when the endpoint is saturated (real-time) or apply back-pressure |
| alert/clip options | same as `python -m peoplecount run` (`--threshold --hold --clear --clip-pre --clip-post --merge-gap --webhook …`) |

Live stats every 5 s: input fps, processed fps, dropped, errors, end-to-end latency (p50/p95), endpoint round-trip and server ms/frame.
Ctrl-C and SIGTERM (Docker/systemd stop) finish in-flight requests and save everything (episodes, counts, clips).

## Layout
```
cloudstream/
  server/   app.py (FastAPI), Dockerfile, requirements.txt   the container; one file works for SageMaker AND Vertex
  deploy/   build_push_ecr.sh, build_push_gar.sh, sagemaker_deploy.py, vertex_deploy.py
  sources.py transports.py codec.py pipeline.py bench.py cli.py   the streaming client
tests/test_cloudstream.py   offline tests (fake endpoint, out-of-order replies, loops, drops, errors)
```
Protocol (JSON): request `{"frames": [<base64 JPEG>…], "conf": 0.3, "imgsz": 640}` (or Vertex `{"instances": [{"image": …}]}`);
response `{"results": [{"boxes": [[x1,y1,x2,y2,conf], …]}, …], "server_ms": …}` (Vertex: `"predictions"`). Boxes are in the pixel
coordinates of the frame that was sent. The server detects **people only** by default (`CLASSES=0`, like the local pipeline).

## Bandwidth and cost sizing
A 640 px JPEG is about 55 KB. One camera at 30 fps ≈ 1.7 MB/s (about 13 Mbit/s upstream). 50 cameras at 5 fps each = 250 frames/s
≈ 14 MB/s (about 110 Mbit/s). At 10 fps per camera and a YOLO-small-class model, expect roughly 2 T4-class GPUs for 50 cameras
(see the sizing discussion in the main README). Endpoints cost per instance-hour; a looped test video needs no camera network.

## Verified vs not verified
* **Tested here (CPU, no cloud account):** the real inference server + HTTP transport end to end with a looping video (4 violations
  across loops, overlapping alerts merged into one clip, bounded latency with frame drops under load, clean shutdown on SIGTERM,
  throughput benchmark); and 10 offline unit tests including fake SageMaker/Vertex clients and out-of-order batch completion.
* **NOT tested:** the Docker build, ECR/Artifact Registry push, the SageMaker and Vertex deploy scripts, and real SageMaker/Vertex
  invocations. They follow the documented APIs but need your credentials/quota, so expect to fix small things. GPU (FP16) and
  throughput numbers on a T4 are estimates until you run `bench` on a real endpoint.
* **Limits to check:** request payload size limits (Vertex online prediction is documented around 1.5 MB per request; SageMaker real-time
  is larger) and per-endpoint request timeouts. The defaults (4 frames × ~55 KB) are far below both.
* **Not implemented:** cross-request dynamic batching on the server, TensorRT in the container, SageMaker's bidirectional streaming
  API, autoscaling configuration, one-process-many-cameras, stream reconnect for RTSP.
