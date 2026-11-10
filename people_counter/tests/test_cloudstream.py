"""Tests for the cloud-endpoint streaming package (cloudstream/). None of them needs a network, a GPU or a cloud account.

How the real system works (so the tests make sense)
---------------------------------------------------
cloudstream plays a video in a loop, sends JPEG frames in batches to a hosted YOLO endpoint (SageMaker / Vertex AI / any HTTP
server), gets bounding boxes back, and then runs tracking + the normal alert system locally. These tests replace the endpoint with
a FAKE one, so they check everything around it: encoding, looping, batching, ordering, dropping, error handling, alerts.

The fake endpoint (see FakeEndpoint below) and the test video
-------------------------------------------------------------
The test video has 100 frames at 20 fps (5 s). Frame number i is a flat gray image with brightness 2*i. The fake endpoint decodes
each JPEG it receives, reads the brightness back to recover the frame number, and "sees" 5 people while that frame number is in
[20, 60), i.e. from t=1 s to t=3 s of the video, and nobody otherwise. So with an alert limit of 3 people, the expected
violation is a known ~1 s .. ~3 s window.

File layout: shared fixtures first, then tests grouped by area: codec, video source, transports, server, pipeline.
"""
import base64
import csv
import io
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cloudstream import codec
from cloudstream.pipeline import StreamPipeline
from cloudstream.sources import LoopingVideoSource
from cloudstream.transports import SageMakerTransport, Transport, VertexTransport

FPS = 20   # frame rate of the synthetic test video


# ======================================================================================================================
# Shared fixtures and fakes
# ======================================================================================================================
@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A 5-second test video: 100 frames at 20 fps; frame i is a flat gray image of brightness 2*i.

    The brightness encodes the frame number so that FakeEndpoint can recover it from a JPEG. Built once and shared (module scope).
    """
    p = str(tmp_path_factory.mktemp("v") / "t.mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (96, 64))
    for i in range(100):
        w.write(np.full((64, 96, 3), 2 * i, np.uint8))
    w.release()
    return p


class FakeEndpoint(Transport):
    """A stand-in for a hosted YOLO endpoint.

    * Reports 5 people while the frame number is in [20, 60) (t = 1..3 s of the video), and nobody otherwise.
    * Sleeps a random 0-30 ms per call, so when several requests are in flight their replies come back OUT OF ORDER, like a real
      network with several requests running at once. The pipeline must still process frames in order.
    """

    def invoke(self, body: bytes) -> bytes:
        time.sleep(random.uniform(0, 0.03))
        results = []
        for b64 in json.loads(body)["frames"]:
            # Decode the JPEG and recover the frame number from its brightness (brightness = 2 * frame number).
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            idx = round(float(img.mean()) / 2)
            people = 5 if 20 <= idx < 60 else 0
            # Five side-by-side, non-overlapping boxes [x1, y1, x2, y2, confidence], so the tracker can keep them apart.
            results.append({"boxes": [[10 + 15 * k, 10, 20 + 15 * k, 40, 0.9] for k in range(people)]})
        return json.dumps({"results": results, "server_ms": 4.0, "model": "fake"}).encode()


# ======================================================================================================================
# Codec: the wire format between client and server (cloudstream/codec.py)
# ======================================================================================================================
def test_encode_frame_resizes_without_upscaling():
    """Frames are shrunk so the longest side is at most `max_side`, keeping the aspect ratio, and are never enlarged."""
    frame = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
    small, jpg = codec.encode_frame(frame, 640, 80)
    assert max(small.shape[:2]) == 640                                                      # 1280x720 -> 640x360
    assert small.shape[1] / small.shape[0] == pytest.approx(1280 / 720, abs=0.01)           # aspect ratio preserved
    assert isinstance(jpg, bytes) and len(jpg) > 0
    assert codec.encode_frame(small, 1000)[0].shape == small.shape                          # already smaller than 1000: untouched


def test_make_request_builds_a_json_batch():
    """A request is JSON with the frames base64-encoded, plus the optional confidence setting."""
    _, jpg = codec.encode_frame(np.zeros((64, 96, 3), np.uint8))
    body = json.loads(codec.make_request([jpg], conf=0.4))
    assert body["conf"] == 0.4 and len(body["frames"]) == 1


def test_parse_response_reads_both_envelopes_and_errors():
    """The reply is read in the plain format ("results") and Vertex's ("predictions"); an error reply raises."""
    # Two frames: the first has one box [x1, y1, x2, y2, confidence], the second has none.
    reply = {"predictions": [{"boxes": [[1, 2, 3, 4, .5]]}, {"boxes": []}], "server_ms": 3}
    boxes, meta = codec.parse_response(json.dumps(reply).encode())
    assert boxes[0].shape == (1, 5) and boxes[1].shape == (0, 5)       # one (N, 5) array per frame; N=0 is allowed
    assert meta["server_ms"] == 3                                      # everything except the boxes is returned as metadata
    with pytest.raises(RuntimeError):
        codec.parse_response(b'{"error": "boom"}')


# ======================================================================================================================
# Video source: the looping player (cloudstream/sources.py)
# ======================================================================================================================
def test_looping_source_wraps_and_keeps_stream_time(video):
    """Three loops of a 100-frame video = 300 frames. The loop counter goes 0,1,2 and stream time keeps counting up."""
    frames = list(LoopingVideoSource(video, rate=FPS, pace=False, loops=3))     # pace=False: no waiting, as fast as possible
    assert len(frames) == 300 and [f.loop for f in frames[::100]] == [0, 1, 2]  # frames 0, 100, 200 start new loops
    assert frames[150].t == pytest.approx(150 / FPS)                            # time is NOT reset at each loop...
    assert [f.seq for f in frames] == list(range(300))                          # ...and the frame counter continues 0..299


def test_looping_source_duration_and_pacing(video):
    """`duration_s` stops the stream after that much stream time, and `pace=True` plays it in real time."""
    t0 = time.perf_counter()
    # 1.5 s at 100 fps = 150 frames: more than the 100 in the file, so the video wraps once and the last frame is in loop 1.
    frames = list(LoopingVideoSource(video, rate=100, pace=True, duration_s=1.5))
    elapsed = time.perf_counter() - t0
    assert len(frames) == 150 and frames[-1].loop == 1
    assert 1.3 < elapsed < 2.4                                                  # paced: took about 1.5 real seconds


def test_looping_source_target_fps_samples_in_stream_time(video):
    """target_fps picks frames evenly in time while the video keeps its own speed and stream clock."""
    src = LoopingVideoSource(video, rate=FPS, pace=False, loops=2, target_fps=5)    # 20 fps file -> detect 5 times per second
    frames = list(src)
    assert src.rate == 5 and len(frames) == 50                                      # 2 loops x 5 s x 5 fps
    assert [f.seq for f in frames[:3]] == [0, 4, 8]                                 # every 4th source frame (20 / 5 = 4)
    assert frames[1].t == pytest.approx(0.2)                                        # time is still in real video seconds
    assert [f.loop for f in frames[::25]] == [0, 1]                                 # the 26th sample is in the second loop
    assert LoopingVideoSource(video, rate=FPS, target_fps=60).rate == FPS           # can't exceed the video's own 20 fps


# ======================================================================================================================
# Transports: how a request reaches SageMaker / Vertex (cloudstream/transports.py). Tested with stand-in cloud clients.
# ======================================================================================================================
def test_sagemaker_transport_calls_invoke_endpoint():
    """SageMakerTransport must call `invoke_endpoint` with the right endpoint name and JSON content type, and return the body."""
    seen = {}

    class FakeSageMakerRuntime:
        def invoke_endpoint(self, **kw):
            seen.update(kw)                                              # remember how we were called
            return {"Body": io.BytesIO(b'{"results": []}')}              # boto3 returns the reply as a readable stream

    out = SageMakerTransport("ep", client=FakeSageMakerRuntime()).invoke(b"{}")
    assert out == b'{"results": []}'
    assert seen["EndpointName"] == "ep" and seen["ContentType"] == "application/json"


def test_vertex_transport_raw_and_predict_modes():
    """VertexTransport has two modes. 'raw' forwards our JSON unchanged; 'predict' converts it to Vertex's instances format."""

    class FakeVertexEndpoint:
        def raw_predict(self, body, headers):
            return SimpleNamespace(content=b'{"predictions": []}')

        def predict(self, instances, parameters=None):
            self.got = (instances, parameters)                          # remember what Vertex would have been sent
            return SimpleNamespace(predictions=[{"boxes": []}])

    ep = FakeVertexEndpoint()
    # raw mode: the reply bytes come back untouched.
    assert VertexTransport("p", "l", "1", "raw", endpoint=ep).invoke(b"{}") == b'{"predictions": []}'
    # predict mode: {"frames": [...], "conf": ...} becomes instances=[{"image": ...}] plus parameters={"conf": ...},
    # and the reply is wrapped back into JSON bytes.
    out = VertexTransport("p", "l", "1", "predict", endpoint=ep).invoke(json.dumps({"frames": ["AAA"], "conf": 0.3}).encode())
    assert json.loads(out) == {"predictions": [{"boxes": []}]}
    assert ep.got[0] == [{"image": "AAA"}] and ep.got[1] == {"conf": 0.3}


# ======================================================================================================================
# Server: the FastAPI inference app (cloudstream/server/app.py), with a stub model instead of YOLO
# ======================================================================================================================
def test_server_app_with_stub_model():
    """Exercise the real HTTP routes (SageMaker and Vertex flavours, and error handling) with a fake model."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")                                        # skip (not fail) if the optional server packages are missing
    from fastapi.testclient import TestClient
    from cloudstream.server.app import create_app

    def loader():
        """Stands in for loading YOLO: a 'model' that finds the same two boxes in every frame."""
        def predict(frames, conf, imgsz):
            return [np.array([[1, 2, 30, 40, 0.9], [5, 5, 20, 20, 0.5]], np.float32) for _ in frames]
        return predict

    img = codec.encode_frame(np.zeros((64, 96, 3), np.uint8))[1]
    body = json.loads(codec.make_request([img, img]))                   # a batch of two frames
    with TestClient(create_app(loader)) as c:
        # Health checks: SageMaker polls /ping, Vertex polls /health.
        assert c.get("/ping").status_code == 200 and c.get("/health").status_code == 200
        # SageMaker-style call: POST /invocations -> one result per frame, with the boxes and the server timing.
        r = c.post("/invocations", json=body).json()
        assert len(r["results"]) == 2 and r["results"][0]["boxes"][0][:4] == [1, 2, 30, 40] and "server_ms" in r
        # Vertex-style call: POST /predict with {"instances": [...]} gets its answer back under "predictions".
        v = c.post("/predict", json={"instances": [{"image": body["frames"][0]}], "parameters": {"conf": 0.5}}).json()
        assert len(v["predictions"]) == 1
        # Bad input is rejected with HTTP 400 instead of crashing: an undecodable image, and an empty request.
        assert c.post("/invocations", json={"frames": ["not-an-image"]}).status_code == 400
        assert c.post("/invocations", json={}).status_code == 400
        # What the server returns can be read back by the client's own parser (the two sides agree on the format).
        parsed, _ = codec.parse_response(json.dumps(r).encode())
        assert parsed[0].shape == (2, 5)


# ======================================================================================================================
# Pipeline: client side end to end (cloudstream/pipeline.py), against the fake endpoint
# ======================================================================================================================
def run_pipeline(video, tmp_path, loops, **kw):
    """Stream the test video through StreamPipeline + FakeEndpoint, for `loops` passes over the video.

    Alert settings: limit 3 people, alert after 0.5 s over the limit, clear after 0.5 s. `on_backlog="block"` makes sure no
    frame is dropped, so results are deterministic. The random seed makes the fake endpoint's delays repeatable.
    Returns (the pipeline object, its summary statistics). Output files go to `tmp_path`.
    """
    random.seed(1)
    src = LoopingVideoSource(video, rate=FPS, pace=False, loops=loops)
    pipe = StreamPipeline(lambda: FakeEndpoint(), label="fake", threshold=3, hold_s=0.5, clear_s=0.5, batch_size=4,
                          max_inflight=4, on_backlog="block", out_dir=str(tmp_path), stats_every=0, **kw)
    summary = pipe.run(src)
    return pipe, summary


def test_pipeline_alerts_like_local_pipeline_and_stays_in_order(video, tmp_path):
    """Full run: the alert system gives the expected violation, files match the local pipeline's format, frames stay in order."""
    pipe, summary = run_pipeline(video, tmp_path, loops=1)
    # Nothing lost: all 100 frames were processed, none dropped, none failed.
    assert summary["frames_processed"] == 100 and summary["frames_dropped"] == 0 and summary["frame_errors"] == 0

    # The episodes file must have the same fields as the one written by `python -m peoplecount run`
    # (that is what keeps the notebooks working on cloudstream output).
    d = json.loads(Path(pipe.paths.episodes).read_text())
    assert {"source", "model", "threshold", "hold_s", "clear_s", "summary", "counts_csv", "episodes", "clips"} <= set(d)
    assert {"max_people", "avg_people", "distinct_tracks"} <= set(d["summary"])

    # Exactly one violation, matching the fake endpoint's 5 people from t=1 s to t=3 s. The bounds are a little loose because
    # the count is smoothed over several frames, which shifts the detected edges by up to ~0.3 s.
    assert len(d["episodes"]) == 1
    ep = d["episodes"][0]
    assert 1.0 <= ep["start_t"] <= 1.7 and 3.0 <= ep["end_t"] <= 3.8 and ep["peak_count"] == 5
    # The alert fires `hold_s` (0.5 s) after the violation started, and an evidence clip was written.
    assert ep["alert_t"] - ep["start_t"] >= 0.5 and ep["clip"] and Path(ep["clip"]).exists()

    # Replies arrive out of order (random delays), but the per-frame count log must be in frame order and complete.
    rows = list(csv.DictReader(open(pipe.paths.counts)))
    frames = [int(r["frame"]) for r in rows]
    assert frames == sorted(frames) and len(frames) == 100
    # The alert log has exactly one ALERT followed by one CLEARED.
    kinds = [json.loads(line)["kind"] for line in open(pipe.paths.alerts)]
    assert kinds == ["ALERT", "CLEARED"]


def test_pipeline_records_every_loop_and_merges_overlapping_clips(video, tmp_path):
    """A looped video repeats the violation every pass: both must be recorded, and their clips must merge into one."""
    pipe, summary = run_pipeline(video, tmp_path, loops=2)                       # two passes = 10 s of stream time
    d = json.loads(Path(pipe.paths.episodes).read_text())
    assert len(d["episodes"]) == 2                                              # one violation in each pass
    assert d["episodes"][1]["start_t"] > 5.5                                    # the second pass starts at t=5 s (+ ~1 s into it)
    # The two clip windows overlap (the first ends ~4.7 s, the second starts ~3.2 s) -> ONE clip covering both episodes.
    assert len(d["clips"]) == 1 and d["clips"][0]["episodes"] == [1, 2]
    assert d["episodes"][0]["clip"] == d["episodes"][1]["clip"]


def test_pipeline_drops_instead_of_lagging_when_endpoint_is_slow(video, tmp_path):
    """Real-time behaviour: if the endpoint is too slow the pipeline must DROP frames, keeping latency bounded."""

    class Slow(FakeEndpoint):
        def invoke(self, body):
            time.sleep(0.25)                                                    # every request takes an extra 0.25 s
            return super().invoke(body)

    # Frames arrive at 20/s (pace=True), but one request at a time can only carry 2 frames per ~0.25 s = ~8 frames/s.
    src = LoopingVideoSource(video, rate=FPS, pace=True, duration_s=2.0)
    pipe = StreamPipeline(lambda: Slow(), label="slow", batch_size=2, max_inflight=1, on_backlog="drop",
                          out_dir=str(tmp_path), stats_every=0, clips=False)
    s = pipe.run(src)
    # Some frames were dropped, some processed, and the 95th-percentile delay stayed bounded (no growing backlog).
    assert s["frames_dropped"] > 0 and s["frames_processed"] > 0 and s["latency_ms"]["p95"] < 1500


def test_pipeline_survives_endpoint_errors(video, tmp_path):
    """Network errors (throttling, timeouts) must not crash the stream: failed batches are counted and skipped."""

    class Flaky(FakeEndpoint):
        def invoke(self, body):
            if random.random() < 0.3:                                           # ~30% of requests fail
                raise ConnectionError("throttled")
            return super().invoke(body)

    random.seed(3)
    src = LoopingVideoSource(video, rate=FPS, pace=False, loops=1)
    pipe = StreamPipeline(lambda: Flaky(), label="flaky", batch_size=4, max_inflight=2, on_backlog="block", retries=0,
                          out_dir=str(tmp_path), stats_every=0, clips=False)
    s = pipe.run(src)
    # Some frames failed, and every frame is accounted for exactly once: processed OR failed (nothing silently lost).
    assert s["frame_errors"] > 0 and s["frames_processed"] + s["frame_errors"] == 100


def test_pipeline_with_target_fps_keeps_alert_times_in_stream_time(video, tmp_path):
    """Detecting on fewer frames per second must not change WHEN alerts happen: times stay in real video seconds."""
    random.seed(2)
    src = LoopingVideoSource(video, rate=FPS, pace=False, loops=1, target_fps=10)    # 20 fps video -> 10 detections per second
    pipe = StreamPipeline(lambda: FakeEndpoint(), label="fake", threshold=3, hold_s=0.5, clear_s=0.5, batch_size=2,
                          max_inflight=3, on_backlog="block", out_dir=str(tmp_path), stats_every=0)
    s = pipe.run(src)
    assert s["frames_processed"] == 50                                              # half of the 100 frames
    # Same ~1 s .. ~3 s violation as with every frame (bounds are a bit wider: fewer samples = coarser edges).
    ep = json.loads(Path(pipe.paths.episodes).read_text())["episodes"]
    assert len(ep) == 1 and 1.0 <= ep[0]["start_t"] <= 2.2 and 3.0 <= ep[0]["end_t"] <= 4.0
