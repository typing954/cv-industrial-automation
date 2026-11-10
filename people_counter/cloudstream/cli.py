import argparse

from .transports import make_factory


def _endpoint_args(s):
    g = s.add_argument_group("endpoint")
    g.add_argument("--endpoint-type", choices=["http", "sagemaker", "vertex"], required=True)
    g.add_argument("--url", default="http://localhost:8080", help="[http] base URL of the inference server")
    g.add_argument("--header", action="append", help="[http] extra header 'Name: value' (e.g. 'Authorization: Bearer …'); repeatable")
    g.add_argument("--endpoint-name", help="[sagemaker] endpoint name")
    g.add_argument("--region", help="[sagemaker] AWS region")
    g.add_argument("--project", help="[vertex] GCP project id")
    g.add_argument("--location", default="us-central1", help="[vertex] region")
    g.add_argument("--endpoint-id", help="[vertex] endpoint id")
    g.add_argument("--vertex-mode", choices=["raw", "predict"], default="raw")
    g.add_argument("--timeout", type=float, default=30.0)
    s.add_argument("--conf", type=float, default=0.3)
    s.add_argument("--send-size", type=int, default=640, help="resize frames so the longest side is this many px before sending")
    s.add_argument("--jpeg-quality", type=int, default=80)


def main(argv=None):
    p = argparse.ArgumentParser(prog="cloudstream", description="Stream video to a hosted YOLO endpoint.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stream", help="loop a video (or read a camera) through the endpoint with tracking + alerts + clips")
    s.add_argument("source", help="video file (played in a loop), webcam index, or rtsp:// URL")
    _endpoint_args(s)
    g = s.add_argument_group("stream")
    g.add_argument("--target-fps", type=float, help="run detection at this many frames per second (default: every frame). Picked evenly in "
                   "time, so a 30 FPS video with --target-fps 5 sends 5 frames/s and still plays in real time")
    g.add_argument("--fps", type=float, help="playback speed of the video in frames/s (default: its own fps). Higher = faster/high-FPS stream; "
                   "for choosing the detection rate use --target-fps")
    g.add_argument("--max-speed", action="store_true", help="don't pace: push as fast as the endpoint accepts (throughput test)")
    g.add_argument("--duration", type=float, default=30.0, help="stop after this many seconds of stream time (0 = until Ctrl-C)")
    g.add_argument("--loops", type=int, default=0, help="stop after this many passes over the video (0 = no limit)")
    g.add_argument("--batch-size", type=int, default=4, help="frames per request")
    g.add_argument("--max-inflight", type=int, default=4, help="concurrent requests in flight")
    g.add_argument("--max-wait-ms", type=float, default=50.0, help="send a partial batch if its oldest frame waited this long")
    g.add_argument("--on-backlog", choices=["drop", "block"], help="when the endpoint can't keep up: drop frames (real-time) or block. "
                   "Default: drop when paced, block with --max-speed")
    g.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    a = s.add_argument_group("alerts (identical to `python -m peoplecount run`)")
    a.add_argument("--threshold", type=int, default=10)
    a.add_argument("--hold", type=float, default=2.0)
    a.add_argument("--clear", type=float, default=3.0)
    a.add_argument("--cooldown", type=float, default=0.0)
    a.add_argument("--webhook")
    a.add_argument("--no-clips", action="store_true")
    a.add_argument("--clip-pre", type=float, default=3.0)
    a.add_argument("--clip-post", type=float, default=3.0)
    a.add_argument("--merge-gap", type=float, default=1.0)
    a.add_argument("--out-dir", default="samples")
    a.add_argument("--save-video", action="store_true", help="also write the full annotated video")
    a.add_argument("--stats-every", type=float, default=5.0)
    s.set_defaults(f="stream")

    b = sub.add_parser("bench", help="measure the endpoint's max throughput (frames/s) for batch × concurrency settings")
    b.add_argument("video")
    _endpoint_args(b)
    b.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    b.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    b.add_argument("--seconds", type=float, default=10.0, help="per setting")
    b.add_argument("--out-dir", default="samples")
    b.set_defaults(f="bench")

    v = sub.add_parser("serve", help="run the inference server locally (needs fastapi + uvicorn + ultralytics)")
    v.set_defaults(f="serve")

    a = p.parse_args(argv)
    if a.f == "serve":
        from .server.app import main as serve
        return serve()
    factory = make_factory(a)
    if a.f == "bench":
        from .bench import bench
        return bench(a, factory)

    from .pipeline import StreamPipeline
    from .sources import open_source
    pace = not a.max_speed
    import signal

    def _graceful(*_):                         # Ctrl-C, `docker stop`, systemd stop (SIGTERM): finish and save instead of dying
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _graceful)    # also re-enables Ctrl-C when launched as a background job (SIGINT ignored)
    signal.signal(signal.SIGTERM, _graceful)
    src = open_source(a.source, rate=a.fps, pace=pace, loops=a.loops, duration_s=a.duration, target_fps=a.target_fps)
    label = {"http": "http", "sagemaker": f"sagemaker-{a.endpoint_name}", "vertex": f"vertex-{a.endpoint_id}"}[a.endpoint_type]
    pipe = StreamPipeline(
        factory, label=label, send_max_side=a.send_size, jpeg_quality=a.jpeg_quality, batch_size=a.batch_size,
        max_inflight=a.max_inflight, max_wait_ms=a.max_wait_ms, on_backlog=a.on_backlog or ("drop" if pace else "block"),
        conf=a.conf, tracker=a.tracker, threshold=a.threshold, hold_s=a.hold, clear_s=a.clear, cooldown_s=a.cooldown,
        webhook=a.webhook, clips=not a.no_clips, clip_pre_s=a.clip_pre, clip_post_s=a.clip_post, merge_gap_s=a.merge_gap,
        out_dir=a.out_dir, save_video=a.save_video, stats_every=a.stats_every)
    pipe.run(src)


if __name__ == "__main__":
    main()
