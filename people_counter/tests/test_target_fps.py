"""Tests for `target_fps` in peoplecount/video.py (VideoSource).

`target_fps=N` means "run detection on N frames per second of video". VideoSource reads the file at its own speed but only hands
back frames picked EVENLY IN TIME at that rate. Frame i of a video with F fps happens at t = i / F seconds, so with target_fps=5
a frame is taken every 0.2 s of video, whatever F is (this also works when F is not a multiple of N, e.g. 25 fps -> 10 fps).

Every test uses one synthetic video: 100 frames at 25 fps (= 4 seconds). VideoSource yields (frame_index, time_in_seconds, frame).
"""
import time

import cv2
import numpy as np
import pytest

from peoplecount.video import VideoSource


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A 4-second test video: 100 frames at 25 fps, each a small solid-gray image (frame i has gray level i).

    Built once per test module (scope="module") and shared by all tests, because writing a video file is slow.
    """
    p = str(tmp_path_factory.mktemp("v") / "t.mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for i in range(100):
        w.write(np.full((48, 64, 3), i, np.uint8))
    w.release()
    return p


def test_default_is_every_frame(video):
    """Without target_fps every frame is returned, in order, and the effective rate is the video's own 25 fps."""
    s = VideoSource(video)
    assert s.eff_fps == 25 and [i for i, _, _ in s] == list(range(100))


def test_target_fps_picks_frames_evenly_in_time(video):
    """25 fps video, target 5 fps -> every 5th frame (25 / 5 = 5), i.e. one frame each 0.2 s of video."""
    s = VideoSource(video, target_fps=5)
    got = list(s)
    assert s.eff_fps == 5 and [i for i, _, _ in got] == list(range(0, 100, 5))     # frame indices 0, 5, 10, ... 95
    assert [round(t, 3) for _, t, _ in got[:4]] == [0.0, 0.2, 0.4, 0.6]            # stream time of each returned frame


def test_non_integer_ratio_still_hits_the_requested_rate(video):
    """25 fps -> 10 fps is 2.5 source frames per sample, which a plain 'every Nth frame' could not do. Time-based sampling can."""
    got = list(VideoSource(video, target_fps=10))
    assert len(got) == 40                                                           # 4 s x 10 fps
    # The gap between returned frames alternates between 2 and 3 frames (average 2.5).
    gaps = {b[0] - a[0] for a, b in zip(got, got[1:])}
    assert gaps == {2, 3}
    # And every returned frame stays close to its ideal slot on the 0.1 s grid (0.0, 0.1, 0.2, ...); 0.04 s = one 25 fps frame.
    assert all(abs(t - k * 0.1) < 0.04 for k, (_, t, _) in enumerate(got))


def test_target_above_video_fps_uses_every_frame(video):
    """You can't detect on more frames per second than the video has: asking for 60 on a 25 fps video gives all 25."""
    s = VideoSource(video, target_fps=60)
    assert s.eff_fps == 25 and len(list(s)) == 100


def test_invalid_target_fps(video):
    """Zero and negative rates make no sense and must raise an error instead of silently doing something odd."""
    with pytest.raises(ValueError):
        VideoSource(video, target_fps=0)
    with pytest.raises(ValueError):
        VideoSource(video, target_fps=-3)


def test_max_frames_counts_sampled_frames(video):
    """`max_frames` limits how many frames are RETURNED (after sampling), not how many are read from the file."""
    assert len(list(VideoSource(video, target_fps=5, max_frames=7))) == 7


def test_realtime_with_target_fps_paces_and_does_not_drop(video):
    """`realtime=True` plays the video at its natural speed: 5 fps for 4 s of video should take about 4 real seconds."""
    s = VideoSource(video, target_fps=5, realtime=True)
    t0 = time.perf_counter()
    got = list(s)
    el = time.perf_counter() - t0                                                   # wall-clock seconds the loop took
    # 20 frames returned (4 s x 5 fps), none dropped (this loop does no work per frame), and it took ~4 s (3.4-4.6 allows slack).
    assert len(got) == 20 and s.dropped == 0 and 3.4 < el < 4.6


def test_realtime_drops_when_the_detector_is_too_slow(video):
    """In real-time mode a slow detector must cause frames to be DROPPED (counted in `dropped`) rather than queued up."""
    s = VideoSource(video, target_fps=10, realtime=True)                            # one frame is due every 0.1 s
    n = 0
    for _ in s:
        n += 1
        time.sleep(0.35)                           # pretend the detector needs 0.35 s per frame, 3.5x slower than the 0.1 s slot
        if n == 6:
            break
    assert s.dropped > 0                           # it skipped slots it could not keep up with
