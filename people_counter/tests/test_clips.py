import cv2
import numpy as np

from peoplecount.clips import ClipRecorder

FPS = 10


def feed(rec, alerts, seconds=60):
    """alerts: {alert_time: (start_t, episode)} -- fired right after the frame at that time is added."""
    for i in range(seconds * FPS):
        t = i / FPS
        rec.add_frame(t, np.full((48, 64, 3), i % 255, np.uint8))
        if round(t, 3) in alerts:
            s, ep = alerts[round(t, 3)]
            rec.on_alert(s, t, ep)
    rec.close()


def dur(path):
    c = cv2.VideoCapture(path)
    return c.get(cv2.CAP_PROP_FRAME_COUNT) / c.get(cv2.CAP_PROP_FPS)


def test_single_alert_gets_3s_before_violation_and_3s_after_alert(tmp_path):
    rec = ClipRecorder(tmp_path, pre_s=3, post_s=3, hold_s=2)
    feed(rec, {12.0: (10.0, 1)})                          # violation 10s, alert 12s -> clip 7s..15s
    assert len(rec.clips) == 1
    c = rec.clips[0]
    assert abs(c["start_t"] - 7.0) < 0.15 and abs(c["end_t"] - 15.0) < 0.15
    assert abs(dur(c["path"]) - 8.0) < 0.5


def test_overlapping_alerts_merge_into_one_clip(tmp_path):
    rec = ClipRecorder(tmp_path, pre_s=3, post_s=3, hold_s=2)
    feed(rec, {12.0: (10.0, 1), 17.0: (15.0, 2), 22.0: (20.0, 3)})   # each pre-roll overlaps the previous clip
    assert len(rec.clips) == 1 and rec.clips[0]["episodes"] == [1, 2, 3]
    assert abs(rec.clips[0]["start_t"] - 7.0) < 0.15 and abs(rec.clips[0]["end_t"] - 25.0) < 0.15
    assert abs(dur(rec.clips[0]["path"]) - 18.0) < 0.6    # continuous, no gaps or duplicates


def test_distant_alerts_make_separate_clips(tmp_path):
    rec = ClipRecorder(tmp_path, pre_s=3, post_s=3, hold_s=2)
    feed(rec, {12.0: (10.0, 1), 42.0: (40.0, 2)})
    assert len(rec.clips) == 2 and [c["episodes"] for c in rec.clips] == [[1], [2]]


def test_alert_near_stream_start_is_clipped_to_available_footage(tmp_path):
    rec = ClipRecorder(tmp_path, pre_s=3, post_s=3, hold_s=2)
    feed(rec, {2.0: (0.5, 1)})
    assert rec.clips[0]["start_t"] == 0.0


def test_gap_just_inside_merge_window_merges_just_outside_splits(tmp_path):
    a = ClipRecorder(tmp_path / "a", pre_s=3, post_s=3, hold_s=2, merge_gap_s=1.0)
    feed(a, {12.0: (10.0, 1), 22.5: (20.5, 2)})           # 2nd clip would start at 17.5; 1st ends 15 -> gap 2.5 > 1
    assert len(a.clips) == 2
    b = ClipRecorder(tmp_path / "b", pre_s=3, post_s=3, hold_s=2, merge_gap_s=1.0)
    feed(b, {12.0: (10.0, 1), 20.5: (18.5, 2)})           # 2nd would start 15.5; gap 0.5 <= 1 -> merge
    assert len(b.clips) == 1
