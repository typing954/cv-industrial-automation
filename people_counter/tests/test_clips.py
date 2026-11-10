"""Tests for the evidence-clip recorder in peoplecount/clips.py (ClipRecorder).

What a clip is
--------------
When an alert fires, the recorder saves a video from `pre_s` seconds BEFORE the violation started until `post_s` seconds AFTER the
alert was raised. Alerts whose clips would overlap (or sit within `merge_gap_s` of each other) are merged into ONE clip instead of
creating near-duplicates.

Timing vocabulary used in the comments below
--------------------------------------------
  violation start : the moment the count first went over the limit          (`start_t` passed to on_alert)
  alert time      : violation start + hold_s (here hold_s=2), when the alert fires
  clip window     : [violation start - pre_s, alert time + post_s]
The frames are 10 per second, so any start/end time is only accurate to about one frame (0.1 s); the assertions allow 0.15 s.
"""
import cv2
import numpy as np

from peoplecount.clips import ClipRecorder

FPS = 10   # frames per second of the synthetic stream used in every test


def feed(rec, alerts, seconds=60):
    """Play a synthetic 10 fps stream into the recorder and fire the given alerts.

    `alerts` maps alert time -> (violation start time, episode number), e.g. {12.0: (10.0, 1)} means "at t=12 s an alert fires
    for a violation that began at t=10 s". Each alert is delivered right AFTER the frame at that time has been added, exactly
    as the real pipeline does it. The stream is closed at the end so every clip file is finished and readable.
    """
    for i in range(seconds * FPS):
        t = i / FPS
        rec.add_frame(t, np.full((48, 64, 3), i % 255, np.uint8))   # tiny gray frame; the content does not matter
        if round(t, 3) in alerts:
            start_t, episode = alerts[round(t, 3)]
            rec.on_alert(start_t, t, episode)
    rec.close()


def dur(path):
    """Length in seconds of a saved video file (frame count / fps)."""
    c = cv2.VideoCapture(path)
    return c.get(cv2.CAP_PROP_FRAME_COUNT) / c.get(cv2.CAP_PROP_FPS)


def test_single_alert_gets_3s_before_violation_and_3s_after_alert(tmp_path):
    """One alert -> one clip that runs from 3 s before the violation to 3 s after the alert."""
    rec = ClipRecorder(tmp_path, pre_s=3, post_s=3, hold_s=2)
    feed(rec, {12.0: (10.0, 1)})                          # violation at 10 s, alert at 12 s -> clip window 7 s .. 15 s
    assert len(rec.clips) == 1
    c = rec.clips[0]
    assert abs(c["start_t"] - 7.0) < 0.15 and abs(c["end_t"] - 15.0) < 0.15
    assert abs(dur(c["path"]) - 8.0) < 0.5                # 15 - 7 = 8 s of video actually written to the file


def test_overlapping_alerts_merge_into_one_clip(tmp_path):
    """Alerts whose windows overlap must be merged into a single continuous clip, not three separate ones."""
    rec = ClipRecorder(tmp_path, pre_s=3, post_s=3, hold_s=2)
    # Windows would be [7,15], [12,20] and [17,25]. Each one starts before the previous one ends, so they chain together.
    feed(rec, {12.0: (10.0, 1), 17.0: (15.0, 2), 22.0: (20.0, 3)})
    assert len(rec.clips) == 1 and rec.clips[0]["episodes"] == [1, 2, 3]     # one clip covers all three alerts
    assert abs(rec.clips[0]["start_t"] - 7.0) < 0.15 and abs(rec.clips[0]["end_t"] - 25.0) < 0.15
    assert abs(dur(rec.clips[0]["path"]) - 18.0) < 0.6    # 25 - 7 = 18 s: continuous, with no gaps and no duplicated footage


def test_distant_alerts_make_separate_clips(tmp_path):
    """Alerts far apart in time must NOT be merged: each gets its own clip."""
    rec = ClipRecorder(tmp_path, pre_s=3, post_s=3, hold_s=2)
    feed(rec, {12.0: (10.0, 1), 42.0: (40.0, 2)})         # windows [7,15] and [37,45] are 22 s apart
    assert len(rec.clips) == 2 and [c["episodes"] for c in rec.clips] == [[1], [2]]


def test_alert_near_stream_start_is_clipped_to_available_footage(tmp_path):
    """If the violation starts less than 3 s into the stream there is no footage to go back to; the clip starts at t=0."""
    rec = ClipRecorder(tmp_path, pre_s=3, post_s=3, hold_s=2)
    feed(rec, {2.0: (0.5, 1)})                            # wanted start = 0.5 - 3 = -2.5 s, but the stream only begins at 0
    assert rec.clips[0]["start_t"] == 0.0


def test_gap_just_inside_merge_window_merges_just_outside_splits(tmp_path):
    """The `merge_gap_s` tolerance: clips separated by a small gap (<= 1 s) are merged, larger gaps are kept apart."""
    # Case A. First clip ends at 12 + 3 = 15 s. The second alert's window would start at 20.5 - 3 = 17.5 s.
    # The gap is 2.5 s, larger than merge_gap_s=1 -> two clips.
    a = ClipRecorder(tmp_path / "a", pre_s=3, post_s=3, hold_s=2, merge_gap_s=1.0)
    feed(a, {12.0: (10.0, 1), 22.5: (20.5, 2)})
    assert len(a.clips) == 2

    # Case B. Same first clip (ends at 15 s). The second window would start at 18.5 - 3 = 15.5 s.
    # The gap is only 0.5 s, within merge_gap_s=1 -> merged into one clip.
    b = ClipRecorder(tmp_path / "b", pre_s=3, post_s=3, hold_s=2, merge_gap_s=1.0)
    feed(b, {12.0: (10.0, 1), 20.5: (18.5, 2)})
    assert len(b.clips) == 1
