"""Tests for the alert rules in peoplecount/alert.py (AlertManager).

How AlertManager works, in one paragraph
----------------------------------------
It is fed one people count per processed frame, together with that frame's stream time in seconds:
`update(count, t)`. An *episode* starts when the count goes ABOVE the threshold. An ALERT event is raised only once the count has
stayed above it for `hold_s` seconds (so a one-frame blip is ignored). The episode ends, and a CLEARED event is raised, once the
count has been at or below the threshold for `clear_s` seconds. Every episode is stored in `am.episodes` with the time the limit
was first exceeded (`start_t`), the alert time, the end, the duration and the peak.

Test style: the helper `run()` plays a list of counts at one sample per second, so the position of a number in the list IS its
time in seconds (index 4 = t=4 s). The comments next to each list spell out that timeline.
"""
from peoplecount.alert import AlertManager


def run(am, counts, dt=1.0):
    """Feed `counts` to the AlertManager, one sample every `dt` seconds (sample i is at t = i * dt).

    Returns a list of (time, event kind, episode number) for every event raised, in order, e.g. [(4.0, "ALERT", 1)].
    """
    return [(i * dt, e.kind, e.episode) for i, c in enumerate(counts) if (e := am.update(c, i * dt))]


def test_sustained_alert_and_clear():
    """The basic life cycle: alert after `hold_s` over the limit, cleared after `clear_s` back under it."""
    am = AlertManager(5, hold_s=2, clear_s=3)
    # Limit is 5. Counts: 3 people until t=1, 8 people from t=2 to t=9, then 3 again from t=10.
    seq = [8 if 2 <= t < 10 else 3 for t in range(20)]
    events = run(am, seq)
    # Over the limit since t=2 -> held for 2 s -> ALERT at t=4. Back under at t=10 -> held for 3 s -> CLEARED at t=13.
    assert [(t, kind) for t, kind, _ in events] == [(4, "ALERT"), (13, "CLEARED")]


def test_blip_does_not_alert():
    """A single sample over the limit is shorter than `hold_s`, so it must not raise an alert."""
    am = AlertManager(5, hold_s=3)
    # Only t=5 is over the limit (9 > 5); it lasts 1 s but 3 s are required.
    assert run(am, [2, 2, 2, 2, 2, 9, 2, 2, 2, 2]) == []


def test_every_episode_is_recorded_not_just_the_first():
    """Regression test: the old default (30 s cooldown) hid every violation after the first. Now all are recorded."""
    am = AlertManager(5, hold_s=1, clear_s=2)            # cooldown_s defaults to 0 = no suppression
    # Three separate violations, each followed by a quiet period long enough to clear:
    #   t=3..6   9 people      t=12..16  12 people      t=22..24  7 people
    seq = [2] * 3 + [9] * 4 + [2] * 5 + [12] * 5 + [2] * 5 + [7] * 3 + [2] * 4
    run(am, seq)
    # Each episode produces one ALERT and one CLEARED, in order.
    assert [e.kind for e in am.events] == ["ALERT", "CLEARED"] * 3
    assert [ep["episode"] for ep in am.episodes] == [1, 2, 3]


def test_violation_start_peak_and_duration():
    """The episode record must say WHEN the limit was first exceeded (not when the alert fired), plus peak and duration."""
    am = AlertManager(5, hold_s=2, clear_s=2)
    #        t=0 1  2  3  4  5  6  7  8  9
    seq = [1, 1, 7, 8, 9, 6, 6, 2, 2, 2]
    #   over the limit from t=2 (first exceeded) ... alert needs 2 s -> raised at t=4 ... last sample over the limit is t=6
    run(am, seq)
    ep = am.episodes[0]
    assert ep["start_t"] == 2 and ep["alert_t"] == 4
    assert ep["peak_count"] == 9 and ep["peak_t"] == 4                       # highest count was 9, seen at t=4
    # The episode ends at the LAST moment still over the limit (t=6), not when CLEARED fires (t=9).
    assert ep["end_t"] == 6 and ep["duration_s"] == 4 and ep["start_timecode"] == "00:00:02.000"
    ev = am.events[0]
    assert ev.start_t == 2 and ev.t == 4          # the ALERT event carries the first-exceeded time AND the time it was raised


def test_short_dip_below_limit_stays_one_episode():
    """A dip below the limit shorter than `clear_s` must not split one violation into two episodes."""
    am = AlertManager(5, hold_s=1, clear_s=3)
    # Over at t=0..2, dips to 3 at t=3..4 (only 2 s, but 3 s are needed to clear), over again t=5..7, then quiet from t=8.
    run(am, [9, 9, 9, 3, 3, 9, 9, 9, 2, 2, 2, 2])
    assert len(am.episodes) == 1


def test_stream_end_closes_open_episode():
    """If the video ends in the middle of a violation, `finish()` must close and record it (reason "stream_end")."""
    am = AlertManager(5, hold_s=1)
    run(am, [9, 9, 9, 9])                       # over the limit the whole time (t=0..3): alert raised at t=1, never cleared
    assert am.active
    e = am.finish(4.0, 9)                       # the stream ends at t=4
    # The episode ends at the last moment it was still over the limit, t=3.
    assert e.kind == "CLEARED" and e.reason == "stream_end" and am.episodes[0]["end_t"] == 3


def test_cooldown_still_available():
    """`cooldown_s` is optional (default 0). When set, a new alert too soon after the last one is suppressed."""
    am = AlertManager(5, hold_s=0, clear_s=0, cooldown_s=100)
    assert am.update(9, 0).kind == "ALERT"       # hold_s=0 -> alert immediately
    assert am.update(1, 1).kind == "CLEARED"     # clear_s=0 -> cleared immediately
    assert am.update(9, 2) is None               # over the limit again after 2 s, but the 100 s cooldown suppresses it
