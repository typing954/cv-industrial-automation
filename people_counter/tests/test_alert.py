from peoplecount.alert import AlertManager


def run(am, counts, dt=1.0):
    return [(i * dt, e.kind, e.episode) for i, c in enumerate(counts) if (e := am.update(c, i * dt))]


def test_sustained_alert_and_clear():
    am = AlertManager(5, hold_s=2, clear_s=3)
    seq = [8 if 2 <= t < 10 else 3 for t in range(20)]
    assert [(t, k) for t, k, _ in run(am, seq)] == [(4, "ALERT"), (13, "CLEARED")]


def test_blip_does_not_alert():
    am = AlertManager(5, hold_s=3)
    assert run(am, [2, 2, 2, 2, 2, 9, 2, 2, 2, 2]) == []


def test_every_episode_is_recorded_not_just_the_first():
    am = AlertManager(5, hold_s=1, clear_s=2)            # default cooldown = 0
    seq = [2] * 3 + [9] * 4 + [2] * 5 + [12] * 5 + [2] * 5 + [7] * 3 + [2] * 4
    run(am, seq)
    assert [e.kind for e in am.events] == ["ALERT", "CLEARED"] * 3
    assert [ep["episode"] for ep in am.episodes] == [1, 2, 3]


def test_violation_start_peak_and_duration():
    am = AlertManager(5, hold_s=2, clear_s=2)
    seq = [1, 1, 7, 8, 9, 6, 6, 2, 2, 2]                 # exceeds at t=2, alert at t=4, last over t=6
    run(am, seq)
    ep = am.episodes[0]
    assert ep["start_t"] == 2 and ep["alert_t"] == 4 and ep["peak_count"] == 9 and ep["peak_t"] == 4
    assert ep["end_t"] == 6 and ep["duration_s"] == 4 and ep["start_timecode"] == "00:00:02.000"
    ev = am.events[0]
    assert ev.start_t == 2 and ev.t == 4          # ALERT event reports when the limit was FIRST exceeded


def test_short_dip_below_limit_stays_one_episode():
    am = AlertManager(5, hold_s=1, clear_s=3)
    run(am, [9, 9, 9, 3, 3, 9, 9, 9, 2, 2, 2, 2])         # 2 s dip < clear_s -> same episode
    assert len(am.episodes) == 1


def test_stream_end_closes_open_episode():
    am = AlertManager(5, hold_s=1)
    run(am, [9, 9, 9, 9])
    assert am.active
    e = am.finish(4.0, 9)
    assert e.kind == "CLEARED" and e.reason == "stream_end" and am.episodes[0]["end_t"] == 3


def test_cooldown_still_available():
    am = AlertManager(5, hold_s=0, clear_s=0, cooldown_s=100)
    assert am.update(9, 0).kind == "ALERT"
    assert am.update(1, 1).kind == "CLEARED"
    assert am.update(9, 2) is None                         # suppressed by explicit cooldown
