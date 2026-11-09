from peoplecount.alert import AlertManager


def test_sustained_alert_and_clear():
    am = AlertManager(5, hold_s=2, clear_s=3, cooldown_s=0)
    seq = [(t, 8 if 2 <= t < 10 else 3) for t in range(0, 20)]
    ev = [(t, e.kind) for t, c in seq if (e := am.update(c, t))]
    assert ev == [(4, "ALERT"), (13, "CLEARED")]


def test_blip_does_not_alert():
    am = AlertManager(5, hold_s=3, cooldown_s=0)
    assert not [am.update(9 if t == 5 else 2, t) for t in range(10) if am.update(9 if t == 5 else 2, t)]


def test_cooldown_after_clear():
    am = AlertManager(5, hold_s=0, clear_s=0, cooldown_s=100)
    assert am.update(9, 0).kind == "ALERT"
    assert am.update(1, 1).kind == "CLEARED"
    assert am.update(9, 2) is None  # suppressed
