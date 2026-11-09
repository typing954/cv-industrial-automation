"""Threshold alerting with debounce (sustained breach), hysteresis-style clearing and cooldown."""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, asdict
from typing import Callable, Optional


@dataclass
class AlertEvent:
    kind: str          # "ALERT" | "CLEARED"
    t: float           # stream time (s)
    count: int
    threshold: int
    wall_time: float


class ConsoleSink:
    def __call__(self, e: AlertEvent):
        tag = "\033[91m🚨 ALERT\033[0m" if e.kind == "ALERT" else "\033[92m✅ CLEARED\033[0m"
        print(f"[{e.t:8.2f}s] {tag}: {e.count} people (threshold {e.threshold})", flush=True)


class JsonlSink:
    def __init__(self, path: str):
        self.path = path

    def __call__(self, e: AlertEvent):
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(e)) + "\n")


class WebhookSink:
    """POSTs the event as JSON (Slack-compatible 'text' field included). Failures never stop the pipeline."""

    def __init__(self, url: str, timeout: float = 2.0):
        self.url, self.timeout = url, timeout

    def __call__(self, e: AlertEvent):
        body = asdict(e)
        body["text"] = f"{e.kind}: {e.count} people in room (limit {e.threshold})"
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=self.timeout).read()
        except Exception as ex:  # noqa: BLE001
            print(f"[webhook] failed: {ex}", flush=True)


class AlertManager:
    """ALERT fires when count > threshold continuously for `hold_s`.
    CLEARED fires when count <= threshold - `hysteresis` continuously for `clear_s`.
    After an ALERT, re-alert is suppressed for `cooldown_s` unless it was cleared in between."""

    def __init__(self, threshold: int, hold_s: float = 2.0, clear_s: float = 3.0, hysteresis: int = 0,
                 cooldown_s: float = 30.0, sinks: Optional[list[Callable[[AlertEvent], None]]] = None):
        self.threshold, self.hold_s, self.clear_s = threshold, hold_s, clear_s
        self.hysteresis, self.cooldown_s = hysteresis, cooldown_s
        self.sinks = sinks or []
        self.active = False
        self.events: list[AlertEvent] = []
        self._over_since: Optional[float] = None
        self._under_since: Optional[float] = None
        self._last_alert: float = -1e18

    def update(self, count: int, t: float) -> Optional[AlertEvent]:
        ev = None
        if count > self.threshold:
            self._under_since = None
            if self._over_since is None:
                self._over_since = t
            if (not self.active and t - self._over_since >= self.hold_s
                    and t - self._last_alert >= self.cooldown_s):
                self.active, self._last_alert = True, t
                ev = self._emit("ALERT", t, count)
        else:
            self._over_since = None
            if self.active and count <= self.threshold - self.hysteresis:
                if self._under_since is None:
                    self._under_since = t
                if t - self._under_since >= self.clear_s:
                    self.active = False
                    ev = self._emit("CLEARED", t, count)
            else:
                self._under_since = None
        return ev

    def _emit(self, kind: str, t: float, count: int) -> AlertEvent:
        e = AlertEvent(kind, round(t, 3), count, self.threshold, time.time())
        self.events.append(e)
        for s in self.sinks:
            s(e)
        return e
