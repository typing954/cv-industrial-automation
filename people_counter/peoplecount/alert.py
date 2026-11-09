"""Threshold alerting. Every *episode* of the count exceeding the limit is recorded.

Episode lifecycle
  start_t   : first moment the (smoothed) count went above the threshold  <- "time it exceeded the limit"
  ALERT     : raised once the count has stayed above the threshold for `hold_s` (debounce against blips)
  CLEARED   : raised once the count has stayed at/below the limit for `clear_s`; the episode's end time is the
              last moment it was still over, so short dips below the limit do not split one episode in two.
A new episode (= a new ALERT) can start immediately after a CLEARED (`cooldown_s` defaults to 0, no suppression).
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Optional


def timecode(t: float) -> str:
    """Stream time in seconds -> HH:MM:SS.mmm"""
    t = max(float(t), 0.0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


@dataclass
class AlertEvent:
    kind: str                  # "ALERT" | "CLEARED"
    episode: int               # 1, 2, 3 ... one per exceedance episode
    t: float                   # stream time of this event (s)
    timecode: str
    count: int                 # count when the event fired
    threshold: int
    start_t: float             # when the limit was FIRST exceeded in this episode
    start_timecode: str
    peak_count: int            # highest count so far in the episode
    peak_t: float
    wall_time: float           # epoch seconds when the event was raised
    wall_time_iso: str
    end_t: Optional[float] = None          # CLEARED only: last moment still over the limit
    end_timecode: Optional[str] = None
    duration_s: Optional[float] = None     # CLEARED only: end_t - start_t
    reason: str = ""                       # "stream_end" if the video ended mid-episode


class ConsoleSink:
    def __call__(self, e: AlertEvent):
        if e.kind == "ALERT":
            print(f"\033[91m🚨 ALERT #{e.episode}\033[0m {e.count} people (limit {e.threshold}) | exceeded at "
                  f"{e.start_timecode} | alert raised {e.timecode}", flush=True)
        else:
            print(f"\033[92m✅ CLEARED #{e.episode}\033[0m over limit {e.start_timecode} -> {e.end_timecode} "
                  f"({e.duration_s:.1f}s), peak {e.peak_count}" + (f" [{e.reason}]" if e.reason else ""), flush=True)


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
        body["text"] = (f"{e.kind} #{e.episode}: {e.count} people (limit {e.threshold}), "
                        f"exceeded at {e.start_timecode}, peak {e.peak_count}")
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=self.timeout).read()
        except Exception as ex:  # noqa: BLE001
            print(f"[webhook] failed: {ex}", flush=True)


class AlertManager:
    def __init__(self, threshold: int, hold_s: float = 2.0, clear_s: float = 3.0, hysteresis: int = 0,
                 cooldown_s: float = 0.0, sinks: Optional[list[Callable[[AlertEvent], None]]] = None):
        self.threshold, self.hold_s, self.clear_s = threshold, hold_s, clear_s
        self.hysteresis, self.cooldown_s = hysteresis, cooldown_s
        self.sinks = sinks or []
        self.active = False
        self.events: list[AlertEvent] = []
        self.episodes: list[dict] = []          # one dict per episode (see _start_episode); end fields filled on clear
        self._cur: Optional[dict] = None
        self._over_since: Optional[float] = None
        self._under_since: Optional[float] = None
        self._last_over: float = 0.0
        self._cand_peak: tuple[int, float] = (0, 0.0)
        self._last_alert: float = -1e18

    # -- main entry: call once per processed frame -------------------------------------------------------------
    def update(self, count: int, t: float) -> Optional[AlertEvent]:
        ev = None
        if count > self.threshold:
            self._under_since = None
            self._last_over = t
            if self._over_since is None:
                self._over_since, self._cand_peak = t, (count, t)
            elif count > self._cand_peak[0]:
                self._cand_peak = (count, t)
            if self.active:
                if count > self._cur["peak_count"]:
                    self._cur["peak_count"], self._cur["peak_t"] = count, t
            elif t - self._over_since >= self.hold_s and t - self._last_alert >= self.cooldown_s:
                ev = self._start_episode(t, count)
        else:
            self._over_since = None
            if self.active and count <= self.threshold - self.hysteresis:
                if self._under_since is None:
                    self._under_since = t
                if t - self._under_since >= self.clear_s:
                    ev = self._end_episode(t, count)
            else:
                self._under_since = None
        return ev

    def finish(self, t: float, count: int = 0) -> Optional[AlertEvent]:
        """Call when the stream ends so an episode still in progress is closed and recorded."""
        if self.active:
            return self._end_episode(t, count, reason="stream_end")
        return None

    # -- internals -----------------------------------------------------------------------------------------------
    def _start_episode(self, t, count):
        self.active, self._last_alert = True, t
        peak, peak_t = self._cand_peak
        self._cur = {"episode": len(self.episodes) + 1, "threshold": self.threshold,
                     "start_t": round(self._over_since, 3), "start_timecode": timecode(self._over_since),
                     "alert_t": round(t, 3), "alert_timecode": timecode(t), "peak_count": peak, "peak_t": round(peak_t, 3),
                     "end_t": None, "end_timecode": None, "duration_s": None, "clip": None,
                     "start_wall_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())}
        self.episodes.append(self._cur)
        return self._emit("ALERT", t, count)

    def _end_episode(self, t, count, reason=""):
        c = self._cur
        end_t = max(self._last_over, c["start_t"])
        c.update(end_t=round(end_t, 3), end_timecode=timecode(end_t), duration_s=round(end_t - c["start_t"], 3))
        self.active, self._under_since = False, None
        return self._emit("CLEARED", t, count, reason)

    def _emit(self, kind: str, t: float, count: int, reason: str = "") -> AlertEvent:
        c, now = self._cur, time.time()
        e = AlertEvent(kind, c["episode"], round(t, 3), timecode(t), count, self.threshold, c["start_t"],
                       c["start_timecode"], c["peak_count"], c["peak_t"], now,
                       time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
                       c["end_t"] if kind == "CLEARED" else None, c["end_timecode"] if kind == "CLEARED" else None,
                       c["duration_s"] if kind == "CLEARED" else None, reason)
        self.events.append(e)
        for s in self.sinks:
            s(e)
        return e
