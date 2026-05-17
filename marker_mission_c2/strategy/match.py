"""Match clock + per-tick counter for the strategy runtime.

A small piece of state, separate from :class:`StrategySettings` so it
can be mutated freely at runtime without persisting through the
settings JSON. Owned by the runner and read by the web UI.

Two concerns wrapped in one module:

  * **Match clock** — operator-driven. Press *Start match* in the UI;
    the strategy records the start wall-clock time. ``remaining_s``
    counts down to zero against ``settings.match.duration_s``. *Stop
    match* clears the start time. Nothing in the strategy stack
    auto-stops at zero (that decision is operator-policy); the UI
    just shows the timer turning red.

  * **Tick counter** — runner-driven. Increments once per
    :meth:`SwarmRunner._run` iteration. Lets the UI prove the
    strategy is alive ("312 ticks · 1.0 Hz · last 0.3 s ago")
    without the operator having to ssh in for ``journalctl``.

Thread-safe: the runner is async, the web app handlers run on Flask
worker threads — :class:`MatchState` is mutated by both, so writes
hold ``self._lock``.
"""
from __future__ import annotations

import threading
import time
from typing import Optional


class MatchState:
    """Mutable runtime state. Construct once at strategy startup, share
    the same instance between the runner (writes ticks) and the web
    app (reads everything, writes match-start/stop).
    """

    def __init__(self, duration_s: float) -> None:
        self._lock = threading.Lock()
        self._duration_s = float(duration_s)
        # Wall-clock seconds (time.time()) when /api/match/start was
        # hit; None means "no match running".
        self._match_start_wall: Optional[float] = None
        # Per-tick instrumentation. monotonic, not wall-clock, so
        # NTP nudges don't make the tick rate jump.
        self._tick_count: int = 0
        self._first_tick_mono: Optional[float] = None
        self._last_tick_mono: Optional[float] = None

    # ---------------------- match clock ----------------------

    def start_match(self) -> None:
        with self._lock:
            self._match_start_wall = time.time()

    def stop_match(self) -> None:
        with self._lock:
            self._match_start_wall = None

    def set_duration_s(self, duration_s: float) -> None:
        with self._lock:
            self._duration_s = float(duration_s)

    # ---------------------- tick counter ---------------------

    def record_tick(self) -> None:
        """Called by :class:`SwarmRunner` once per loop iteration."""
        with self._lock:
            now = time.monotonic()
            self._tick_count += 1
            if self._first_tick_mono is None:
                self._first_tick_mono = now
            self._last_tick_mono = now

    # ---------------------- read-only views ------------------

    def snapshot(self) -> dict:
        """Return a JSON-safe snapshot. Used by ``/api/match`` and
        embedded in ``/api/state`` so the UI's poll picks up both
        clock + tick info in one round trip."""
        with self._lock:
            now_wall = time.time()
            now_mono = time.monotonic()
            match_running = self._match_start_wall is not None
            elapsed_s: Optional[float] = None
            remaining_s: Optional[float] = None
            if match_running:
                elapsed_s = max(0.0, now_wall - self._match_start_wall)
                remaining_s = max(0.0, self._duration_s - elapsed_s)
            # tick stats
            uptime_s: Optional[float] = None
            last_age_s: Optional[float] = None
            hz: Optional[float] = None
            if self._first_tick_mono is not None:
                uptime_s = now_mono - self._first_tick_mono
                if uptime_s > 0 and self._tick_count > 1:
                    hz = (self._tick_count - 1) / uptime_s
            if self._last_tick_mono is not None:
                last_age_s = now_mono - self._last_tick_mono
            return {
                "match": {
                    "running": match_running,
                    "start_time_wall": self._match_start_wall,
                    "duration_s": self._duration_s,
                    "elapsed_s": elapsed_s,
                    "remaining_s": remaining_s,
                    "expired": (remaining_s is not None and remaining_s <= 0.0),
                },
                "ticks": {
                    "count": self._tick_count,
                    "uptime_s": uptime_s,
                    "last_age_s": last_age_s,
                    "hz": hz,
                },
            }
