"""Ring-buffer of recent strategy events for the operator UI.

The runner writes events as side-effects of its tick loop (role
changes, script pushes, safety overrides, arm/disarm, match
start/stop). The web app reads ``recent()`` from the Flask thread.
Bounded capacity (default 500) so a long-running strategy doesn't
grow unbounded.

Thread-safety: the runner is asyncio + a single coroutine that
calls :meth:`add`; the web reads from Flask worker threads. A
plain ``threading.Lock`` is sufficient because the asyncio side
doesn't block on the lock long enough to matter.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional


@dataclass(frozen=True)
class Event:
    """One logged event. Keep small — there can be hundreds in flight."""
    seq: int           # monotonically increasing, set by EventLog.add
    t_mono: float      # time.monotonic() — for ordering across boots
    t_wall: float      # time.time() — for display ("12:34:56")
    kind: str          # see EventLog.KINDS
    drone: Optional[str]
    msg: str
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "t_mono": self.t_mono,
            "t_wall": self.t_wall,
            "kind": self.kind,
            "drone": self.drone,
            "msg": self.msg,
            "payload": self.payload,
        }


class EventLog:
    """Append-only ring buffer with seq-based long-polling support."""

    # The set of kinds the runner actually emits; the UI uses these
    # for colour coding. Adding a new one only requires extending the
    # JS palette — Python-side is free-form.
    KINDS = ("role_change", "script_push", "safety",
             "arm", "disarm", "match_start", "match_stop", "info")

    def __init__(self, capacity: int = 500) -> None:
        self._buf: Deque[Event] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0

    def add(self, kind: str, msg: str,
            drone: Optional[str] = None,
            payload: Optional[dict] = None) -> None:
        """Append an event. ``kind`` should be from :attr:`KINDS` but
        we don't enforce — easier to add new kinds without churn."""
        with self._lock:
            self._seq += 1
            self._buf.append(Event(
                seq=self._seq,
                t_mono=time.monotonic(),
                t_wall=time.time(),
                kind=str(kind),
                drone=drone,
                msg=str(msg),
                payload=dict(payload or {}),
            ))

    def recent(self, limit: int = 50,
               since_seq: Optional[int] = None) -> list[dict]:
        """Return up to ``limit`` recent events, newest first. When
        ``since_seq`` is given, only return events with ``seq >
        since_seq`` (long-poll pattern — UI sends back the last seq
        it saw)."""
        with self._lock:
            items = list(self._buf)
        items.reverse()
        if since_seq is not None:
            items = [e for e in items if e.seq > int(since_seq)]
        if limit > 0:
            items = items[:limit]
        return [e.to_dict() for e in items]

    def latest_seq(self) -> int:
        with self._lock:
            return self._seq
