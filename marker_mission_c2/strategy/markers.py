"""Slot-based marker observation tracker.

The SDC26 arena has 6 physical target markers, each with two ArUco faces:

    slot 1 -> ids (31 blue / 41 red)
    slot 2 -> ids (32 blue / 42 red)
    ...
    slot 6 -> ids (36 blue / 46 red)

Whichever face a scout currently sees tells you which team is holding the
slot. There is no "captured = not visible" — capture is about which side
is up, full stop.

Public surface:

    tracker.ingest(fc_name, visible_marker_ids)
        Called every tick by the runner with the FC's marker observations.
        Updates the affected slot's holder + freshness.

    tracker.slot_status(slot)
        Returns the slot's current ``SlotStatus``.

    tracker.snapshot()
        All known slots.

    tracker.to_dict()
        JSON-friendly serialisation for the web UI.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .settings import ALL_SLOTS, face_id, slot_for_face, team_for_face

logger = logging.getLogger(__name__)

# How long an observation stays "fresh" (visible) before we consider the
# slot's holder stale. Used purely for UI flagging — the holder itself
# only changes when the *opposite* face becomes the more recent observation.
FRESHNESS_S = 1.5

# v7 §1.4.3 — after any capture (in either direction) the box is locked for
# 5 s; no further capture can register until the lock expires. The C2 cannot
# observe the lock directly, so we conservatively start the timer the moment
# the marker tracker SEES a holder change (which is at-or-after the FC-side
# flip), arming this many seconds of "skip me" for the planner.
CAPTURE_LOCK_S = 5.0


@dataclass
class SlotStatus:
    slot: int
    holder: str = "unknown"                # "red" | "blue" | "unknown"
    last_observed_face_id: Optional[int] = None
    last_seen_unix_s: float = 0.0
    last_seen_by: Optional[str] = None
    # Per-face freshness (when did we last see each side?)
    red_face_seen_unix_s: float = 0.0
    blue_face_seen_unix_s: float = 0.0
    seen_by_recent: List[str] = field(default_factory=list)
    # Tracks the last time the holder actually flipped, for UI / events.
    holder_changed_unix_s: Optional[float] = None
    # v7 capture lock: wall-clock until which this box is locked (no further
    # captures register). 0 means "not locked".
    lock_until_unix_s: float = 0.0


class MarkerTracker:
    """Thread-safe per-slot capture-state tracker."""

    def __init__(self, *, active_slots: Iterable[int] = ALL_SLOTS) -> None:
        self._lock = threading.RLock()
        self._slots: Dict[int, SlotStatus] = {}
        self._per_fc_last_seen: Dict[Tuple[str, int], float] = {}
        self._set_active_slots_locked(tuple(active_slots))

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_active_slots(self, slots: Iterable[int]) -> None:
        with self._lock:
            self._set_active_slots_locked(tuple(slots))

    def _set_active_slots_locked(self, slots: Tuple[int, ...]) -> None:
        wanted = {int(s) for s in slots if 1 <= int(s) <= 6}
        # Drop slots no longer active.
        for s in list(self._slots.keys()):
            if s not in wanted:
                del self._slots[s]
        # Add new ones with unknown holder.
        for s in wanted:
            self._slots.setdefault(s, SlotStatus(slot=s))

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, fc_name: str, visible_ids: Iterable[int]) -> None:
        """Record observations from a single FC."""
        if not fc_name:
            return
        now = time.time()
        with self._lock:
            for raw in visible_ids:
                try:
                    mid = int(raw)
                except (TypeError, ValueError):
                    continue
                slot = slot_for_face(mid)
                team = team_for_face(mid)
                if slot is None or team is None:
                    continue  # not a target face — ignore (e.g. wall markers)
                if slot not in self._slots:
                    # Not in active_slots — ignore quietly.
                    continue
                status = self._slots[slot]
                status.last_observed_face_id = mid
                status.last_seen_unix_s = now
                status.last_seen_by = fc_name
                if team == "red":
                    status.red_face_seen_unix_s = now
                else:
                    status.blue_face_seen_unix_s = now
                if status.holder != team:
                    status.holder = team
                    status.holder_changed_unix_s = now
                    # v7: arm the 5-s capture lock the moment we observe a
                    # holder change. Slightly conservative (we observe a few
                    # frames after the actual flip), which is fine — it just
                    # tells the planner to skip this slot for the lock window.
                    status.lock_until_unix_s = now + CAPTURE_LOCK_S
                self._per_fc_last_seen[(fc_name, mid)] = now
            # Recompute "seen_by_recent" for every active slot.
            self._recompute_locked(now)

    def _recompute_locked(self, now: float) -> None:
        for status in self._slots.values():
            recent: List[str] = []
            for (fc, m_id), ts in self._per_fc_last_seen.items():
                if slot_for_face(m_id) == status.slot and (now - ts) <= FRESHNESS_S:
                    recent.append(fc)
            status.seen_by_recent = sorted(set(recent))

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def slot_status(self, slot: int) -> Optional[SlotStatus]:
        with self._lock:
            s = self._slots.get(int(slot))
            if s is None:
                return None
            return _copy(s)

    def snapshot(self) -> Dict[int, SlotStatus]:
        with self._lock:
            return {k: _copy(v) for k, v in self._slots.items()}

    def to_dict(self, *, our_team: str = "red") -> Dict[str, dict]:
        """JSON-friendly view, keyed by slot id.

        ``our_team`` is supplied so the UI can compute a "captured_by_us"
        flag without re-deriving holder semantics on the client side.
        """
        with self._lock:
            now = time.time()
            self._recompute_locked(now)
            enemy = "blue" if our_team == "red" else "red"
            out: Dict[str, dict] = {}
            for slot, s in self._slots.items():
                age = (
                    now - s.last_seen_unix_s if s.last_seen_unix_s > 0 else None
                )
                captured_by_us = s.holder == our_team
                captured_by_enemy = s.holder == enemy
                out[str(slot)] = {
                    "slot": s.slot,
                    "red_face_id": face_id(s.slot, "red"),
                    "blue_face_id": face_id(s.slot, "blue"),
                    "holder": s.holder,
                    "captured_by_us": captured_by_us,
                    "captured_by_enemy": captured_by_enemy,
                    "last_observed_face_id": s.last_observed_face_id,
                    "last_seen_unix_s": s.last_seen_unix_s,
                    "last_seen_age_s": age,
                    "last_seen_by": s.last_seen_by,
                    "red_face_seen_unix_s": s.red_face_seen_unix_s,
                    "blue_face_seen_unix_s": s.blue_face_seen_unix_s,
                    "seen_by_recent": list(s.seen_by_recent),
                    "holder_changed_unix_s": s.holder_changed_unix_s,
                    "lock_until_unix_s": s.lock_until_unix_s,
                    "locked": s.lock_until_unix_s > now,
                    "lock_remaining_s": (max(0.0, s.lock_until_unix_s - now)
                                         if s.lock_until_unix_s > 0 else 0.0),
                }
            return out

    # ------------------------------------------------------------------
    # Convenience for roles
    # ------------------------------------------------------------------

    def slot_locked(self, slot: int, now: Optional[float] = None) -> bool:
        """True while the slot is inside the 5-s post-capture lock window."""
        t = time.time() if now is None else now
        with self._lock:
            s = self._slots.get(int(slot))
            return bool(s and s.lock_until_unix_s > t)

    def slot_holder(self, slot: int) -> str:
        """Return the slot's current holder ("red"|"blue"|"unknown")."""
        with self._lock:
            s = self._slots.get(int(slot))
            return s.holder if s else "unknown"


def _copy(s: SlotStatus) -> SlotStatus:
    return SlotStatus(
        slot=s.slot,
        holder=s.holder,
        last_observed_face_id=s.last_observed_face_id,
        last_seen_unix_s=s.last_seen_unix_s,
        last_seen_by=s.last_seen_by,
        red_face_seen_unix_s=s.red_face_seen_unix_s,
        blue_face_seen_unix_s=s.blue_face_seen_unix_s,
        seen_by_recent=list(s.seen_by_recent),
        holder_changed_unix_s=s.holder_changed_unix_s,
        lock_until_unix_s=s.lock_until_unix_s,
    )
