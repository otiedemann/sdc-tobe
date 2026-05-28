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

# v7 §1.4.3 Special Maneuver ("sudden-death" / instant-win): holding ALL six
# boxes in our colour continuously for this many seconds ends the match.
SPECIAL_MANEUVER_HOLD_S = 5.0


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


def _slot_home_team(slot: int) -> str:
    """v7 §1.4.2 Table 1 — boxes 1-3 RED home, 4-6 BLUE home."""
    return "red" if 1 <= int(slot) <= 3 else "blue"


class MarkerTracker:
    """Thread-safe per-slot capture-state tracker + capture scoreboard."""

    def __init__(self, *, active_slots: Iterable[int] = ALL_SLOTS) -> None:
        self._lock = threading.RLock()
        self._slots: Dict[int, SlotStatus] = {}
        self._per_fc_last_seen: Dict[Tuple[str, int], float] = {}
        # v7 scoreboard (estimator — see _score_capture). Each entry counts
        # how many enemy-captures we've observed by that team. This is an
        # upper-bound estimate: it does not yet validate the "drone reached
        # home before recapture" rule, nor does it apply the 5/10-pt
        # bonuses (the 1-pt baseline is the conservative floor).
        self._caps: Dict[str, int] = {"red": 0, "blue": 0}
        # Last few capture events for the UI / future 10-pt detection.
        self._cap_events: List[dict] = []
        # v7 §1.4.3 Special Maneuver: holding ALL six boxes our colour for
        # ≥ 5 s ends the match (= "sudden-death" / instant win). We track
        # the timestamp the all-ours state first became true (per team);
        # cleared the instant any slot is not that team's colour. The
        # runner polls match_status(our_team, now) on every tick and emits
        # the "match_won" event once the dwell crosses the threshold.
        self._all_ours_since: Dict[str, float] = {"red": 0.0, "blue": 0.0}
        # True once we've emitted match_won for a given team this match —
        # we only emit it once so the event log doesn't fill up with
        # duplicates while the win condition continues to hold.
        self._won_emitted: Dict[str, bool] = {"red": False, "blue": False}
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
                    prev = status.holder
                    status.holder = team
                    status.holder_changed_unix_s = now
                    # v7: arm the 5-s capture lock the moment we observe a
                    # holder change. Slightly conservative (we observe a few
                    # frames after the actual flip), which is fine — it just
                    # tells the planner to skip this slot for the lock window.
                    status.lock_until_unix_s = now + CAPTURE_LOCK_S
                    # Score-keeping: an enemy capture (away from the slot's
                    # home colour) earns the attacking team 1 pt baseline.
                    # A home recapture earns 0 (regs §1.4.3) but we still
                    # log the event for the UI / future 10-pt detection.
                    self._record_capture(status.slot, prev, team, now, fc_name)
                self._per_fc_last_seen[(fc_name, mid)] = now
            # Recompute "seen_by_recent" for every active slot.
            self._recompute_locked(now)

    # v7 §1.4.3 sync window for double-strike detection. Two captures by
    # the same team on DIFFERENT slots within this window count as a
    # 10-pt event (5 pts each); both flips get retro-upgraded the moment
    # the second one lands.
    SYNC_WINDOW_S: float = 1.0

    def _record_capture(self, slot: int, prev: str, new: str,
                        now: float, observed_by: Optional[str]) -> None:
        """v7 scoring: record an observed holder change.

        v7 §1.4.3 awards points by play kind, not per flip:

            * **1 pt**   single drone captures an enemy box and returns home
            * **5 pts**  capture moment + return with all our drones outside
                         home (= "full sortie")
            * **10 pts** two different enemy boxes flipped by the same team
                         within ≤ 1 s (= "double-strike")

        The C2 doesn't see the drone-return condition directly, so 1-pt
        is the conservative floor. 10-pt is detectable here: when a new
        flip lands, scan the last SYNC_WINDOW_S of capture events; if
        the same team already flipped a DIFFERENT slot in that window,
        upgrade BOTH flips to 5 pts (= 10 pts total for the pair).

        Home recaptures (flip back to the slot's home team) are 0 pts.
        """
        home = _slot_home_team(slot)
        is_home_recap = (new == home)
        # Baseline: 1 pt for an enemy capture, 0 pts for a home recap.
        pts = 0 if is_home_recap else 1
        sync_pair_slot: Optional[int] = None
        if not is_home_recap and new in self._caps:
            # Look for a sync-pair partner — a flip by the same team, on
            # a DIFFERENT slot, inside the 1-s window.
            for prev_ev in reversed(self._cap_events):
                if now - prev_ev.get("unix_s", 0.0) > self.SYNC_WINDOW_S:
                    break
                if (prev_ev.get("new") == new
                        and prev_ev.get("slot") != int(slot)
                        and not prev_ev.get("home_recap")
                        and prev_ev.get("play_kind") != "double_strike_10"):
                    sync_pair_slot = int(prev_ev["slot"])
                    # Retro-upgrade the prior flip: was 1 pt, becomes 5.
                    bump = 5 - int(prev_ev.get("pts", 1))
                    if bump > 0:
                        self._caps[new] += bump
                    prev_ev["pts"] = 5
                    prev_ev["play_kind"] = "double_strike_10"
                    prev_ev["sync_pair_slot"] = int(slot)
                    pts = 5  # this flip also counts as the 5-pt half
                    break
            self._caps[new] += pts
        ev = {
            "unix_s": now, "slot": int(slot), "prev": prev, "new": new,
            "home_recap": is_home_recap, "observed_by": observed_by,
            "pts": pts,
            "play_kind": ("home_recap_0" if is_home_recap
                          else "double_strike_10" if sync_pair_slot is not None
                          else "single_strike_1"),
            "sync_pair_slot": sync_pair_slot,
        }
        self._cap_events.append(ev)
        # Keep only the most recent ~50 events.
        if len(self._cap_events) > 50:
            del self._cap_events[: len(self._cap_events) - 50]

    def score(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._caps)

    # ------------------------------------------------------------------
    # Special Maneuver / instant-win detection
    # ------------------------------------------------------------------

    def match_status(self, our_team: str, now: float) -> dict:
        """Return the v7 Special-Maneuver dwell state for ``our_team``.

        Returns a dict with::

            all_ours        bool     — all 6 slots are currently our colour
            since           float|0  — unix ts the all-ours state started
            dwell_s         float    — seconds the all-ours state has held
            threshold_s     float    — regs §1.4.3 = 5 s
            won             bool     — dwell_s >= threshold (instant win)
            just_won        bool     — first tick where ``won`` is True
                                       (use to emit a one-shot event)

        Mutates internal state: if the all-ours condition just became
        true we start the timer here; if it just became false we clear
        it. Idempotent within a tick (re-calling for the same team and
        the same world view returns the same answer).
        """
        with self._lock:
            our_team = (our_team or "").lower()
            if our_team not in ("red", "blue"):
                return {"all_ours": False, "since": 0.0, "dwell_s": 0.0,
                        "threshold_s": SPECIAL_MANEUVER_HOLD_S,
                        "won": False, "just_won": False}
            holders = [s.holder for s in self._slots.values()]
            all_ours = bool(holders) and all(h == our_team for h in holders)
            since = self._all_ours_since.get(our_team, 0.0)
            if all_ours and since <= 0.0:
                # First tick of all-ours — start the dwell timer.
                self._all_ours_since[our_team] = now
                since = now
            elif not all_ours and since > 0.0:
                # Lost the all-ours condition — reset both timer + won flag.
                self._all_ours_since[our_team] = 0.0
                self._won_emitted[our_team] = False
                since = 0.0
            dwell = (now - since) if (all_ours and since > 0.0) else 0.0
            won = bool(all_ours and dwell >= SPECIAL_MANEUVER_HOLD_S)
            just_won = bool(won and not self._won_emitted.get(our_team, False))
            if just_won:
                self._won_emitted[our_team] = True
            return {
                "all_ours": all_ours,
                "since": float(since),
                "dwell_s": float(dwell),
                "threshold_s": float(SPECIAL_MANEUVER_HOLD_S),
                "won": won,
                "just_won": just_won,
            }

    def capture_events(self, limit: int = 20) -> List[dict]:
        with self._lock:
            return list(self._cap_events[-limit:])

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
