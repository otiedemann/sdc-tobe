"""SDC26 match-level state machine — aligned with SDC Regulations v7 (v5.0.0).

Runs once per SwarmRunner tick BEFORE individual role decisions.
Transitions between macro phases and coordinates global strategy
decisions (wave dispatch, Special Maneuver push, time management).

Regulation references (§ = SDC Regulations v5.0.0 / "v7"):
  §1.1  Playing field: 20 m × 10 m × 6 m.
          Red zone (5 m) | Neutral zone (10 m) | Blue zone (5 m).
  §1.2  Target objects: 6 boxes total; slots 1-3 in red zone (default red,
          marker IDs 41-43), slots 4-6 in blue zone (default blue, IDs 34-36).
          RFID detection height: ~1-2 m above box.
  §1.3  Time Limit: 10 minutes (600 s). Match ends when time elapses or
          Special Maneuver triggers.
  §1.4.3  Capture mechanics:
            - drone must hover ≥ MIN_CAPTURE_HOVER_S (2 s) over the box
            - no defending drone (RFID) may be detected at that box
            - box is LOCKED for BOX_LOCK_S (5 s) after a successful capture
            - home-zone recaptures are IMMEDIATE but score 0 game points
  §1.4.3  Scoring:
            1 pt  — capture enemy box; capturing drone returns home before
                    that box is recaptured by the enemy.
            5 pts — capture enemy box AND at the moment of capture ALL team
                    drones are OUTSIDE the home zone; then all drones return
                    home before recapture. Requires ≥2 drones on the team.
            10 pts — two DIFFERENT drones capture two DIFFERENT enemy boxes
                     within < 1 second of each other. Requires ≥2 drones.
                     NOTE: our coordinator dispatches on the same 1-second tick
                     (same departure), but arrival at different boxes is NOT
                     guaranteed to be within 1 s. The <1 s window may not be
                     reliably met. No travel-time compensation is implemented.
  §1.4.1  Special Maneuver (instant win):
            control ALL 6 boxes continuously for ≥ SPECIAL_MANEUVER_HOLD_S
            (5 s) → match immediately won. A 10-second resolution window
            (INSTANT_WIN_RESOLUTION_S) follows for open scoring attempts.
  §1.3/§1.4  "Sitting dead duck" (passive continuous hovering directly OVER
              boxes) is EXPLICITLY PROHIBITED → disqualification. All drone
              flyovers must be brief passes, never continuous hover on a box.

Phase flow::

    PRE_MATCH
        │ arm()
        ▼
    SCOUTING ──── ≥1 enemy slot observed ──→ WAVE_ATTACK
        ▲                                        │
        │                           all attackers still │
        │                           airborne            ▼
        └───────────────────────── RECOVERY_WAIT (rearm)
                                                  │ ≥ MIN_WAVE_DRONES idle
                                                  ▼
                                            WAVE_ATTACK (loop)

    any phase ──── 5/6 slots ours & ≥1 enemy remaining ──→ INSTANT_WIN_PUSH
    INSTANT_WIN_PUSH  ──── all 6 acquired ───────────────→ HOLDING_ALL
    HOLDING_ALL ──── held < SPECIAL_MANEUVER_HOLD_S (5 s) → (wait)
    HOLDING_ALL ──── held ≥ 5 s ─────────────────────────→ (log Special Maneuver)
    HOLDING_ALL ──── enemy recaptures ───────────────────→ WAVE_ATTACK
    any phase ──── match_elapsed ≥ 570 s ────────────────→ END_LAND

Notes on 5-pt capture (§1.4.3):
  All team drones must be outside home zone at the MOMENT of capture.
  The coordinator launches all roles simultaneously, so scouts/anchors
  are normally already airborne. Condition is met as long as no drone
  explicitly parks on the home pad while an attack is in progress.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from .markers import MarkerTracker
    from .runner import SwarmRunner
    from .settings import StrategySettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regulation-derived constants  (§ = SDC Regulations v5.0.0 / "v7")
#
# Key changes from sdc_regulations_v50.pdf → SDC_Regulations_v7.pdf:
#
#   1. Detection: "NFT tag" (v50) → "RFID tag" (v7).  Code already uses RFID.
#
#   2. Target positions: v50 required positions to VARY per match so teams
#      could not pre-program fixed setups. v7 REMOVED this rule — target
#      positions are NOW FIXED across matches. Flight scripts may therefore
#      use pre-configured slot coordinates.
#
#   3. Capture stability vs box lock:
#      v50: box had to remain in capturing team's colour for ≥5 s to
#           "validate" the capture (stability window).
#      v7:  capture is INSTANT once drone hovers ≥2 s (RFID detected) AND
#           no defending drone is present; box is then LOCKED for 5 s —
#           meaning the opposing team cannot immediately re-capture it.
#           → BOX_LOCK_S = 5.0 implements the v7 lock, not a validation wait.
#
#   4. Home zone validation: v50 required the drone to "enter and remain
#      inside [home zone] for at least 5 seconds."  v7 only requires the
#      drone to "return home" with no dwell time specified.
#      → HOME_ZONE_VALIDATION_S = 0 (removed in v7).
#
#   5. 5-pt rule changed:
#      v50: "Single drone captures box, entire team returns to home base
#            together."
#      v7:  "At the moment of capture, ALL drones of your team are OUTSIDE
#            your home zone; then all return home before that box is
#            recaptured." Requires ≥2 drones.
#
#   6. §1.4.4 Additional rules (v7 only):
#      - One drone can only run ONE scoring attempt at a time.
#      - A drone may only start a NEW attempt after detected back in its
#        own home zone (RTH required between attempts).
#      - One capture attempt scores ONCE: 1, 5, or 10 pts — no stacking.
#      - Priority: Special Maneuver > 10 > 5 > 1.
#
#   7. Challenge points (v7 Table 2 has an internal discrepancy):
#      Changelog states max 25 pts = 15 match + 5 jury + 5 report.
#      Table 2 shows jury = 10, total = 25 (15+10+5=30 ≠ 25).
#      Text says max 25. Jury is likely 5 pts; the "10" in Table 2 appears
#      to be a typo.  Not modelled in strategy code.
# ---------------------------------------------------------------------------

# §1.4.3  Minimum RFID hover time required for a capture to register.
#         (v50 also had 2 s; unchanged.)
MIN_CAPTURE_HOVER_S: float = 2.0

# §1.4.3  How long a box is LOCKED (cannot be recaptured) after a successful
#         capture. (v50 had a "capture stability" window of the same 5 s but
#         with different semantics — v7 clarifies this as a lock, not a
#         validation hold.)
BOX_LOCK_S: float = 5.0

# §1.4.3 / v7 change: home zone validation dwell time.
#         v50 required drone to stay ≥5 s in home zone; v7 removed this.
#         Set to 0 to document the removal.  The FC's home-zone detection
#         (RFID reader at take-off pad) triggers immediately on arrival.
HOME_ZONE_VALIDATION_S: float = 0.0  # v7: no dwell requirement

# §1.4.1  How long ALL 6 boxes must be held continuously to trigger the
#         Special Maneuver instant win.
SPECIAL_MANEUVER_HOLD_S: float = 5.0

# §1.4.1  Resolution window after instant win before match is fully over.
INSTANT_WIN_RESOLUTION_S: float = 10.0

# §1.3  Total match duration in seconds.
MATCH_DURATION_S: float = 600.0

# Land-safe margin: stop dispatching new attacks this many seconds before end.
SAFE_LAND_MARGIN_S: float = 30.0

# Minimum idle attackers required to send a wave (1 = always send; ≥2 needed
# for the 10-pt simultaneous bonus, but we don't wait for it).
MIN_WAVE_DRONES: int = 1

# Cooldown (s) between coordinator assignment decisions to prevent hammering.
COORDINATOR_COOLDOWN_S: float = 2.0

# §1.4.4  Game point priority for the same capture event (v7 only):
#   Special Maneuver > 10 > 5 > 1.  Scores do NOT stack for the same attempt.
SCORE_PRIORITY = ("special_maneuver", "10pt", "5pt", "1pt")

# Slot grouping per §1.2 / Table 1:
#   slots 1-3 are in the RED team zone (default colour = red, marker IDs 41-43)
#   slots 4-6 are in the BLUE team zone (default colour = blue, marker IDs 34-36)
#   Target positions are FIXED across matches (v7 removed the variable-position
#   rule that existed in v50).
RED_ZONE_SLOTS: tuple[int, ...] = (1, 2, 3)
BLUE_ZONE_SLOTS: tuple[int, ...] = (4, 5, 6)
ALL_SLOTS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


# ---------------------------------------------------------------------------
# Phase enum & snapshot
# ---------------------------------------------------------------------------


class MatchPhase(str, Enum):
    PRE_MATCH = "pre_match"
    SCOUTING = "scouting"
    WAVE_ATTACK = "wave_attack"
    INSTANT_WIN_PUSH = "instant_win_push"   # ≥5/6 slots ours, pushing for last
    HOLDING_ALL = "holding_all"             # all 6 ours; counting 5-s timer
    RECOVERY_WAIT = "recovery_wait"         # all attackers airborne; wait for RTH
    END_LAND = "end_land"                   # near time limit; no new attacks


@dataclass
class MatchStateSnapshot:
    phase: str
    phase_entered_unix_s: float
    match_started_unix_s: Optional[float]
    match_elapsed_s: Optional[float]
    slots_ours: List[int]
    slots_enemy: List[int]
    slots_unknown: List[int]
    idle_attackers: List[str]
    running_attackers: List[str]
    scout_fcs: List[str]
    anchor_fcs: List[str]
    last_coordinator_unix_s: Optional[float]
    wave_count: int
    points_estimated: int
    # Special Maneuver tracking
    all_slots_hold_s: Optional[float]    # seconds we have held all 6 slots
    special_maneuver_achieved: bool      # True once 5-s hold completed


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class MatchStateMachine:
    """Runs on every SwarmRunner tick; assigns targets to idle attackers
    based on match phase and MarkerTracker state.

    Must be called from the async loop (SwarmRunner._tick_once) AFTER
    the C2 overview has been refreshed and markers ingested.
    """

    def __init__(self) -> None:
        self._phase = MatchPhase.PRE_MATCH
        self._phase_entered_unix_s: float = time.time()
        self._match_started_unix_s: Optional[float] = None
        self._last_coordinator_unix_s: float = 0.0
        self._wave_count: int = 0
        self._points_estimated: int = 0
        self._last_wave_slots: Set[int] = set()
        # Special Maneuver: when did we first hold all 6 slots in current run?
        self._all_slots_since: Optional[float] = None
        self._special_maneuver_achieved: bool = False

    # ------------------------------------------------------------------ public

    def on_armed(self) -> None:
        """Called when the SwarmRunner is armed by the operator."""
        if self._phase == MatchPhase.PRE_MATCH:
            self._match_started_unix_s = time.time()
            self._all_slots_since = None
            self._special_maneuver_achieved = False
            self._transition(MatchPhase.SCOUTING, "runner armed")

    def on_disarmed(self) -> None:
        """Called when the runner is disarmed."""
        self._transition(MatchPhase.PRE_MATCH, "runner disarmed")
        self._match_started_unix_s = None
        self._all_slots_since = None
        self._special_maneuver_achieved = False

    @property
    def phase(self) -> MatchPhase:
        return self._phase

    @property
    def match_elapsed_s(self) -> Optional[float]:
        if self._match_started_unix_s is None:
            return None
        return time.time() - self._match_started_unix_s

    def snapshot(self, *,
                 markers: "MarkerTracker",
                 settings: "StrategySettings",
                 runner: "SwarmRunner") -> MatchStateSnapshot:
        s = settings
        our_team = s.markers.our_team
        active = list(s.markers.active_slots)
        slots_ours = [sl for sl in active if markers.slot_holder(sl) == our_team]
        enemy_team = "blue" if our_team == "red" else "red"
        slots_enemy = [sl for sl in active if markers.slot_holder(sl) == enemy_team]
        slots_unknown = [sl for sl in active if markers.slot_holder(sl) == "unknown"]

        rs_map = runner.role_states()
        attacker_fcs = [d.fc_name for d in s.drones
                        if d.role == "attacker" and d.enabled]
        idle_atk = [fc for fc in attacker_fcs
                    if getattr(rs_map.get(fc), "phase", "") in ("", "idle", "done")]
        running_atk = [fc for fc in attacker_fcs if fc not in idle_atk]
        scout_fcs = [d.fc_name for d in s.drones if d.role == "scout" and d.enabled]
        anchor_fcs = [d.fc_name for d in s.drones if d.role == "anchor" and d.enabled]

        all_slots_hold_s: Optional[float] = None
        if self._all_slots_since is not None:
            all_slots_hold_s = time.time() - self._all_slots_since

        return MatchStateSnapshot(
            phase=self._phase.value,
            phase_entered_unix_s=self._phase_entered_unix_s,
            match_started_unix_s=self._match_started_unix_s,
            match_elapsed_s=self.match_elapsed_s,
            slots_ours=slots_ours,
            slots_enemy=slots_enemy,
            slots_unknown=slots_unknown,
            idle_attackers=idle_atk,
            running_attackers=running_atk,
            scout_fcs=scout_fcs,
            anchor_fcs=anchor_fcs,
            last_coordinator_unix_s=self._last_coordinator_unix_s or None,
            wave_count=self._wave_count,
            points_estimated=self._points_estimated,
            all_slots_hold_s=all_slots_hold_s,
            special_maneuver_achieved=self._special_maneuver_achieved,
        )

    # ------------------------------------------------------------------ tick

    def tick(self, *,
             runner: "SwarmRunner",
             markers: "MarkerTracker",
             settings: "StrategySettings") -> None:
        """Called once per SwarmRunner tick (1 s by default).

        Mutates per-drone RoleState via runner.assign_target() when
        appropriate. Transitions the match phase.
        """
        if not runner.is_armed():
            if self._phase != MatchPhase.PRE_MATCH:
                self.on_disarmed()
            return

        if self._phase == MatchPhase.PRE_MATCH:
            self.on_armed()
            return

        s = settings
        our_team = s.markers.our_team
        active = list(s.markers.active_slots)

        # ── §1.3 time management ──────────────────────────────────────
        elapsed = self.match_elapsed_s or 0.0
        if elapsed >= MATCH_DURATION_S - SAFE_LAND_MARGIN_S:
            if self._phase != MatchPhase.END_LAND:
                self._transition(MatchPhase.END_LAND,
                                 f"match near end ({elapsed:.0f}/{MATCH_DURATION_S:.0f}s)")
            return  # no new attacks in END_LAND

        # ── classify slots ────────────────────────────────────────────
        enemy_team = "blue" if our_team == "red" else "red"
        enemy_slots = [sl for sl in active if markers.slot_holder(sl) == enemy_team]
        our_slots = [sl for sl in active if markers.slot_holder(sl) == our_team]
        total = len(active)

        # ── §1.4.1 Special Maneuver tracking: all-6 hold timer ───────
        if len(our_slots) == total and total > 0:
            # We hold every active slot.
            if self._all_slots_since is None:
                self._all_slots_since = time.time()
                logger.info("match_state: holding ALL %d slots — starting "
                            "%.0f-second Special Maneuver timer",
                            total, SPECIAL_MANEUVER_HOLD_S)
                runner.events.add(
                    "coordinator",
                    f"ALL {total} slots captured — Special Maneuver timer started "
                    f"({SPECIAL_MANEUVER_HOLD_S:.0f}s needed)",
                )
            hold_s = time.time() - self._all_slots_since
            if hold_s >= SPECIAL_MANEUVER_HOLD_S and not self._special_maneuver_achieved:
                self._special_maneuver_achieved = True
                logger.info(
                    "match_state: SPECIAL MANEUVER achieved — all %d slots "
                    "held for ≥%.0f s → instant win!",
                    total, SPECIAL_MANEUVER_HOLD_S,
                )
                runner.events.add(
                    "coordinator",
                    f"*** SPECIAL MANEUVER *** all {total} slots held ≥"
                    f"{SPECIAL_MANEUVER_HOLD_S:.0f}s → INSTANT WIN",
                )
            if self._phase != MatchPhase.HOLDING_ALL:
                self._transition(MatchPhase.HOLDING_ALL,
                                 f"all {total} slots ours — holding for Special Maneuver")
        else:
            # Lost at least one slot — reset the hold timer.
            if self._all_slots_since is not None:
                logger.info("match_state: lost hold of all slots — resetting timer")
                runner.events.add("coordinator",
                                  "lost all-slot hold — Special Maneuver timer reset")
            self._all_slots_since = None

            # ── §1.4.1 Instant-win push: 5/6 → push for last box ─────
            if len(our_slots) >= total - 1 and enemy_slots:
                if self._phase not in (MatchPhase.INSTANT_WIN_PUSH,
                                       MatchPhase.HOLDING_ALL):
                    self._transition(MatchPhase.INSTANT_WIN_PUSH,
                                     f"{len(our_slots)}/{total} slots ours; "
                                     f"push for last: {enemy_slots}")
            elif self._phase in (MatchPhase.INSTANT_WIN_PUSH,
                                  MatchPhase.HOLDING_ALL):
                # Lost instant-win position after enemy recapture.
                self._transition(MatchPhase.WAVE_ATTACK,
                                 "lost instant-win position — back to wave attack")

        # ── phase-specific tick ───────────────────────────────────────
        if self._phase == MatchPhase.SCOUTING:
            self._tick_scouting(runner, markers, s, enemy_slots)
        elif self._phase == MatchPhase.WAVE_ATTACK:
            self._tick_wave_attack(runner, markers, s, enemy_slots)
        elif self._phase == MatchPhase.INSTANT_WIN_PUSH:
            self._tick_instant_win(runner, markers, s, enemy_slots)
        elif self._phase == MatchPhase.HOLDING_ALL:
            self._tick_holding_all(runner, markers, s)
        elif self._phase == MatchPhase.RECOVERY_WAIT:
            self._tick_recovery(runner, markers, s, enemy_slots)

    # ---------------------------------------------------------------- phases

    def _tick_scouting(self,
                       runner: "SwarmRunner",
                       markers: "MarkerTracker",
                       s: "StrategySettings",
                       enemy_slots: List[int]) -> None:
        """Wait for the scout to observe at least one enemy slot."""
        if enemy_slots:
            self._transition(MatchPhase.WAVE_ATTACK,
                             f"scout found enemy slots: {enemy_slots}")

    def _tick_wave_attack(self,
                          runner: "SwarmRunner",
                          markers: "MarkerTracker",
                          s: "StrategySettings",
                          enemy_slots: List[int]) -> None:
        """Dispatch idle attackers to enemy slots.

        Dispatches all idle attackers simultaneously on the same tick so
        they depart together. Whether the <1 s simultaneous-capture window
        (§1.4.3, 10-pt rule) is achieved depends on flight geometry —
        no travel-time compensation is implemented.
        """
        now = time.time()
        if now - self._last_coordinator_unix_s < COORDINATOR_COOLDOWN_S:
            return

        rs_map = runner.role_states()
        attacker_fcs = [d.fc_name for d in s.drones
                        if d.role == "attacker" and d.enabled]
        idle_atk = [fc for fc in attacker_fcs
                    if getattr(rs_map.get(fc), "phase", "") in ("", "idle", "done")]

        if not idle_atk or not enemy_slots:
            return

        # One attacker per enemy slot (zip truncates to the shorter list).
        targets = sorted(enemy_slots)[:len(idle_atk)]
        if len(targets) < MIN_WAVE_DRONES:
            return

        wave_size = len(targets)
        # §1.4.3: 10-pt bonus if ≥2 drones capture ≥2 different boxes
        # within <1 s. We dispatch simultaneously; arrival simultaneity
        # is best-effort only.
        if wave_size >= 2:
            bonus_label = "SIMULTANEOUS launch (10-pt attempt §1.4.3)"
        else:
            bonus_label = "single attacker (1-pt §1.4.3)"

        logger.info("coordinator: wave #%d — %s — %s → slots %s",
                    self._wave_count + 1, bonus_label,
                    [fc for fc in idle_atk[:wave_size]], targets)

        for fc, slot in zip(idle_atk, targets):
            runner.assign_target(fc, slot)
            runner.events.add(
                "coordinator",
                f"wave #{self._wave_count + 1}: {fc} → slot {slot} [{bonus_label}]",
                drone=fc,
            )

        self._wave_count += 1
        self._last_wave_slots = set(targets)
        self._last_coordinator_unix_s = now
        # Conservative estimate; actual pts depend on RTH + recapture race.
        self._points_estimated += 10 if wave_size >= 2 else 1

    def _tick_instant_win(self,
                          runner: "SwarmRunner",
                          markers: "MarkerTracker",
                          s: "StrategySettings",
                          enemy_slots: List[int]) -> None:
        """Push ALL available drones onto remaining enemy slot(s) to achieve
        full-board control and start the Special Maneuver hold timer (§1.4.1).
        """
        now = time.time()
        if now - self._last_coordinator_unix_s < COORDINATOR_COOLDOWN_S:
            return

        if not enemy_slots:
            # All slots captured — the tick() logic above will handle
            # transitioning to HOLDING_ALL on next tick.
            return

        rs_map = runner.role_states()
        # Use both attackers and anchors for the final push.
        push_fcs = [d.fc_name for d in s.drones
                    if d.role in ("attacker", "anchor") and d.enabled]
        idle_all = [fc for fc in push_fcs
                    if getattr(rs_map.get(fc), "phase", "") in ("", "idle", "done")]

        if not idle_all:
            return

        # Pile every available drone onto the first enemy slot.
        target = sorted(enemy_slots)[0]
        for fc in idle_all:
            runner.assign_target(fc, target)
            runner.events.add(
                "coordinator",
                f"INSTANT WIN push: {fc} → slot {target} "
                f"(§1.4.1: {len(enemy_slots)} slot(s) remaining)",
                drone=fc,
            )
        self._last_coordinator_unix_s = now

    def _tick_holding_all(self,
                          runner: "SwarmRunner",
                          markers: "MarkerTracker",
                          s: "StrategySettings") -> None:
        """All 6 slots are ours. Monitor the §1.4.1 Special Maneuver
        SPECIAL_MANEUVER_HOLD_S (5 s) continuous-hold requirement.

        The state machine already logs when the timer completes and fires
        the special-maneuver announcement. This tick just ensures no new
        unnecessary attacks are dispatched while we hold.

        If the hold is broken (enemy recapture), the parent tick() already
        handles the transition back to WAVE_ATTACK.
        """
        if self._special_maneuver_achieved:
            # Nothing more to do — match win has been announced.
            return
        if self._all_slots_since is not None:
            hold_s = time.time() - self._all_slots_since
            remaining = max(0.0, SPECIAL_MANEUVER_HOLD_S - hold_s)
            if int(hold_s) % 1 == 0:  # log roughly every second
                logger.debug("match_state: HOLDING_ALL %.1f / %.0f s "
                             "(%.1f s remaining for Special Maneuver)",
                             hold_s, SPECIAL_MANEUVER_HOLD_S, remaining)

    def _tick_recovery(self,
                       runner: "SwarmRunner",
                       markers: "MarkerTracker",
                       s: "StrategySettings",
                       enemy_slots: List[int]) -> None:
        """Wait until ≥ MIN_WAVE_DRONES attackers have returned home (idle)."""
        rs_map = runner.role_states()
        attacker_fcs = [d.fc_name for d in s.drones
                        if d.role == "attacker" and d.enabled]
        idle_atk = [fc for fc in attacker_fcs
                    if getattr(rs_map.get(fc), "phase", "") in ("", "idle", "done")]
        if len(idle_atk) >= MIN_WAVE_DRONES and enemy_slots:
            self._transition(MatchPhase.WAVE_ATTACK,
                             f"{len(idle_atk)} attacker(s) ready, "
                             f"{len(enemy_slots)} enemy slot(s) available")

    # ----------------------------------------------------------------- helpers

    def _transition(self, new_phase: MatchPhase, reason: str) -> None:
        if new_phase == self._phase:
            return
        logger.info("match_state: %s → %s (%s)",
                    self._phase.value, new_phase.value, reason)
        self._phase = new_phase
        self._phase_entered_unix_s = time.time()
