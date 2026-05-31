"""Role definitions: the per-drone behaviour contract.

A *role* is a small state machine that, given the current world view, decides
whether to push a fresh mission script to the FC. The runner ticks every
known drone once per loop and asks its role to ``decide``.

Roles supported in this iteration:
- ``idle``     — no script, drone stays put on the ground.
- ``scout``    — fly to the neutral zone, rotate 360°, push marker
                 observations to the tracker (handled by the runner via the
                 C2 ``visible_marker_ids`` stream).
- ``attacker`` — given a target marker assigned by the operator, fly toward
                 it, capture it, and return home once a scout confirms the
                 marker has been knocked over.

Concrete role classes live in :mod:`.scout` and :mod:`.attacker`; this module
is the registry and shared dataclasses.
"""
from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .markers import MarkerTracker
from .settings import DroneSettings, MatchSettings

logger = logging.getLogger(__name__)


# Minimum interval between re-pushes of the same script (avoids hammering FC
# while it's transitioning between phases).
MIN_PUSH_INTERVAL_S = 2.0


# ---------------------------------------------------------------------------
# Plain-data view of a drone's C2 state (one item from /api/c2/overview).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DroneState:
    fc_name: str
    connection_ok: bool = False
    drone_connected: bool = False
    phase: str = "unknown"
    visible_marker_ids: tuple[int, ...] = ()
    active_marker_id: Optional[int] = None
    mission_running: bool = False
    # v7 §1.4.4 needs to know whether the drone is currently inside its
    # own home zone before allowing a new scoring attempt. We pull the
    # arena-frame position straight from /api/state (sim + real FC both
    # publish it) and the runner derives in_home_now from drone.team.
    world_position_m: Optional[tuple[float, float, float]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_overview(cls, fc_name: str, item: Dict[str, Any]) -> "DroneState":
        if not isinstance(item, dict):
            return cls(fc_name=fc_name, raw={})
        state = item.get("state") or {}
        if not isinstance(state, dict):
            state = {}
        vmids_raw = state.get("visible_marker_ids") or []
        try:
            vmids = tuple(int(x) for x in vmids_raw)
        except (TypeError, ValueError):
            vmids = ()
        phase = str(state.get("phase") or "unknown")
        active = state.get("active_marker_id")
        try:
            active_id = int(active) if active is not None else None
        except (TypeError, ValueError):
            active_id = None
        pos_raw = state.get("world_position_m")
        pos_t: Optional[tuple[float, float, float]] = None
        if isinstance(pos_raw, (list, tuple)) and len(pos_raw) >= 3:
            try:
                pos_t = (float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
            except (TypeError, ValueError):
                pos_t = None
        return cls(
            fc_name=fc_name,
            connection_ok=bool(item.get("connection_ok", False)),
            drone_connected=bool(item.get("drone_connected", False)),
            phase=phase,
            visible_marker_ids=vmids,
            active_marker_id=active_id,
            mission_running=phase not in ("", "unknown", "init", "idle", "landed"),
            world_position_m=pos_t,
            raw=item,
        )


# v7 §1.1 home-zone bounds (regs-fixed): the playing field is 20 m along
# the long (Y) axis split into red (5 m) + neutral (10 m) + blue (5 m), with
# the 10 m width spanning x in [-5, +5]. Used by the runner to decide whether
# a drone is currently INSIDE its own home zone for the §1.4.4 gate
# ("a drone may only start a new attempt after it has been detected back in
# its own home zone").
def in_home_zone(team: str, x: float, y: float) -> bool:
    if abs(x) > 5.0:
        return False
    if team == "red":
        return -10.0 <= y <= -5.0
    if team == "blue":
        return 5.0 <= y <= 10.0
    return False


def home_park_xy(team: str, lane: int = 0, n_lanes: int = 1) -> tuple[float, float]:
    """Where a drone parks when recalled home to BANK a scoring attempt.

    |y|=6 m — comfortably inside the home zone (y in ±[5,10]) yet ~1.5 m clear
    of the boxes at |y|=7.5 so a parked/rotating drone is never a "dead duck"
    over a target (regs §1.3). The x is SPREAD across the 10 m-wide home zone
    by ``lane`` (this drone's index) of ``n_lanes`` so the whole team doesn't
    converge on a single point during bank — stacking everyone on x=0 made
    them cross each other's altitude bands and triggered a collision-avoidance
    storm that stalled the next attack. Lanes span x in [-3.5, +3.5].
    """
    y = -6.0 if team == "red" else 6.0
    if n_lanes <= 1:
        return (0.0, y)
    span = 7.0   # -3.5 .. +3.5
    x = -span / 2.0 + span * (lane / (n_lanes - 1))
    return (round(x, 2), y)


def enemy_heading_deg(team: str) -> float:
    """Absolute arena heading (for the FC ``YAW`` verb) that faces the ENEMY.

    The arena long axis is Y; red's home is -Y and its targets are +Y, blue is
    the mirror. heading 0 deg = facing +Y, 180 deg = facing -Y. So red attacks
    facing 0, blue facing 180. Used to orient an attacker before its body-frame
    cruise so it never drives the wrong way (e.g. into the home wall) because
    of an inherited heading from a previous run / interrupted mission.
    """
    return 0.0 if team == "red" else 180.0


# ---------------------------------------------------------------------------
# Per-drone role-runtime state, kept in memory by the runner.
# ---------------------------------------------------------------------------


@dataclass
class RoleState:
    fc_name: str
    role: str = "idle"
    phase: str = "idle"
    target_slot: Optional[int] = None              # 1..6, set by operator
    target_assigned_unix_s: Optional[float] = None
    last_attack_marker_id: Optional[int] = None    # face id we last APPROACHed
    last_pushed_script: str = ""
    last_pushed_unix_s: float = 0.0
    last_decision_reason: str = ""
    phase_started_unix_s: float = field(default_factory=time.time)
    history: list[dict] = field(default_factory=list)

    def reset_for_role(self, new_role: str) -> None:
        self.role = new_role
        self.phase = "idle"
        self.target_slot = None
        self.target_assigned_unix_s = None
        self.last_attack_marker_id = None
        self.last_pushed_script = ""
        self.last_pushed_unix_s = 0.0
        self.last_decision_reason = "role changed"
        self.phase_started_unix_s = time.time()
        self.history.clear()

    def advance_phase(self, new_phase: str, reason: str = "") -> None:
        if new_phase == self.phase:
            return
        self.history.append(
            {
                "from": self.phase,
                "to": new_phase,
                "reason": reason,
                "unix_s": time.time(),
            }
        )
        self.phase = new_phase
        self.phase_started_unix_s = time.time()
        self.last_decision_reason = reason or self.last_decision_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fc_name": self.fc_name,
            "role": self.role,
            "phase": self.phase,
            "target_slot": self.target_slot,
            "target_assigned_unix_s": self.target_assigned_unix_s,
            "last_attack_marker_id": self.last_attack_marker_id,
            "last_pushed_script": self.last_pushed_script,
            "last_pushed_unix_s": self.last_pushed_unix_s,
            "last_pushed_age_s": (
                time.time() - self.last_pushed_unix_s
                if self.last_pushed_unix_s
                else None
            ),
            "last_decision_reason": self.last_decision_reason,
            "phase_age_s": time.time() - self.phase_started_unix_s,
            "history_tail": self.history[-5:],
        }


# ---------------------------------------------------------------------------
# Decision: what the runner should do for this drone right now.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    kind: str          # "push" | "stop" | "noop"
    script: str = ""
    new_phase: str = ""
    reason: str = ""


def noop(reason: str = "") -> Decision:
    return Decision(kind="noop", reason=reason)


def push(script: str, *, new_phase: str = "", reason: str = "") -> Decision:
    return Decision(kind="push", script=script, new_phase=new_phase, reason=reason)


def stop_cmd(reason: str = "") -> Decision:
    return Decision(kind="stop", reason=reason)


# ---------------------------------------------------------------------------
# Role context (everything a role needs to decide).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleContext:
    drone: DroneSettings
    match: MatchSettings
    state: DroneState
    markers: MarkerTracker
    role_state: RoleState
    our_team: str                     # "red" | "blue"
    active_slots: tuple[int, ...]     # which slots are in play (1..6)
    # Per-drone CRUISE altitude after height-deconfliction (distinct per
    # drone so two drones never share a cruise height — see runner's
    # deconflict step). None means "use the drone's own *_alt_m setting".
    # Roles should prefer this for their cruise/loiter HEIGHT; the final
    # capture descent still drops to the box-detect band regardless.
    cruise_alt_m: Optional[float] = None
    # v7 §1.4.4 gate: True iff drone.world_position_m is currently inside
    # this drone's TEAM home zone (red y in [-10,-5] / blue y in [5,10],
    # x in [-5,+5]). Computed by the runner from drone.team + state.world_
    # position_m. Roles must refuse to start a NEW scoring attempt while
    # this is False (the regs require the drone be detected back home).
    in_home_now: bool = False
    # Team-level scoring phase (set by the runner's team coordinator):
    #   "sortie" — attack: every drone OUT of our home zone (scout at centre,
    #              defender in neutral, attackers on enemy boxes). This is the
    #              state in which a capture earns 5 pts (all drones outside
    #              home at the moment of capture).
    #   "bank"   — we just captured an enemy box: recall EVERY drone home to
    #              secure the attempt (5 pts require all drones return home
    #              before the box is recaptured). No new attacks start.
    # Roles branch on this so the whole team pulses out-and-back together.
    team_phase: str = "sortie"
    # This drone's DISTINCT home-park point (x spread across the home zone by
    # the runner so the team doesn't pile on one spot during bank). Roles use
    # it for the bank/return-home target. None -> fall back to home_park_xy().
    home_park_xy: Optional[tuple[float, float]] = None


class Role(abc.ABC):
    """Base class for a role. Stateless — all state lives in ``RoleState``."""

    name: str = "abstract"

    @abc.abstractmethod
    def decide(self, ctx: RoleContext) -> Decision:
        """Return the action the runner should take for this drone now."""


# ---------------------------------------------------------------------------
# Registry. Concrete roles register themselves at module-import time.
# ---------------------------------------------------------------------------


_REGISTRY: Dict[str, Role] = {}


def register(role: Role) -> Role:
    _REGISTRY[role.name] = role
    return role


def get(name: str) -> Optional[Role]:
    return _REGISTRY.get(name)


def all_roles() -> Dict[str, Role]:
    return dict(_REGISTRY)


# Idle role: never pushes anything. Used as the default safe state.


class _IdleRole(Role):
    name = "idle"

    def decide(self, ctx: RoleContext) -> Decision:
        if ctx.role_state.phase != "idle":
            ctx.role_state.advance_phase("idle", "role=idle")
        return noop("idle")


register(_IdleRole())
