"""Operator-tunable strategy settings.

Stored in its own JSON file so :class:`marker_mission_c2.config.C2Config`
(which describes the FCs, ports, polls, etc.) doesn't need to grow new
fields every time the strategy gets a new knob. The two concerns also
have very different change frequencies — fleet topology is set up once
per box, team colour / per-drone roles change match-to-match.

The single canonical knob is ``team_color`` ∈ ``{red, blue}``. From
that we derive ``own_target_ids`` / ``enemy_target_ids`` using the
SDC26 ID convention baked into ``tools/sphinx-arena/default_target_layout.json``:

    red  team → 41 42 43 (live) + 44 45 46 (spares)
    blue team → 31 32 33 (live) + 34 35 36 (spares)

Operators can override either set explicitly via the JSON, or flip the
``live_targets_only`` flag to ignore spares — useful when the planner
should only consider boxes that were physically deployed.

Resolution order, like :mod:`marker_mission_c2.config`:

    1. explicit ``path=`` argument to :func:`load`
    2. ``$MARKER_MISSION_C2_STRATEGY_SETTINGS`` env var
    3. ``./marker_mission_c2/strategy/settings.json``
    4. ``./marker_mission_c2/strategy/settings.example.json``
    5. defaults (red team, live targets only, no per-FC roles)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple


# Canonical SDC26 target ID ranges. The xx1/xx2/xx3 ids are the live
# boxes; xx4/xx5/xx6 are spares (``enabled=false`` in
# default_target_layout.json). Wall markers occupy 1..16 — those aren't
# owned by either team.
RED_TARGET_IDS:  FrozenSet[int] = frozenset(range(41, 47))   # 41..46
BLUE_TARGET_IDS: FrozenSet[int] = frozenset(range(31, 37))   # 31..36
RED_LIVE_IDS:    FrozenSet[int] = frozenset({41, 42, 43})
BLUE_LIVE_IDS:   FrozenSet[int] = frozenset({31, 32, 33})
WALL_MARKER_IDS: FrozenSet[int] = frozenset(range(1, 17))    # 1..16


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TeamColor(str, Enum):
    RED = "red"
    BLUE = "blue"

    @classmethod
    def parse(cls, raw) -> "TeamColor":
        if isinstance(raw, cls):
            return raw
        s = str(raw or "").strip().lower()
        if s in ("red", "r"):
            return cls.RED
        if s in ("blue", "b"):
            return cls.BLUE
        raise ValueError(
            f"unknown team color {raw!r} — use 'red' or 'blue'"
        )


class Role(str, Enum):
    """What a drone is supposed to be doing right now.

    The planner uses this to pick the right task family. ``IDLE`` is
    the fallback for drones the operator hasn't assigned a job to yet
    — they sit on the ground or hover in place.
    """
    ATTACKER = "attacker"
    SCOUT = "scout"
    DEFENDER = "defender"
    IDLE = "idle"

    @classmethod
    def parse(cls, raw) -> "Role":
        if isinstance(raw, cls):
            return raw
        s = str(raw or "idle").strip().lower()
        for r in cls:
            if r.value == s:
                return r
        raise ValueError(
            f"unknown role {raw!r} — use attacker / scout / defender / idle"
        )


# ---------------------------------------------------------------------------
# Sub-dataclasses (one per concern, kept flat-ish for readable JSON)
# ---------------------------------------------------------------------------

@dataclass
class ArenaSettings:
    """Arena geometry the strategy treats as ground truth.

    Width is along X (red on -X, blue on +X). Depth is along Y. Origin
    is the floor centre, matching marker_mission's convention.

    ``safety_margin_m`` is how close to a wall the planner allows
    before clipping back. Geofence enforcement lives in
    :mod:`marker_mission_c2.strategy.safety`; this field is its
    single source of truth.

    Each zone is a 2D **rectangle** defined by an X-range and a Y-range
    (both inclusive). The X-ranges partition the arena into three
    bands across width; the Y-ranges further restrict each zone to a
    sub-span of depth (default: the full depth, ``-D/2 .. +D/2``, so
    zones behave exactly like the old full-depth bands unless the
    operator opts in).

      red_home_x_m / red_home_y_m       red's territory (LEFT, -X side)
      neutral_zone_x_m / neutral_y_m    middle band (scouts park here)
      blue_home_x_m / blue_home_y_m     blue's territory (RIGHT, +X side)

    Helpers on :class:`StrategySettings` pick the "ours" / "theirs"
    rectangle based on ``team_color`` so the rest of the code is
    team-agnostic.
    """
    width_m: float = 10.0
    depth_m: float = 20.0
    safety_margin_m: float = 0.5
    red_home_x_m:     Tuple[float, float] = (-5.0, -2.0)
    red_home_y_m:     Tuple[float, float] = (-10.0, 10.0)
    blue_home_x_m:    Tuple[float, float] = ( 2.0,  5.0)
    blue_home_y_m:    Tuple[float, float] = (-10.0, 10.0)
    neutral_zone_x_m: Tuple[float, float] = (-2.0,  2.0)
    neutral_zone_y_m: Tuple[float, float] = (-10.0, 10.0)


@dataclass
class DroneAssignment:
    """Per-FC role + cruise altitude.

    Collision avoidance is primarily by altitude layering — each
    drone has its own ``altitude_m`` so two drones cruising the same
    X-Y line stay safely above / below each other.

    Roles drive task selection in the planner (attacker / scout /
    defender / idle).
    """
    role: Role = Role.IDLE
    altitude_m: float = 1.5


@dataclass
class AttackSettings:
    """Parameters for the 10-point simultaneous-strike maneuver.

    Attackers stage at ``hover_alt_m`` above their assigned target,
    wait for their partner to be in position (within
    ``sync_window_s`` seconds), then both drop to ``strike_alt_m``
    simultaneously to score the 10-point bonus, then RTH.

    ``pair_targets_override`` lets you pin which marker *pairs*
    qualify (e.g. only diagonal pairs). ``None`` derives every
    2-combination of our own live targets automatically.

    ``home_zone_clear`` makes the attack abort if any *other* friendly
    drone is inside our home zone — the operator's rule that the only
    friendlies allowed in the home zone during the strike are the
    two attackers.
    """
    hover_alt_m: float = 3.5
    strike_alt_m: float = 1.5
    sync_window_s: float = 1.5
    pair_targets_override: Optional[List[List[int]]] = None
    home_zone_clear: bool = True


@dataclass
class DefenderSettings:
    """Reactive defender behaviour.

    When an enemy drone gets within ``intercept_radius_m`` (metres,
    Euclidean) of one of our own targets, the defender abandons its
    neutral-zone wait and approaches that target to reclaim it.
    """
    intercept_radius_m: float = 2.5


@dataclass
class MatchSettings:
    """Match-clock configuration. The clock state itself
    (start_time, running) is runtime-only — see
    :class:`marker_mission_c2.strategy.match.MatchState` — but the
    *duration* of a match is config the operator sets up before
    pressing Start."""
    duration_s: float = 600.0   # 10 min default, tweak in the UI / JSON


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------

@dataclass
class StrategySettings:
    """Editable knobs that drive the planner / safety / runner.

    Loaded once at startup and exposed via :func:`load` /
    :func:`save`. The runner holds one instance and gives the
    planner/safety a read-only reference. To change at runtime,
    mutate the instance and the next tick will see the new values
    (no re-instantiation needed — every derived property is
    computed on access).
    """

    # ---- team ----
    team_color: TeamColor = TeamColor.RED
    own_target_ids_override:   Optional[FrozenSet[int]] = None
    enemy_target_ids_override: Optional[FrozenSet[int]] = None
    live_targets_only: bool = True

    # ---- nested sections ----
    arena:    ArenaSettings    = field(default_factory=ArenaSettings)
    drones:   Dict[str, DroneAssignment] = field(default_factory=dict)
    attack:   AttackSettings   = field(default_factory=AttackSettings)
    defender: DefenderSettings = field(default_factory=DefenderSettings)
    match:    MatchSettings    = field(default_factory=MatchSettings)

    # ---------------- derived: target id sets ----------------

    @property
    def own_target_ids(self) -> FrozenSet[int]:
        if self.own_target_ids_override is not None:
            return frozenset(self.own_target_ids_override)
        full = (RED_TARGET_IDS if self.team_color == TeamColor.RED
                else BLUE_TARGET_IDS)
        if not self.live_targets_only:
            return full
        live = (RED_LIVE_IDS if self.team_color == TeamColor.RED
                else BLUE_LIVE_IDS)
        return frozenset(full & live)

    @property
    def enemy_target_ids(self) -> FrozenSet[int]:
        if self.enemy_target_ids_override is not None:
            return frozenset(self.enemy_target_ids_override)
        full = (BLUE_TARGET_IDS if self.team_color == TeamColor.RED
                else RED_TARGET_IDS)
        if not self.live_targets_only:
            return full
        live = (BLUE_LIVE_IDS if self.team_color == TeamColor.RED
                else RED_LIVE_IDS)
        return frozenset(full & live)

    # ---------------- derived: zones ----------------

    @property
    def our_home_x_m(self) -> Tuple[float, float]:
        return (self.arena.red_home_x_m if self.team_color == TeamColor.RED
                else self.arena.blue_home_x_m)

    @property
    def our_home_y_m(self) -> Tuple[float, float]:
        return (self.arena.red_home_y_m if self.team_color == TeamColor.RED
                else self.arena.blue_home_y_m)

    @property
    def enemy_home_x_m(self) -> Tuple[float, float]:
        return (self.arena.blue_home_x_m if self.team_color == TeamColor.RED
                else self.arena.red_home_x_m)

    @property
    def enemy_home_y_m(self) -> Tuple[float, float]:
        return (self.arena.blue_home_y_m if self.team_color == TeamColor.RED
                else self.arena.red_home_y_m)

    @staticmethod
    def _in_rect(x: float, y: Optional[float],
                 x_range: Tuple[float, float],
                 y_range: Tuple[float, float]) -> bool:
        if not (x_range[0] <= float(x) <= x_range[1]):
            return False
        # y is optional for backward compat with callers that only
        # care about the X axis (e.g. plotting a horizontal band).
        if y is None:
            return True
        return y_range[0] <= float(y) <= y_range[1]

    def is_in_home_zone(self, x: float, y: Optional[float] = None) -> bool:
        return self._in_rect(x, y, self.our_home_x_m, self.our_home_y_m)

    def is_in_enemy_zone(self, x: float, y: Optional[float] = None) -> bool:
        return self._in_rect(x, y, self.enemy_home_x_m, self.enemy_home_y_m)

    def is_in_neutral_zone(self, x: float, y: Optional[float] = None) -> bool:
        return self._in_rect(x, y,
                             self.arena.neutral_zone_x_m,
                             self.arena.neutral_zone_y_m)

    def is_within_arena(self, x: float, y: float,
                        margin_m: Optional[float] = None) -> bool:
        """True iff (x, y) is at least ``margin_m`` from every wall.
        Defaults to ``arena.safety_margin_m``."""
        m = float(self.arena.safety_margin_m
                  if margin_m is None else margin_m)
        hw = self.arena.width_m / 2.0
        hd = self.arena.depth_m / 2.0
        return (-hw + m <= float(x) <= hw - m
                and -hd + m <= float(y) <= hd - m)

    # ---------------- derived: per-FC ----------------

    def role_for(self, fc_name: str) -> Role:
        a = self.drones.get(fc_name)
        return a.role if a is not None else Role.IDLE

    def cruise_altitude_for(self, fc_name: str, default_m: float = 1.5) -> float:
        a = self.drones.get(fc_name)
        return a.altitude_m if a is not None else float(default_m)

    def fcs_with_role(self, role: Role) -> List[str]:
        return [name for name, a in self.drones.items() if a.role == role]

    # ---------------- derived: attack pairs ----------------

    def attack_pair_targets(self) -> List[Tuple[int, int]]:
        """Marker-pair ids that count as a simultaneous 2-target strike.

        Uses ``attack.pair_targets_override`` if set; otherwise every
        2-combination of our live target ids in sorted order.
        """
        if self.attack.pair_targets_override is not None:
            out: List[Tuple[int, int]] = []
            for p in self.attack.pair_targets_override:
                if len(p) != 2:
                    raise ValueError(
                        f"attack.pair_targets_override entry {p!r} "
                        f"must have exactly 2 ids"
                    )
                out.append((int(p[0]), int(p[1])))
            return out
        return list(combinations(sorted(self.own_target_ids), 2))

    # ---------------- classification helpers ----------------

    def is_own_target(self, marker_id: int) -> bool:
        return int(marker_id) in self.own_target_ids

    def is_enemy_target(self, marker_id: int) -> bool:
        return int(marker_id) in self.enemy_target_ids

    def classify_marker(self, marker_id: int) -> str:
        """Return one of ``'own' | 'enemy' | 'wall' | 'unknown'``."""
        mid = int(marker_id)
        if mid in self.own_target_ids:
            return "own"
        if mid in self.enemy_target_ids:
            return "enemy"
        if mid in WALL_MARKER_IDS:
            return "wall"
        return "unknown"

    # ---------------- serialisation ----------------

    def to_dict(self) -> dict:
        return {
            "team_color": self.team_color.value,
            "own_target_ids_override": (
                sorted(self.own_target_ids_override)
                if self.own_target_ids_override is not None else None
            ),
            "enemy_target_ids_override": (
                sorted(self.enemy_target_ids_override)
                if self.enemy_target_ids_override is not None else None
            ),
            "live_targets_only": bool(self.live_targets_only),
            "arena": {
                "width_m": self.arena.width_m,
                "depth_m": self.arena.depth_m,
                "safety_margin_m": self.arena.safety_margin_m,
                "red_home_x_m":     list(self.arena.red_home_x_m),
                "red_home_y_m":     list(self.arena.red_home_y_m),
                "blue_home_x_m":    list(self.arena.blue_home_x_m),
                "blue_home_y_m":    list(self.arena.blue_home_y_m),
                "neutral_zone_x_m": list(self.arena.neutral_zone_x_m),
                "neutral_zone_y_m": list(self.arena.neutral_zone_y_m),
            },
            "drones": {
                name: {
                    "role": a.role.value,
                    "altitude_m": a.altitude_m,
                }
                for name, a in self.drones.items()
            },
            "attack": {
                "hover_alt_m":    self.attack.hover_alt_m,
                "strike_alt_m":   self.attack.strike_alt_m,
                "sync_window_s":  self.attack.sync_window_s,
                "pair_targets_override": (
                    [list(p) for p in self.attack.pair_targets_override]
                    if self.attack.pair_targets_override is not None
                    else None
                ),
                "home_zone_clear": bool(self.attack.home_zone_clear),
            },
            "defender": {
                "intercept_radius_m": self.defender.intercept_radius_m,
            },
            "match": {
                "duration_s": self.match.duration_s,
            },
        }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

_DIR = Path(__file__).parent
DEFAULT_PATH = _DIR / "settings.json"
EXAMPLE_PATH = _DIR / "settings.example.json"


def _tuple2(v, default: Tuple[float, float]) -> Tuple[float, float]:
    if v is None:
        return default
    try:
        return (float(v[0]), float(v[1]))
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ValueError(f"expected [min, max] pair, got {v!r}: {e}") from e


def _from_dict(d: dict) -> StrategySettings:
    arena_d = d.get("arena") or {}
    # Default Y ranges span the full depth so old configs (no Y fields)
    # behave exactly like the old "full-depth bands". Width / depth
    # default first so the Y default uses the actual configured depth.
    _depth = float(arena_d.get("depth_m", 20.0))
    _full_y = (-_depth / 2.0, _depth / 2.0)
    arena = ArenaSettings(
        width_m=float(arena_d.get("width_m", 10.0)),
        depth_m=_depth,
        safety_margin_m=float(arena_d.get("safety_margin_m", 0.5)),
        red_home_x_m=_tuple2(arena_d.get("red_home_x_m"), (-5.0, -2.0)),
        red_home_y_m=_tuple2(arena_d.get("red_home_y_m"), _full_y),
        blue_home_x_m=_tuple2(arena_d.get("blue_home_x_m"), (2.0, 5.0)),
        blue_home_y_m=_tuple2(arena_d.get("blue_home_y_m"), _full_y),
        neutral_zone_x_m=_tuple2(arena_d.get("neutral_zone_x_m"), (-2.0, 2.0)),
        neutral_zone_y_m=_tuple2(arena_d.get("neutral_zone_y_m"), _full_y),
    )

    drones_d = d.get("drones") or {}
    drones: Dict[str, DroneAssignment] = {}
    for name, v in drones_d.items():
        if not isinstance(v, dict):
            raise ValueError(
                f"drones[{name!r}] must be an object with role/altitude_m, got {v!r}"
            )
        drones[str(name)] = DroneAssignment(
            role=Role.parse(v.get("role", "idle")),
            altitude_m=float(v.get("altitude_m", 1.5)),
        )

    attack_d = d.get("attack") or {}
    pair_override = attack_d.get("pair_targets_override")
    if pair_override is not None:
        pair_override = [
            [int(x) for x in p] for p in pair_override
        ]
    attack = AttackSettings(
        hover_alt_m=float(attack_d.get("hover_alt_m", 3.5)),
        strike_alt_m=float(attack_d.get("strike_alt_m", 1.5)),
        sync_window_s=float(attack_d.get("sync_window_s", 1.5)),
        pair_targets_override=pair_override,
        home_zone_clear=bool(attack_d.get("home_zone_clear", True)),
    )

    defender_d = d.get("defender") or {}
    defender = DefenderSettings(
        intercept_radius_m=float(defender_d.get("intercept_radius_m", 2.5)),
    )

    match_d = d.get("match") or {}
    match = MatchSettings(
        duration_s=float(match_d.get("duration_s", 600.0)),
    )

    return StrategySettings(
        team_color=TeamColor.parse(d.get("team_color", "red")),
        own_target_ids_override=(
            frozenset(int(x) for x in d["own_target_ids_override"])
            if d.get("own_target_ids_override") else None
        ),
        enemy_target_ids_override=(
            frozenset(int(x) for x in d["enemy_target_ids_override"])
            if d.get("enemy_target_ids_override") else None
        ),
        live_targets_only=bool(d.get("live_targets_only", True)),
        arena=arena,
        drones=drones,
        attack=attack,
        defender=defender,
        match=match,
    )


def load(path: Optional[str] = None) -> StrategySettings:
    """Resolve a settings file and load it. Returns the package defaults
    when no file is found anywhere — never raises just because no file
    exists. Raises ``ValueError`` only if a file *does* exist but is
    malformed.
    """
    env = os.environ.get("MARKER_MISSION_C2_STRATEGY_SETTINGS")
    candidates: list[Optional[Path]] = [
        Path(path) if path else None,
        Path(env) if env else None,
        DEFAULT_PATH,
        EXAMPLE_PATH,
    ]
    chosen: Optional[Path] = next(
        (c for c in candidates if c is not None and c.exists()),
        None,
    )
    if chosen is None:
        return StrategySettings()
    try:
        return _from_dict(json.loads(chosen.read_text()))
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"failed to parse {chosen}: {e}") from e


def save(settings: StrategySettings, path: Optional[str] = None) -> Path:
    """Persist ``settings`` to ``path`` (or :data:`DEFAULT_PATH`).
    Returns the path written. Creates parent dirs as needed.
    """
    target = Path(path) if path else DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(settings.to_dict(), indent=2) + "\n")
    return target
