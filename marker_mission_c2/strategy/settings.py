"""Operator-tunable strategy settings (team colour, target ID sets, …).

Stored in its own JSON file so :class:`marker_mission_c2.config.C2Config`
(which describes the FCs, ports, polls, etc.) doesn't need to grow new
fields every time the strategy gets a new knob. The two concerns also
have very different change frequencies — fleet topology is set up once
per box, team colour changes match-to-match.

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
    5. defaults (red team, live targets only)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import FrozenSet, Optional


# Canonical SDC26 target ID ranges. The xx1/xx2/xx3 ids are the live
# boxes; xx4/xx5/xx6 are spares (``enabled=false`` in
# default_target_layout.json). Wall markers occupy 1..16 — those aren't
# owned by either team.
RED_TARGET_IDS:  FrozenSet[int] = frozenset(range(41, 47))   # 41..46
BLUE_TARGET_IDS: FrozenSet[int] = frozenset(range(31, 37))   # 31..36
RED_LIVE_IDS:    FrozenSet[int] = frozenset({41, 42, 43})
BLUE_LIVE_IDS:   FrozenSet[int] = frozenset({31, 32, 33})
WALL_MARKER_IDS: FrozenSet[int] = frozenset(range(1, 17))    # 1..16


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


@dataclass
class StrategySettings:
    """Live, mutable knobs the strategy reads at run time.

    Construct once at startup, hand a reference to the planner / safety
    / runner. To re-tune mid-flight, just mutate the instance — every
    derived property recomputes on access so the next tick sees the new
    value (no recreation needed). Persist with :func:`save`.
    """

    team_color: TeamColor = TeamColor.RED

    # If set, override the team's default ID set (rare — use only for
    # one-off arena layouts that differ from the SDC26 spec).
    own_target_ids_override: Optional[FrozenSet[int]] = None
    enemy_target_ids_override: Optional[FrozenSet[int]] = None

    # If True, exclude spare IDs (xx4, xx5, xx6) from both sets so the
    # planner only considers physically-deployed targets.
    live_targets_only: bool = True

    # ---- derived sets (computed on access so live mutation works) ----

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

    # ---- classification helpers used by planners / safety ----

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

    # ---- serialisation ----

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
        }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

_DIR = Path(__file__).parent
DEFAULT_PATH = _DIR / "settings.json"
EXAMPLE_PATH = _DIR / "settings.example.json"


def _from_dict(d: dict) -> StrategySettings:
    """Parse a settings dict. Raises ValueError on bad team_color."""
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
