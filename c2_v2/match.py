"""Match-level state for C2 V2: which team we are, which arena is active, and
the live box-state tracker.

Thread-safe. The pool (asyncio thread) feeds marker observations in via
``ingest``; the Flask thread reads ``to_client`` and flips team/arena via the
setters. Team + arena are persisted to a small JSON so they survive a restart
at the venue.

Reuses the proven ``MarkerTracker`` (box holder / 5-s lock / 1-5-10 scoring /
Special-Maneuver dwell) and ``arena_state`` (live real/gvz geometry).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from marker_mission_c2.strategy import arena_state
from marker_mission_c2.strategy.markers import MarkerTracker
from marker_mission_c2.strategy.settings import ALL_SLOTS

from .tunables import TUNE_SPEC, Tunables, coerce, from_dict as tunables_from_dict

logger = logging.getLogger("c2_v2.match")

VALID_TEAMS = ("red", "blue")
VALID_ROLES = ("idle", "scout", "attacker", "defender")
_STATE_PATH = Path(__file__).resolve().parent / "c2v2_state.json"

# Default roles for a 5-drone fleet (operator can change live): the simplified
# scout-less strategy — 3 fixed-lane attackers + 2 neutral-zone defenders. The
# attackers + defenders do their own scouting (each watches the boxes it works).
_DEFAULT_ROLE_BY_LANE = ("attacker", "attacker", "attacker", "defender", "defender")


class MatchState:
    def __init__(self, our_team: str = "red",
                 arena_name: Optional[str] = None,
                 fc_names: Optional[list] = None,
                 state_path: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self._path = state_path or _STATE_PATH
        persisted = self._load()
        team = (persisted.get("our_team") or our_team or "red").lower()
        self.our_team = team if team in VALID_TEAMS else "red"
        arena = arena_name or persisted.get("arena") or arena_state.default_name()
        try:
            arena_state.set_active(arena)
        except Exception as exc:
            logger.warning("c2_v2: could not set arena %r: %s", arena, exc)
        self.tracker = MarkerTracker(active_slots=ALL_SLOTS)

        # Commanding is DISARMED until the operator explicitly arms. The runner
        # observes always but never pushes a script while disarmed.
        self.armed = False
        # AUTO: the coordinator assigns roles from the game state. MANUAL: the
        # operator assigns roles. Persisted.
        self.auto = bool(persisted.get("auto", False))
        # Latest coordinator strategy line (for the UI).
        self.coord_summary = ""
        # Live-tunable strategy + capture parameters (operator-editable).
        self.tunables: Tunables = tunables_from_dict(persisted.get("tunables") or {})
        # Per-drone config: {fc_name: {"role", "enabled", "lane"}}. ``lane`` is a
        # stable 0-based index used for altitude deconfliction + RTH fan-out.
        self.drones: dict[str, dict] = {}
        pd = persisted.get("drones") or {}
        for lane, name in enumerate(fc_names or []):
            saved = pd.get(name) or {}
            role = saved.get("role")
            if role not in VALID_ROLES:
                role = (_DEFAULT_ROLE_BY_LANE[lane]
                        if lane < len(_DEFAULT_ROLE_BY_LANE) else "idle")
            self.drones[name] = {
                "role": role,
                "enabled": bool(saved.get("enabled", True)),
                "lane": lane,
            }

    # ------------------------------------------------------------------ setters
    def set_team(self, team: str) -> bool:
        t = (team or "").lower()
        if t not in VALID_TEAMS:
            return False
        with self._lock:
            self.our_team = t
            self._save()
        return True

    def set_arena(self, name: str) -> bool:
        try:
            arena_state.set_active(name)
        except Exception as exc:
            logger.warning("c2_v2: set_arena(%r) failed: %s", name, exc)
            return False
        with self._lock:
            self._save()
        return True

    def set_role(self, fc_name: str, role: str) -> bool:
        r = (role or "").lower()
        if r not in VALID_ROLES:
            return False
        with self._lock:
            d = self.drones.get(fc_name)
            if d is None:
                return False
            d["role"] = r
            self._save()
        return True

    def set_enabled(self, fc_name: str, enabled: bool) -> bool:
        with self._lock:
            d = self.drones.get(fc_name)
            if d is None:
                return False
            d["enabled"] = bool(enabled)
            self._save()
        return True

    def set_armed(self, armed: bool) -> None:
        with self._lock:
            self.armed = bool(armed)
            self._save()

    def set_auto(self, auto: bool) -> None:
        with self._lock:
            self.auto = bool(auto)
            self._save()

    def set_tunable(self, key: str, value) -> bool:
        cv = coerce(key, value)
        if cv is None:
            return False
        with self._lock:
            setattr(self.tunables, key, cv)
            self._save()
        return True

    def holders(self) -> dict:
        """{slot -> effective (decayed) holder} for the coordinator."""
        return {s: self.tracker.effective_holder(s) for s in range(1, 7)}

    def drone_cfg(self, fc_name: str) -> dict:
        with self._lock:
            return dict(self.drones.get(fc_name) or {})

    @property
    def enemy_team(self) -> str:
        return "blue" if self.our_team == "red" else "red"

    # ------------------------------------------------------------------ ingest
    def ingest(self, fc_name: str, visible_marker_ids: Iterable[int]) -> None:
        self.tracker.ingest(fc_name, visible_marker_ids)

    # ------------------------------------------------------------------ read
    def to_client(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            team = self.our_team
        boxes = self.tracker.to_dict(our_team=team)
        score = self.tracker.score()
        special = self.tracker.match_status(team, now)
        ours = sum(1 for b in boxes.values() if b["captured_by_us"])
        enemy = sum(1 for b in boxes.values() if b["captured_by_enemy"])
        return {
            "our_team": team,
            "enemy_team": self.enemy_team,
            "arena": arena_state.active_name(),
            "arena_options": arena_state.arena_labels(),
            "armed": self.armed,
            "auto": self.auto,
            "coord_summary": self.coord_summary,
            "tunables": self.tunables.to_dict(),
            "tune_spec": TUNE_SPEC,
            "drones": {k: dict(v) for k, v in self.drones.items()},
            "boxes": boxes,
            "boxes_ours": ours,
            "boxes_enemy": enemy,
            "score": score,           # {"red": n, "blue": n} (estimate)
            "special": special,       # all_ours / dwell_s / won
            "events": self.tracker.capture_events(limit=12),
        }

    # ------------------------------------------------------------------ persist
    def _load(self) -> Dict[str, Any]:
        try:
            if self._path.exists():
                d = json.loads(self._path.read_text())
                return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            pass
        return {}

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps({
                "our_team": self.our_team,
                "arena": arena_state.active_name(),
                "auto": self.auto,
                "tunables": self.tunables.to_dict(),
                # NOTE: armed is deliberately NOT persisted — every restart comes
                # up DISARMED for safety.
                "drones": {k: {"role": v["role"], "enabled": v["enabled"]}
                           for k, v in self.drones.items()},
            }, indent=2))
        except OSError as exc:
            logger.warning("c2_v2: could not persist state: %s", exc)
