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

logger = logging.getLogger("c2_v2.match")

VALID_TEAMS = ("red", "blue")
_STATE_PATH = Path(__file__).resolve().parent / "c2v2_state.json"


class MatchState:
    def __init__(self, our_team: str = "red",
                 arena_name: Optional[str] = None,
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
            self._path.write_text(json.dumps(
                {"our_team": self.our_team, "arena": arena_state.active_name()},
                indent=2))
        except OSError as exc:
            logger.warning("c2_v2: could not persist state: %s", exc)
