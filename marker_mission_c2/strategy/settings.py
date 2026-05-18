"""Per-drone strategy settings, persisted to JSON.

The strategy layer is configured via a single JSON file:

    marker_mission_c2/strategy/settings.json     (gitignored, runtime)
    marker_mission_c2/strategy/settings.example.json   (checked in)

If no settings file exists the loader bootstraps one from the FC list in the
C2 config so the operator just has to pick teams + roles in the UI.

Settings shape::

    {
      "match": {
        "neutral_red":     {"x": -1.5, "y": 0.0},
        "neutral_blue":    {"x":  1.5, "y": 0.0},
        "home_red":        {"x": -4.0, "y": 0.0, "alt": 0.8},
        "home_blue":       {"x":  4.0, "y": 0.0, "alt": 0.8},
        "approach_distance_m":  0.5,
        "capture_ascend_m":     1.5,
        "capture_forward_m":    0.5,
        "capture_hover_s":      3.0,
        "scout_hover_s":        12.0
      },
      "markers": {
        "red_live_ids":  [44, 45, 46],
        "blue_live_ids": [31, 32, 33]
      },
      "drones": {
        "<fc-name>": {
          "team":          "red"|"blue"|null,
          "role":          "idle"|"scout"|"attacker",
          "enabled":       true,
          "scout_alt_m":   1.8,
          "attack_alt_m":  1.0,
          "home_alt_m":    0.8
        }
      }
    }

The store is thread-safe (mutations go through a single ``RLock``) and writes
atomically to avoid half-written files on crash.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_DEFAULT_PATH = _HERE / "settings.json"
_EXAMPLE_PATH = _HERE / "settings.example.json"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


VALID_TEAMS = ("red", "blue")
VALID_ROLES = ("idle", "scout", "attacker")


@dataclass(frozen=True)
class Point2D:
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class Point3D:
    x: float = 0.0
    y: float = 0.0
    alt: float = 0.8


@dataclass(frozen=True)
class MatchSettings:
    neutral_red: Point2D = field(default_factory=lambda: Point2D(-1.5, 0.0))
    neutral_blue: Point2D = field(default_factory=lambda: Point2D(1.5, 0.0))
    home_red: Point3D = field(default_factory=lambda: Point3D(-4.0, 0.0, 0.8))
    home_blue: Point3D = field(default_factory=lambda: Point3D(4.0, 0.0, 0.8))
    approach_distance_m: float = 0.5
    capture_ascend_m: float = 1.5
    capture_forward_m: float = 0.5
    capture_hover_s: float = 3.0
    scout_hover_s: float = 12.0


@dataclass(frozen=True)
class MarkerSettings:
    red_live_ids: tuple[int, ...] = (44, 45, 46)
    blue_live_ids: tuple[int, ...] = (31, 32, 33)


@dataclass(frozen=True)
class DroneSettings:
    fc_name: str
    team: Optional[str] = None      # "red" | "blue" | None (unassigned)
    role: str = "idle"              # one of VALID_ROLES
    enabled: bool = True
    scout_alt_m: float = 1.8
    attack_alt_m: float = 1.0
    home_alt_m: float = 0.8


@dataclass(frozen=True)
class StrategySettings:
    match: MatchSettings = field(default_factory=MatchSettings)
    markers: MarkerSettings = field(default_factory=MarkerSettings)
    drones: tuple[DroneSettings, ...] = ()


# ---------------------------------------------------------------------------
# (de)serialisation
# ---------------------------------------------------------------------------


def _point2d_to_dict(p: Point2D) -> Dict[str, float]:
    return {"x": float(p.x), "y": float(p.y)}


def _point3d_to_dict(p: Point3D) -> Dict[str, float]:
    return {"x": float(p.x), "y": float(p.y), "alt": float(p.alt)}


def _point2d_from(raw: Any, default: Point2D) -> Point2D:
    if not isinstance(raw, dict):
        return default
    try:
        return Point2D(
            x=float(raw.get("x", default.x)),
            y=float(raw.get("y", default.y)),
        )
    except (TypeError, ValueError):
        return default


def _point3d_from(raw: Any, default: Point3D) -> Point3D:
    if not isinstance(raw, dict):
        return default
    try:
        return Point3D(
            x=float(raw.get("x", default.x)),
            y=float(raw.get("y", default.y)),
            alt=float(raw.get("alt", default.alt)),
        )
    except (TypeError, ValueError):
        return default


def _coerce_int_list(raw: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)):
        return default
    out: List[int] = []
    for x in raw:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return tuple(out) if out else default


def _coerce_team(raw: Any) -> Optional[str]:
    if raw in VALID_TEAMS:
        return raw
    return None


def _coerce_role(raw: Any) -> str:
    if raw in VALID_ROLES:
        return raw
    return "idle"


def _drone_to_dict(d: DroneSettings) -> Dict[str, Any]:
    return {
        "team": d.team,
        "role": d.role,
        "enabled": bool(d.enabled),
        "scout_alt_m": float(d.scout_alt_m),
        "attack_alt_m": float(d.attack_alt_m),
        "home_alt_m": float(d.home_alt_m),
    }


def _drone_from(fc_name: str, raw: Any) -> DroneSettings:
    if not isinstance(raw, dict):
        return DroneSettings(fc_name=fc_name)
    return DroneSettings(
        fc_name=fc_name,
        team=_coerce_team(raw.get("team")),
        role=_coerce_role(raw.get("role")),
        enabled=bool(raw.get("enabled", True)),
        scout_alt_m=float(raw.get("scout_alt_m", 1.8)),
        attack_alt_m=float(raw.get("attack_alt_m", 1.0)),
        home_alt_m=float(raw.get("home_alt_m", 0.8)),
    )


def _settings_to_dict(s: StrategySettings) -> Dict[str, Any]:
    return {
        "match": {
            "neutral_red": _point2d_to_dict(s.match.neutral_red),
            "neutral_blue": _point2d_to_dict(s.match.neutral_blue),
            "home_red": _point3d_to_dict(s.match.home_red),
            "home_blue": _point3d_to_dict(s.match.home_blue),
            "approach_distance_m": float(s.match.approach_distance_m),
            "capture_ascend_m": float(s.match.capture_ascend_m),
            "capture_forward_m": float(s.match.capture_forward_m),
            "capture_hover_s": float(s.match.capture_hover_s),
            "scout_hover_s": float(s.match.scout_hover_s),
        },
        "markers": {
            "red_live_ids": list(s.markers.red_live_ids),
            "blue_live_ids": list(s.markers.blue_live_ids),
        },
        "drones": {d.fc_name: _drone_to_dict(d) for d in s.drones},
    }


def _settings_from_dict(raw: Any, *, known_fc_names: Iterable[str]) -> StrategySettings:
    raw = raw if isinstance(raw, dict) else {}
    m_raw = raw.get("match") or {}
    m_defaults = MatchSettings()
    match = MatchSettings(
        neutral_red=_point2d_from(m_raw.get("neutral_red"), m_defaults.neutral_red),
        neutral_blue=_point2d_from(m_raw.get("neutral_blue"), m_defaults.neutral_blue),
        home_red=_point3d_from(m_raw.get("home_red"), m_defaults.home_red),
        home_blue=_point3d_from(m_raw.get("home_blue"), m_defaults.home_blue),
        approach_distance_m=float(
            m_raw.get("approach_distance_m", m_defaults.approach_distance_m)
        ),
        capture_ascend_m=float(
            m_raw.get("capture_ascend_m", m_defaults.capture_ascend_m)
        ),
        capture_forward_m=float(
            m_raw.get("capture_forward_m", m_defaults.capture_forward_m)
        ),
        capture_hover_s=float(
            m_raw.get("capture_hover_s", m_defaults.capture_hover_s)
        ),
        scout_hover_s=float(m_raw.get("scout_hover_s", m_defaults.scout_hover_s)),
    )

    markers_raw = raw.get("markers") or {}
    m_defaults_marker = MarkerSettings()
    markers = MarkerSettings(
        red_live_ids=_coerce_int_list(
            markers_raw.get("red_live_ids"), m_defaults_marker.red_live_ids
        ),
        blue_live_ids=_coerce_int_list(
            markers_raw.get("blue_live_ids"), m_defaults_marker.blue_live_ids
        ),
    )

    drones_raw = raw.get("drones") or {}
    drones: List[DroneSettings] = []
    # Preserve a stable order based on known FC names + any extras from disk.
    seen: set[str] = set()
    for fc in known_fc_names:
        drones.append(_drone_from(fc, drones_raw.get(fc)))
        seen.add(fc)
    for fc in drones_raw.keys():
        if fc in seen:
            continue
        drones.append(_drone_from(fc, drones_raw.get(fc)))

    return StrategySettings(
        match=match, markers=markers, drones=tuple(drones)
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SettingsStore:
    """Thread-safe JSON-backed strategy settings.

    Construct once at app startup; share the instance across the runner and
    the Flask app.
    """

    def __init__(
        self,
        *,
        path: Optional[os.PathLike[str] | str] = None,
        fc_names: Optional[Iterable[str]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._path = Path(path) if path else _DEFAULT_PATH
        self._fc_names = tuple(fc_names) if fc_names else ()
        self._settings: StrategySettings = self._load_initial()

    # ----- I/O ----------------------------------------------------------------

    def _load_initial(self) -> StrategySettings:
        for candidate in (self._path, _EXAMPLE_PATH):
            if candidate.exists():
                try:
                    raw = json.loads(candidate.read_text())
                    logger.info("strategy: loaded settings from %s", candidate)
                    return _settings_from_dict(raw, known_fc_names=self._fc_names)
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning("strategy: failed to load %s: %s", candidate, e)
        # Nothing on disk — bootstrap from FC list.
        bootstrap = _settings_from_dict({}, known_fc_names=self._fc_names)
        try:
            self._write_atomic(_settings_to_dict(bootstrap))
        except OSError as e:
            logger.warning("strategy: failed to write bootstrap settings: %s", e)
        return bootstrap

    def _write_atomic(self, data: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # NamedTemporaryFile in the same dir => same filesystem => atomic rename.
        fd, tmp = tempfile.mkstemp(
            prefix=".settings.", suffix=".json.tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ----- read API -----------------------------------------------------------

    def snapshot(self) -> StrategySettings:
        with self._lock:
            return self._settings

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return _settings_to_dict(self._settings)

    def drone(self, fc_name: str) -> Optional[DroneSettings]:
        with self._lock:
            for d in self._settings.drones:
                if d.fc_name == fc_name:
                    return d
            return None

    # ----- write API ----------------------------------------------------------

    def update_drone(self, fc_name: str, **changes: Any) -> DroneSettings:
        """Patch a single drone's settings. Unknown keys are ignored."""
        with self._lock:
            existing = self.drone(fc_name) or DroneSettings(fc_name=fc_name)
            allowed: Dict[str, Any] = {}
            if "team" in changes:
                allowed["team"] = _coerce_team(changes["team"])
            if "role" in changes:
                allowed["role"] = _coerce_role(changes["role"])
            if "enabled" in changes:
                allowed["enabled"] = bool(changes["enabled"])
            for key in ("scout_alt_m", "attack_alt_m", "home_alt_m"):
                if key in changes:
                    try:
                        allowed[key] = float(changes[key])
                    except (TypeError, ValueError):
                        continue
            patched = replace(existing, **allowed)
            new_drones = []
            seen = False
            for d in self._settings.drones:
                if d.fc_name == fc_name:
                    new_drones.append(patched)
                    seen = True
                else:
                    new_drones.append(d)
            if not seen:
                new_drones.append(patched)
            self._settings = replace(self._settings, drones=tuple(new_drones))
            self._write_atomic(_settings_to_dict(self._settings))
            return patched

    def update_match(self, **changes: Any) -> MatchSettings:
        """Patch match-level fields. Points accept dict {x, y[, alt]}."""
        with self._lock:
            m = self._settings.match
            allowed: Dict[str, Any] = {}
            if "neutral_red" in changes:
                allowed["neutral_red"] = _point2d_from(changes["neutral_red"], m.neutral_red)
            if "neutral_blue" in changes:
                allowed["neutral_blue"] = _point2d_from(changes["neutral_blue"], m.neutral_blue)
            if "home_red" in changes:
                allowed["home_red"] = _point3d_from(changes["home_red"], m.home_red)
            if "home_blue" in changes:
                allowed["home_blue"] = _point3d_from(changes["home_blue"], m.home_blue)
            for key in (
                "approach_distance_m",
                "capture_ascend_m",
                "capture_forward_m",
                "capture_hover_s",
                "scout_hover_s",
            ):
                if key in changes:
                    try:
                        allowed[key] = float(changes[key])
                    except (TypeError, ValueError):
                        continue
            patched = replace(m, **allowed)
            self._settings = replace(self._settings, match=patched)
            self._write_atomic(_settings_to_dict(self._settings))
            return patched

    def update_markers(self, **changes: Any) -> MarkerSettings:
        with self._lock:
            cur = self._settings.markers
            allowed: Dict[str, Any] = {}
            if "red_live_ids" in changes:
                allowed["red_live_ids"] = _coerce_int_list(
                    changes["red_live_ids"], cur.red_live_ids
                )
            if "blue_live_ids" in changes:
                allowed["blue_live_ids"] = _coerce_int_list(
                    changes["blue_live_ids"], cur.blue_live_ids
                )
            patched = replace(cur, **allowed)
            self._settings = replace(self._settings, markers=patched)
            self._write_atomic(_settings_to_dict(self._settings))
            return patched
