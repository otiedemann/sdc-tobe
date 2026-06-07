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
        "our_team":     "red"|"blue",
        "active_slots": [1, 2, 3, 4, 5, 6]
      },
      "drones": {
        "<fc-name>": {
          "team":          "red"|"blue"|null,
          "role":          "idle"|"scout"|"attacker",
          "enabled":       true,
          "scout_alt_m":   1.0,
          "attack_alt_m":  1.0,
          "home_alt_m":    0.8
        }
      }
    }

Each physical target marker has two ArUco faces: ``3<slot>`` (blue holder)
and ``4<slot>`` (red holder). Whichever face a scout sees is the side
currently "up", i.e. that team's score. There are 6 slots in play
(numbered 1..6), so live IDs are 31..36 and 41..46.

The store is thread-safe (mutations go through a single ``RLock``) and writes
atomically to avoid half-written files on crash.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_DEFAULT_PATH = _HERE / "settings.json"
_EXAMPLE_PATH = _HERE / "settings.example.json"


# ---------------------------------------------------------------------------
# Constants / helpers about the target marker scheme
# ---------------------------------------------------------------------------


VALID_TEAMS = ("red", "blue")
VALID_ROLES = ("idle", "scout", "attacker", "defender")

# All slots ever used in the SDC26 layout (1..6).
ALL_SLOTS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

# Helpers: face id <-> (slot, team).


def slot_for_face(marker_id: int) -> Optional[int]:
    """Return slot 1..6 for a target-face id, or None if not a target."""
    if 31 <= marker_id <= 36:
        return marker_id - 30
    if 41 <= marker_id <= 46:
        return marker_id - 40
    return None


def team_for_face(marker_id: int) -> Optional[str]:
    """Return "red"|"blue" for a target-face id, or None if not a target."""
    if 31 <= marker_id <= 36:
        return "blue"
    if 41 <= marker_id <= 46:
        return "red"
    return None


def face_id(slot: int, team: str) -> int:
    """Return the marker id shown when ``team`` holds ``slot``."""
    base = 40 if team == "red" else 30
    return base + int(slot)


def enemy_team(our_team: str) -> str:
    return "blue" if our_team == "red" else "red"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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
    # Home zones live at the back walls (depth-axis ends), not the side
    # walls — SDC26 arena puts red at -y, blue at +y. Cruise altitude
    # 3 m so the drone clears all 1 m-tall target boxes during RTH.
    home_red: Point3D = field(default_factory=lambda: Point3D(0.0, -8.0, 3.0))
    home_blue: Point3D = field(default_factory=lambda: Point3D(0.0, 8.0, 3.0))
    approach_distance_m: float = 0.5
    capture_ascend_m: float = 1.5
    capture_forward_m: float = 0.5
    capture_hover_s: float = 3.0
    # Sustained hover (s) the drone holds in its OWN home zone at the end
    # of every attack/defend run. Per SDC26 regs the drone must return to
    # the home zone and remain inside to validate the capture (>=5 s) and
    # the drone must NEVER land during a match. We end runs with this hover
    # instead of LAND. Default ~ a full 10-min match so the drone keeps
    # holding at home (the FC only safety-lands once the script ends, i.e.
    # after the match). The FC's own arena guard keeps it inside the zone.
    home_hover_s: float = 600.0
    # Hover seconds between consecutive SCOUT rotations within a single
    # scout script. Short = fast cadence; long = more time for the
    # marker tracker to register the latest rotation's observations.
    # NOTE: scout.py currently uses one continuous RC drive instead of
    # chained SCOUT verbs, so this knob is unused — kept for backwards
    # compat with persisted settings.
    scout_hover_s: float = 2.0
    # Yaw stick value (RC counts; -100..+100) for the scout role's
    # continuous yaw drive. Bypasses cfg.scout_yaw_stick on the FC —
    # the strategy uses an RC step rather than the SCOUT verb so the
    # drone rotates without the per-cycle brake.
    scout_yaw_stick: int = 45
    # How long one scout push drives the yaw stick for. Drone keeps
    # rotating for this many seconds, then the script ends and the
    # FC safety-lands. The strategy then re-pushes (TAKEOFF + RC ...).
    # Default 600s = ~10 minutes per push; battery dies around 25 min.
    scout_drive_duration_s: float = 600.0
    # When true the scout flies to the arena centre (0,0) via the FC's
    # existing TO verb before starting its yaw drive, so it watches all
    # six slots from the middle. Disable if TO is unreliable on your FC
    # (the scout then just rotates wherever it took off — which is the
    # arena centre in the sim, where the drone spawns at 0,0).
    scout_center: bool = True
    # Which side-wall markers the scout GO_HOMEs onto to reach the arena
    # centre. These are the mid-field side markers (x = ±5, y = 0) present in
    # both the real/competition and GVZ halls — IDs 11 (right) and 15 (left).
    # The scout APPROACHes whichever it can see. Editable per hall: if a venue
    # uses different centre markers, set them here without a code change.
    scout_center_markers: tuple[int, ...] = (11, 15)


@dataclass(frozen=True)
class MarkerSettings:
    our_team: str = "red"                     # "red" | "blue"
    active_slots: tuple[int, ...] = ALL_SLOTS

    def enemy_team(self) -> str:
        return enemy_team(self.our_team)

    def our_face_id(self, slot: int) -> int:
        return face_id(slot, self.our_team)

    def enemy_face_id(self, slot: int) -> int:
        return face_id(slot, self.enemy_team())

    def all_face_ids(self) -> tuple[int, ...]:
        out: List[int] = []
        for s in self.active_slots:
            out.append(face_id(s, "red"))
            out.append(face_id(s, "blue"))
        return tuple(out)


@dataclass(frozen=True)
class DroneSettings:
    fc_name: str
    team: Optional[str] = None      # "red" | "blue" | None (unassigned)
    role: str = "idle"              # one of VALID_ROLES
    enabled: bool = True
    # Default cruise altitude for the scout role (m). 3.0 puts the drone
    # comfortably above the 2 m lower wall markers, well below the 4 m
    # upper wall markers, and clear of the 1.4 m box tops — so the camera
    # is LOW (~1 m) on purpose: the box faces sit at ~1 m, so the scout must
    # fly level with them to read the slot status (from 3-4 m the faces are
    # past the 8 m vision range once you add the vertical offset, and seen at
    # a steep angle). The runner's deconfliction places scouts in a dedicated
    # low band (red 1.0 m / blue 1.3 m) below the movers' cruise band.
    scout_alt_m: float = 1.0
    attack_alt_m: float = 1.0
    home_alt_m: float = 0.8
    # Scout rotation speed: the yaw-stick value (1..100) for the continuous spin.
    # The drone rotates at ~ (this/100) × the FC's MaxRotationSpeed (~150°/s), so
    # higher = faster scanning. Per-drone + editable in the dashboard. 70 ≈ 105°/s
    # (~3.4 s per turn). Was a single match-level value (45); now per-drone.
    scout_yaw_stick: int = 70
    # Per-drone RTH arrival bearing (deg, -80..+80) for the GO_HOME on the home
    # wall marker: the angle around the marker this drone returns to, so several
    # attackers fan out instead of converging. None -> the runner auto-assigns
    # (-45/0/+45 across the attackers). Set explicitly to pin a drone's lane.
    go_home_angle_deg: Optional[float] = None


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


def _coerce_team(raw: Any) -> Optional[str]:
    if raw in VALID_TEAMS:
        return raw
    return None


def _coerce_role(raw: Any) -> str:
    if raw in VALID_ROLES:
        return raw
    return "idle"


def _coerce_slot_list(raw: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)):
        return default
    out: List[int] = []
    seen: set[int] = set()
    for x in raw:
        try:
            v = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= v <= 6 and v not in seen:
            out.append(v)
            seen.add(v)
    return tuple(out) if out else default


def _coerce_marker_ids(raw: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    """Parse a list/tuple of marker IDs, or a comma/space-separated string
    (dashboard convenience, e.g. "11,15"). Keeps any positive integer ID,
    dedups, preserves order; falls back to ``default`` if nothing valid."""
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    if not isinstance(raw, (list, tuple)):
        return default
    out: List[int] = []
    seen: set[int] = set()
    for x in raw:
        try:
            v = int(x)
        except (TypeError, ValueError):
            continue
        if v > 0 and v not in seen:
            out.append(v)
            seen.add(v)
    return tuple(out) if out else default


def _drone_to_dict(d: DroneSettings) -> Dict[str, Any]:
    return {
        "team": d.team,
        "role": d.role,
        "enabled": bool(d.enabled),
        "scout_alt_m": float(d.scout_alt_m),
        "attack_alt_m": float(d.attack_alt_m),
        "home_alt_m": float(d.home_alt_m),
        "scout_yaw_stick": int(d.scout_yaw_stick),
        "go_home_angle_deg": (None if d.go_home_angle_deg is None
                              else float(d.go_home_angle_deg)),
    }


def _coerce_go_home_angle(raw: Any) -> Optional[float]:
    """Optional RTH approach angle. Empty/None/"auto" -> None (auto-assign);
    else a float clamped to the GO_HOME-legal [-80, +80] deg."""
    if raw is None or raw == "" or raw == "auto":
        return None
    try:
        return max(-80.0, min(80.0, float(raw)))
    except (TypeError, ValueError):
        return None


def _drone_from(fc_name: str, raw: Any) -> DroneSettings:
    if not isinstance(raw, dict):
        return DroneSettings(fc_name=fc_name)
    return DroneSettings(
        fc_name=fc_name,
        team=_coerce_team(raw.get("team")),
        role=_coerce_role(raw.get("role")),
        enabled=bool(raw.get("enabled", True)),
        scout_alt_m=float(raw.get("scout_alt_m", 1.0)),
        attack_alt_m=float(raw.get("attack_alt_m", 1.0)),
        home_alt_m=float(raw.get("home_alt_m", 0.8)),
        scout_yaw_stick=max(1, min(100, int(raw.get("scout_yaw_stick", 70)))),
        go_home_angle_deg=_coerce_go_home_angle(raw.get("go_home_angle_deg")),
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
            "home_hover_s": float(s.match.home_hover_s),
            "scout_hover_s": float(s.match.scout_hover_s),
            "scout_yaw_stick": int(s.match.scout_yaw_stick),
            "scout_drive_duration_s": float(s.match.scout_drive_duration_s),
            "scout_center": bool(s.match.scout_center),
            "scout_center_markers": list(s.match.scout_center_markers),
        },
        "markers": {
            "our_team": s.markers.our_team,
            "active_slots": list(s.markers.active_slots),
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
        home_hover_s=float(m_raw.get("home_hover_s", m_defaults.home_hover_s)),
        scout_hover_s=float(m_raw.get("scout_hover_s", m_defaults.scout_hover_s)),
        scout_yaw_stick=max(-100, min(100, int(
            m_raw.get("scout_yaw_stick", m_defaults.scout_yaw_stick)
        ))),
        scout_drive_duration_s=float(
            m_raw.get("scout_drive_duration_s", m_defaults.scout_drive_duration_s)
        ),
        scout_center=bool(m_raw.get("scout_center", m_defaults.scout_center)),
        scout_center_markers=_coerce_marker_ids(
            m_raw.get("scout_center_markers"), m_defaults.scout_center_markers
        ),
    )

    markers_raw = raw.get("markers") or {}
    m_defaults_marker = MarkerSettings()
    our_team_raw = markers_raw.get("our_team")
    our_team = our_team_raw if our_team_raw in VALID_TEAMS else m_defaults_marker.our_team
    markers = MarkerSettings(
        our_team=our_team,
        active_slots=_coerce_slot_list(
            markers_raw.get("active_slots"), m_defaults_marker.active_slots
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
    """Thread-safe JSON-backed strategy settings."""

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
        bootstrap = _settings_from_dict({}, known_fc_names=self._fc_names)
        try:
            self._write_atomic(_settings_to_dict(bootstrap))
        except OSError as e:
            logger.warning("strategy: failed to write bootstrap settings: %s", e)
        return bootstrap

    def _write_atomic(self, data: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
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

    def reset_drones(self) -> int:
        """Drop every drone from the roster and persist the empty state.

        Returns the number of drones removed. The runner's auto-adopt path
        will re-populate the roster from the C2 overview on the next tick,
        with fresh DroneSettings defaults (so e.g. an out-of-date persisted
        ``scout_alt_m`` reverts to the current code default).
        """
        with self._lock:
            removed = len(self._settings.drones)
            self._settings = replace(self._settings, drones=())
            self._write_atomic(_settings_to_dict(self._settings))
            return removed

    # ----- write API ----------------------------------------------------------

    def update_drone(self, fc_name: str, **changes: Any) -> DroneSettings:
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
            if "scout_yaw_stick" in changes:
                try:
                    allowed["scout_yaw_stick"] = max(1, min(100, int(changes["scout_yaw_stick"])))
                except (TypeError, ValueError):
                    pass
            if "go_home_angle_deg" in changes:
                # None / "" / "auto" -> auto-assign; else clamped float.
                allowed["go_home_angle_deg"] = _coerce_go_home_angle(
                    changes["go_home_angle_deg"])
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
                "home_hover_s",
                "scout_hover_s",
                "scout_drive_duration_s",
            ):
                if key in changes:
                    try:
                        allowed[key] = float(changes[key])
                    except (TypeError, ValueError):
                        continue
            if "scout_yaw_stick" in changes:
                try:
                    val = int(changes["scout_yaw_stick"])
                except (TypeError, ValueError):
                    val = None
                if val is not None:
                    # Clamp to the RC channel range.
                    allowed["scout_yaw_stick"] = max(-100, min(100, val))
            if "scout_center" in changes:
                allowed["scout_center"] = bool(changes["scout_center"])
            if "scout_center_markers" in changes:
                allowed["scout_center_markers"] = _coerce_marker_ids(
                    changes["scout_center_markers"], m.scout_center_markers)
            patched = replace(m, **allowed)
            self._settings = replace(self._settings, match=patched)
            self._write_atomic(_settings_to_dict(self._settings))
            return patched

    def update_markers(self, **changes: Any) -> MarkerSettings:
        with self._lock:
            cur = self._settings.markers
            allowed: Dict[str, Any] = {}
            if "our_team" in changes:
                t = _coerce_team(changes["our_team"])
                if t is not None:
                    allowed["our_team"] = t
            if "active_slots" in changes:
                allowed["active_slots"] = _coerce_slot_list(
                    changes["active_slots"], cur.active_slots
                )
            patched = replace(cur, **allowed)
            self._settings = replace(self._settings, markers=patched)
            self._write_atomic(_settings_to_dict(self._settings))
            return patched
