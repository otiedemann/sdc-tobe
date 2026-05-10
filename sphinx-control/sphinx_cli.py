"""Reference catalogue and runtime wrappers for `sphinx-cli`.

The Parrot Sphinx control reference page
(https://developer.parrot.com/docs/sphinx/all_control_references.html)
lists every module exposed at runtime and the params/actions each
supports. This file:

  1. Encodes a curated subset of those modules — enough to drive a
     UI without forcing the operator to memorise sphinx-cli flags.
  2. Wraps `sphinx-cli param` and `sphinx-cli action` invocations.

The catalogue is INTENTIONALLY incomplete; covering every option
would duplicate the upstream docs. Add entries here as the team
finds itself reaching for them. Any module/param NOT in the
catalogue is still callable via the generic /api/sphinx-cli/raw
endpoint — the catalogue is for ergonomics, not gating.

`pysphinx` would let us introspect modules at runtime, but that
requires sourcing /opt/parrot-sphinx/usr/bin/parrot-sphinx-setenv.sh
in this venv first; for now we take the simpler subprocess path.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Literal


# ─── Catalogue ──────────────────────────────────────────────────────
# Each Param has:
#   key        — the parameter name (passed after the module to
#                sphinx-cli param). Some params are nested ("sky/sky").
#   kind       — "string" | "bool" | "float" | "int" | "enum"
#   options    — for enum, the allowed values
#   description — operator-facing help text
# Each Action has:
#   key        — the action name (after `-m <module>`)
#   args       — list of (name, kind) tuples for the positional args
#   description
#
# Categories follow the Parrot docs index.

@dataclass
class Param:
    key: str
    kind: str
    description: str = ""
    options: list[str] = field(default_factory=list)
    default: Any = None


@dataclass
class Action:
    key: str
    description: str = ""
    args: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Module:
    name: str          # The -m value passed to sphinx-cli
    label: str         # Human-readable
    category: str      # UI grouping
    description: str = ""
    params: list[Param] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    # docs URL fragment under
    # https://developer.parrot.com/docs/sphinx/<docs_slug>.html
    docs_slug: str = ""


# Curated catalogue. Add modules/params as the team needs them; the
# raw endpoint covers anything not listed here.
MODULES: list[Module] = [
    # ── World / environment ────────────────────────────────────────
    Module(
        name="world",
        label="World",
        category="World",
        description="Top-level world rendering / day-night-cycle hooks.",
        docs_slug="ref_world",
        params=[
            Param("sky/sky", "enum", "Sky preset (cloudy, indoor, night, …)",
                  options=["clear", "cloudy", "indoor", "night", "overcast",
                           "stormy", "sunny", "foggy"]),
            Param("actors", "string",
                  "AMS actors. Set 'pause' to true/false to freeze/unfreeze "
                  "all PerimeterPath actors at once."),
        ],
    ),
    Module(
        name="atmosphere",
        label="Atmosphere",
        category="World",
        docs_slug="ref_atmosphere",
        params=[
            Param("temperature_kelvin", "float",
                  "Ambient temperature; affects barometer & lift."),
            Param("pressure_pascal", "float", "Sea-level pressure."),
        ],
    ),
    Module(
        name="wind",
        label="Wind",
        category="World",
        docs_slug="ref_wind",
        params=[
            Param("velocity", "string",
                  "Wind vector \"vx vy vz\" in m/s. Negative = headwind."),
            Param("turbulence", "float",
                  "Turbulence intensity 0..1; default 0."),
        ],
    ),
    Module(
        name="light",
        label="Light",
        category="World",
        docs_slug="ref_light",
        params=[
            Param("sun/azimuth_deg", "float", "Sun direction (compass)."),
            Param("sun/elevation_deg", "float", "Sun elevation above horizon."),
            Param("sun/intensity", "float", "Sun intensity (relative)."),
        ],
    ),
    Module(
        name="physics",
        label="Physics",
        category="World",
        docs_slug="ref_physics",
        params=[
            Param("gravity_z", "float",
                  "Gravity along world Z (m/s²); default -9.81."),
            Param("update_rate_hz", "int",
                  "Physics solver tick rate; default 1000."),
        ],
    ),

    # ── Drone hardware ────────────────────────────────────────────
    Module(
        name="lipo_battery",
        label="Li-Po Battery",
        category="Drone hardware",
        docs_slug="ref_lipo_battery",
        params=[
            Param("voltage_volts", "float", "Cell voltage."),
            Param("capacity_mah", "int", "Pack capacity."),
            Param("charge_state", "float",
                  "State of charge 0..1 (1 = full)."),
        ],
        actions=[
            Action("set_charge_state", "Force the battery to a SOC.",
                   args=[("soc", "float")]),
        ],
    ),
    Module(
        name="smart_battery",
        label="Smart Battery",
        category="Drone hardware",
        docs_slug="ref_smart_battery",
        params=[
            Param("charge_state", "float",
                  "State of charge 0..1 (1 = full)."),
            Param("temperature_celsius", "float", "Pack temperature."),
        ],
    ),
    Module(
        name="motors",
        label="Motors",
        category="Drone hardware",
        docs_slug="ref_motors",
        params=[
            Param("max_thrust_factor", "float",
                  "Multiplier on per-motor max thrust; 1.0 = nominal."),
        ],
    ),
    Module(
        name="gimbal_motor",
        label="Gimbal motor",
        category="Drone hardware",
        docs_slug="ref_gimbal",
    ),
    Module(
        name="camera",
        label="Camera",
        category="Drone hardware",
        docs_slug="ref_camera",
        params=[
            Param("exposure_compensation", "float",
                  "EV adjustment, default 0."),
        ],
    ),
    Module(
        name="handling",
        label="Handling",
        category="Drone hardware",
        docs_slug="ref_handling",
        params=[
            Param("max_horizontal_speed_mps", "float", "Cruise speed cap."),
            Param("max_vertical_speed_mps", "float",   "Climb/descend cap."),
        ],
    ),

    # ── Sensors ────────────────────────────────────────────────────
    Module(
        name="gps",
        label="GPS",
        category="Sensors",
        docs_slug="ref_gps",
        params=[
            Param("nb_satellites", "int", "Number of locked satellites."),
            Param("noise_horizontal_m", "float",
                  "Horizontal-position noise stddev."),
            Param("noise_vertical_m", "float",
                  "Vertical-position noise stddev."),
        ],
    ),
    Module(
        name="imu",
        label="IMU",
        category="Sensors",
        docs_slug="ref_imu",
        params=[
            Param("gyro_noise_rad_s", "float", "Gyro angular-rate noise."),
            Param("accel_noise_m_s2", "float", "Accelerometer noise."),
        ],
    ),
    Module(
        name="magneto",
        label="Magnetometer",
        category="Sensors",
        docs_slug="ref_magneto",
        params=[
            Param("declination_deg", "float", "Magnetic declination."),
        ],
    ),
    Module(
        name="barometer",
        label="Barometer",
        category="Sensors",
        docs_slug="ref_barometer",
        params=[
            Param("noise_pascal", "float", "Pressure-reading noise."),
        ],
    ),
    Module(
        name="time-of-flight",
        label="Time-of-Flight",
        category="Sensors",
        docs_slug="ref_tof",
    ),
    Module(
        name="ultrasound",
        label="Ultrasound",
        category="Sensors",
        docs_slug="ref_ultrasound",
    ),

    # ── System ─────────────────────────────────────────────────────
    Module(
        name="omniscient",
        label="Omniscient",
        category="System",
        description="Ground-truth pose access; teleport drones at will.",
        docs_slug="ref_omniscient",
        actions=[
            Action("pose",
                   "Teleport drone to (x y z roll pitch yaw) — "
                   "world frame, metres / radians.",
                   args=[("x", "float"), ("y", "float"), ("z", "float"),
                         ("roll",  "float"), ("pitch", "float"),
                         ("yaw",   "float")]),
            Action("set_mode",
                   "Switch the omniscient component mode.",
                   args=[("mode", "string")]),
        ],
    ),
    Module(
        name="actors",
        label="Actors (AMS)",
        category="System",
        docs_slug="ref_actors",
        params=[
            Param("pause", "bool",
                  "Freeze (true) or release (false) all path actors."),
        ],
    ),
    Module(
        name="fwman",
        label="Firmware manager",
        category="System",
        docs_slug="ref_fwman",
    ),
    Module(
        name="iplink",
        label="IP link",
        category="System",
        docs_slug="ref_iplink",
    ),
    Module(
        name="wireless",
        label="Wireless link",
        category="System",
        docs_slug="ref_wireless",
        params=[
            Param("loss_rate", "float",
                  "Packet loss probability 0..1; default 0."),
            Param("latency_ms", "float",
                  "Added one-way latency, ms."),
        ],
    ),
    Module(
        name="spherical_coordinates",
        label="Spherical coordinates",
        category="System",
        docs_slug="ref_spherical_coordinates",
        params=[
            Param("latitude_deg", "float", "Reference latitude."),
            Param("longitude_deg", "float", "Reference longitude."),
            Param("altitude_m", "float", "Reference altitude."),
        ],
    ),
]


def catalogue() -> dict[str, Any]:
    """Return the catalogue serialisable for the web UI."""
    return {
        "modules": [
            {
                "name": m.name,
                "label": m.label,
                "category": m.category,
                "description": m.description,
                "docs_slug": m.docs_slug,
                "params": [
                    {"key": p.key, "kind": p.kind, "options": p.options,
                     "description": p.description, "default": p.default}
                    for p in m.params
                ],
                "actions": [
                    {"key": a.key, "args": a.args, "description": a.description}
                    for a in m.actions
                ],
            }
            for m in MODULES
        ],
    }


# ─── Runtime wrappers ───────────────────────────────────────────────


def _sphinx_cli_available() -> bool:
    return shutil.which("sphinx-cli") is not None


def call_param(module: str, param: str, value: str | None = None,
               timeout: float = 4.0) -> dict[str, Any]:
    """Invoke `sphinx-cli param -m <module> <param> [value]`.

    If `value` is None the call READS the current value; otherwise it
    SETS. Returns {ok, stdout, stderr, rc, cmd}.
    """
    if not _sphinx_cli_available():
        return {"ok": False, "error": "sphinx-cli not on PATH",
                "stdout": "", "stderr": "", "rc": -1, "cmd": []}
    cmd = ["sphinx-cli", "param", "-m", module, param]
    if value is not None and value != "":
        cmd.append(str(value))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": r.returncode == 0,
            "rc": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "cmd": cmd,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": str(e), "stdout": "", "stderr": "",
                "rc": -1, "cmd": cmd}


def call_action(module: str, action: str, args: list[str] | None = None,
                timeout: float = 4.0) -> dict[str, Any]:
    """Invoke `sphinx-cli action -m <module> <action> [args ...]`."""
    if not _sphinx_cli_available():
        return {"ok": False, "error": "sphinx-cli not on PATH",
                "stdout": "", "stderr": "", "rc": -1, "cmd": []}
    cmd = ["sphinx-cli", "action", "-m", module, action]
    if args:
        cmd.extend(str(a) for a in args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": r.returncode == 0,
            "rc": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "cmd": cmd,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": str(e), "stdout": "", "stderr": "",
                "rc": -1, "cmd": cmd}


def call_raw(args: list[str], timeout: float = 4.0) -> dict[str, Any]:
    """Pass-through: run `sphinx-cli <args ...>` literally. Use for
    `list-modules`, `info`, or anything not in the catalogue."""
    if not _sphinx_cli_available():
        return {"ok": False, "error": "sphinx-cli not on PATH",
                "stdout": "", "stderr": "", "rc": -1, "cmd": []}
    cmd = ["sphinx-cli", *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": r.returncode == 0,
            "rc": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "cmd": cmd,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": str(e), "stdout": "", "stderr": "",
                "rc": -1, "cmd": cmd}
