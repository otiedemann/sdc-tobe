"""
Tiny mission-control language.

Parses an operator-typed text script into a list of ``Step`` objects
that ``MissionController._advance_script`` walks one at a time. The
language is one-command-per-line, case-insensitive, with ``#`` comments
and blank lines ignored.

Eighteen commands. Suffix convention:

  * ``_RC`` — raw RC stick at the given deflection for a duration.
    Open-loop, fast, may drift; use when you want "stick at X% for
    Y seconds" semantics.
  * ``_IMU`` — closed-loop position/angle target using Anafi's
    on-board sensor fusion (IMU + baro + optical flow). Accurate,
    symmetric +/-, but discrete (the step blocks until the firmware
    confirms completion).

::

    TAKEOFF
    APPROACH [<marker-id>] [<distance>]
    HOOVER   [<seconds>]
    AWAIT    <marker-id> <timeout-seconds>
    PAUSE    <seconds>
    LAND
    HEIGHT   [<height>]
    TO       <x> <y> [<z>]
    DANCE    [<seconds>] [<mode>]                  mode in {wobble, spin, random}
    LR_RC    <rc> [<seconds>]                      +right / -left,   rc in [-100, +100]
    FB_RC    <rc> [<seconds>]                      +forward / -back, rc in [-100, +100]
    UD_RC    <rc> [<seconds>]                      +up / -down,      rc in [-100, +100]
    YAW_RC   <rc> [<seconds>]                      +cw / -ccw,       rc in [-100, +100]
    LR_IMU   <meters>                              +right / -left,   |m| in [0.01, 5.0]
    FB_IMU   <meters>                              +forward / -back, |m| in [0.01, 5.0]
    UD_IMU   <meters>                              +up / -down,      |m| in [0.01, 5.0]
    YAW_IMU  <deg>                                 +cw / -ccw,       deg in [-180, +180]
    RC       <lr> <fb> <ud> <yaw> [<seconds>]      all four sticks at once
    SCOUT                                          slow 360° yaw spin (cw, stick=15)

LR_RC / FB_RC / UD_RC / YAW_RC / RC are raw-RC steps: pin the listed
stick(s) to the given value(s) for the given duration (default 1 s;
fractional values such as ``1.5`` are accepted), and leave the other
channels at 0. Followed by an automatic IMU-based brake so the drone
settles before the next step.

LR_IMU / FB_IMU / UD_IMU / YAW_IMU are closed-loop moves: the firmware
flies the drone exactly the requested distance / angle and stops.
Symmetric in both directions (no Anafi velocity-controller asymmetry).
The step blocks until the move completes.

LR / FB / UD / RC are raw-RC steps: pin the listed stick(s) to the
given value(s) for the given duration (default 1 second; fractional
values such as ``1.5`` are accepted), and leave the other channels
at 0. The FC ceiling / arena guard still clamp at the wire — a
UD +100 above the ceiling gets reduced to 0 by the FC. Useful for
hand-tuning a position or jogging the drone during a script.

YAW is different: it's a discrete rotation by the given number of
degrees (positive = CW from above, negative = CCW), not a stick rate.
``YAW 90`` rotates exactly 90° and then advances; the step takes as
long as the firmware needs to complete the rotation. If you really
do want a raw yaw-stick rate (e.g. for very small / slow turns),
use the multi-axis ``RC`` command whose yaw field is still a stick
in [-100, +100].

AWAIT behaves like HOOVER (HOLD-style station-keeping if the previous
step was APPROACH, otherwise IDLE) but exits early as soon as the
named marker becomes visible. The timeout is a hard upper bound.

PAUSE is unconditional IDLE for the given seconds (rc = 0 throughout).
Useful for fixed pauses regardless of what came before.

TO drives the drone to an arena-frame coordinate. With two arguments
the current altitude is preserved; with three the third becomes the
target altitude. If the world-position estimate goes stale mid-step
the controller falls back to a yaw-search until a fresh fix arrives,
so the drone can never run open-loop on a dead estimate.

Omitted arguments fall back to ``MissionConfig`` defaults at parse time
(passed in via the ``defaults`` dict). Invalid syntax raises
``ScriptError`` with a 1-based line number so the UI can highlight the
offending line.

Round-trip ``format(parse(text, defaults))`` returns a canonicalised
form (uppercase command, normalised whitespace) that re-parses to the
same Step list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


VALID_DANCE_MODES = ("wobble", "spin", "random")
DEFAULT_DANCE_MODE = "wobble"


class ScriptError(ValueError):
    """Raised by ``parse`` when a script line can't be parsed.

    ``line_no`` is 1-based to match the UI's textarea numbering.
    """

    def __init__(self, line_no: int, message: str):
        self.line_no = int(line_no)
        self.message = str(message)
        super().__init__(f"line {self.line_no}: {self.message}")


@dataclass
class Step:
    kind: str                           # "TAKEOFF" / "APPROACH" / ...
    marker_id: Optional[int] = None
    distance: Optional[float] = None
    seconds: Optional[float] = None
    height: Optional[float] = None
    mode: Optional[str] = None
    # TO step: arena-frame target. world_x / world_y are required;
    # height is reused for the optional z (None means "keep current
    # altitude"). yaw is the optional arena-frame target yaw in
    # degrees (CW from +y / front wall). The literal "auto" picks
    # the best-view yaw at step entry (the yaw that maximises the
    # number of well-placed reference markers visible from the
    # target). Default is "auto".
    world_x: Optional[float] = None
    world_y: Optional[float] = None
    yaw: Optional[object] = None       # float or "auto" or None
    # Raw RC sticks for kind="RC" (FB / UD / LR / RC commands). Each
    # axis in [-100, +100]; clamped at parse time. Zero on every non-RC
    # step. seconds carries the duration as for HOOVER / PAUSE.
    rc_lr: int = 0
    rc_fb: int = 0
    rc_ud: int = 0
    rc_yaw: int = 0
    # Discrete rotation in degrees for kind="YAW" (the YAW_IMU
    # command). Range [-180, +180]; positive = CW. None on every
    # non-YAW step. No seconds — api.rotate is synchronous and the
    # step advances when it returns.
    rotation_deg: Optional[int] = None
    # Closed-loop position-relative move for kind="MOVE_IMU"
    # (FB_IMU / LR_IMU / UD_IMU). ``move_direction`` is one of
    # "forward"/"back"/"left"/"right"/"up"/"down" (sign baked in,
    # ``move_distance_m`` is always positive). Synchronous via
    # api.move which wraps Olympe's moveBy.
    move_direction: Optional[str] = None
    move_distance_m: Optional[float] = None
    line_no: int = 0                    # 1-based source line, for diagnostics


def _required_default(defaults: dict, key: str, line_no: int) -> object:
    if key not in defaults:
        raise ScriptError(line_no,
                          f"missing default for '{key}' (caller bug)")
    return defaults[key]


def _parse_int(token: str, line_no: int, what: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise ScriptError(line_no,
                          f"{what} expects an integer, got {token!r}")


def _parse_float(token: str, line_no: int, what: str) -> float:
    try:
        return float(token)
    except ValueError:
        raise ScriptError(line_no,
                          f"{what} expects a number, got {token!r}")


def parse(text: str, defaults: dict) -> List[Step]:
    """Parse ``text`` into a list of Steps.

    ``defaults`` must supply: ``marker_id``, ``distance``,
    ``hold_seconds``, ``height``, ``dance_seconds``, ``dance_mode``.
    """
    out: List[Step] = []
    for raw_line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0].upper()
        args = tokens[1:]
        if cmd == "TAKEOFF":
            if args:
                raise ScriptError(raw_line_no,
                                  f"TAKEOFF takes no arguments, got {args}")
            out.append(Step(kind="TAKEOFF", line_no=raw_line_no))
        elif cmd == "APPROACH":
            if len(args) > 2:
                raise ScriptError(raw_line_no,
                                  f"APPROACH takes 0-2 arguments, got {len(args)}")
            mid = (_parse_int(args[0], raw_line_no, "APPROACH marker-id")
                   if len(args) >= 1
                   else int(_required_default(defaults, "marker_id", raw_line_no)))
            dist = (_parse_float(args[1], raw_line_no, "APPROACH distance")
                    if len(args) >= 2
                    else float(_required_default(defaults, "distance", raw_line_no)))
            out.append(Step(kind="APPROACH",
                            marker_id=mid, distance=dist,
                            line_no=raw_line_no))
        elif cmd == "HOOVER":
            if len(args) > 1:
                raise ScriptError(raw_line_no,
                                  f"HOOVER takes 0-1 arguments, got {len(args)}")
            sec = (_parse_float(args[0], raw_line_no, "HOOVER seconds")
                   if len(args) >= 1
                   else float(_required_default(defaults, "hold_seconds", raw_line_no)))
            if sec < 0:
                raise ScriptError(raw_line_no,
                                  f"HOOVER seconds must be >= 0, got {sec}")
            out.append(Step(kind="HOOVER", seconds=sec, line_no=raw_line_no))
        elif cmd == "AWAIT":
            if len(args) != 2:
                raise ScriptError(raw_line_no,
                                  f"AWAIT takes exactly 2 arguments "
                                  f"(<marker-id> <timeout-seconds>), got "
                                  f"{len(args)}")
            mid = _parse_int(args[0], raw_line_no, "AWAIT marker-id")
            sec = _parse_float(args[1], raw_line_no, "AWAIT timeout-seconds")
            if sec < 0:
                raise ScriptError(raw_line_no,
                                  f"AWAIT timeout must be >= 0, got {sec}")
            out.append(Step(kind="AWAIT",
                            marker_id=mid, seconds=sec,
                            line_no=raw_line_no))
        elif cmd == "PAUSE":
            if len(args) != 1:
                raise ScriptError(raw_line_no,
                                  f"PAUSE takes exactly 1 argument "
                                  f"(<seconds>), got {len(args)}")
            sec = _parse_float(args[0], raw_line_no, "PAUSE seconds")
            if sec < 0:
                raise ScriptError(raw_line_no,
                                  f"PAUSE seconds must be >= 0, got {sec}")
            out.append(Step(kind="PAUSE", seconds=sec, line_no=raw_line_no))
        elif cmd == "LAND":
            if args:
                raise ScriptError(raw_line_no,
                                  f"LAND takes no arguments, got {args}")
            out.append(Step(kind="LAND", line_no=raw_line_no))
        elif cmd == "HEIGHT":
            if len(args) > 1:
                raise ScriptError(raw_line_no,
                                  f"HEIGHT takes 0-1 arguments, got {len(args)}")
            h = (_parse_float(args[0], raw_line_no, "HEIGHT")
                 if len(args) >= 1
                 else float(_required_default(defaults, "height", raw_line_no)))
            out.append(Step(kind="HEIGHT", height=h, line_no=raw_line_no))
        elif cmd == "TO":
            if len(args) not in (2, 3, 4):
                raise ScriptError(raw_line_no,
                                  f"TO takes 2-4 arguments "
                                  f"(<x> <y> [<z>] [<yaw>|auto]), "
                                  f"got {len(args)}")
            wx = _parse_float(args[0], raw_line_no, "TO x")
            wy = _parse_float(args[1], raw_line_no, "TO y")
            wz = (_parse_float(args[2], raw_line_no, "TO z")
                  if len(args) >= 3 else None)
            # Yaw default is "auto": the controller picks the
            # best-view yaw at step entry. A numeric 4th arg is
            # honoured as the explicit arena-frame yaw target.
            yaw_arg: object = "auto"
            if len(args) == 4:
                if args[3].lower() == "auto":
                    yaw_arg = "auto"
                else:
                    yaw_arg = _parse_float(args[3], raw_line_no, "TO yaw")
            out.append(Step(kind="TO",
                            world_x=wx, world_y=wy, height=wz,
                            yaw=yaw_arg,
                            line_no=raw_line_no))
        elif cmd == "TO_HOME":
            # TO_HOME [<height>] — navigate back to the world position
            # snapshotted at TAKEOFF completion. Sidesteps absolute-
            # offset errors in the ArUco solution: as long as motion
            # deltas are correct (verified vs GPS), the drone returns
            # to its actual physical takeoff spot in the arena.
            #
            # Optional height arg overrides the snapshotted Z (useful
            # for "cruise back at 3m, then LAND descends").
            if len(args) > 1:
                raise ScriptError(raw_line_no,
                                  f"TO_HOME takes 0-1 arguments "
                                  f"([<height>]), got {len(args)}")
            wz = (_parse_float(args[0], raw_line_no, "TO_HOME z")
                  if len(args) == 1 else None)
            out.append(Step(kind="TO_HOME",
                            height=wz,
                            line_no=raw_line_no))
        elif cmd in ("LR_RC", "FB_RC", "UD_RC", "YAW_RC"):
            # Raw RC stick for ``seconds``. ``rc`` is the protocol
            # value [-100, +100]; the FC ceiling, arena guard and
            # watchdog still clamp at the wire. Followed by an
            # automatic IMU-based brake (see controller._step_rc).
            if len(args) not in (1, 2):
                raise ScriptError(raw_line_no,
                                  f"{cmd} takes 1-2 arguments "
                                  f"(<rc> [<seconds>]), got {len(args)}")
            rc = _parse_int(args[0], raw_line_no, f"{cmd} rc")
            if not (-100 <= rc <= 100):
                raise ScriptError(raw_line_no,
                                  f"{cmd} rc must be in [-100, +100], "
                                  f"got {rc}")
            sec = (_parse_float(args[1], raw_line_no, f"{cmd} seconds")
                   if len(args) >= 2 else 1.0)
            if sec < 0:
                raise ScriptError(raw_line_no,
                                  f"{cmd} seconds must be >= 0, got {sec}")
            step = Step(kind="RC", seconds=sec, line_no=raw_line_no)
            if cmd == "LR_RC":
                step.rc_lr = rc
            elif cmd == "FB_RC":
                step.rc_fb = rc
            elif cmd == "UD_RC":
                step.rc_ud = rc
            else:  # YAW_RC
                step.rc_yaw = rc
            out.append(step)
        elif cmd == "YAW_IMU":
            # Discrete rotation by N degrees, range [-180, +180].
            # Positive = CW (looking down). Synchronous on the FC
            # side — the step advances when api.rotate returns. No
            # seconds argument because the rotation has its own
            # duration determined by the firmware's yaw rate.
            if len(args) != 1:
                raise ScriptError(raw_line_no,
                                  f"YAW_IMU takes exactly 1 argument "
                                  f"(<deg>, +cw / -ccw, "
                                  f"range [-180, +180]), got {len(args)}")
            deg = _parse_int(args[0], raw_line_no, "YAW_IMU deg")
            if not (-180 <= deg <= 180):
                raise ScriptError(raw_line_no,
                                  f"YAW_IMU deg must be in [-180, +180], "
                                  f"got {deg}")
            out.append(Step(kind="YAW", rotation_deg=deg,
                            line_no=raw_line_no))
        elif cmd in ("FB_IMU", "LR_IMU", "UD_IMU"):
            # Closed-loop position-relative move using Anafi's
            # moveBy (sensor-fused — IMU + baro + optical flow).
            # Symmetric +/- (unlike the open-loop *_RC family,
            # which is bottlenecked by Anafi's asymmetric velocity
            # controller). The step blocks until the firmware
            # confirms the move has completed.
            if len(args) != 1:
                raise ScriptError(raw_line_no,
                                  f"{cmd} takes exactly 1 argument "
                                  f"(<meters>), got {len(args)}")
            meters = _parse_float(args[0], raw_line_no, f"{cmd} meters")
            if abs(meters) < 0.01:
                raise ScriptError(raw_line_no,
                                  f"{cmd} |meters| must be >= 0.01, "
                                  f"got {meters}")
            if abs(meters) > 5.0:
                raise ScriptError(raw_line_no,
                                  f"{cmd} |meters| must be <= 5.0 "
                                  f"(safety cap), got {meters}")
            direction_map = {
                "FB_IMU": ("forward", "back"),
                "LR_IMU": ("right", "left"),
                "UD_IMU": ("up", "down"),
            }
            pos_dir, neg_dir = direction_map[cmd]
            direction = pos_dir if meters >= 0 else neg_dir
            out.append(Step(
                kind="MOVE_IMU",
                move_direction=direction,
                move_distance_m=abs(float(meters)),
                line_no=raw_line_no,
            ))
        elif cmd == "SCOUT":
            # Slow 360° yaw spin for visual situational awareness.
            # Yaw stick is hardcoded at 15 (a gentle rate that still
            # completes a full turn in ~10-15 s on Anafi indoor —
            # slow enough that the operator can actually see the
            # surroundings the camera passes through). No
            # arguments; advances once telemetry confirms the yaw
            # has accumulated 360°, then enters the standard brake
            # phase like the other raw-RC steps.
            if args:
                raise ScriptError(raw_line_no,
                                  f"SCOUT takes no arguments, got {args}")
            out.append(Step(kind="SCOUT", line_no=raw_line_no))
        elif cmd == "RC":
            # All four sticks at once: RC <lr> <fb> <ud> <yaw> [<seconds>].
            # Order matches the drone-stick convention
            # (roll, pitch, throttle, yaw → lr, fb, ud, yaw). Use
            # LR/FB/UD/YAW individually if you need only one axis.
            if len(args) not in (4, 5):
                raise ScriptError(raw_line_no,
                                  f"RC takes 4-5 arguments "
                                  f"(<lr> <fb> <ud> <yaw> [<seconds>]), "
                                  f"got {len(args)}")
            lr = _parse_int(args[0], raw_line_no, "RC lr")
            fb = _parse_int(args[1], raw_line_no, "RC fb")
            ud = _parse_int(args[2], raw_line_no, "RC ud")
            yaw = _parse_int(args[3], raw_line_no, "RC yaw")
            for axis_name, axis_val in (("lr", lr), ("fb", fb),
                                         ("ud", ud), ("yaw", yaw)):
                if not (-100 <= axis_val <= 100):
                    raise ScriptError(raw_line_no,
                                      f"RC {axis_name} must be in "
                                      f"[-100, +100], got {axis_val}")
            sec = (_parse_float(args[4], raw_line_no, "RC seconds")
                   if len(args) >= 5 else 1.0)
            if sec < 0:
                raise ScriptError(raw_line_no,
                                  f"RC seconds must be >= 0, got {sec}")
            out.append(Step(kind="RC", seconds=sec,
                            rc_lr=lr, rc_fb=fb, rc_ud=ud, rc_yaw=yaw,
                            line_no=raw_line_no))
        elif cmd == "DANCE":
            if len(args) > 2:
                raise ScriptError(raw_line_no,
                                  f"DANCE takes 0-2 arguments, got {len(args)}")
            sec = (_parse_float(args[0], raw_line_no, "DANCE seconds")
                   if len(args) >= 1
                   else float(_required_default(defaults, "dance_seconds", raw_line_no)))
            if sec < 0:
                raise ScriptError(raw_line_no,
                                  f"DANCE seconds must be >= 0, got {sec}")
            mode = (args[1].lower() if len(args) >= 2
                    else str(_required_default(defaults, "dance_mode", raw_line_no)).lower())
            if mode not in VALID_DANCE_MODES:
                raise ScriptError(raw_line_no,
                                  f"DANCE mode must be one of "
                                  f"{VALID_DANCE_MODES}, got {mode!r}")
            out.append(Step(kind="DANCE",
                            seconds=sec, mode=mode,
                            line_no=raw_line_no))
        else:
            raise ScriptError(raw_line_no, f"unknown command {cmd!r}")
    return out


def format(steps: List[Step]) -> str:
    """Canonicalised textual form of a Step list. Round-trips through parse."""
    lines: List[str] = []
    for s in steps:
        if s.kind == "TAKEOFF":
            lines.append("TAKEOFF")
        elif s.kind == "APPROACH":
            lines.append(f"APPROACH {s.marker_id} {s.distance:g}")
        elif s.kind == "HOOVER":
            lines.append(f"HOOVER {s.seconds:g}")
        elif s.kind == "AWAIT":
            lines.append(f"AWAIT {s.marker_id} {s.seconds:g}")
        elif s.kind == "PAUSE":
            lines.append(f"PAUSE {s.seconds:g}")
        elif s.kind == "LAND":
            lines.append("LAND")
        elif s.kind == "HEIGHT":
            lines.append(f"HEIGHT {s.height:g}")
        elif s.kind == "TO":
            # Default yaw is "auto" (omitted from canonical form).
            # A numeric yaw forces the 4-arg form; we must emit z
            # alongside, since the grammar is positional. Steps with
            # a numeric yaw and no z come from the parser's 4-arg
            # branch which always sets z, so this is well-defined.
            parts = [f"TO {s.world_x:g} {s.world_y:g}"]
            numeric_yaw = (s.yaw is not None and s.yaw != "auto")
            if s.height is not None or numeric_yaw:
                parts.append(f"{s.height:g}" if s.height is not None
                             else "0")
            if numeric_yaw:
                parts.append(f"{float(s.yaw):g}")
            lines.append(" ".join(parts))
        elif s.kind == "TO_HOME":
            if s.height is not None:
                lines.append(f"TO_HOME {s.height:g}")
            else:
                lines.append("TO_HOME")
        elif s.kind == "RC":
            # Round-trip rule:
            #   - exactly one of lr/fb/ud/yaw non-zero → the matching
            #     single-axis short form (LR_RC/FB_RC/UD_RC/YAW_RC)
            #   - everything else (multi-axis or all-zero) → RC form
            axes = (s.rc_lr, s.rc_fb, s.rc_ud, s.rc_yaw)
            nonzero = sum(1 for v in axes if v != 0)
            if nonzero == 1 and s.rc_lr != 0:
                lines.append(f"LR_RC {s.rc_lr} {s.seconds:g}")
            elif nonzero == 1 and s.rc_fb != 0:
                lines.append(f"FB_RC {s.rc_fb} {s.seconds:g}")
            elif nonzero == 1 and s.rc_ud != 0:
                lines.append(f"UD_RC {s.rc_ud} {s.seconds:g}")
            elif nonzero == 1 and s.rc_yaw != 0:
                lines.append(f"YAW_RC {s.rc_yaw} {s.seconds:g}")
            else:
                lines.append(
                    f"RC {s.rc_lr} {s.rc_fb} {s.rc_ud} {s.rc_yaw} "
                    f"{s.seconds:g}"
                )
        elif s.kind == "YAW":
            lines.append(f"YAW_IMU {int(s.rotation_deg or 0)}")
        elif s.kind == "MOVE_IMU":
            # Re-encode the direction back into the signed-meters form
            # of the appropriate single-axis IMU command.
            dir_to_cmd = {
                "forward": ("FB_IMU", +1),
                "back":    ("FB_IMU", -1),
                "right":   ("LR_IMU", +1),
                "left":    ("LR_IMU", -1),
                "up":      ("UD_IMU", +1),
                "down":    ("UD_IMU", -1),
            }
            name, sign = dir_to_cmd.get(s.move_direction or "",
                                         ("FB_IMU", +1))
            meters = sign * float(s.move_distance_m or 0.0)
            lines.append(f"{name} {meters:g}")
        elif s.kind == "SCOUT":
            lines.append("SCOUT")
        elif s.kind == "DANCE":
            lines.append(f"DANCE {s.seconds:g} {s.mode}")
        else:
            raise ValueError(f"unknown step kind {s.kind!r}")
    return "\n".join(lines)


HARDCODED_DEFAULT_SCRIPT = "TAKEOFF\nAPPROACH\nHOOVER\nLAND\n"


def defaults_from_cfg(cfg) -> dict:
    """Build the defaults dict that ``parse`` expects from a MissionConfig."""
    return {
        "marker_id":     cfg.target_marker_id,
        "distance":      cfg.target_distance_m,
        "hold_seconds":  cfg.hold_time_s,
        "height":        cfg.default_height_m,
        "dance_seconds": cfg.default_dance_seconds_s,
        "dance_mode":    DEFAULT_DANCE_MODE,
    }


def newest_flight_script(flights_root: Optional[Path]) -> Optional[str]:
    """Return the contents of the newest ``mission_script.txt`` under
    ``flights_root``, or None if there is no such file. Used by the
    page-load priority chain for the textarea draft.
    """
    if flights_root is None or not flights_root.is_dir():
        return None
    newest_mtime = -1.0
    newest_text: Optional[str] = None
    for d in flights_root.iterdir():
        if not d.is_dir():
            continue
        p = d / "mission_script.txt"
        if not p.is_file():
            continue
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if mt > newest_mtime:
            try:
                newest_text = p.read_text()
                newest_mtime = mt
            except OSError:
                continue
    return newest_text


def load_priority_script(active_path: Optional[Path],
                         flights_root: Optional[Path]) -> str:
    """Return the textarea contents per the priority chain:
    1. active draft at ``active_path`` (operator's most recent typing)
    2. operator-pinned default-by-name (set on /, "Set as default")
    3. newest per-flight ``mission_script.txt`` under ``flights_root``
    4. ``HARDCODED_DEFAULT_SCRIPT``

    Default-by-name sits in the middle: the active draft is the
    operator's WIP and MUST stay first (overriding it on every
    restart would obliterate ongoing edits). Default kicks in when
    there's no active draft -- fresh install or operator wiped it.
    """
    if active_path is not None:
        try:
            if active_path.is_file():
                return active_path.read_text()
        except OSError:
            pass
    # Operator-pinned default-by-name.
    try:
        from .config import get_default, MISSION_SCRIPTS_DIR
        name = get_default("mission_script")
        if name:
            p = MISSION_SCRIPTS_DIR / f"{name}.txt"
            if p.is_file():
                return p.read_text()
    except Exception as e:
        print(f"[mission_script] default-by-name load failed: {e}")
    flight_text = newest_flight_script(flights_root)
    if flight_text is not None:
        return flight_text
    return HARDCODED_DEFAULT_SCRIPT
