"""
Tiny mission-control language.

Parses an operator-typed text script into a list of ``Step`` objects
that ``MissionController._advance_script`` walks one at a time. The
language is one-command-per-line, case-insensitive, with ``#`` comments
and blank lines ignored.

Nine commands:

    TAKEOFF
    APPROACH [<marker-id>] [<distance>]
    HOOVER   [<seconds>]
    AWAIT    <marker-id> <timeout-seconds>
    PAUSE    <seconds>
    LAND
    HEIGHT   [<height>]
    TO       <x> <y> [<z>]
    DANCE    [<seconds>] [<mode>]            mode in {wobble, spin, random}

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
    1. active draft at ``active_path``
    2. newest per-flight ``mission_script.txt`` under ``flights_root``
    3. ``HARDCODED_DEFAULT_SCRIPT``
    """
    if active_path is not None:
        try:
            if active_path.is_file():
                return active_path.read_text()
        except OSError:
            pass
    flight_text = newest_flight_script(flights_root)
    if flight_text is not None:
        return flight_text
    return HARDCODED_DEFAULT_SCRIPT
