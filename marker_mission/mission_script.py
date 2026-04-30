"""
Tiny mission-control language.

Parses an operator-typed text script into a list of ``Step`` objects
that ``MissionController._advance_script`` walks one at a time. The
language is one-command-per-line, case-insensitive, with ``#`` comments
and blank lines ignored.

Six commands:

    TAKEOFF
    APPROACH [<marker-id>] [<distance>]
    HOOVER   [<seconds>]
    LAND
    HEIGHT   [<height>]
    DANCE    [<seconds>] [<mode>]            mode in {wobble, spin, random}

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
        elif s.kind == "LAND":
            lines.append("LAND")
        elif s.kind == "HEIGHT":
            lines.append(f"HEIGHT {s.height:g}")
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
