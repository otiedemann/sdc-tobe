"""Detect the active graphical (GNOME / etc.) session so UE4 can be
spawned inside it.

Why this matters:
  When sphinx-control runs as a systemd service (or under any non-
  graphical context), it inherits no DISPLAY / WAYLAND_DISPLAY /
  XAUTHORITY. A naked ``parrot-ue4-empty`` then has nowhere to draw —
  it crashes, runs offscreen, or grabs a stray virtual framebuffer. The
  user can't see the simulator, and tools that stream the active
  session (Sunshine, RDP, VNC) have nothing to capture.

  This module asks ``loginctl`` for the currently-active graphical
  session and produces the env vars a child process needs to render
  into it. The launcher splices those env vars into the UE4 child's
  ``subprocess.Popen(env=...)`` call.

Cross-platform: ``loginctl`` only exists on systemd-based Linux. On
macOS / FreeBSD / non-systemd Linux this module returns ``None`` and
the launcher falls back to its old behavior (inherit the parent's
environment unchanged) — useful for the dry-run dev path.
"""
from __future__ import annotations

import logging
import pwd
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("sphinx-control.session")


@dataclass
class SessionInfo:
    """Snapshot of an active graphical session."""

    session_id: str
    user: str
    uid: int
    session_type: str  # "x11" or "wayland"
    display: str       # e.g. ":0" or "wayland-0"
    seat: str
    env: dict[str, str]

    def summary(self) -> dict[str, str | int]:
        return {
            "session_id": self.session_id,
            "user": self.user,
            "uid": self.uid,
            "type": self.session_type,
            "display": self.display,
            "seat": self.seat,
        }


def detect_active_session() -> SessionInfo | None:
    """Return the active graphical session, or None if there isn't one
    (or if loginctl is unavailable). Only sessions with ``Active=yes``
    AND ``Type∈{x11,wayland}`` qualify; SSH / tty / lock-screen
    sessions are skipped.

    On hosts with multiple users logged in, returns the first
    qualifying session in ``loginctl``'s output. If you have a
    multi-seat setup and the wrong session is picked, override via
    ``SPHINX_CONTROL_SESSION_USER`` env var.
    """
    if shutil.which("loginctl") is None:
        return None
    try:
        out = subprocess.check_output(
            ["loginctl", "list-sessions", "--no-legend"],
            text=True, timeout=3,
        )
    except subprocess.SubprocessError as e:
        log.debug("loginctl list-sessions failed: %s", e)
        return None

    import os
    preferred_user = os.environ.get("SPHINX_CONTROL_SESSION_USER")

    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        session_id, uid_str, user = parts[0], parts[1], parts[2]
        if preferred_user and user != preferred_user:
            continue
        info = _show_session(session_id)
        if not info:
            continue
        if info.get("Active") != "yes":
            continue
        session_type = info.get("Type", "")
        if session_type not in {"x11", "wayland"}:
            continue
        try:
            uid = int(uid_str)
        except ValueError:
            continue
        env = _env_for_session(uid, user, session_type, info)
        return SessionInfo(
            session_id=session_id,
            user=user,
            uid=uid,
            session_type=session_type,
            display=info.get("Display", ""),
            seat=info.get("Seat", ""),
            env=env,
        )
    return None


def _show_session(session_id: str) -> dict[str, str] | None:
    try:
        out = subprocess.check_output(
            ["loginctl", "show-session", session_id],
            text=True, timeout=3,
        )
    except subprocess.SubprocessError:
        return None
    info: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    return info


def _env_for_session(
    uid: int, user: str, session_type: str, info: dict[str, str]
) -> dict[str, str]:
    """Build the env-var overrides a child needs to render in this
    session. Only includes keys we actually resolved — never leave
    blank values like ``DISPLAY=`` or downstream rendering breaks in
    confusing ways."""
    try:
        home = pwd.getpwuid(uid).pw_dir
    except KeyError:
        home = f"/home/{user}"
    runtime_dir = f"/run/user/{uid}"
    env: dict[str, str] = {
        "XDG_RUNTIME_DIR": runtime_dir,
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime_dir}/bus",
        "HOME": home,
        "USER": user,
        "LOGNAME": user,
    }
    display = info.get("Display") or ""
    if session_type == "wayland":
        # On Wayland, Display is e.g. "wayland-0". Some compositors
        # leave it blank in loginctl; default to wayland-0 which is
        # what GNOME / KDE on Ubuntu use by default.
        env["WAYLAND_DISPLAY"] = display or "wayland-0"
        # XDG_SESSION_TYPE helps Qt / SDL choose the right backend.
        env["XDG_SESSION_TYPE"] = "wayland"
    else:  # x11
        env["DISPLAY"] = display or ":0"
        env["XDG_SESSION_TYPE"] = "x11"
        # XAUTHORITY: try the usual paths in priority order. If none
        # exist, leave it unset and let the child's X server reject
        # the connection — a clearer failure mode than pointing at a
        # non-existent file.
        for candidate in (
            f"{home}/.Xauthority",
            f"/run/user/{uid}/gdm/Xauthority",
            f"/run/user/{uid}/Xauthority",
        ):
            if Path(candidate).is_file():
                env["XAUTHORITY"] = candidate
                break
    return env


def parse_explicit(spec: str) -> dict[str, str] | None:
    """Parse an explicit override like ``x11::0`` or ``wayland:wayland-0``.

    Use case: operator wants to pin a specific display without trusting
    auto-detection. Returns env-var overrides or None if the spec
    doesn't parse."""
    if not spec or ":" not in spec:
        return None
    sess_type, _, display = spec.partition(":")
    sess_type = sess_type.strip().lower()
    display = display.strip()
    if sess_type not in {"x11", "wayland"} or not display:
        return None
    if sess_type == "x11":
        env = {"DISPLAY": display, "XDG_SESSION_TYPE": "x11"}
    else:
        env = {"WAYLAND_DISPLAY": display, "XDG_SESSION_TYPE": "wayland"}
    return env
