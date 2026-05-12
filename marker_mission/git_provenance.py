"""
Capture the git commit (sha + branch + dirty marker) of the
marker_mission package at runtime so per-flight directories can
record what code was actually flying.

Used by mission.py at flight-start to write ``git_commit.txt``
into the flight directory. Returns an empty string when git
isn't available, the package isn't a checkout, or anything else
that would normally break -- callers can treat that as "unknown"
and the flight artefacts are still produced.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def _git(args: list, cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True, text=True, timeout=2.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def describe_marker_mission_commit() -> str:
    """Return a short, human-readable provenance string for the
    marker_mission package's git checkout. Examples:

    - ``commit=8c06530 branch=marker_mission``
    - ``commit=8c06530 branch=marker_mission dirty=+modified``
    - ``commit=8c06530+modified branch=marker_mission`` (alt form
      from describe --dirty)

    Empty string when git can't tell us anything.
    """
    pkg_dir = Path(__file__).resolve().parent
    sha = _git(["rev-parse", "--short=10", "HEAD"], pkg_dir)
    if sha is None:
        return ""
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], pkg_dir) or "?"
    # `git status --porcelain` is empty <=> clean tree.
    status = _git(["status", "--porcelain"], pkg_dir)
    dirty = (status is not None and status != "")
    line = f"commit={sha} branch={branch}"
    if dirty:
        line += " dirty=true"
    # Optional: the most recent commit message subject for quick
    # at-a-glance triage when reviewing many flights.
    subject = _git(["log", "-1", "--format=%s"], pkg_dir)
    if subject:
        line += f" subject={subject!r}"
    return line
