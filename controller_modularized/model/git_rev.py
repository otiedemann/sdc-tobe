"""Snapshot the repo revision at startup. Stamped into every flight-log
header so post-flight analysis can match a log to the exact code that
produced it."""
from __future__ import annotations

import subprocess
from pathlib import Path


def read_git_revision() -> dict:
    """Return a small dict describing the repo state at C2 startup.
    Safe under: no .git dir, detached HEAD, git not on PATH."""
    info: dict = {}
    repo = Path(__file__).resolve().parent.parent.parent

    def _run(args: list[str]) -> str:
        try:
            r = subprocess.run(args, cwd=repo, capture_output=True,
                               text=True, timeout=2)
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            pass
        return ""

    info["sha"]       = _run(["git", "rev-parse", "HEAD"])
    info["short_sha"] = _run(["git", "rev-parse", "--short", "HEAD"])
    info["branch"]    = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    info["dirty"]     = bool(_run(["git", "status", "--porcelain"]))
    info["subject"]   = _run(["git", "log", "-1", "--pretty=%s"])
    return info
