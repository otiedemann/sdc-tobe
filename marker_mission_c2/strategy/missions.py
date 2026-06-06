"""Mission-step recorder.

Every time the strategy pushes a mission script to a flight controller it is
recorded here — the EXACT text the C2 sent to ``marker_mission`` — so the
operator can read it back and copy-paste it onto a live drone for testing.

Two sinks, both optional and best-effort (a logging failure must never break
the tick loop):

* an in-memory ring (newest last) exposed via ``recent()`` for the dashboard
  / ``GET /api/missions``;
* an append-only text FILE in a copy-paste-friendly layout — a ``#`` comment
  header (the marker_mission DSL ignores ``#`` lines, so the whole block can
  be pasted as-is) followed by the raw verb lines, then a blank separator.

The file looks like::

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [#7  14:03:11]  red3  ·  attacker: capture slot 4 (id=34)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TAKEOFF
    UD_RC 100 1.2
    HEIGHT 2.70
    ...
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

_RULE = "━" * 64   # ━ heavy horizontal rule (in a # comment, ignored by FC)


class MissionLog:
    """Thread-safe recorder of pushed mission scripts (ring + optional file)."""

    def __init__(self, path: Optional[str | Path] = None, capacity: int = 200) -> None:
        self._path: Optional[Path] = Path(path) if path else None
        self._lock = threading.Lock()
        self._ring: Deque[Dict[str, Any]] = deque(maxlen=max(1, capacity))
        self._seq = 0
        if self._path is not None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                # Truncate at startup so each run starts with a clean file.
                header = (
                    f"# marker_mission mission-step log — started "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"# Each block below is exactly what the C2 sent to a "
                    f"flight controller.\n"
                    f"# '#' lines are comments (ignored by the FC) — paste a "
                    f"whole block onto a live drone.\n\n"
                )
                self._path.write_text(header)
            except OSError:
                self._path = None   # unwritable -> ring only, never crash

    @property
    def path(self) -> Optional[str]:
        return str(self._path) if self._path else None

    def record(self, fc_name: str, reason: str, script: str) -> None:
        """Append one pushed mission. Best-effort; never raises."""
        script = (script or "").rstrip("\n")
        ts = time.time()
        with self._lock:
            self._seq += 1
            seq = self._seq
            entry = {
                "seq": seq,
                "unix": ts,
                "time": time.strftime("%H:%M:%S", time.localtime(ts)),
                "fc": fc_name,
                "reason": reason or "",
                "lines": script.count("\n") + 1 if script else 0,
                "script": script,
            }
            self._ring.append(entry)
            if self._path is not None:
                try:
                    block = (
                        f"# {_RULE}\n"
                        f"# [#{seq}  {entry['time']}]  {fc_name}  ·  "
                        f"{entry['reason']}\n"
                        f"# {_RULE}\n"
                        f"{script}\n\n"
                    )
                    with self._path.open("a") as f:
                        f.write(block)
                except OSError:
                    pass   # keep the ring even if the file goes away

    def recent(self, limit: int = 50, fc: Optional[str] = None) -> List[Dict[str, Any]]:
        """Newest-first list of recent missions (optionally filtered by fc)."""
        with self._lock:
            items = list(self._ring)
        if fc:
            items = [e for e in items if e["fc"] == fc]
        items.reverse()                       # newest first
        return items[: max(1, limit)]
