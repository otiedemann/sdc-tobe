"""Runtime settings persisted across C2 restarts.

Currently tracks:
  * ``disabled_fcs`` — names of flight controllers the operator has
    silenced from the /settings page. Disabled FCs are not polled,
    don't appear on the overview, aren't included in any fleet
    broadcast (start-all, emergency-land-all, arena/tune/scripts
    push) and don't render in any checkbox row.

Storage is a small JSON file under ``marker_mission_c2/runtime/``
so it never collides with the operator-canonical ``config.json``.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_RUNTIME_DIR = _PKG_DIR / "runtime"


@dataclass
class Settings:
    disabled_fcs: set[str] = field(default_factory=set)
    # Active arena draft used by the overview viz (and writable from
    # the /arena page). None = no draft loaded yet; viz shows "no
    # arena loaded".
    arena_draft: Optional[dict] = None


class SettingsStore:
    """JSON-backed, thread-safe settings persistence.

    Writes are atomic (.tmp + rename). The lock guards both the
    in-memory ``Settings`` and the file write so a reader never sees
    a partially-mutated value.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = (Path(path) if path
                     else _DEFAULT_RUNTIME_DIR / "settings.json")
        self._lock = threading.Lock()
        self._settings = Settings()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ----------------------------------------------------------- load
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError):
            # Corrupt or unreadable file — fall back to defaults
            # rather than crashing the C2 on startup. The next save
            # rewrites the file with the clean shape.
            return
        if not isinstance(raw, dict):
            return
        disabled = raw.get("disabled_fcs") or []
        if isinstance(disabled, list):
            self._settings.disabled_fcs = {
                str(n) for n in disabled if isinstance(n, str)}
        arena = raw.get("arena_draft")
        if isinstance(arena, dict):
            self._settings.arena_draft = arena

    # ----------------------------------------------------------- write
    def _save_locked(self) -> None:
        # Caller must hold self._lock.
        tmp = self.path.with_suffix(".json.tmp")
        payload = {
            "disabled_fcs": sorted(self._settings.disabled_fcs),
            "arena_draft": self._settings.arena_draft,
        }
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)

    # --------------------------------------------------------- queries
    def is_fc_enabled(self, name: str) -> bool:
        with self._lock:
            return name not in self._settings.disabled_fcs

    def disabled_fcs(self) -> set[str]:
        with self._lock:
            return set(self._settings.disabled_fcs)

    def arena_draft(self) -> Optional[dict]:
        with self._lock:
            import copy
            return copy.deepcopy(self._settings.arena_draft)

    # --------------------------------------------------------- mutators
    def set_fc_enabled(self, name: str, enabled: bool) -> None:
        with self._lock:
            if enabled:
                self._settings.disabled_fcs.discard(name)
            else:
                self._settings.disabled_fcs.add(name)
            self._save_locked()

    def set_arena_draft(self, payload: Optional[dict]) -> None:
        with self._lock:
            self._settings.arena_draft = (
                payload if isinstance(payload, dict) else None)
            self._save_locked()
