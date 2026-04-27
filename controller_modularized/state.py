"""Shared mutable runtime state. Module-level attributes here are READ
+ WRITTEN by routes, services, and background threads. Always go through
the module — ``state.active_drone_id = "2"`` — never ``from .state import
active_drone_id`` (which copies the binding).
"""
from __future__ import annotations

import json
import os
import threading

from controller_modularized.config import (
    DEFAULT_DRONES,
    DRONES_CONFIG_PATH,
)


# ── Drone fleet (loaded at import time, mutable in place) ─────────
def _load() -> dict:
    if DRONES_CONFIG_PATH.exists():
        try:
            with open(DRONES_CONFIG_PATH) as f:
                cfg = json.load(f)
            for did, info in cfg.items():
                if not all(k in info for k in ("name", "type", "base")):
                    print(f"[CONFIG] Invalid drone entry {did}, using defaults")
                    return dict(DEFAULT_DRONES)
            print(f"[CONFIG] Loaded {len(cfg)} drones from {DRONES_CONFIG_PATH}")
            return cfg
        except Exception as e:
            print(f"[CONFIG] Error loading {DRONES_CONFIG_PATH}: {e}, using defaults")
    else:
        with open(DRONES_CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_DRONES, f, indent=2)
        print(f"[CONFIG] Created default config at {DRONES_CONFIG_PATH}")
    return dict(DEFAULT_DRONES)


DRONES: dict = _load()
active_drone_id: str = "1"

# PI_API_BASE env var overrides the base URL from drones_config.json
_env_base = os.getenv("PI_API_BASE")
if _env_base:
    PI_BASE: str = _env_base.rstrip("/")
    DRONES[active_drone_id]["base"] = PI_BASE
else:
    PI_BASE = DRONES[active_drone_id]["base"]


def save_drones_config(drones: dict) -> None:
    with open(DRONES_CONFIG_PATH, "w") as f:
        json.dump(drones, f, indent=2)


# ── Pause state ──────────────────────────────────────────────────
_global_paused: bool = False
_global_paused_at: float = 0.0
_global_paused_src: str = ""
_pause_lock = threading.Lock()


# ── Connection state (populated by the heartbeat thread) ─────────
conn_state: dict = {}
