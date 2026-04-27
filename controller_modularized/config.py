"""Static configuration: env vars, ports, timeouts, paths.

These values are evaluated once at import time and never change at
runtime. Mutable state lives in ``state.py`` instead.
"""
from __future__ import annotations

import os
from pathlib import Path


# ── HTTP server ────────────────────────────────────────────────────
HTTP_HOST: str = "0.0.0.0"
HTTP_PORT: int = 8090

# ── Pi-side request timeouts ───────────────────────────────────────
TIMEOUT_CMD: float = float(os.getenv("PI_TIMEOUT_CMD", "8.0"))
TIMEOUT_STATUS: float = float(os.getenv("PI_TIMEOUT_STATUS", "0.5"))
TIMEOUT_FAST: float = float(os.getenv("PI_TIMEOUT_FAST", "1.5"))
TIMEOUT_SLOW: float = float(os.getenv("PI_TIMEOUT_SLOW", "15.0"))

# ── Video forwarding ───────────────────────────────────────────────
VIDEO_UDP_FORWARD_PORT: int = int(os.getenv("VIDEO_UDP_FORWARD_PORT", "55004"))
VIDEO_FPS: int = int(os.getenv("VIDEO_FPS", "30"))
VIDEO_JPEG_QUALITY: int = int(os.getenv("VIDEO_JPEG_QUALITY", "70"))

# ── Drone fleet config file ───────────────────────────────────────
DRONES_CONFIG_PATH: Path = Path(__file__).resolve().parent / "drones_config.json"

DEFAULT_DRONES: dict = {
    "1": {"name": "Anafi 1", "type": "anafi", "base": "http://flightctrl1:8080"},
    "2": {"name": "Anafi 2", "type": "anafi", "base": "http://flightctrl2:8080"},
    "3": {"name": "Anafi 3", "type": "anafi", "base": "http://flightctrl3:8080"},
    "4": {"name": "Anafi 4", "type": "anafi", "base": "http://flightctrl4:8080"},
}

# ── Per-flight automatic logging ──────────────────────────────────
FLIGHT_LOG_DIR: Path = Path(os.getenv("FLIGHT_LOG_DIR", "flight_logs")).resolve()
FLIGHT_LOG_HZ: float = float(os.getenv("FLIGHT_LOG_HZ", "5.0"))

# ── C2 version banner ─────────────────────────────────────────────
C2_CODE_VERSION: str = "2026-04-24-cr (FC version endpoint + C2/FC mismatch check)"
