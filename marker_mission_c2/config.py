"""C2 server config: dataclass + JSON loader + validator.

Resolution order for `load_config()`:
    1. explicit path argument
    2. env var ``MARKER_MISSION_C2_CONFIG``
    3. ``./marker_mission_c2/config.json``
    4. ``./marker_mission_c2/config.example.json`` (always shipped)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class C2ConfigError(ValueError):
    pass


_PKG_DIR = Path(__file__).resolve().parent
_EXAMPLE_PATH = _PKG_DIR / "config.example.json"


@dataclass(frozen=True)
class BindCfg:
    host: str = "127.0.0.1"
    port: int = 8090


@dataclass(frozen=True)
class FCSpec:
    name: str
    host: str
    port: int = 8080

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class C2Config:
    bind: BindCfg
    fcs: tuple[FCSpec, ...]
    emergency_land_key: str = "?"
    state_poll_hz: float = 5.0
    ui_refresh_hz: float = 2.0
    calibration_sync_seconds: float = 30.0
    fc_request_timeout_s: float = 2.0
    library_dir: str = "marker_mission_c2/calibrations"
    arena_drafts_dir: str = "marker_mission_c2/arena_drafts"
    log_level: str = "INFO"

    def library_path(self) -> Path:
        p = Path(self.library_dir)
        return p if p.is_absolute() else (_PKG_DIR.parent / p)

    def arena_drafts_path(self) -> Path:
        p = Path(self.arena_drafts_dir)
        return p if p.is_absolute() else (_PKG_DIR.parent / p)


def _validate(raw: dict, source: str) -> C2Config:
    try:
        bind_raw = raw.get("bind") or {}
        bind = BindCfg(
            host=str(bind_raw.get("host", "127.0.0.1")),
            port=int(bind_raw.get("port", 8090)),
        )
        if not (1 <= bind.port <= 65535):
            raise C2ConfigError(f"{source}: bind.port out of range")

        fcs_raw = raw.get("fcs") or []
        if not isinstance(fcs_raw, list) or not fcs_raw:
            raise C2ConfigError(f"{source}: fcs[] is empty")
        fcs: list[FCSpec] = []
        seen: set[str] = set()
        for i, item in enumerate(fcs_raw):
            if not isinstance(item, dict):
                raise C2ConfigError(f"{source}: fcs[{i}] not an object")
            name = str(item.get("name") or "").strip()
            host = str(item.get("host") or "").strip()
            port = int(item.get("port", 8080))
            if not name:
                raise C2ConfigError(f"{source}: fcs[{i}].name missing")
            if not host:
                raise C2ConfigError(f"{source}: fcs[{i}].host missing")
            if name in seen:
                raise C2ConfigError(f"{source}: duplicate fc name {name!r}")
            if not (1 <= port <= 65535):
                raise C2ConfigError(
                    f"{source}: fcs[{i}].port out of range ({port})")
            seen.add(name)
            fcs.append(FCSpec(name=name, host=host, port=port))

        emergency_key = str(raw.get("emergency_land_key", "?"))
        if len(emergency_key) != 1:
            raise C2ConfigError(
                f"{source}: emergency_land_key must be exactly one character "
                f"(got {emergency_key!r})")

        state_hz = float(raw.get("state_poll_hz", 5.0))
        if not (0.1 <= state_hz <= 10.0):
            raise C2ConfigError(
                f"{source}: state_poll_hz {state_hz} out of [0.1, 10]")
        ui_hz = float(raw.get("ui_refresh_hz", 2.0))
        if not (0.1 <= ui_hz <= 10.0):
            raise C2ConfigError(
                f"{source}: ui_refresh_hz {ui_hz} out of [0.1, 10]")
        cal_secs = float(raw.get("calibration_sync_seconds", 30.0))
        if not (1.0 <= cal_secs <= 3600.0):
            raise C2ConfigError(
                f"{source}: calibration_sync_seconds {cal_secs} out of "
                "[1, 3600]")
        timeout = float(raw.get("fc_request_timeout_s", 2.0))
        if not (0.1 <= timeout <= 30.0):
            raise C2ConfigError(
                f"{source}: fc_request_timeout_s {timeout} out of [0.1, 30]")

        return C2Config(
            bind=bind,
            fcs=tuple(fcs),
            emergency_land_key=emergency_key,
            state_poll_hz=state_hz,
            ui_refresh_hz=ui_hz,
            calibration_sync_seconds=cal_secs,
            fc_request_timeout_s=timeout,
            library_dir=str(raw.get("library_dir",
                                    "marker_mission_c2/calibrations")),
            arena_drafts_dir=str(raw.get("arena_drafts_dir",
                                         "marker_mission_c2/arena_drafts")),
            log_level=str(raw.get("log_level", "INFO")),
        )
    except (TypeError, ValueError) as e:
        raise C2ConfigError(f"{source}: {e}") from e


def load_config(explicit_path: Optional[str] = None) -> C2Config:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    env_path = os.environ.get("MARKER_MISSION_C2_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(_PKG_DIR / "config.json")
    candidates.append(_EXAMPLE_PATH)

    for path in candidates:
        if path.exists():
            try:
                raw = json.loads(path.read_text())
            except json.JSONDecodeError as e:
                raise C2ConfigError(f"{path}: invalid JSON: {e}") from e
            return _validate(raw, str(path))
    raise C2ConfigError(
        f"no config file found (tried: {[str(p) for p in candidates]})")
