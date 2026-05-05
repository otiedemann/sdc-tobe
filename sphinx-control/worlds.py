"""World/UE4-app registry plus drone-profile registry.

Reads both lists from the YAML config (see config.example.yaml) and
exposes typed accessors. Files referenced by config (UE4 binaries,
drone descriptors, optional config-files) are existence-checked at
startup so the dashboard can mark broken entries instead of failing
silently mid-spawn.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WorldEntry:
    name: str
    binary: str
    description: str = ""
    config_file: str | None = None  # Sphinx --config-file YAML, optional
    # Extra UE4 command-line args (e.g. -ams-path=PerimeterPath,Malcolm,...)
    # appended verbatim to the parrot-ue4-* invocation.
    extra_ue4_args: list[str] | None = None
    available: bool = True
    unavailable_reason: str | None = None


@dataclass
class DroneProfile:
    name: str
    descriptor: str  # e.g. anafi2.drone
    descriptor_path: str  # absolute path resolved at load time
    description: str = ""
    available: bool = True
    unavailable_reason: str | None = None


class Registry:
    def __init__(
        self,
        worlds: list[WorldEntry],
        profiles: list[DroneProfile],
    ) -> None:
        self.worlds = {w.name: w for w in worlds}
        self.profiles = {p.name: p for p in profiles}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Registry":
        sphinx_cfg = cfg.get("sphinx", {}) or {}
        descriptor_dir = Path(
            sphinx_cfg.get(
                "drone_descriptor_dir",
                "/opt/parrot-sphinx/usr/share/sphinx/drones",
            )
        )

        worlds: list[WorldEntry] = []
        for raw in cfg.get("ue4_apps", []) or []:
            entry = WorldEntry(
                name=str(raw["name"]),
                binary=str(raw["binary"]),
                description=str(raw.get("description", "")),
                config_file=raw.get("config_file"),
                extra_ue4_args=list(raw.get("extra_ue4_args") or []),
            )
            if shutil.which(entry.binary) is None and not Path(entry.binary).is_file():
                entry.available = False
                entry.unavailable_reason = f"binary not found: {entry.binary}"
            elif entry.config_file:
                cfg_path = _resolve_optional_path(entry.config_file)
                if not cfg_path.is_file():
                    entry.available = False
                    entry.unavailable_reason = f"config_file missing: {cfg_path}"
                else:
                    entry.config_file = str(cfg_path)
            worlds.append(entry)

        profiles: list[DroneProfile] = []
        for raw in cfg.get("drone_profiles", []) or []:
            descriptor = str(raw["descriptor"])
            descriptor_path = (descriptor_dir / descriptor).resolve()
            profile = DroneProfile(
                name=str(raw["name"]),
                descriptor=descriptor,
                descriptor_path=str(descriptor_path),
                description=str(raw.get("description", "")),
            )
            if not descriptor_path.is_file():
                profile.available = False
                profile.unavailable_reason = (
                    f"descriptor missing: {descriptor_path}"
                )
            profiles.append(profile)

        return cls(worlds=worlds, profiles=profiles)

    def world(self, name: str) -> WorldEntry:
        if name not in self.worlds:
            raise KeyError(f"unknown world: {name}")
        return self.worlds[name]

    def profile(self, name: str) -> DroneProfile:
        if name not in self.profiles:
            raise KeyError(f"unknown drone profile: {name}")
        return self.profiles[name]

    def list_worlds(self) -> list[WorldEntry]:
        return list(self.worlds.values())

    def list_profiles(self) -> list[DroneProfile]:
        return list(self.profiles.values())


def _resolve_optional_path(p: str) -> Path:
    """Expand ~/$VARS and resolve relative paths against the
    sphinx-control/ directory (so example config_file values can be
    written relative to where the service is checked out)."""
    expanded = os.path.expandvars(os.path.expanduser(p))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = (Path(__file__).resolve().parent / candidate).resolve()
    return candidate
