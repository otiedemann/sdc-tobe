"""C2 V2 runtime config — the five real flight controllers and poll cadence.

No simulation: V2 connects straight to flightctrl1-5 on :8080 (operator
requirement #4). The FC list is fixed by default but overridable on the CLI for
bench testing against a subset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Reuse the proven FC endpoint spec (host/port -> base_url) rather than
# re-deriving it.
from marker_mission_c2.config import FCSpec

# The competition fleet: five drones, one marker_mission FC each, all on :8080.
DEFAULT_FC_HOSTS: tuple[str, ...] = (
    "flightctrl1", "flightctrl2", "flightctrl3", "flightctrl4", "flightctrl5",
)
FC_PORT: int = 8080


def default_fcs() -> List[FCSpec]:
    """The five real flight controllers (flightctrl1-5:8080)."""
    return [FCSpec(name=h, host=h, port=FC_PORT) for h in DEFAULT_FC_HOSTS]


def fcs_from_hosts(hosts: list[str]) -> List[FCSpec]:
    """Build FCSpecs from explicit host[:port] strings (CLI override).

    Accepts ``flightctrl2`` or ``flightctrl2:8080`` or ``192.168.0.5:8080``;
    the name is the host (so it maps to the flightctrlN hostname for the UI)."""
    out: List[FCSpec] = []
    for raw in hosts:
        h = raw.strip()
        if not h:
            continue
        if ":" in h:
            host, _, p = h.rpartition(":")
            try:
                port = int(p)
            except ValueError:
                host, port = h, FC_PORT
        else:
            host, port = h, FC_PORT
        out.append(FCSpec(name=host, host=host, port=port))
    return out


@dataclass(frozen=True)
class PollConfig:
    # /api/state poll rate per FC. The FC publishes at ~10 Hz; 5 Hz is plenty
    # for strategy + a smooth dashboard without hammering the link.
    state_hz: float = 5.0
    # Slow-cadence reads (serial number, WiFi SSID) — they rarely change.
    identity_period_s: float = 10.0
    wlan_period_s: float = 8.0
    # Per-request HTTP timeout. Short, so a momentarily-unreachable FC marks
    # itself disconnected fast (the v7 link-continuity / safety logic cares).
    request_timeout_s: float = 1.5
    # A drone is considered CONNECTED only if a /api/state read succeeded within
    # this window. Past it, the dashboard shows it offline.
    stale_after_s: float = 3.0
