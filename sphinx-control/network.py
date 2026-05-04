"""Multi-instance network strategies for Sphinx drones.

Sphinx 2.x ties one drone to one Sphinx process and (by default) one IP
address. To run several drones on the same host we have two options:

* **ports** mode — every drone shares the host's IP but listens on a
  unique port (`base_port + instance_id`). External clients use
  ``host:port`` to address each drone. No root, no kernel features —
  just argparse-driven port allocation. Good default for development
  and Tailscale-fronted access (the host's tailnet IP plus a port is
  the addressable endpoint).

* **netns** mode — each drone lives in its own Linux network
  namespace, with a unique IP in a private subnet (default
  10.202.0.0/24). A bridge connects the namespaces to the host,
  Tailscale advertises the subnet via ``--advertise-routes`` so external
  tailnet members can route to drone IPs directly. Closer to "real
  fleet" but needs CAP_NET_ADMIN (typically root or sudo).

Both modes return a ``DroneEndpoint`` describing what the management UI
should display. The *launcher* layer translates that into Sphinx CLI
flags; this module is just the IP/port allocator.
"""
from __future__ import annotations

import ipaddress
import socket
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class DroneEndpoint:
    """How a particular Sphinx drone instance should be addressed."""

    instance_id: int
    ip: str
    port: int | None
    netns: str | None  # populated only in netns mode

    def display(self) -> str:
        if self.netns:
            return f"{self.ip} (netns {self.netns})"
        if self.port:
            return f"{self.ip}:{self.port}"
        return self.ip


class PortsMode:
    """Single-IP, distinct-ports allocator."""

    def __init__(self, base_port: int, bind_ip: str | None) -> None:
        self.base_port = int(base_port)
        self.bind_ip = bind_ip or _detect_primary_ip()

    def endpoint_for(self, instance_id: int) -> DroneEndpoint:
        return DroneEndpoint(
            instance_id=instance_id,
            ip=self.bind_ip,
            port=self.base_port + instance_id,
            netns=None,
        )

    def setup(self, instance_id: int) -> None:
        # Nothing to do in ports mode — the launcher will tell Sphinx
        # which port to bind via CLI flags.
        return None

    def teardown(self, instance_id: int) -> None:
        return None


class NetnsMode:
    """Per-drone Linux network namespace allocator.

    Each drone gets a netns named ``sphinx-drone-<instance_id>`` with
    one veth pair joined to a bridge on the host. All netns/veth/bridge
    operations require CAP_NET_ADMIN (typically root). On unprivileged
    hosts the setup methods raise so the launcher falls back to ports.

    NOTE: This mode is INTENTIONALLY NOT auto-invoked unless explicitly
    selected in config. Network namespace surgery on a misconfigured
    host can take down the management interface itself; we only do it
    on demand.
    """

    def __init__(self, subnet: str, bridge_name: str, ip_offset: int = 10) -> None:
        self.subnet = ipaddress.ip_network(subnet, strict=False)
        self.bridge_name = bridge_name
        self.ip_offset = int(ip_offset)

    def endpoint_for(self, instance_id: int) -> DroneEndpoint:
        ip = str(self.subnet.network_address + self.ip_offset + instance_id)
        return DroneEndpoint(
            instance_id=instance_id,
            ip=ip,
            port=None,
            netns=f"sphinx-drone-{instance_id}",
        )

    def setup(self, instance_id: int) -> None:
        ep = self.endpoint_for(instance_id)
        prefix = self.subnet.prefixlen
        ns = ep.netns
        veth_h = f"veth-h{instance_id}"
        veth_d = f"veth-d{instance_id}"

        # Idempotent: each command may already have run if a previous
        # spawn of this drone didn't tear down. We log+continue rather
        # than abort.
        cmds: list[list[str]] = [
            ["ip", "netns", "add", ns],
            ["ip", "link", "add", veth_h, "type", "veth", "peer", "name", veth_d],
            ["ip", "link", "set", veth_d, "netns", ns],
            ["ip", "link", "set", veth_h, "master", self.bridge_name],
            ["ip", "link", "set", veth_h, "up"],
            ["ip", "-n", ns, "addr", "add", f"{ep.ip}/{prefix}", "dev", veth_d],
            ["ip", "-n", ns, "link", "set", veth_d, "up"],
            ["ip", "-n", ns, "link", "set", "lo", "up"],
        ]
        for cmd in cmds:
            try:
                subprocess.run(
                    ["sudo", "-n", *cmd],
                    check=True, capture_output=True, text=True, timeout=5,
                )
            except subprocess.CalledProcessError as e:
                # Common case: object already exists. Bubble up only if
                # the error looks substantive (e.g. permissions).
                if "File exists" in (e.stderr or "") or "exists" in (e.stderr or ""):
                    continue
                raise RuntimeError(
                    f"netns setup failed for drone {instance_id}: {e.stderr}"
                ) from e
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"netns setup timed out for drone {instance_id}: {cmd}"
                ) from e

    def teardown(self, instance_id: int) -> None:
        ns = f"sphinx-drone-{instance_id}"
        veth_h = f"veth-h{instance_id}"
        for cmd in (
            ["ip", "link", "delete", veth_h],
            ["ip", "netns", "delete", ns],
        ):
            subprocess.run(
                ["sudo", "-n", *cmd],
                capture_output=True, text=True, timeout=5,
            )

    def ensure_bridge(self) -> None:
        """Idempotent bridge creation. Caller should invoke once at
        service startup before launching any netns drones."""
        cmds: list[list[str]] = [
            ["ip", "link", "add", "name", self.bridge_name, "type", "bridge"],
            ["ip", "link", "set", self.bridge_name, "up"],
            ["ip", "addr", "add",
             f"{self.subnet.network_address + 1}/{self.subnet.prefixlen}",
             "dev", self.bridge_name],
        ]
        for cmd in cmds:
            try:
                subprocess.run(
                    ["sudo", "-n", *cmd],
                    check=True, capture_output=True, text=True, timeout=5,
                )
            except subprocess.CalledProcessError as e:
                if "File exists" in (e.stderr or "") or "exists" in (e.stderr or ""):
                    continue
                raise RuntimeError(f"bridge setup failed: {e.stderr}") from e


def _detect_primary_ip() -> str:
    """Find the host's primary outbound IP without sending real traffic.
    Falls back to 127.0.0.1 if even DNS sockets fail."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def listening_pids_on_port(port: int) -> list[int]:
    """Best-effort: who's listening on this TCP port. Used by the
    `connections` view to show which clients are bound to a drone."""
    try:
        out = subprocess.check_output(
            ["ss", "-tnHp", f"sport = :{port}"],
            text=True, timeout=2,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    pids: list[int] = []
    for line in out.splitlines():
        # ss output: "...users:((\"name\",pid=12345,fd=...)..."
        if "pid=" not in line:
            continue
        try:
            after = line.split("pid=", 1)[1]
            pids.append(int(after.split(",", 1)[0]))
        except (ValueError, IndexError):
            continue
    return pids


def established_peers_to_port(port: int) -> list[str]:
    """Best-effort: who is currently CONNECTED to this drone's port.
    Returns peer "host:port" strings."""
    try:
        out = subprocess.check_output(
            ["ss", "-tnH", "state", "established", f"sport = :{port}"],
            text=True, timeout=2,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    peers: list[str] = []
    for line in out.splitlines():
        cols = line.split()
        # ss columns: Recv-Q Send-Q Local-Addr:Port Peer-Addr:Port
        if len(cols) >= 4:
            peers.append(cols[3])
    return peers
