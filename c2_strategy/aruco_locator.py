"""
ArUco position receiver for SDC26 C2 strategy.

Listens for position updates from pi_position nodes (via UDP port 5005)
or directly from the relay WebSocket.  Transforms pi_position world
coordinates to arena coordinates (arena_x = world_x + 10).

For simulator mode, positions come from the sim telemetry instead.
"""

import asyncio
import json
import logging
import time
from typing import Optional

from .models import Vec2, Vec3, TargetBox

log = logging.getLogger("aruco_locator")


class ArucoLocator:
    """
    Receives and fuses ArUco-based position updates for drones and targets.

    Coordinate transform: arena_x = world_x + 10, arena_y = world_y.
    Target markers have IDs >= target_id_min (default 30 from pi_position).
    """

    def __init__(
        self,
        target_id_min: int = 30,
        target_id_max: int = 50,
        stale_timeout: float = 2.0,
    ) -> None:
        self.target_id_min = target_id_min
        self.target_id_max = target_id_max
        self.stale_timeout = stale_timeout

        # Camera (drone) positions keyed by source name (e.g. "pi-cam-1")
        self._cam_positions: dict[str, dict] = {}

        # Discovered target boxes keyed by marker ID string
        self._target_positions: dict[str, Vec3] = {}

        # All known target boxes (shared with strategy engine)
        self.target_boxes: dict[str, TargetBox] = {}

        # Callbacks
        self._on_target_discovered: list = []

        self._running = False
        self._udp_transport: Optional[asyncio.DatagramTransport] = None

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    @staticmethod
    def world_to_arena(wx: float, wy: float, wz: float = 0.0) -> Vec3:
        """Convert pi_position world coords to arena coords."""
        return Vec3(x=wx + 10.0, y=wy, z=wz)

    @staticmethod
    def world_to_arena_2d(wx: float, wy: float) -> Vec2:
        return Vec2(x=wx + 10.0, y=wy)

    # ------------------------------------------------------------------
    # Process incoming position data
    # ------------------------------------------------------------------

    def process_position_message(self, data: dict, source: str = "unknown") -> None:
        """
        Process a position message from pi_position.

        Expected JSON format:
        {
            "cam": [x, y, z],         # camera world position
            "dir": [dx, dy, dz],      # camera direction
            "targets": {"30": [x,y,z], ...},  # target markers
            "ref_markers": [0, 5, 6], # visible ref markers
            "stale": false
        }
        """
        now = time.monotonic()

        # Camera position
        cam = data.get("cam")
        if cam and len(cam) >= 3:
            arena = self.world_to_arena(cam[0], cam[1], cam[2])
            direction = data.get("dir", [0, 0, 0])
            self._cam_positions[source] = {
                "position": arena,
                "direction": direction,
                "stale": data.get("stale", False),
                "ref_markers": data.get("ref_markers", []),
                "marker_weights": data.get("marker_weights", {}),
                "updated_at": now,
            }

        # Target markers
        targets = data.get("targets", {})
        for marker_id_str, pos in targets.items():
            try:
                mid = int(marker_id_str)
            except ValueError:
                continue

            if self.target_id_min <= mid <= self.target_id_max and len(pos) >= 3:
                arena_pos = self.world_to_arena(pos[0], pos[1], pos[2])
                self._target_positions[marker_id_str] = arena_pos
                self._register_target(marker_id_str, mid, arena_pos)

    def _register_target(self, box_id: str, marker_id: int, position: Vec3) -> None:
        """Register or update a discovered target box."""
        pos2d = Vec2(x=position.x, y=position.y)

        if box_id not in self.target_boxes:
            self.target_boxes[box_id] = TargetBox(
                box_id=box_id,
                aruco_marker_id=marker_id,
                position=pos2d,
                discovered=True,
            )
            log.info(
                f"Target discovered: marker {marker_id} at "
                f"({pos2d.x:.1f}, {pos2d.y:.1f})"
            )
            for cb in self._on_target_discovered:
                try:
                    cb(self.target_boxes[box_id])
                except Exception:
                    pass
        else:
            # Update position (may drift slightly)
            self.target_boxes[box_id].position = pos2d

    # ------------------------------------------------------------------
    # Simulator mode: update from telemetry
    # ------------------------------------------------------------------

    def update_drone_position_from_telemetry(
        self, drone_id: str, x: float, y: float, z: float = 0.0
    ) -> None:
        """
        Update drone position from simulator telemetry (already in arena coords).
        """
        now = time.monotonic()
        self._cam_positions[drone_id] = {
            "position": Vec3(x=x, y=y, z=z),
            "direction": [0, 0, 0],
            "stale": False,
            "ref_markers": [],
            "marker_weights": {},
            "updated_at": now,
        }

    def update_target_from_sim(
        self, box_id: str, marker_id: int, x: float, y: float
    ) -> None:
        """Register a target box from simulator data (already arena coords)."""
        pos = Vec3(x=x, y=y, z=0.0)
        self._target_positions[str(marker_id)] = pos
        self._register_target(box_id, marker_id, pos)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_drone_position(self, source: str) -> Optional[Vec2]:
        """Get latest position for a drone/camera source."""
        entry = self._cam_positions.get(source)
        if not entry:
            return None
        age = time.monotonic() - entry["updated_at"]
        if age > self.stale_timeout:
            return None
        p = entry["position"]
        return Vec2(x=p.x, y=p.y)

    def get_drone_altitude(self, source: str) -> Optional[float]:
        entry = self._cam_positions.get(source)
        if not entry:
            return None
        return entry["position"].z

    def get_all_drone_positions(self) -> dict[str, Vec2]:
        """Get all non-stale drone positions."""
        now = time.monotonic()
        result = {}
        for src, entry in self._cam_positions.items():
            if (now - entry["updated_at"]) <= self.stale_timeout:
                p = entry["position"]
                result[src] = Vec2(x=p.x, y=p.y)
        return result

    def get_discovered_targets(self) -> dict[str, TargetBox]:
        return dict(self.target_boxes)

    def on_target_discovered(self, callback) -> None:
        """Register a callback for when a new target is discovered."""
        self._on_target_discovered.append(callback)

    # ------------------------------------------------------------------
    # UDP listener
    # ------------------------------------------------------------------

    async def start_udp_listener(self, host: str = "0.0.0.0", port: int = 5005) -> None:
        """Start listening for UDP position packets from pi_position nodes."""
        loop = asyncio.get_running_loop()

        class _Protocol(asyncio.DatagramProtocol):
            def __init__(self, locator: ArucoLocator):
                self.locator = locator

            def datagram_received(self, data: bytes, addr: tuple) -> None:
                try:
                    msg = json.loads(data.decode("utf-8"))
                    source = f"{addr[0]}:{addr[1]}"
                    self.locator.process_position_message(msg, source=source)
                except Exception as e:
                    log.debug(f"Bad UDP packet from {addr}: {e}")

        transport, _ = await loop.create_datagram_endpoint(
            lambda: _Protocol(self),
            local_addr=(host, port),
        )
        self._udp_transport = transport
        self._running = True
        log.info(f"ArUco UDP listener started on {host}:{port}")

    async def stop(self) -> None:
        self._running = False
        if self._udp_transport:
            self._udp_transport.close()
            self._udp_transport = None
        log.info("ArUco locator stopped")

    # ------------------------------------------------------------------
    # WebSocket listener (connect to relay.py)
    # ------------------------------------------------------------------

    async def connect_relay_ws(self, url: str = "ws://localhost:8000/ws") -> None:
        """
        Connect to the ArUco relay WebSocket to receive position updates.
        Runs until cancelled.
        """
        try:
            import websockets
        except ImportError:
            log.error("websockets package required for relay connection")
            return

        self._running = True
        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    log.info(f"Connected to ArUco relay at {url}")
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            # Relay may wrap in {"node": ..., "data": ...}
                            if "data" in msg and "cam" in msg["data"]:
                                source = msg.get("node", "relay")
                                self.process_position_message(msg["data"], source)
                            elif "cam" in msg:
                                self.process_position_message(msg, source="relay")
                        except Exception as e:
                            log.debug(f"Bad WS message: {e}")
            except Exception as e:
                log.warning(f"Relay WS connection failed: {e}, retrying in 3s")
                await asyncio.sleep(3.0)

    def summary(self) -> dict:
        """Return a summary of current positioning state."""
        now = time.monotonic()
        drones = {}
        for src, entry in self._cam_positions.items():
            age = now - entry["updated_at"]
            p = entry["position"]
            drones[src] = {
                "x": round(p.x, 2),
                "y": round(p.y, 2),
                "z": round(p.z, 2),
                "age_s": round(age, 1),
                "stale": age > self.stale_timeout,
            }
        targets = {}
        for tid, box in self.target_boxes.items():
            if box.position:
                targets[tid] = {
                    "x": round(box.position.x, 2),
                    "y": round(box.position.y, 2),
                    "marker_id": box.aruco_marker_id,
                }
        return {"drones": drones, "targets": targets}
