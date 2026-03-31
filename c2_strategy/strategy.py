"""
Game strategy engine for SDC26.

Implements the main state machine:
  SETUP → SCOUTING → SIMULTANEOUS_ATTACK → TEAM_RETURN → REPEAT / INSTANT_WIN

Strategy priorities (from the v4.0 regulations):
1. Special Capture combo  — capture then team return for 5pt bonus
2. Simultaneous Attack    — 2+ drones capture at same tick → 10pt bonus
3. ArUco dominance        — discover all boxes, pick optimal attack order
4. Defensive orbit patrol — avoid sitting in home zone doing nothing
5. Instant Win provocation — if controlling 5+, rush the 6th
"""

import asyncio
import logging
import math
import time
from typing import Optional

from .arena_config import ArenaConfig
from .aruco_locator import ArucoLocator
from .drone_client import FleetClient
from .models import (
    DronePhase,
    DroneRole,
    DroneState,
    GamePhase,
    GameState,
    TargetBox,
    Vec2,
)
from .scoring import ScoreTracker

log = logging.getLogger("strategy")


class StrategyEngine:
    """
    Central strategy engine.  Runs as an async loop, reading positions
    from ArucoLocator, issuing commands via FleetClient, and tracking
    score via ScoreTracker.
    """

    TICK_INTERVAL = 0.25  # seconds between strategy ticks

    # RC speed constants (−100 to 100 range)
    RC_SPEED = 40          # forward/lateral speed for navigation
    RC_YAW_SCAN = 35       # yaw speed during scanning rotation
    RC_YAW_NAV = 0         # yaw during straight-line navigation
    SCAN_DURATION = 6.0    # seconds for a full scan rotation
    ARRIVAL_THRESHOLD = 0.8  # meters — "close enough" to waypoint

    def __init__(
        self,
        config: ArenaConfig,
        fleet: FleetClient,
        locator: ArucoLocator,
        scorer: ScoreTracker,
    ) -> None:
        self.config = config
        self.fleet = fleet
        self.locator = locator
        self.scorer = scorer

        self.state = GameState(
            our_team=config.our_team,
            phase=GamePhase.SETUP,
        )
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Strategy tuning
        self.hover_altitude_m = 1.5
        self.attack_speed = 50
        self.return_speed = 60

        # Per-drone scout state
        self._scout_waypoints: dict[str, list[Vec2]] = {}
        self._scout_wp_index: dict[str, int] = {}
        self._scan_start: dict[str, float] = {}  # when current scan started

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.ensure_future(self._loop())
            log.info("Strategy engine started")

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        log.info("Strategy engine stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Strategy tick error")
            await asyncio.sleep(self.TICK_INTERVAL)

    async def _tick(self) -> None:
        """One strategy iteration."""
        # 1. Update drone states from telemetry + locator
        await self._update_drone_states()

        # 2. Sync discovered targets
        self._sync_targets()

        # 3. Check scoring
        completed = self.scorer.check_captures()
        for ev in completed:
            self.state.captures_log.append(ev.model_dump())
            log.info(f"Capture complete: box {ev.box_id} by {ev.drone_id} (+{ev.points})")

        self.state.score_us = self.scorer.our_score
        self.state.score_them = self.scorer.their_score

        # 4. Check instant win
        controlled = {
            b.box_id for b in self.state.target_boxes.values()
            if b.captured_by == self.config.our_team
        }
        if self.scorer.check_instant_win(controlled):
            self.state.phase = GamePhase.GAME_OVER
            log.info("INSTANT WIN!")
            return

        # 5. Run phase-specific logic
        phase = self.state.phase
        if phase == GamePhase.SETUP:
            await self._phase_setup()
        elif phase == GamePhase.SCOUTING:
            await self._phase_scouting()
        elif phase == GamePhase.SIMULTANEOUS_ATTACK:
            await self._phase_simultaneous_attack()
        elif phase == GamePhase.TEAM_RETURN:
            await self._phase_team_return()
        elif phase == GamePhase.DEFEND:
            await self._phase_defend()
        elif phase == GamePhase.INSTANT_WIN_ATTEMPT:
            await self._phase_instant_win_attempt()

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    async def _update_drone_states(self) -> None:
        """Poll telemetry and merge with locator data."""
        telemetry = await self.fleet.poll_all_telemetry()

        for drone_id, ds in self.state.drones.items():
            telem = telemetry.get(drone_id, {})
            if telem:
                ds.last_telemetry_at = time.monotonic()
                ds.connected = True
                ds.battery_pct = telem.get("bat", telem.get("battery", ds.battery_pct))
                ds.flying = telem.get("flying", ds.flying)
                ds.altitude_m = telem.get("h", telem.get("altitude", ds.altitude_m))
                raw_heading = telem.get("yaw", telem.get("heading", ds.heading_deg))
                ds.heading_deg = raw_heading % 360  # normalize to [0, 360)

                # Position from telemetry (sim mode gives x/y directly)
                tx = telem.get("x")
                ty = telem.get("y")
                if tx is not None and ty is not None:
                    ds.position = Vec2(x=float(tx), y=float(ty))
                    # Also feed into locator for unified view
                    self.locator.update_drone_position_from_telemetry(
                        drone_id, float(tx), float(ty),
                        float(telem.get("z", telem.get("h", 0)))
                    )

            # Try ArUco locator position (overwrites sim if available)
            aruco_pos = self.locator.get_drone_position(drone_id)
            if aruco_pos:
                ds.position = aruco_pos
                alt = self.locator.get_drone_altitude(drone_id)
                if alt is not None:
                    ds.altitude_m = alt

            # Mark as lost if no updates
            if ds.connected and (time.monotonic() - ds.last_telemetry_at) > 5.0:
                ds.connected = False
                ds.phase = DronePhase.LOST
                log.warning(f"Drone {drone_id} lost (no telemetry)")

    def _sync_targets(self) -> None:
        """Merge locator discovered targets into game state."""
        for box_id, target in self.locator.get_discovered_targets().items():
            if box_id not in self.state.target_boxes:
                self.state.target_boxes[box_id] = target
            elif target.position:
                self.state.target_boxes[box_id].position = target.position
                self.state.target_boxes[box_id].discovered = True

    # ------------------------------------------------------------------
    # Phase: SETUP — initialize fleet, wait for ready
    # ------------------------------------------------------------------

    async def _phase_setup(self) -> None:
        """
        Initialize drones. Transition to SCOUTING when all connected.
        If drones are already flying (manual takeoff), go straight to scouting.
        """
        connected = [ds for ds in self.state.drones.values() if ds.connected]
        flying = [ds for ds in self.state.drones.values() if ds.flying]

        if not connected:
            return

        # If any drones are already flying, transition immediately
        if flying:
            log.info(f"{len(flying)} drones flying → transitioning to SCOUTING")
            self.state.phase = GamePhase.SCOUTING
            self.state.start_time = time.monotonic()
            for ds in self.state.drones.values():
                ds.role = DroneRole.SCOUT
            return

        # All connected but none flying — take off
        if len(connected) == len(self.state.drones):
            log.info("All drones connected → taking off and transitioning to SCOUTING")
            self.state.phase = GamePhase.SCOUTING
            self.state.start_time = time.monotonic()

            for drone_id, ds in self.state.drones.items():
                if not ds.flying:
                    await self.fleet.takeoff(drone_id)
                    ds.phase = DronePhase.TAKING_OFF
                    ds.role = DroneRole.SCOUT

    # ------------------------------------------------------------------
    # Phase: SCOUTING — discover target boxes
    # ------------------------------------------------------------------

    def _generate_scout_waypoints(self, drone_id: str, idx: int, total: int) -> list[Vec2]:
        """
        Generate a patrol route for one scout drone.
        Drones spread across the arena in lanes, visiting the enemy zone
        and then sweeping back through the middle.
        """
        w = self.config.width_m
        h = self.config.height_m
        enemy = self.config.enemy_zone()
        home = self.config.home_zone()

        # Divide arena into horizontal lanes per drone
        lane_y = h * (idx + 1) / (total + 1)
        lane_y = max(1.5, min(lane_y, h - 1.5))

        # Build waypoint loop: home-side center → enemy zone → sweep → back
        mid_x = w / 2
        enemy_cx = (enemy[0] + enemy[2]) / 2
        home_cx = (home[0] + home[2]) / 2

        waypoints = [
            Vec2(x=mid_x, y=lane_y),             # center of arena
            Vec2(x=enemy_cx - 2, y=lane_y),       # approach enemy zone
            Vec2(x=enemy_cx + 2, y=lane_y),       # deep into enemy zone
            Vec2(x=enemy_cx, y=h - lane_y),        # sweep to other side
            Vec2(x=mid_x, y=h - lane_y),           # back through center
            Vec2(x=home_cx + 2, y=lane_y),         # near home
        ]
        return waypoints

    async def _phase_scouting(self) -> None:
        """
        Fly drones around to discover target boxes.
        Each drone follows a patrol route, scanning (rotating) at each waypoint.
        Once enough targets found, transition to attack.
        """
        discovered = [
            b for b in self.state.target_boxes.values() if b.discovered and b.position
        ]

        # If we have targets and enough drones, plan simultaneous attack
        flying_drones = [d for d in self.state.drones.values() if d.flying]
        if len(discovered) >= min(len(flying_drones), 2):
            # Stop all drones before transitioning
            for did in self.state.drones:
                await self._rc_stop(did)
            log.info(
                f"Found {len(discovered)} targets, planning simultaneous attack"
            )
            self._assign_attack_targets(discovered)
            self.state.phase = GamePhase.SIMULTANEOUS_ATTACK
            return

        # Initialize scout waypoints if needed
        drone_ids = list(self.state.drones.keys())
        total = len(drone_ids)
        for i, drone_id in enumerate(drone_ids):
            if drone_id not in self._scout_waypoints:
                self._scout_waypoints[drone_id] = self._generate_scout_waypoints(drone_id, i, total)
                self._scout_wp_index[drone_id] = 0

        # Drive each scout drone
        now = time.monotonic()
        for drone_id, ds in self.state.drones.items():
            if not ds.flying or not ds.connected:
                continue
            ds.role = DroneRole.SCOUT

            waypoints = self._scout_waypoints.get(drone_id, [])
            if not waypoints:
                continue
            wp_idx = self._scout_wp_index.get(drone_id, 0)
            target = waypoints[wp_idx % len(waypoints)]

            if not ds.position:
                # No position yet — spin in place to scan for ArUco markers
                ds.phase = DronePhase.ORBIT_PATROL
                await self._rc_scan(drone_id)
                continue

            dist = ds.position.distance_to(target)

            if dist <= self.ARRIVAL_THRESHOLD:
                # Arrived at waypoint — do a scan rotation
                if drone_id not in self._scan_start:
                    self._scan_start[drone_id] = now
                    ds.phase = DronePhase.ORBIT_PATROL
                    log.info(f"{drone_id} arrived at waypoint {wp_idx}, scanning...")

                scan_elapsed = now - self._scan_start[drone_id]
                if scan_elapsed < self.SCAN_DURATION:
                    # Still scanning — rotate in place
                    await self._rc_scan(drone_id)
                else:
                    # Scan complete — advance to next waypoint
                    self._scan_start.pop(drone_id, None)
                    self._scout_wp_index[drone_id] = (wp_idx + 1) % len(waypoints)
                    next_wp = waypoints[self._scout_wp_index[drone_id]]
                    ds.phase = DronePhase.FLYING_TO_TARGET
                    ds.assigned_target = f"wp{self._scout_wp_index[drone_id]}"
                    log.info(f"{drone_id} → next waypoint ({next_wp.x:.1f}, {next_wp.y:.1f})")
            else:
                # Navigate toward waypoint using continuous rc
                ds.phase = DronePhase.FLYING_TO_TARGET
                await self._rc_navigate(drone_id, target)

    # ------------------------------------------------------------------
    # Phase: SIMULTANEOUS_ATTACK — coordinate multi-drone capture
    # ------------------------------------------------------------------

    async def _phase_simultaneous_attack(self) -> None:
        """
        Fly assigned drones to their target boxes using continuous rc.
        Wait until all are hovering, then start capture timers simultaneously.
        """
        attackers = {
            did: ds for did, ds in self.state.drones.items()
            if ds.role == DroneRole.ATTACKER and ds.assigned_target
        }

        if not attackers:
            self.state.phase = GamePhase.SCOUTING
            return

        all_in_position = True
        for drone_id, ds in attackers.items():
            target = self.state.target_boxes.get(ds.assigned_target)
            if not target or not target.position or not ds.position:
                all_in_position = False
                continue

            dist = ds.position.distance_to(target.position)
            if dist > self.config.approach_tolerance_m:
                all_in_position = False
                ds.phase = DronePhase.FLYING_TO_TARGET
                # Continuously navigate toward target every tick
                await self._rc_navigate(drone_id, target.position)
            else:
                ds.phase = DronePhase.HOVERING_ON_TARGET
                await self._rc_stop(drone_id)

        # If all attackers are hovering, start capture timers
        if all_in_position and len(attackers) > 0:
            for drone_id, ds in attackers.items():
                if ds.assigned_target:
                    self.scorer.start_capture(ds.assigned_target, drone_id)

            # Check if captures complete
            captures = self.scorer.check_captures()
            if captures:
                for ev in captures:
                    box = self.state.target_boxes.get(ev.box_id)
                    if box:
                        box.captured = True
                        box.captured_by = self.config.our_team
                    self.state.captures_log.append(ev.model_dump())

                # Transition to team return for bonus points
                log.info("Captures complete → TEAM_RETURN for bonus")
                self.state.phase = GamePhase.TEAM_RETURN
                for ds in self.state.drones.values():
                    ds.role = DroneRole.RETURNER
                    ds.phase = DronePhase.RETURNING_HOME
                    ds.assigned_target = None

    # ------------------------------------------------------------------
    # Phase: TEAM_RETURN — all drones return home for special capture bonus
    # ------------------------------------------------------------------

    async def _phase_team_return(self) -> None:
        """Bring all drones to home zone for special capture bonus (5pt)."""
        all_home = True

        for drone_id, ds in self.state.drones.items():
            if not ds.position:
                all_home = False
                continue

            if self.config.is_in_home_zone(ds.position):
                ds.phase = DronePhase.IN_HOME_ZONE
                await self._rc_stop(drone_id)
            else:
                all_home = False
                ds.phase = DronePhase.RETURNING_HOME
                # Continuously navigate home every tick
                await self._rc_navigate(drone_id, ds.home)

        self.state.all_in_home = all_home

        if all_home:
            bonus = self.scorer.apply_team_return_bonus()
            if bonus > 0:
                log.info(f"Team return bonus: +{bonus} points!")
                self.state.score_us = self.scorer.our_score

            # Check if we should attempt instant win
            controlled = {
                b.box_id for b in self.state.target_boxes.values()
                if b.captured_by == self.config.our_team
            }
            if len(controlled) >= 5:
                self.state.phase = GamePhase.INSTANT_WIN_ATTEMPT
            else:
                # Go back to scouting/attack cycle
                self.state.phase = GamePhase.SCOUTING
                for ds in self.state.drones.values():
                    ds.role = DroneRole.SCOUT
                    ds.phase = DronePhase.IDLE

    # ------------------------------------------------------------------
    # Phase: DEFEND — orbit patrol to avoid idle penalties
    # ------------------------------------------------------------------

    async def _phase_defend(self) -> None:
        """
        Defensive mode: patrol near captured boxes to discourage enemy.
        Drones orbit captured boxes rather than sitting idle.
        """
        captured = [
            b for b in self.state.target_boxes.values()
            if b.captured_by == self.config.our_team and b.position
        ]

        for drone_id, ds in self.state.drones.items():
            if ds.role == DroneRole.DEFENDER and ds.position:
                if captured:
                    nearest = min(captured, key=lambda b: ds.position.distance_to(b.position))
                    if ds.position.distance_to(nearest.position) > 2.0:
                        ds.phase = DronePhase.FLYING_TO_TARGET
                        await self._rc_navigate(drone_id, nearest.position)
                    else:
                        # Orbit: gentle yaw + lateral movement
                        ds.phase = DronePhase.ORBIT_PATROL
                        await self.fleet.command_drone(
                            drone_id, "rc", lr=15, fb=10, ud=0, yaw=30
                        )

    # ------------------------------------------------------------------
    # Phase: INSTANT_WIN_ATTEMPT — rush remaining boxes
    # ------------------------------------------------------------------

    async def _phase_instant_win_attempt(self) -> None:
        """
        We control 5+ boxes — send all drones to capture the remaining ones.
        """
        uncaptured = [
            b for b in self.state.target_boxes.values()
            if b.captured_by != self.config.our_team and b.discovered and b.position
        ]

        if not uncaptured:
            return

        available = [
            (did, ds) for did, ds in self.state.drones.items()
            if ds.flying and ds.connected
        ]

        for i, (drone_id, ds) in enumerate(available):
            target = uncaptured[i % len(uncaptured)]
            ds.assigned_target = target.box_id
            ds.role = DroneRole.ATTACKER

            if ds.position and target.position:
                dist = ds.position.distance_to(target.position)
                if dist <= self.config.approach_tolerance_m:
                    ds.phase = DronePhase.HOVERING_ON_TARGET
                    await self._rc_stop(drone_id)
                    self.scorer.start_capture(target.box_id, drone_id)
                else:
                    ds.phase = DronePhase.FLYING_TO_TARGET
                    await self._rc_navigate(drone_id, target.position)

    # ------------------------------------------------------------------
    # Target assignment
    # ------------------------------------------------------------------

    def _assign_attack_targets(self, targets: list[TargetBox]) -> None:
        """
        Assign drones to targets for simultaneous attack.
        Uses a greedy nearest-drone approach.
        """
        available_drones = [
            (did, ds) for did, ds in self.state.drones.items()
            if ds.flying and ds.connected and ds.position
        ]

        # Sort targets by priority (uncaptured enemy boxes first)
        enemy_targets = [t for t in targets if t.captured_by != self.config.our_team]
        if not enemy_targets:
            enemy_targets = targets

        assignments: list[tuple[str, str]] = []
        used_drones: set[str] = set()

        for target in enemy_targets:
            if not target.position:
                continue
            # Find nearest available drone
            best_did = None
            best_dist = float("inf")
            for did, ds in available_drones:
                if did in used_drones or not ds.position:
                    continue
                d = ds.position.distance_to(target.position)
                if d < best_dist:
                    best_dist = d
                    best_did = did

            if best_did:
                assignments.append((best_did, target.box_id))
                used_drones.add(best_did)

            if len(assignments) >= len(available_drones):
                break

        # Apply assignments
        for drone_id, box_id in assignments:
            ds = self.state.drones[drone_id]
            ds.role = DroneRole.ATTACKER
            ds.assigned_target = box_id
            ds.phase = DronePhase.FLYING_TO_TARGET
            log.info(f"Assigned {drone_id} → target {box_id}")

        # Remaining drones become defenders
        for did, ds in self.state.drones.items():
            if did not in used_drones and ds.flying:
                ds.role = DroneRole.DEFENDER
                ds.phase = DronePhase.ORBIT_PATROL

    # ------------------------------------------------------------------
    # Movement helpers (continuous RC-based)
    # ------------------------------------------------------------------

    async def _rc_navigate(self, drone_id: str, target: Vec2) -> None:
        """
        Send rc command to fly toward a 2D target position.
        Computes heading from current position and converts to
        forward/lateral rc values relative to the drone's heading.
        """
        ds = self.state.drones.get(drone_id)
        if not ds or not ds.position:
            return

        dx = target.x - ds.position.x
        dy = target.y - ds.position.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.1:
            await self._rc_stop(drone_id)
            return

        # Target bearing in world frame (radians, 0 = +Y, CW positive)
        target_angle = math.atan2(dx, dy)

        # Drone heading in radians
        heading_rad = ds.heading_deg * math.pi / 180

        # Relative angle from drone heading to target
        rel = target_angle - heading_rad
        # Normalize to [-pi, pi]
        rel = (rel + math.pi) % (2 * math.pi) - math.pi

        # Decompose into forward/lateral
        speed = self.RC_SPEED
        fb = int(speed * math.cos(rel))   # forward/back
        lr = int(speed * math.sin(rel))   # left/right

        # Slow down when close
        if dist < 2.0:
            factor = max(0.3, dist / 2.0)
            fb = int(fb * factor)
            lr = int(lr * factor)

        await self.fleet.command_drone(
            drone_id, "rc", lr=lr, fb=fb, ud=0, yaw=0
        )

    async def _rc_scan(self, drone_id: str) -> None:
        """Rotate in place to scan for ArUco markers."""
        await self.fleet.command_drone(
            drone_id, "rc", lr=0, fb=0, ud=0, yaw=self.RC_YAW_SCAN
        )

    async def _rc_stop(self, drone_id: str) -> None:
        """Stop all movement."""
        await self.fleet.command_drone(
            drone_id, "rc", lr=0, fb=0, ud=0, yaw=0
        )

    async def _fly_to(self, drone_id: str, target: Vec2) -> None:
        """Navigate to a target — dispatches to rc_navigate for continuous control."""
        await self._rc_navigate(drone_id, target)

    # ------------------------------------------------------------------
    # Manual overrides
    # ------------------------------------------------------------------

    async def manual_command(self, drone_id: str, action: str, params: dict) -> dict:
        """Execute a manual override command."""
        if action == "takeoff":
            result = await self.fleet.takeoff(drone_id)
            if drone_id in self.state.drones:
                self.state.drones[drone_id].flying = True
                self.state.drones[drone_id].phase = DronePhase.TAKING_OFF
            return result
        elif action == "land":
            result = await self.fleet.land(drone_id)
            if drone_id in self.state.drones:
                self.state.drones[drone_id].flying = False
                self.state.drones[drone_id].phase = DronePhase.LANDED
            return result
        elif action == "emergency":
            result = await self.fleet.emergency(drone_id)
            if drone_id in self.state.drones:
                self.state.drones[drone_id].flying = False
                self.state.drones[drone_id].phase = DronePhase.LANDED
            return result
        elif action == "goto":
            x = params.get("x", 0)
            y = params.get("y", 0)
            z = params.get("z", self.hover_altitude_m)
            return await self.fleet.command_drone(
                drone_id, "move_to", x=x, y=y, z=z
            )
        elif action == "move":
            return await self.fleet.command_drone(
                drone_id, "move",
                direction=params.get("direction", "forward"),
                distance_cm=params.get("distance", 50),
            )
        else:
            return {"ok": False, "error": f"Unknown action: {action}"}

    def set_phase(self, phase_str: str) -> bool:
        """Manually set the game phase."""
        try:
            new_phase = GamePhase(phase_str)
            self.state.phase = new_phase
            log.info(f"Phase manually set to {new_phase.value}")
            return True
        except ValueError:
            return False

    def assign_target(self, drone_id: str, box_id: str) -> bool:
        """Manually assign a drone to a target box."""
        ds = self.state.drones.get(drone_id)
        if not ds:
            return False
        ds.assigned_target = box_id
        ds.role = DroneRole.ATTACKER
        ds.phase = DronePhase.FLYING_TO_TARGET
        return True

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_fleet_from_config(self) -> None:
        """Set up drones and fleet client from arena config."""
        home_positions = self.config.default_home_positions(
            len(self.config.drone_configs)
        )
        for i, dc in enumerate(self.config.drone_configs):
            # Add to fleet client
            self.fleet.add_drone(
                drone_id=dc.drone_id,
                api_base_url=dc.api_base_url,
                drone_type=dc.drone_type,
                sim_drone_id=dc.sim_drone_id,
            )
            # Add to game state
            home = home_positions[i] if i < len(home_positions) else Vec2(x=1, y=1)
            self.state.drones[dc.drone_id] = DroneState(
                drone_id=dc.drone_id,
                home=home,
                api_base_url=dc.api_base_url,
                sim_drone_id=dc.sim_drone_id,
            )

    def get_state_snapshot(self) -> dict:
        """Return the full game state as a dict for broadcasting."""
        return {
            "phase": self.state.phase.value,
            "our_team": self.state.our_team,
            "score_us": self.state.score_us,
            "score_them": self.state.score_them,
            "all_in_home": self.state.all_in_home,
            "drones": {
                did: ds.model_dump() for did, ds in self.state.drones.items()
            },
            "targets": {
                bid: tb.model_dump() for bid, tb in self.state.target_boxes.items()
            },
            "captures": self.state.captures_log[-20:],
            "scoring": self.scorer.summary(),
            "locator": self.locator.summary(),
            "pillar_markers": {
                mid: m.model_dump() for mid, m in self.config.pillar_markers.items()
            },
        }
