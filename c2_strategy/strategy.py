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
                ds.heading_deg = telem.get("yaw", telem.get("heading", ds.heading_deg))

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
        """Initialize drones. Transition to SCOUTING when all connected."""
        all_connected = all(
            ds.connected for ds in self.state.drones.values()
        )
        if all_connected and len(self.state.drones) > 0:
            log.info("All drones connected → transitioning to SCOUTING")
            self.state.phase = GamePhase.SCOUTING
            self.state.start_time = time.monotonic()

            # Take off all drones
            for drone_id, ds in self.state.drones.items():
                if not ds.flying:
                    await self.fleet.takeoff(drone_id)
                    ds.phase = DronePhase.TAKING_OFF
                    ds.role = DroneRole.SCOUT

    # ------------------------------------------------------------------
    # Phase: SCOUTING — discover target boxes
    # ------------------------------------------------------------------

    async def _phase_scouting(self) -> None:
        """
        Fly drones around to discover target boxes.
        Once enough targets found, transition to attack.
        """
        discovered = [
            b for b in self.state.target_boxes.values() if b.discovered and b.position
        ]

        # If we have targets and enough drones, plan simultaneous attack
        num_drones = len([d for d in self.state.drones.values() if d.flying])
        if len(discovered) >= min(num_drones, 2):
            log.info(
                f"Found {len(discovered)} targets, planning simultaneous attack"
            )
            self._assign_attack_targets(discovered)
            self.state.phase = GamePhase.SIMULTANEOUS_ATTACK
            return

        # Otherwise assign scout patrol routes
        for drone_id, ds in self.state.drones.items():
            if ds.role == DroneRole.SCOUT and ds.phase != DronePhase.FLYING_TO_TARGET:
                # Simple patrol: fly to enemy zone to scan for boxes
                enemy = self.config.enemy_zone()
                cx = (enemy[0] + enemy[2]) / 2
                cy = (enemy[1] + enemy[3]) / 2
                # Spread drones across enemy zone
                idx = list(self.state.drones.keys()).index(drone_id)
                spread_y = enemy[1] + 2.0 + idx * 2.0
                spread_y = min(spread_y, enemy[3] - 1.0)

                ds.phase = DronePhase.FLYING_TO_TARGET
                await self._fly_to(drone_id, Vec2(x=cx, y=spread_y))

    # ------------------------------------------------------------------
    # Phase: SIMULTANEOUS_ATTACK — coordinate multi-drone capture
    # ------------------------------------------------------------------

    async def _phase_simultaneous_attack(self) -> None:
        """
        Fly assigned drones to their target boxes.
        Wait until all are hovering, then start capture timers simultaneously.
        """
        attackers = {
            did: ds for did, ds in self.state.drones.items()
            if ds.role == DroneRole.ATTACKER and ds.assigned_target
        }

        if not attackers:
            # No attackers assigned, go back to scouting
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
                if ds.phase != DronePhase.FLYING_TO_TARGET:
                    ds.phase = DronePhase.FLYING_TO_TARGET
                    await self._fly_to(drone_id, target.position)
            else:
                ds.phase = DronePhase.HOVERING_ON_TARGET

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
            else:
                all_home = False
                if ds.phase != DronePhase.RETURNING_HOME:
                    ds.phase = DronePhase.RETURNING_HOME
                # Fly home
                home_pos = ds.home
                await self._fly_to(drone_id, home_pos)

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
                # Find nearest captured box to patrol
                if captured:
                    nearest = min(captured, key=lambda b: ds.position.distance_to(b.position))
                    if ds.position.distance_to(nearest.position) > 2.0:
                        await self._fly_to(drone_id, nearest.position)
                    else:
                        # Orbit: gentle yaw rotation while hovering
                        ds.phase = DronePhase.ORBIT_PATROL
                        await self.fleet.command_drone(
                            drone_id, "rc", lr=15, fb=0, ud=0, yaw=30
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
            # All captured! Just wait for timer
            return

        # Assign all available drones to remaining targets
        available = [
            (did, ds) for did, ds in self.state.drones.items()
            if ds.flying and ds.connected
        ]

        for i, (drone_id, ds) in enumerate(available):
            target = uncaptured[i % len(uncaptured)]
            ds.assigned_target = target.box_id
            ds.role = DroneRole.ATTACKER
            ds.phase = DronePhase.FLYING_TO_TARGET
            await self._fly_to(drone_id, target.position)

            # Check if in position → start capture
            if ds.position and target.position:
                if ds.position.distance_to(target.position) <= self.config.approach_tolerance_m:
                    ds.phase = DronePhase.HOVERING_ON_TARGET
                    self.scorer.start_capture(target.box_id, drone_id)

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
    # Movement helpers
    # ------------------------------------------------------------------

    async def _fly_to(self, drone_id: str, target: Vec2) -> None:
        """Command a drone to fly to a 2D position."""
        ds = self.state.drones.get(drone_id)
        if not ds or not ds.position:
            return

        # Use move_to for sim/olympe (continuous position)
        client = self.fleet.drones.get(drone_id)
        if client and client.drone_type in ("simulator", "anafi"):
            await self.fleet.command_drone(
                drone_id, "move_to",
                x=target.x, y=target.y, z=self.hover_altitude_m,
            )
        else:
            # For Tello: compute relative movement
            dx = target.x - ds.position.x
            dy = target.y - ds.position.y
            # Convert to go command (cm)
            await self.fleet.command_drone(
                drone_id, "go",
                x=int(dx * 100), y=int(dy * 100), z=0,
                speed=self.attack_speed,
            )

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
        }
