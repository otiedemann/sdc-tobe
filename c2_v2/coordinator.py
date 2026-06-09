"""C2 V2 coordinator — Phase 4: the autonomous swarm brain.

Decides each drone's ROLE from the live game state, in AUTO mode. The strategy
is shaped by two hard facts:

  1. **Never-land architecture** (Phase 3): a drone flies one infinite role
     script; changing its role costs a STOP -> land -> relaunch. So role
     allocation must be STABLE — we re-task only when the situation genuinely
     demands it, and the runner adds a per-drone cooldown on top.

  2. **v7 scoring** (priority Special > 10 > 5 > 1). With independent infinite
     scripts we cannot reliably orchestrate the sub-second 10-pt double-strike
     or the "all drones outside home at the instant of capture" 5-pt window. So
     the achievable winning plan is:
        * ATTACKERS continuously re-capture the 3 ENEMY boxes (each capture +
          home = 1 pt; the boxes flip back when we leave, so the loop keeps
          scoring all match), AND
        * DEFENDERS hold presence near OUR boxes so the enemy can't capture
          them (capture needs "no defending drone in that box"),
     converging on the **Special Maneuver**: hold ALL 6 boxes for 5 s = instant
     win. The coordinator biases the attacker/defender split toward that goal
     and, when we're one box short, throws extra attackers at the last enemy box.

Pure decision function — no I/O. The runner applies the plan with hysteresis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DroneView:
    fc: str
    connected: bool
    enabled: bool
    current_role: str


@dataclass
class Plan:
    roles: Dict[str, str]      # fc -> desired role
    summary: str               # human-readable strategy line


def _our_slots(team: str):
    return (1, 2, 3) if team == "red" else (4, 5, 6)


def _enemy_slots(team: str):
    return (4, 5, 6) if team == "red" else (1, 2, 3)


class Coordinator:
    def plan(self, our_team: str, holders: Dict[int, str],
             drones: Dict[str, DroneView], baseline_def: int = 1) -> Plan:
        """holders: {slot -> 'red'|'blue'|'unknown'} (effective, decayed).
        drones: {fc -> DroneView}. Returns the desired role per LIVE drone."""
        enemy = "blue" if our_team == "red" else "red"
        live = [v for v in drones.values() if v.connected and v.enabled]
        roles: Dict[str, str] = {}
        if not live:
            return Plan(roles={}, summary="no live drones")

        # 1) SCOUT is OPTIONAL in the simplified strategy. Keep a scout ONLY if
        #    the operator currently has one assigned (don't force one onto a
        #    scout-less fleet); attackers + defenders do their own scouting.
        live_sorted = sorted(live, key=lambda v: v.fc)
        scout = next((v for v in live_sorted if v.current_role == "scout"), None)
        if scout is not None:
            roles[scout.fc] = "scout"
        movers = [v for v in live_sorted if scout is None or v.fc != scout.fc]
        M = len(movers)
        if M == 0:
            return Plan(roles=roles, summary="scout only")

        # 2) Read the board.
        our = _our_slots(our_team)
        ens = _enemy_slots(our_team)
        our_threatened = [s for s in our if holders.get(s) == enemy]
        enemy_open = [s for s in ens if holders.get(s) != our_team]   # not ours yet
        enemy_held_by_us = len(ens) - len(enemy_open)
        boxes_ours = sum(1 for s in our if holders.get(s) == our_team) + enemy_held_by_us

        # 3) Decide the attacker / defender split (target counts).
        special_push = boxes_ours >= 5 and len(enemy_open) >= 1
        if special_push:
            # ONE box short of the instant win: go all-out to grab the last
            # enemy box; only hold back a defender for a box we've already lost.
            def_n = min(M, len(our_threatened))
        else:
            # Cover every threatened own box, plus a baseline home presence
            # (operator-tunable) so the enemy can't walk into an undefended box.
            baseline = baseline_def if M >= 3 else 0
            def_n = min(M, len(our_threatened) + baseline)
        atk_n = M - def_n
        if atk_n <= 0:                    # always keep at least one attacker
            atk_n, def_n = 1, M - 1

        # 4) Assign roles to movers STABLY: keep movers in their current role
        #    where it fits the target counts, only changing the minimum number.
        want = {"attacker": atk_n, "defender": def_n}
        # First pass: keep current attacker/defender assignments while quota left.
        unassigned: List[DroneView] = []
        for v in movers:
            r = v.current_role
            if r in want and want[r] > 0:
                roles[v.fc] = r
                want[r] -= 1
            else:
                unassigned.append(v)
        # Second pass: fill remaining quota.
        for v in unassigned:
            r = "attacker" if want["attacker"] > 0 else "defender"
            roles[v.fc] = r
            want[r] -= 1

        scout_lbl = scout.fc.replace("flightctrl", "fc") if scout else "none"
        summary = (f"{atk_n} atk · {def_n} def · scout {scout_lbl} "
                   f"| ours {boxes_ours}/6"
                   + (f" · threatened {sorted(our_threatened)}" if our_threatened else "")
                   + (" · ONE OFF SPECIAL — all-out" if special_push else "")
                   + (" · holding all enemy boxes" if not enemy_open and not special_push else ""))
        return Plan(roles=roles, summary=summary)
