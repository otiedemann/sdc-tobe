from __future__ import annotations

import json

from .models import MatchState


SYSTEM_PROMPT = """You are the tactical strategy engine for a Swarm Drone Challenge team.

Your task is to assign safe target arena coordinates to 2 to 5 drones.

Goal: maximize score while preserving safety and defending owned target boxes.

Competition priorities:
1. Prevent opponent validation of our own targets.
2. Recover own targets that are opponent-colored before the 5 second validation window completes.
3. Attempt synchronized captures of multiple enemy targets when own targets are safe.
4. Threaten instant win by making all six targets our color for at least 5 seconds.
5. Prefer simultaneous captures over isolated 1-point captures.
6. Avoid passive hovering directly above targets except short capture/recovery holds.
7. Preserve at least two operational drones.
8. Stay inside arena bounds and keep drones separated.

Coordinate frame:
- x: -10 to 10 meters
- y: 0 to 10 meters
- z: 0 to 6 meters
- left zone is x=-10..-5
- neutral zone is x=-5..5
- right zone is x=5..10

Use these objective types only:
- stage_attack
- capture_target
- recover_target
- defense_patrol
- offset_loiter
- return_home
- hold_home
- regroup
- land

Capture/recovery altitude should normally be 1.2 to 1.8 meters.
Transit/staging/patrol altitude should normally be 1.6 to 2.5 meters.

Return JSON only. Do not include prose outside JSON.
"""


def build_user_prompt(state: MatchState) -> str:
    payload = state.to_prompt_dict()
    return f"""Given the current match state, return the next command for each active own drone.

Return exactly this JSON shape:
{{
  "mode": "short_mode_name",
  "strategy_summary": "one sentence summary",
  "commands": [
    {{
      "drone_id": "own_1",
      "role": "attacker|defender|flex|failsafe",
      "objective": "stage_attack|capture_target|recover_target|defense_patrol|offset_loiter|return_home|hold_home|regroup|land",
      "target_id": "target id or null",
      "target_position": {{ "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0 }},
      "hold_seconds": 0.0,
      "priority": 0,
      "reason": "short reason"
    }}
  ]
}}

Current match state:
{json.dumps(payload, indent=2)}
"""
