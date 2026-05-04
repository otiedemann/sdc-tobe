# AI-Driven C2 Strategy Prototype

This module prototypes a strategic controller for Swarm Drone Challenge 2026. It controls 2 to 5 own drones by producing target arena coordinates, while mock enemy drones move in the same arena.

The controller uses a hybrid architecture:

- OpenRouter LLM proposes tactical commands from the current match state.
- A deterministic validator rejects unsafe or invalid commands.
- A fallback policy keeps the match playable when the LLM fails, times out, or returns invalid JSON.
- A browser-based 3D wireframe display shows drones, targets, commands, score, and events.

## Coordinate Frame

```text
x = -10..10
y = 0..10
z = 0..6
left zone    = x -10..-5
neutral zone = x -5..5
right zone   = x 5..10
```

The own home side is configurable as `left` or `right`. Target box coordinates are loaded from `config.example.json`.

## Run With LLM

PowerShell:

```powershell
$env:OPENROUTER_API_KEY="your-key"
python run_simulation.py --config config.example.json
```

Then open:

```text
http://127.0.0.1:8765
```

Override the model:

```powershell
python run_simulation.py --model openai/gpt-4.1-mini
```

## Run Without LLM

```powershell
python run_simulation.py --no-llm
```

This uses the deterministic fallback policy only.

## Useful Options

```powershell
python run_simulation.py --drones 3 --enemy-drones 4 --home-side right --no-llm
```

## Validation

```powershell
python -m unittest discover -s tests -q
python -m compileall .
```

## Planner Contract

The LLM receives the current match state and returns JSON commands:

```json
{
  "mode": "simultaneous_attack",
  "strategy_summary": "Stage three attackers while one defender patrols owned targets.",
  "commands": [
    {
      "drone_id": "own_1",
      "role": "attacker",
      "objective": "capture_target",
      "target_id": "right_1",
      "target_position": { "x": 8.5, "y": 2.0, "z": 1.4, "yaw": 0.0 },
      "hold_seconds": 2.0,
      "priority": 90,
      "reason": "Capture enemy target."
    }
  ]
}
```

Supported objectives:

```text
stage_attack
capture_target
recover_target
defense_patrol
offset_loiter
return_home
hold_home
regroup
land
```

## Strategy Doctrine

```text
Map/know configured targets first.
Defend owned target boxes always.
Recover compromised own targets before opponent validation.
Use synchronized multi-drone attacks when own targets are safe.
Prefer simultaneous captures over isolated standard captures.
Keep at least two drones operational.
Validate all LLM commands before dispatch.
```

## Display

The 3D display uses Three.js from a CDN. It draws:

- arena wireframe,
- left/neutral/right zones,
- own drones in blue,
- enemy drones in red,
- target boxes colored by current owner,
- command lines and ghost destination markers for own drones,
- score, match time, current planner mode, and recent events.
