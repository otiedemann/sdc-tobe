from __future__ import annotations

import unittest

from strategy.config import load_config
from strategy.controller import StrategyController
from strategy.fallback_policy import plan_fallback
from strategy.models import DroneCommand, Pose
from strategy.simulation import Simulation
from strategy.validator import validate_commands


class StrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("config.example.json")
        self.config.llm.enabled = False
        self.simulation = Simulation(self.config, StrategyController(use_llm=False))

    def test_fallback_commands_all_active_drones(self) -> None:
        state = self.simulation._build_state_unlocked()
        plan = plan_fallback(state)
        self.assertEqual(len(plan.commands), len(state.active_own_drones()))

    def test_validator_clamps_out_of_bounds_command(self) -> None:
        state = self.simulation._build_state_unlocked()
        command = DroneCommand(
            drone_id="own_1",
            role="attacker",
            objective="stage_attack",
            target_position=Pose(99.0, -20.0, 99.0, 0.0),
        )
        result = validate_commands(state, [command])
        self.assertLessEqual(result.commands[0].target_position.x, 10.0)
        self.assertGreaterEqual(result.commands[0].target_position.y, 0.0)
        self.assertLessEqual(result.commands[0].target_position.z, 2.8)


if __name__ == "__main__":
    unittest.main()
