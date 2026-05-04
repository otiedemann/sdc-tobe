from __future__ import annotations

import argparse
from pathlib import Path

from strategy.config import load_config
from strategy.controller import StrategyController
from strategy.llm_planner import LlmPlanner
from strategy.openrouter_client import OpenRouterClient
from strategy.simulation import Simulation
from strategy.web_server import StrategyWebServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SDC26 strategy prototype.")
    parser.add_argument("--config", default="config.example.json", help="Path to config JSON.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host for the 3D display.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port for the 3D display.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM and use fallback policy only.")
    parser.add_argument("--model", help="Override OpenRouter model from config.")
    parser.add_argument("--drones", type=int, choices=range(2, 6), help="Override own drone count.")
    parser.add_argument("--enemy-drones", type=int, choices=range(0, 6), help="Override enemy drone count.")
    parser.add_argument("--home-side", choices=("left", "right"), help="Override own home side.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config.llm.model = args.model
    if args.drones is not None:
        config.simulation.own_drone_count = args.drones
    if args.enemy_drones is not None:
        config.simulation.enemy_drone_count = args.enemy_drones
    if args.home_side:
        config.team.home_side = args.home_side
        for target in config.targets:
            target.owner = config.team.color if target.side == config.team.home_side else config.team.opponent_color
            target.validated_for_owner = target.owner

    use_llm = config.llm.enabled and not args.no_llm
    llm_planner = None
    if use_llm:
        client = OpenRouterClient(config.llm, config.openrouter)
        llm_planner = LlmPlanner(client)

    controller = StrategyController(llm_planner=llm_planner, use_llm=use_llm)
    simulation = Simulation(config, controller)
    simulation.start_background()

    web_dir = Path(__file__).parent / "web"
    print(f"Serving 3D display at http://{args.host}:{args.port}")
    if use_llm:
        print(f"Using OpenRouter model: {config.llm.model}")
    else:
        print("LLM disabled; using deterministic fallback policy.")
    StrategyWebServer(simulation, args.host, args.port, web_dir).serve_forever()


if __name__ == "__main__":
    main()
