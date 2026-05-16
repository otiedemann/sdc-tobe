"""Entry point: ``python -m marker_mission_c2 [--config PATH]``."""
from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import C2ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="marker_mission_c2",
        description="Local C2 server for multiple marker_mission flight "
                    "controllers.",
    )
    ap.add_argument("--config", "-c", default=None,
                    help="Path to a JSON config file. Default search: "
                         "$MARKER_MISSION_C2_CONFIG, "
                         "./marker_mission_c2/config.json, "
                         "./marker_mission_c2/config.example.json.")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except C2ConfigError as e:
        print(f"[c2] config error: {e}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from .server import run_server
    return run_server(cfg)


if __name__ == "__main__":
    sys.exit(main())
