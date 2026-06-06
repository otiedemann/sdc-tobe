"""Entry point: ``python -m marker_mission_sim --config sim_config.json``."""
from __future__ import annotations

import sys

from .server import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
