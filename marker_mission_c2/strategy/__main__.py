"""``python -m marker_mission_c2.strategy`` entrypoint."""
from __future__ import annotations

import sys

from .app import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
