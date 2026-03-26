#!/usr/bin/env python3
"""
Video receiver for Tello UDP stream on port 11111.
Run this on the viewer PC.

Usage:
  python3 video_receiver_11111.py

Requires ffplay (ffmpeg package).
"""

import shutil
import subprocess
import sys

PORT = 11111


def main():
    ffplay = shutil.which("ffplay")
    if not ffplay:
        print("ffplay not found. Install ffmpeg first.")
        print("macOS: brew install ffmpeg")
        print("Ubuntu/RPi: sudo apt install -y ffmpeg")
        sys.exit(1)

    cmd = [
        ffplay,
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-framedrop",
        "-probesize", "32",
        "-analyzeduration", "0",
        f"udp://0.0.0.0:{PORT}",
    ]
    print("Starting video receiver:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
