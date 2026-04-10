#!/usr/bin/env bash
# Start aruco_position main.py with all parameters set to their defaults.
# Edit values here to override without changing the source code.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/main.py" \
    --src 0 \
    --target-ip 127.0.0.1 \
    --detect balanced \
    --min-ref-weight 0.0 \
    --min-ref-count 1 \
    --outlier-thresh 2.5 \
    --pose-hold 0.8 \
    --target-z-pos -1.5
    # --arena-config "$SCRIPT_DIR/arena_config.json"   # auto-detected when present
    # --calib "$SCRIPT_DIR/logi_calibration.npz"       # use camera calibration file
    # --verbose                                        # enable verbose position output
    # --preview                                        # show local OpenCV preview window
    # --force-headless                                 # disable GUI even if display is available
    # --pos-kalman                                     # enable per-axis Kalman filter in aruco position module
