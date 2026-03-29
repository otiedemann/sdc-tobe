# Camera Calibration Guide

This module uses OpenCV's chessboard-based camera calibration to compute the intrinsic camera parameters (focal length,
principal point) and lens distortion coefficients. These parameters are required for accurate ArUco marker-based
positioning.

## The Calibration Pattern

The calibration uses a printed chessboard pattern (`calibration_chessboard.png`) with:

- **Grid**: 9 columns x 6 rows of interior corners (10x7 squares)
- **Square size**: 25mm (0.025m) per square

Print `calibration_chessboard.png` on A4/Letter paper and mount it on a flat, rigid surface (cardboard or foam board).
The pattern must be perfectly flat for accurate calibration.

## Running Calibration

```bash
cd aruco-position
python calibrate_camera.py
```

The script provides an interactive menu with these options:

| Option | Description                                  |
|--------|----------------------------------------------|
| 1      | Full calibration: capture images + calibrate |
| 2      | Capture images only                          |
| 3      | Calibrate from existing images               |
| 4      | Test existing calibration                    |
| 5      | Load existing calibration                    |
| 6      | Show checkerboard creation guide             |

## Full Calibration Workflow

### 1. Image Capture

When you select option 1 or 2, the script opens your camera and displays a live preview:

- **Green corners** = chessboard detected
- **Red text** = no corners found, move the checkerboard

Press **SPACE** to capture an image when corners are detected in green. Press **Q** to quit early.

**Best practices for capturing calibration images:**

- Capture 15-25 images for reliable results
- Vary the checkerboard position and angle throughout the frame
- Include images near corners and edges
- Include images at different distances
- Ensure good lighting and avoid glare on the pattern

### 2. Calibration

After capturing images, the script processes them to find corners and computes the camera parameters using
`cv2.calibrateCamera()`.

**Output includes:**

- **RMS re-projection error** - lower is better (< 1.0 px is good, < 0.5 is excellent)
- **Camera matrix** - intrinsic parameters:
    - `fx`, `fy` - focal lengths in pixels
    - `cx`, `cy` - principal point coordinates
- **Distortion coefficients**:
    - `k1, k2` - radial distortion
    - `p1, p2` - tangential distortion
    - `k3` - thin prism distortion

### 3. Testing

Option 1 prompts to test the calibration by showing side-by-side original vs undistorted video. This visually confirms
the correction is working.

## Camera Sources

The script supports multiple camera sources:

| Source     | Input         | Notes                                   |
|------------|---------------|-----------------------------------------|
| USB webcam | `0`           | Default, system camera index            |
| IP camera  | RTSP/HTTP URL | e.g., `rtsp://192.168.1.100:554/stream` |
| DJI Tello  | `tello`       | Requires `djitellopy` package           |

## Output Files

Calibration results are saved as:

| File                      | Format        | Contents                               |
|---------------------------|---------------|----------------------------------------|
| `camera_calibration.npz`  | NumPy archive | All parameters (used by other scripts) |
| `camera_calibration.json` | JSON          | Human-readable parameters              |

## Using Calibration in Other Scripts

```python
import numpy as np

# Load calibration
data = np.load("camera_calibration.npz")
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

# Undistort an image
import cv2

undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
```

## Recalibration

Recalibrate when:

- Switching cameras or lenses
- Significant temperature changes affect the lens
- After physical camera adjustments
- RMS error increases noticeably in positioning accuracy

## Troubleshooting

**"No corners detected"**

- Ensure the full chessboard is visible in frame
- Improve lighting; avoid shadows on the pattern
- Verify the printed square size matches the configured value

**High re-projection error (> 1.0)**

- Capture more images with better variety
- Ensure the chessboard is perfectly flat
- Check that the square size measurement is accurate

**Camera matrix appears unreasonable**

- Verify image resolution is consistent
- Check that all squares are equal size on the print
