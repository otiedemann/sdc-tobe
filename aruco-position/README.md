# ArUco Position Tracking

This module provides ArUco marker-based positioning and camera calibration for the Swarm Drone Challenge 2026.

## Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Requirements

- numpy==2.4.2
- opencv-python==4.13.0.92

## Scripts

- **stream.py** - Video streaming functionality
- **calibrate_camera.py** - Camera calibration utility
- **aruco_positioning.py** - ArUco marker detection and positioning system

## Usage

### Camera Calibration

```bash
python calibrate_camera.py
```

### Video Streaming

```bash
python stream.py
```

### ArUco Positioning

```bash
python aruco_positioning.py
```
