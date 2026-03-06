import cv2
from cv2 import aruco


class AruCoPositioning:
    def __init__(self, camera_matrix, dist_coeffs, marker_size=0.5):
        """
        Initialize ArUco positioning system for SDC26

        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Camera distortion coefficients
            marker_size: Size of ArUco markers in meters (default: 0.5m as per regulations)
        """
        self.marker_size = marker_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        # ArUco dictionary: DICT_4X4_50 as per regulations
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()

        # Marker positions in world coordinates (based on Figure 3 and 4)
        # Origin (marker 0) is at back wall center
        # Coordinate system: X-right, Y-up, Z-forward (from marker 0)
        self.marker_positions = self._initialize_marker_positions()

    def _initialize_marker_positions(self):
        """
        Define 3D positions of all ArUco markers based on SDC26 regulations
        Field: 20m length × 10m width × 6m height
        Markers at 2m and 4m heights on pillars
        """
        positions = {}

        # Marker 0 at origin (back wall center)
        positions[0] = np.array([0.0, 3.0, 0.0])  # Center height

        # Back wall pillars (Z=0)
        positions[18] = np.array([-5.0, 4.0, 0.0])  # Left pillar, upper (4m)
        positions[17] = np.array([-5.0, 2.0, 0.0])  # Left pillar, lower (2m)
        positions[20] = np.array([-1.67, 4.0, 0.0])
        positions[19] = np.array([-1.67, 2.0, 0.0])
        positions[22] = np.array([1.67, 4.0, 0.0])
        positions[21] = np.array([1.67, 2.0, 0.0])
        positions[24] = np.array([5.0, 4.0, 0.0])  # Right pillar, upper
        positions[23] = np.array([5.0, 2.0, 0.0])  # Right pillar, lower

        # Front wall pillars (Z=20m)
        positions[12] = np.array([-5.0, 4.0, 20.0])
        positions[11] = np.array([-5.0, 2.0, 20.0])
        positions[10] = np.array([-1.67, 4.0, 20.0])
        positions[9] = np.array([-1.67, 2.0, 20.0])
        positions[8] = np.array([1.67, 4.0, 20.0])
        positions[7] = np.array([1.67, 2.0, 20.0])
        positions[6] = np.array([5.0, 4.0, 20.0])
        positions[5] = np.array([5.0, 2.0, 20.0])

        # Left side pillars (X=-5m)
        positions[16] = np.array([-5.0, 4.0, 6.67])
        positions[15] = np.array([-5.0, 2.0, 6.67])
        positions[14] = np.array([-5.0, 4.0, 13.33])
        positions[13] = np.array([-5.0, 2.0, 13.33])

        # Right side pillars (X=5m)
        positions[2] = np.array([5.0, 4.0, 6.67])
        positions[1] = np.array([5.0, 2.0, 6.67])
        positions[4] = np.array([5.0, 4.0, 13.33])
        positions[3] = np.array([5.0, 2.0, 13.33])

        return positions

    def detect_markers(self, frame):
        """
        Detect ArUco markers in frame

        Args:
            frame: Input image/frame

        Returns:
            corners: Detected marker corners
            ids: Detected marker IDs
            rejected: Rejected candidates
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids, rejected

    def estimate_pose(self, corners, ids):
        """
        Estimate camera pose from detected markers

        Args:
            corners: Detected marker corners
            ids: Detected marker IDs

        Returns:
            position: Camera position in world coordinates (x, y, z)
            rotation: Camera rotation matrix
            camera_direction: Unit vector of camera viewing direction
        """
        if ids is None or len(ids) == 0:
            return None, None, None

        # Estimate pose for each marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs
        )

        # Use first detected marker with known position
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.marker_positions:
                # Get marker pose in camera frame
                rvec = rvecs[i]
                tvec = tvecs[i]

                # Convert rotation vector to matrix
                R_marker_to_cam, _ = cv2.Rodrigues(rvec)

                # Camera position in marker frame
                cam_pos_marker = -R_marker_to_cam.T @ tvec.T

                # Transform to world coordinates
                marker_world_pos = self.marker_positions[marker_id]

                # Camera orientation in world frame
                R_cam_to_world = R_marker_to_cam.T

                # Camera position in world frame
                camera_position = marker_world_pos + R_cam_to_world @ cam_pos_marker.flatten()

                # Camera direction (negative Z-axis in camera frame, transformed to world)
                camera_direction = R_cam_to_world @ np.array([0, 0, -1])
                camera_direction = camera_direction / np.linalg.norm(camera_direction)

                return camera_position, R_cam_to_world, camera_direction

        return None, None, None

    def draw_markers(self, frame, corners, ids):
        """Draw detected markers on frame"""
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
        return frame

    def draw_axes(self, frame, corners, ids):
        """Draw 3D axes on detected markers"""
        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                  rvecs[i], tvecs[i], self.marker_size * 0.5)
        return frame


def main():
    """Main loop for continuous position tracking"""
    import sys

    print("ArUco Positioning System - SDC26")
    print("=" * 50)

    # Load or use default calibration
    if len(sys.argv) > 1 and sys.argv[1] == '--load-calibration':
        calib_file = sys.argv[2] if len(sys.argv) > 2 else 'camera_calibration.npz'
        try:
            data = np.load(calib_file)
            camera_matrix = data['camera_matrix']
            dist_coeffs = data['dist_coeffs']
            print(f"✓ Loaded calibration from {calib_file}")
        except:
            print(f"✗ Failed to load calibration from {calib_file}")
            print("Using default parameters")
            camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
            dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=float)
    else:
        # Default calibration parameters
        print("Using default calibration (not recommended)")
        print("Run with: python aruco_positioning.py --load-calibration camera_calibration.npz")
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
        dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=float)

    # Initialize positioning system
    aruco_pos = AruCoPositioning(camera_matrix, dist_coeffs, marker_size=0.5)

    # Get camera source
    print("\nCamera source:")
    print("  USB camera: 0")
    print("  IP camera: rtsp://192.168.1.100:554/stream")
    print("  IP camera: http://192.168.1.100:8080/video")
    camera_input = input("Enter camera source (default 0): ").strip() or "0"
    camera_source = int(camera_input) if camera_input.isdigit() else camera_input

    # Open camera stream
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print(f"Error: Cannot open camera from {camera_source}")
        return

    print("=" * 50)
    print("Press 'q' to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        # Detect markers
        corners, ids, rejected = aruco_pos.detect_markers(frame)

        # Draw markers
        frame = aruco_pos.draw_markers(frame, corners, ids)
        frame = aruco_pos.draw_axes(frame, corners, ids)

        # Estimate pose
        position, rotation, direction = aruco_pos.estimate_pose(corners, ids)

        # Display information
        if position is not None:
            pos_text = f"Position: X={position[0]:.2f}m Y={position[1]:.2f}m Z={position[2]:.2f}m"
            dir_text = f"Direction: X={direction[0]:.2f} Y={direction[1]:.2f} Z={direction[2]:.2f}"

            cv2.putText(frame, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.putText(frame, dir_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            print(f"\r{pos_text} | {dir_text}", end="")
        else:
            cv2.putText(frame, "No markers detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show frame
        cv2.imshow('ArUco Positioning - SDC26', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


import numpy as np

# Load calibration
data = np.load('camera_calibration.npz')
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

# Use in aruco_positioning.py
aruco_pos = AruCoPositioning(camera_matrix, dist_coeffs)

if __name__ == "__main__":
    main()
