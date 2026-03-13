import time

import cv2
import numpy as np
from cv2 import aruco


class ExponentialMovingAverage:
    """Exponential moving average filter for smoothing noisy data"""

    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: Smoothing factor (0-1). Lower = more smoothing, higher = more responsive
                   0.1-0.3 recommended for position tracking
        """
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        """Update filter with new measurement"""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        """Reset filter state"""
        self.value = None


class AruCoPositioning:
    # Pre-computed marker 3D points (class constant)
    MARKER_3D_POINTS = np.array([
        [-0.25, 0.25, 0],  # marker_size/2 = 0.5/2 = 0.25
        [0.25, 0.25, 0],
        [0.25, -0.25, 0],
        [-0.25, -0.25, 0]
    ], dtype=np.float32)

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

        # Cache the detector to avoid recreation every frame
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Marker positions in world coordinates (based on Figure 3 and 4)
        # Origin (marker 0) is at back wall center
        # Coordinate system: X-right, Y-up, Z-forward (from marker 0)
        self.marker_positions = self._initialize_marker_positions()

        # Pre-compute wall orientation matrices
        self._wall_rotations = {
            'back': np.eye(3),  # Z=0
            'front': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # Z=10
            'left': np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # X=-10
            'right': np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])  # X=10
        }

        # Cache wall type for each marker
        self.marker_wall_type = {}
        for marker_id, pos in self.marker_positions.items():
            self.marker_wall_type[marker_id] = self._determine_wall_type(pos)

        self.view_angle_x = 50.0
        self.view_angle_y = 210.0

        # Cache for view matrix
        self._cached_view_matrix = None
        self._cached_angles = (None, None)

        # Cache for pose estimation results
        self._cached_poses = {}

        # Temporal filters for position and direction (alpha=0.2 for smooth tracking)
        self.position_filter = ExponentialMovingAverage(alpha=0.2)
        self.direction_filter = ExponentialMovingAverage(alpha=0.2)

        # Mouse button state for continuous rotation
        self.mouse_button_pressed = None
        self.last_rotation_time = 0

    def _initialize_marker_positions(self):
        """
        Define 3D positions of all ArUco markers based on SDC26 regulations
        Field: 20m length × 10m width × 6m height
        Markers at 2m and 4m heights on pillars
        """
        positions = {}

        # Marker 0 at origin (back wall center)
        positions[0] = np.array([0.0, 3.0, 0.0])

        # Left side pillars (X=10m)
        positions[1] = np.array([10.0, 2.0, 6.667])
        positions[2] = np.array([10.0, 4.0, 6.667])
        positions[3] = np.array([10.0, 2.0, 3.333])
        positions[4] = np.array([10.0, 4.0, 3.333])

        # Back wall pillars (Z=0m)
        positions[5] = np.array([6.0, 2.0, 0.0])
        positions[6] = np.array([6.0, 4.0, 0.0])
        positions[7] = np.array([2.0, 2.0, 0.0])
        positions[8] = np.array([2.0, 4.0, 0.0])
        positions[9] = np.array([-2.0, 2.0, 0.0])
        positions[10] = np.array([-2.0, 4.0, 0.0])
        positions[11] = np.array([-6.0, 2.0, 0.0])
        positions[12] = np.array([-6.0, 4.0, 0.0])

        # Right side pillars (X=-10m)
        positions[13] = np.array([-10.0, 2.0, 3.333])
        positions[14] = np.array([-10.0, 4.0, 3.333])
        positions[15] = np.array([-10.0, 2.0, 6.667])
        positions[16] = np.array([-10.0, 4.0, 6.667])

        # Front wall pillars (Z=10m)
        positions[17] = np.array([-6.0, 2.0, 10.0])
        positions[18] = np.array([-6.0, 4.0, 10.0])
        positions[19] = np.array([-2.0, 2.0, 10.0])
        positions[20] = np.array([-2.0, 4.0, 10.0])
        positions[21] = np.array([2.0, 2.0, 10.0])
        positions[22] = np.array([2.0, 4.0, 10.0])
        positions[23] = np.array([6.0, 2.0, 10.0])
        positions[24] = np.array([6.0, 4.0, 10.0])

        return positions

    def _determine_wall_type(self, pos):
        """Determine which wall a marker is on based on its position"""
        x, z = pos[0], pos[2]
        if abs(z - 0.0) < 0.1:
            return 'back'
        elif abs(z - 10.0) < 0.1:
            return 'front'
        elif abs(x - (-10.0)) < 0.1:
            return 'left'
        elif abs(x - 10.0) < 0.1:
            return 'right'
        return None

    def _get_marker_orientation(self, marker_id):
        """
        Get the rotation matrix for marker's orientation in world frame
        Markers face into the arena from their respective walls

        Args:
            marker_id: ID of the marker

        Returns:
            3x3 rotation matrix for marker orientation in world coordinates
        """
        if marker_id not in self.marker_wall_type:
            return np.eye(3)

        wall_type = self.marker_wall_type[marker_id]
        return self._wall_rotations.get(wall_type, np.eye(3))

    def _calculate_marker_weight(self, corner, distance):
        """
        Calculate weight for a marker based on detection quality
        Higher weight = more reliable measurement

        Args:
            corner: Detected marker corners (4 points)
            distance: Distance from camera to marker

        Returns:
            float: Weight value (0-1), higher is better
        """
        # Factor 1: Marker size in image (larger = closer = more reliable)
        # Calculate area of detected marker in pixels
        pts = corner.reshape(4, 2)
        # Simple area approximation using cross product
        v1 = pts[1] - pts[0]
        v2 = pts[3] - pts[0]
        area = abs(np.cross(v1, v2))

        # Normalize area (typical range: 100-10000 pixels²)
        area_weight = np.clip(area / 5000.0, 0.1, 1.0)

        # Factor 2: Distance (closer markers are more accurate)
        # Weight decreases with distance, using exponential decay
        distance_weight = np.exp(-distance / 10.0)  # 10m decay constant

        # Factor 3: Corner sharpness (how square/well-defined is the marker)
        # Check if corners form a roughly square shape
        edge_lengths = [
            np.linalg.norm(pts[1] - pts[0]),
            np.linalg.norm(pts[2] - pts[1]),
            np.linalg.norm(pts[3] - pts[2]),
            np.linalg.norm(pts[0] - pts[3])
        ]
        avg_edge = np.mean(edge_lengths)
        edge_variance = np.std(edge_lengths) / (avg_edge + 1e-6)
        # Lower variance = more square = better
        shape_weight = np.exp(-edge_variance * 5.0)

        # Combine factors (all weighted equally)
        total_weight = area_weight * distance_weight * shape_weight

        return total_weight

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
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected

    def estimate_pose(self, corners, ids):
        """
        Estimate camera pose from detected markers using multi-marker fusion

        Args:
            corners: Detected marker corners
            ids: Detected marker IDs

        Returns:
            position: Camera position in world coordinates (x, y, z) - filtered
            rotation: Camera rotation matrix
            camera_direction: Unit vector of camera viewing direction - filtered
            target_marker_positions: Dict of {marker_id: position} for target markers (ID >= 30)
        """
        if ids is None or len(ids) == 0:
            self._cached_poses.clear()
            self.position_filter.reset()
            self.direction_filter.reset()
            return None, None, None, {}

        # Estimate pose for each marker and cache results
        self._cached_poses.clear()
        for i, corner in enumerate(corners):
            marker_id = ids.flatten()[i]
            _, rvec, tvec = cv2.solvePnP(
                self.MARKER_3D_POINTS,
                corner.reshape(-1, 2),
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            self._cached_poses[marker_id] = (rvec, tvec)

        # Multi-marker fusion with weighting: collect all position and direction estimates
        positions = []
        directions = []
        rotation_matrices = []
        weights = []
        target_marker_positions = {}
        reference_wall_types = []  # Track wall types for corrections

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.marker_positions:
                # Get cached marker pose in camera frame
                rvec, tvec = self._cached_poses[marker_id]
                tvec = tvec.reshape(3)

                # Convert rotation vector to matrix
                R_marker_cam, _ = cv2.Rodrigues(rvec)

                # Get camera position in marker's local frame
                R_cam_marker = R_marker_cam.T
                t_cam_marker = -R_cam_marker @ tvec

                # Get marker's world position and orientation
                marker_world_pos = self.marker_positions[marker_id]
                R_marker_world = self._get_marker_orientation(marker_id)

                # Transform camera position to world frame
                camera_position = marker_world_pos + R_marker_world @ t_cam_marker

                # Get wall type for corrections
                wall_type = self.marker_wall_type.get(marker_id)

                # Fix X-axis for front and back wall markers
                # OpenCV ArUco X-axis convention requires correction
                if wall_type in ('front', 'back'):
                    # Compensate for X-axis inversion due to rotation
                    camera_position[0] = 2 * marker_world_pos[0] - camera_position[0]

                # Camera orientation in world frame
                R_cam_world = R_marker_world @ R_cam_marker

                # Camera direction: camera looks along positive Z in OpenCV camera frame
                # Direction vector from rotation matrix is already normalized
                camera_direction = R_cam_world @ np.array([0, 0, 1])

                # Fix direction X-axis for front and back wall markers
                if wall_type in ('front', 'back'):
                    camera_direction[0] = -camera_direction[0]

                # Calculate weight based on detection quality
                distance = np.linalg.norm(tvec)
                corner = corners[i]
                weight = self._calculate_marker_weight(corner, distance)

                positions.append(camera_position)
                directions.append(camera_direction)
                rotation_matrices.append(R_cam_world)
                weights.append(weight)
                reference_wall_types.append(wall_type)
            elif marker_id >= 30:
                # Calculate position for target markers (ID >= 30)
                # Get cached marker pose in camera frame
                rvec, tvec = self._cached_poses[marker_id]
                tvec = tvec.reshape(3)

                # Convert rotation vector to matrix
                R_marker_cam, _ = cv2.Rodrigues(rvec)

                # Get target marker position in camera frame
                marker_pos_cam = tvec

                # If we have a valid camera position from known markers,
                # transform target marker position to world coordinates
                # This will be done after camera position is calculated
                target_marker_positions[marker_id] = {
                    'camera_frame': marker_pos_cam,
                    'R_marker_cam': R_marker_cam
                }

        if len(positions) == 0:
            # Clear target marker positions since we can't calculate world positions without reference markers
            return None, None, None, {}

        # Weighted fusion of multiple marker estimates
        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-6)  # Normalize weights

        # Weighted average of positions
        raw_position = np.zeros(3)
        for pos, w in zip(positions, weights):
            raw_position += pos * w

        # Weighted average of directions
        raw_direction = np.zeros(3)
        for dir_vec, w in zip(directions, weights):
            raw_direction += dir_vec * w
        # Re-normalize direction after weighted averaging
        raw_direction = raw_direction / np.linalg.norm(raw_direction)

        # Apply temporal filtering
        filtered_position = self.position_filter.update(raw_position)
        filtered_direction = self.direction_filter.update(raw_direction)
        # Re-normalize filtered direction
        filtered_direction = filtered_direction / np.linalg.norm(filtered_direction)

        # Use rotation from first marker (could be improved with rotation averaging)
        R_cam_world = rotation_matrices[0]
        primary_wall_type = reference_wall_types[0] if reference_wall_types else None

        # Calculate world positions for target markers (ID >= 30)
        for marker_id, marker_data in target_marker_positions.items():
            # Get target marker position in camera frame (tvec from solvePnP)
            # tvec represents the target marker's position in the camera coordinate system
            marker_pos_cam = marker_data['camera_frame']

            # Transform target marker position from camera frame to world coordinates
            # R_cam_world transforms vectors from camera frame to world frame
            # World position = Camera world position + Rotation * Camera frame vector
            marker_world_pos = filtered_position + R_cam_world @ marker_pos_cam
            # Make a copy to avoid modifying shared array references
            marker_world_pos = marker_world_pos.copy()

            # Apply X-axis correction if primary reference marker was on front/back wall
            # This matches the correction applied to camera position calculation
            if primary_wall_type in ('front', 'back'):
                # Invert X coordinate
                marker_world_pos[0] = 2 * filtered_position[0] - marker_world_pos[0]

            # Update with world coordinates
            target_marker_positions[marker_id] = marker_world_pos

        return filtered_position, R_cam_world, filtered_direction, target_marker_positions

    def draw_markers(self, frame, corners, ids):
        """Draw detected markers on frame"""
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
        return frame

    def draw_axes(self, frame, corners, ids):
        """Draw 3D axes on detected markers using cached pose data"""
        if ids is not None:
            for marker_id in ids.flatten():
                if marker_id in self._cached_poses:
                    rvec, tvec = self._cached_poses[marker_id]
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                      rvec, tvec, self.marker_size * 0.5)
        return frame

    def project_3d_to_2d(self, point_3d, view_matrix, scale=20):
        """
        Project 3D point to 2D for visualization using simple orthographic projection

        Args:
            point_3d: 3D point (x, y, z)
            view_matrix: Rotation matrix for view angle
            scale: Scale factor for display

        Returns:
            2D point (x, y) on screen
        """
        # Apply view rotation - this rotates the 3D point including Z depth
        rotated = view_matrix @ point_3d
        # Orthographic projection: use rotated X and Y for screen position
        # (Z is depth and affects which objects are in front, but not position in ortho projection)
        x = int(rotated[0] * scale + 400)
        y = int(400 - rotated[2] * scale)  # Use Z for vertical (depth goes into screen)
        return (x, y)

    def _get_view_matrix(self):
        """Get or compute cached view matrix based on current angles"""
        current_angles = (self.view_angle_x, self.view_angle_y)
        if self._cached_angles != current_angles:
            # Recompute view matrix
            angle_x = np.radians(self.view_angle_x)
            angle_y = np.radians(self.view_angle_y)

            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)]
            ])

            Ry = np.array([
                [np.cos(angle_y), 0, np.sin(angle_y)],
                [0, 1, 0],
                [-np.sin(angle_y), 0, np.cos(angle_y)]
            ])

            self._cached_view_matrix = Rx @ Ry
            self._cached_angles = current_angles

        return self._cached_view_matrix

    def create_3d_visualization(self, position, direction, target_marker_positions=None):
        """
        Create lightweight 3D wireframe visualization using OpenCV only

        Args:
            position: Camera position (x, y, z)
            direction: Camera direction vector
            target_marker_positions: Dict of positions for target markers (ID >= 30)

        Returns:
            Image of the 3D visualization
        """
        if target_marker_positions is None:
            target_marker_positions = {}
        # Create blank image
        img = np.ones((800, 800, 3), dtype=np.uint8) * 255

        # Get cached view matrix
        view_matrix = self._get_view_matrix()

        # Define arena corners (20m x 10m x 6m)
        # Coordinates: (X, Y, Z) where Y is height
        corners = np.array([
            [-10, 0, 0], [10, 0, 0], [10, 0, 10], [-10, 0, 10],  # Bottom
            [-10, 6, 0], [10, 6, 0], [10, 6, 10], [-10, 6, 10],  # Top
        ])

        # Draw arena edges
        edges = [
            # Bottom rectangle
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top rectangle
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for start_idx, end_idx in edges:
            pt1 = self.project_3d_to_2d(corners[start_idx], view_matrix)
            pt2 = self.project_3d_to_2d(corners[end_idx], view_matrix)
            cv2.line(img, pt1, pt2, (200, 200, 200), 1)

        # Draw coordinate system axes at real origin (0, 0, 0)
        origin = np.array([0.0, 0.0, 0.0])
        axis_length = 3.0

        # X-axis (red)
        x_axis = np.array([axis_length, 0.0, 0.0])
        origin_pt = self.project_3d_to_2d(origin, view_matrix)
        x_pt = self.project_3d_to_2d(x_axis, view_matrix)
        cv2.arrowedLine(img, origin_pt, x_pt, (0, 0, 255), 3, tipLength=0.2)
        cv2.putText(img, 'X', x_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Y-axis (green)
        y_axis = np.array([0.0, axis_length, 0.0])
        y_pt = self.project_3d_to_2d(y_axis, view_matrix)
        cv2.arrowedLine(img, origin_pt, y_pt, (0, 255, 0), 3, tipLength=0.2)
        cv2.putText(img, 'Y', y_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Z-axis (blue)
        z_axis = np.array([0.0, 0.0, axis_length])
        z_pt = self.project_3d_to_2d(z_axis, view_matrix)
        cv2.arrowedLine(img, origin_pt, z_pt, (255, 0, 0), 3, tipLength=0.2)
        cv2.putText(img, 'Z', z_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw markers
        for marker_id, marker_pos in self.marker_positions.items():
            pt = self.project_3d_to_2d(marker_pos, view_matrix)
            cv2.circle(img, pt, 3, (255, 100, 0), -1)
            cv2.putText(img, str(marker_id), (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 0), 1)

            # Draw orientation arrow for each marker
            R_marker = self._get_marker_orientation(marker_id)
            # Marker's forward direction (facing into arena) is along its +Z axis
            marker_forward = R_marker @ np.array([0, 0, 1.0])
            arrow_end = marker_pos + marker_forward
            arrow_pt = self.project_3d_to_2d(arrow_end, view_matrix)
            cv2.arrowedLine(img, pt, arrow_pt, (150, 150, 0), 1, tipLength=0.3)

        # Draw target markers (ID >= 30) in magenta/purple
        for marker_id, marker_pos in target_marker_positions.items():
            pt = self.project_3d_to_2d(marker_pos, view_matrix)
            cv2.circle(img, pt, 5, (255, 0, 255), -1)  # Magenta color
            cv2.putText(img, str(marker_id), (pt[0] + 7, pt[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)

        # Draw camera position and direction
        if position is not None:
            cam_pt = self.project_3d_to_2d(position, view_matrix)
            cv2.circle(img, cam_pt, 8, (0, 0, 255), -1)

            # Draw direction arrow
            if direction is not None:
                arrow_end = position + direction * 2.0
                arrow_pt = self.project_3d_to_2d(arrow_end, view_matrix)
                cv2.arrowedLine(img, cam_pt, arrow_pt, (0, 0, 255), 2, tipLength=0.3)

                # Display position text
                pos_text = f"Pos: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})m"
                cv2.putText(img, pos_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Add title and legend
        cv2.putText(img, 'SDC26 Arena - 3D View', (10, 780),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, 'Blue: Markers | Magenta: Targets | Red: Camera', (10, 760),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Display view angles
        view_text = f"View: X={self.view_angle_x:.1f}deg Y={self.view_angle_y:.1f}deg"
        cv2.putText(img, view_text, (10, 740),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Draw control buttons
        button_y = 20
        button_width = 60
        button_height = 30

        # Up button
        cv2.rectangle(img, (680, button_y), (680 + button_width, button_y + button_height), (200, 200, 200), -1)
        cv2.rectangle(img, (680, button_y), (680 + button_width, button_y + button_height), (100, 100, 100), 2)
        cv2.putText(img, 'UP', (690, button_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Down button
        cv2.rectangle(img, (680, button_y + 80), (680 + button_width, button_y + 80 + button_height), (200, 200, 200),
                      -1)
        cv2.rectangle(img, (680, button_y + 80), (680 + button_width, button_y + 80 + button_height), (100, 100, 100),
                      2)
        cv2.putText(img, 'DOWN', (685, button_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

        # Left button
        cv2.rectangle(img, (610, button_y + 40), (610 + button_width, button_y + 40 + button_height), (200, 200, 200),
                      -1)
        cv2.rectangle(img, (610, button_y + 40), (610 + button_width, button_y + 40 + button_height), (100, 100, 100),
                      2)
        cv2.putText(img, 'LEFT', (615, button_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

        # Right button
        cv2.rectangle(img, (750, button_y + 40), (750 + button_width, button_y + 40 + button_height), (200, 200, 200),
                      -1)
        cv2.rectangle(img, (750, button_y + 40), (750 + button_width, button_y + 40 + button_height), (100, 100, 100),
                      2)
        cv2.putText(img, 'RIGHT', (753, button_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

        return img


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks and hold on the 3D view buttons"""
    aruco_pos = param
    button_y = 20
    button_width = 60
    button_height = 30

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check which button was pressed and store it
        if 680 <= x <= 740 and button_y <= y <= button_y + button_height:
            aruco_pos.mouse_button_pressed = 'up'
        elif 680 <= x <= 740 and button_y + 80 <= y <= button_y + 80 + button_height:
            aruco_pos.mouse_button_pressed = 'down'
        elif 610 <= x <= 670 and button_y + 40 <= y <= button_y + 40 + button_height:
            aruco_pos.mouse_button_pressed = 'left'
        elif 750 <= x <= 810 and button_y + 40 <= y <= button_y + 40 + button_height:
            aruco_pos.mouse_button_pressed = 'right'

    elif event == cv2.EVENT_LBUTTONUP:
        # Release button
        aruco_pos.mouse_button_pressed = None


def main():
    """Main loop for continuous position tracking"""
    import sys

    print("ArUco Positioning System - SDC26")
    print("=" * 50)

    # For optimized printing
    last_print_pos = None

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
    print("Click buttons on 3D view to rotate")
    print("=" * 50)

    # Set up mouse callback for the 3D view window
    cv2.namedWindow('3D Arena View')
    cv2.setMouseCallback('3D Arena View', mouse_callback, aruco_pos)

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
        position, rotation, direction, marker_positions_30plus = aruco_pos.estimate_pose(corners, ids)

        # Display information
        if position is not None:
            pos_text = f"Position: X={position[0]:.2f}m Y={position[1]:.2f}m Z={position[2]:.2f}m"
            dir_text = f"Direction: X={direction[0]:.2f} Y={direction[1]:.2f} Z={direction[2]:.2f}"

            cv2.putText(frame, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.putText(frame, dir_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # Only print if position changed significantly (>1cm) to reduce terminal overhead
            current_pos = (round(position[0], 2), round(position[1], 2), round(position[2], 2))
            if last_print_pos != current_pos:
                marker_30_info = ""
                if marker_positions_30plus:
                    marker_30_info = " | Markers >=30: " + ", ".join(
                        f"ID{mid}({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})"
                        for mid, pos in marker_positions_30plus.items()
                    )
                print(f"\r{pos_text} | {dir_text}{marker_30_info}", end="", flush=True)
                last_print_pos = current_pos

        # Display markers with ID >= 30 on frame
        if marker_positions_30plus:
            y_offset = 90
            for marker_id, pos in marker_positions_30plus.items():
                marker_text = f"Marker {marker_id}: X={pos[0]:.2f}m Y={pos[1]:.2f}m Z={pos[2]:.2f}m"
                cv2.putText(frame, marker_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 255), 2)
                y_offset += 25

        if position is None:
            cv2.putText(frame, "No markers detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            last_print_pos = None

        # Handle continuous rotation if button is held
        current_time = time.time()
        if aruco_pos.mouse_button_pressed is not None:
            # Rotate every 50ms for smooth continuous rotation
            if current_time - aruco_pos.last_rotation_time > 0.05:
                if aruco_pos.mouse_button_pressed == 'up':
                    aruco_pos.view_angle_x += 2
                elif aruco_pos.mouse_button_pressed == 'down':
                    aruco_pos.view_angle_x -= 2
                elif aruco_pos.mouse_button_pressed == 'left':
                    aruco_pos.view_angle_y -= 2
                elif aruco_pos.mouse_button_pressed == 'right':
                    aruco_pos.view_angle_y += 2
                aruco_pos.last_rotation_time = current_time

        # Create and display 3D visualization
        viz_3d = aruco_pos.create_3d_visualization(position, direction, marker_positions_30plus)

        # Show frame and 3D visualization
        cv2.imshow('ArUco Positioning - SDC26', frame)
        cv2.imshow('3D Arena View', viz_3d)

        # Handle keyboard input (don't mask with 0xFF for arrow keys on Windows)
        key = cv2.waitKey(1)

        # Exit on 'q'
        if key == ord('q'):
            break
        # Arrow keys to adjust view angle
        # Windows uses different codes than Linux
        elif key == 2490368 or key == 82:  # Up arrow (Windows/Linux)
            aruco_pos.view_angle_x += 5
        elif key == 2621440 or key == 84:  # Down arrow (Windows/Linux)
            aruco_pos.view_angle_x -= 5
        elif key == 2424832 or key == 81:  # Left arrow (Windows/Linux)
            aruco_pos.view_angle_y -= 5
        elif key == 2555904 or key == 83:  # Right arrow (Windows/Linux)
            aruco_pos.view_angle_y += 5

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
