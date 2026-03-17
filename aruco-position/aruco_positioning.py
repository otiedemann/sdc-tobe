import time

import cv2
import numpy as np
from cv2 import aruco


class KalmanFilter1D:
    """Simple constant velocity Kalman filter for 1D coordinates"""

    def __init__(self, process_variance=1e-3, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state = None  # [pos, vel]
        self.covariance = np.eye(2)
        self.last_time = None

    def update(self, measurement):
        now = time.time()
        if self.state is None:
            self.state = np.array([measurement, 0.0])
            self.last_time = now
            return measurement
        dt = now - self.last_time
        self.last_time = now
        if dt > 0.5:
            self.state[0], self.state[1] = measurement, 0.0
            return measurement
        F = np.array([[1, dt], [0, 1]])
        self.state = F @ self.state
        Q = np.array([[1e-4, 0], [0, 1e-2]]) * self.process_variance
        self.covariance = F @ self.covariance @ F.T + Q
        H = np.array([[1, 0]])
        y = measurement - (H @ self.state)
        S = H @ self.covariance @ H.T + self.measurement_variance
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.covariance = (np.eye(2) - K @ H) @ self.covariance
        return self.state[0]

    def reset(self):
        self.state, self.last_time = None, None


class ExponentialMovingAverage:
    def __init__(self, alpha=0.3):
        self.alpha, self.value = alpha, None
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class AruCoPositioning:
    MARKER_3D_POINTS = np.array([
        [-0.25, 0.25, 0], [0.25, 0.25, 0], [0.25, -0.25, 0], [-0.25, -0.25, 0]
    ], dtype=np.float32)

    def __init__(self, camera_matrix, dist_coeffs, marker_size=0.5):
        self.marker_size = marker_size
        self.camera_matrix, self.dist_coeffs = camera_matrix, dist_coeffs
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_positions = self._initialize_marker_positions()

        self._wall_rotations = {
            'back': np.eye(3),
            'front': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            'left': np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            'right': np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        }

        self.marker_wall_type = {mid: self._determine_wall_type(pos) for mid, pos in self.marker_positions.items()}
        self.view_angle_x, self.view_angle_y = 50.0, 210.0
        self._cached_view_matrix, self._cached_angles = None, (None, None)
        self._cached_poses = {}
        self.kf_pos = [KalmanFilter1D() for _ in range(3)]
        self.direction_filter = ExponentialMovingAverage(alpha=0.2)
        self.target_marker_filters = {}
        self.last_used_markers = set()
        self.mouse_button_pressed, self.last_rotation_time = None, 0

    def _initialize_marker_positions(self):
        # Coordinates relative to Marker 0 (0,0,0)
        positions = {0: np.array([0.0, 0.0, 0.0])}
        # Back Wall
        positions[5], positions[6] = np.array([6.0, -1.0, 0.0]), np.array([6.0, 1.0, 0.0])
        positions[7], positions[8] = np.array([2.0, -1.0, 0.0]), np.array([2.0, 1.0, 0.0])
        positions[9], positions[10] = np.array([-2.0, -1.0, 0.0]), np.array([-2.0, 1.0, 0.0])
        positions[11], positions[12] = np.array([-6.0, -1.0, 0.0]), np.array([-6.0, 1.0, 0.0])
        # Right Wall
        positions[1], positions[2] = np.array([10.0, -1.0, 6.667]), np.array([10.0, 1.0, 6.667])
        positions[3], positions[4] = np.array([10.0, -1.0, 3.333]), np.array([10.0, 1.0, 3.333])
        # Left Wall
        positions[13], positions[14] = np.array([-10.0, -1.0, 3.333]), np.array([-10.0, 1.0, 3.333])
        positions[15], positions[16] = np.array([-10.0, -1.0, 6.667]), np.array([-10.0, 1.0, 6.667])
        # Front Wall
        positions[17], positions[18] = np.array([-6.0, -1.0, 10.0]), np.array([-6.0, 1.0, 10.0])
        positions[19], positions[20] = np.array([-2.0, -1.0, 10.0]), np.array([-2.0, 1.0, 10.0])
        positions[21], positions[22] = np.array([2.0, -1.0, 10.0]), np.array([2.0, 1.0, 10.0])
        positions[23], positions[24] = np.array([6.0, -1.0, 10.0]), np.array([6.0, 1.0, 10.0])
        return positions

    def _determine_wall_type(self, pos):
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

    def _get_marker_orientation(self, mid):
        return self._wall_rotations.get(self.marker_wall_type.get(mid), np.eye(3))

    def _calculate_marker_weight(self, corner, distance):
        pts = corner.reshape(4, 2)
        v1, v2 = pts[1] - pts[0], pts[3] - pts[0]
        area = abs(np.cross(v1, v2))
        area_w = np.clip(area / 5000.0, 0.1, 1.0)
        dist_w = np.exp(-distance / 10.0)
        side_lens = [np.linalg.norm(pts[j] - pts[(j + 1) % 4]) for j in range(4)]
        edge_v = np.std(side_lens) / (np.mean(side_lens) + 1e-6)
        shape_w = np.exp(-edge_v * 5.0)
        return area_w * dist_w * shape_w

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.detector.detectMarkers(gray)

    def estimate_pose(self, corners, ids):
        if ids is None or len(ids) == 0:
            self._cached_poses.clear()
            for kf in self.kf_pos: kf.reset()
            self.direction_filter.reset()
            return None, None, None, {}

        self._cached_poses.clear()
        for i, corner in enumerate(corners):
            mid = ids.flatten()[i]
            ok, rvec, tvec = cv2.solvePnP(self.MARKER_3D_POINTS, corner.reshape(-1, 2), self.camera_matrix,
                                          self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok: self._cached_poses[mid] = (rvec, tvec.reshape(3))

        ref_i, tgt_i = [], []
        for i, mid in enumerate(ids.flatten()):
            if mid in self.marker_positions and mid in self._cached_poses:
                ref_i.append(i)
            elif mid >= 30 and mid in self._cached_poses:
                tgt_i.append(i)

        if not ref_i: return None, None, None, {}

        pos_l, dir_l, rot_l, weight_l, used_ids = [], [], [], [], []
        for idx in ref_i:
            mid = ids.flatten()[idx]
            rvec, tvec = self._cached_poses[mid]
            R_m_c, _ = cv2.Rodrigues(rvec)
            R_m_w = self._get_marker_orientation(mid)
            c_pos_w = self.marker_positions[mid] + R_m_w @ (-R_m_c.T @ tvec)

            wall = self.marker_wall_type.get(mid)
            if wall in ('front', 'back'): c_pos_w[0] = 2 * self.marker_positions[mid][0] - c_pos_w[0]

            R_c_w = R_m_w @ R_m_c
            c_dir_w = R_c_w @ np.array([0, 0, 1])
            if wall in ('front', 'back'): c_dir_w[0] = -c_dir_w[0]

            w = self._calculate_marker_weight(corners[idx], np.linalg.norm(tvec))
            pos_l.append(c_pos_w);
            dir_l.append(c_dir_w);
            rot_l.append(R_c_w);
            weight_l.append(w);
            used_ids.append(mid)

        w_arr = np.array(weight_l) / (sum(weight_l) + 1e-6)
        r_pos, r_dir = sum(p * w for p, w in zip(pos_l, w_arr)), sum(d * w for d, w in zip(dir_l, w_arr))
        r_dir /= (np.linalg.norm(r_dir) + 1e-6)

        if np.any(np.isnan(r_pos)): return None, None, None, {}

        f_pos = np.array([self.kf_pos[j].update(r_pos[j]) for j in range(3)])
        f_dir = self.direction_filter.update(r_dir)
        f_dir /= (np.linalg.norm(f_dir) + 1e-6)

        curr_R = rot_l[np.argmax(weight_l)]
        targets = {}
        for idx in tgt_i:
            tid = ids.flatten()[idx]
            t_w = f_pos + curr_R @ self._cached_poses[tid][1]
            if tid not in self.target_marker_filters: self.target_marker_filters[tid] = ExponentialMovingAverage(
                alpha=0.1)
            targets[tid] = self.target_marker_filters[tid].update(t_w)

        self.last_used_markers = set(used_ids)
        return f_pos, curr_R, f_dir, targets

    def draw_markers(self, f, c, i):
        if i is not None: aruco.drawDetectedMarkers(f, c, i)
        return f

    def draw_axes(self, f, corners, ids):
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid in self._cached_poses:
                    r, t = self._cached_poses[mid]
                    if np.linalg.norm(t) < 15.0:
                        try:
                            cv2.drawFrameAxes(f, self.camera_matrix, self.dist_coeffs, r, t, self.marker_size * 0.4)
                        except:
                            pass
        return f

    def project_3d_to_2d(self, p3d, vm, sc=20):
        shifted = p3d + np.array([0, 2.0, 0])
        rot = vm @ shifted
        return int(rot[0] * sc + 400), int(400 - rot[2] * sc)

    def _get_view_matrix(self):
        cur = (self.view_angle_x, self.view_angle_y)
        if self._cached_angles != cur:
            ax, ay = np.radians(self.view_angle_x), np.radians(self.view_angle_y)
            Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
            Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
            self._cached_view_matrix = Rx @ Ry
            self._cached_angles = cur
        return self._cached_view_matrix

    def create_3d_visualization(self, pos, dr, tgts=None):
        if tgts is None: tgts = {}
        img = np.ones((800, 800, 3), dtype=np.uint8) * 240
        vm = self._get_view_matrix()
        c = np.array([[-10, -3, 0], [10, -3, 0], [10, -3, 10], [-10, -3, 10], [-10, 3, 0], [10, 3, 0], [10, 3, 10],
                      [-10, 3, 10]])
        e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for s, ex in e: cv2.line(img, self.project_3d_to_2d(c[s], vm), self.project_3d_to_2d(c[ex], vm),
                                 (190, 190, 190), 1)
        opt = self.project_3d_to_2d(np.array([0., 0., 0.]), vm)
        for i, (ax, col, l) in enumerate(
                [(np.array([2.5, 0, 0]), (0, 0, 220), 'X'), (np.array([0, 2.5, 0]), (0, 220, 0), 'Y'),
                 (np.array([0, 0, 2.5]), (220, 0, 0), 'Z')]):
            apt = self.project_3d_to_2d(ax, vm)
            cv2.arrowedLine(img, opt, apt, col, 2, tipLength=0.2);
            cv2.putText(img, l, apt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        for mid, mpos in self.marker_positions.items():
            pt = self.project_3d_to_2d(mpos, vm)
            u = mid in self.last_used_markers
            cv2.circle(img, pt, 5 if u else 3, (0, 200, 0) if u else (160, 160, 160), -1)
            cv2.putText(img, str(mid), (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 160, 0) if u else (140, 140, 140), 1)
            if u: cv2.arrowedLine(img, pt, self.project_3d_to_2d(
                mpos + self._get_marker_orientation(mid) @ np.array([0, 0, 0.8]), vm), (0, 180, 0), 1, tipLength=0.3)
        for mid, mpos in tgts.items():
            pt = self.project_3d_to_2d(mpos, vm)
            cv2.circle(img, pt, 6, (200, 0, 200), -1);
            cv2.putText(img, f"ID{mid}", (pt[0] + 7, pt[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)
        if pos is not None:
            cpt = self.project_3d_to_2d(pos, vm);
            cv2.circle(img, cpt, 8, (0, 0, 255), -1)
            if dr is not None: cv2.arrowedLine(img, cpt, self.project_3d_to_2d(pos + dr * 1.5, vm), (0, 0, 255), 2,
                                               tipLength=0.3)
        for l, bx, by in [('UP', 680, 20), ('DOWN', 680, 100), ('LEFT', 610, 60), ('RIGHT', 750, 60)]:
            cv2.rectangle(img, (bx, by), (bx + 60, by + 30), (210, 210, 210), -1);
            cv2.rectangle(img, (bx, by), (bx + 60, by + 30), (140, 140, 140), 1)
            cv2.putText(img, l, (bx + 10, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
        return img


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 680 <= x <= 740 and 20 <= y <= 50:
            param.mouse_button_pressed = 'up'
        elif 680 <= x <= 740 and 100 <= y <= 130:
            param.mouse_button_pressed = 'down'
        elif 610 <= x <= 670 and 60 <= y <= 90:
            param.mouse_button_pressed = 'left'
        elif 750 <= x <= 810 and 60 <= y <= 90:
            param.mouse_button_pressed = 'right'
    elif event == cv2.EVENT_LBUTTONUP:
        param.mouse_button_pressed = None


def main():
    import sys
    cm = np.array([[850., 0., 320.], [0., 850., 240.], [0., 0., 1.]], dtype=float)
    dc = np.zeros(5, dtype=float)
    if len(sys.argv) > 2 and sys.argv[1] == '--load-calibration':
        try:
            d = np.load(sys.argv[2]); cm, dc = d['camera_matrix'], d['dist_coeffs']
        except:
            pass
    ap = AruCoPositioning(cm, dc)
    src = input("Src (0): ").strip() or "0"
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    cv2.namedWindow('Analysis');
    cv2.namedWindow('3D Arena View');
    cv2.setMouseCallback('3D Arena View', mouse_callback, ap)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        corners, ids, rej = ap.detect_markers(frame)
        ap.draw_markers(frame, corners, ids);
        ap.draw_axes(frame, corners, ids)
        pos, rot, dr, tgts = ap.estimate_pose(corners, ids)
        if pos is not None:
            cv2.putText(frame, f"CAM: {pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            y_o = 60
            for tid, tp in tgts.items():
                cv2.putText(frame, f"T{tid}: {tp[0]:.2f} {tp[1]:.2f} {tp[2]:.2f}", (10, y_o), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 0, 200), 2)
                y_o += 25
            print(f"\rRelPos: {pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f} | T: {len(tgts)}", end="")
        if ap.mouse_button_pressed and time.time() - ap.last_rotation_time > 0.05:
            if ap.mouse_button_pressed == 'up':
                ap.view_angle_x += 2
            elif ap.mouse_button_pressed == 'down':
                ap.view_angle_x -= 2
            elif ap.mouse_button_pressed == 'left':
                ap.view_angle_y -= 2
            elif ap.mouse_button_pressed == 'right':
                ap.view_angle_y += 2
            ap.last_rotation_time = time.time()
        cv2.imshow('Analysis', frame);
        cv2.imshow('3D Arena View', ap.create_3d_visualization(pos, dr, tgts))
        if cv2.waitKey(1) == ord('q'): break
    cap.release();
    cv2.destroyAllWindows()


if __name__ == "__main__": main()
