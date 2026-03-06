import glob
import json
import os

import cv2
import numpy as np


class CameraCalibrator:
    def __init__(self, checkerboard_size=(9, 6), square_size=0.025):
        """
        Initialize camera calibrator

        Args:
            checkerboard_size: (cols, rows) - interior corners of checkerboard
            square_size: Size of checkerboard square in meters (default: 25mm)
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size

        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world
        self.imgpoints = []  # 2D points in image plane

    def capture_images(self, num_images=20, camera_source=0, save_dir="calibration_images"):
        """
        Capture calibration images interactively

        Args:
            num_images: Number of images to capture
            camera_source: Camera device ID (int) or IP camera URL (str)
                          Examples: 0, "rtsp://192.168.1.100:554/stream", "http://192.168.1.100:8080/video"
            save_dir: Directory to save captured images
        """
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print(f"Error: Cannot open camera from {camera_source}")
            return False

        print(f"\nCamera Calibration - Image Capture")
        print(f"=" * 50)
        print(f"Target: {num_images} images")
        print(f"Checkerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} interior corners")
        print(f"\nInstructions:")
        print("- Move the checkerboard to different positions and angles")
        print("- Press SPACE when corners are detected (green)")
        print("- Press 'q' to quit early")
        print("=" * 50)

        count = 0

        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()

            # Find checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

            # Draw corners if found
            if ret_corners:
                cv2.drawChessboardCorners(display, self.checkerboard_size, corners, ret_corners)
                status_text = f"Corners detected! Press SPACE to capture ({count}/{num_images})"
                color = (0, 255, 0)
            else:
                status_text = f"No corners detected. Move checkerboard ({count}/{num_images})"
                color = (0, 0, 255)

            cv2.putText(display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Camera Calibration - Capture', display)

            key = cv2.waitKey(1) & 0xFF

            # Capture image on SPACE
            if key == ord(' ') and ret_corners:
                filename = os.path.join(save_dir, f"calib_{count:02d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Captured image {count + 1}/{num_images}: {filename}")
                count += 1

            # Quit on 'q'
            elif key == ord('q'):
                print("\nCapture cancelled by user")
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nCaptured {count} images")
        return count > 0

    def calibrate_from_images(self, image_dir="calibration_images", image_pattern="*.jpg"):
        """
        Calibrate camera from saved images

        Args:
            image_dir: Directory containing calibration images
            image_pattern: Pattern to match image files

        Returns:
            ret: RMS re-projection error
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            rvecs: Rotation vectors
            tvecs: Translation vectors
        """
        images = glob.glob(os.path.join(image_dir, image_pattern))

        if len(images) == 0:
            print(f"Error: No images found in {image_dir}/{image_pattern}")
            return None

        print(f"\nProcessing {len(images)} images for calibration...")

        successful = 0
        img_size = None

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img_size is None:
                img_size = gray.shape[::-1]

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

            if ret:
                self.objpoints.append(self.objp)

                # Refine corner positions
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners_refined)

                successful += 1
                print(f"✓ {os.path.basename(fname)}")
            else:
                print(f"✗ {os.path.basename(fname)} - No corners detected")

        print(f"\nSuccessfully processed: {successful}/{len(images)} images")

        if successful < 3:
            print("Error: Need at least 3 successful images for calibration")
            return None

        # Calibrate camera
        print("\nCalibrating camera...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_size, None, None
        )

        print(f"\n{'=' * 50}")
        print(f"Calibration Results")
        print(f"{'=' * 50}")
        print(f"RMS Re-projection Error: {ret:.4f} pixels")
        print(f"(Lower is better, typically < 1.0 is good)\n")

        print("Camera Matrix:")
        print(camera_matrix)
        print(f"\nFocal Length: fx={camera_matrix[0, 0]:.2f}, fy={camera_matrix[1, 1]:.2f}")
        print(f"Principal Point: cx={camera_matrix[0, 2]:.2f}, cy={camera_matrix[1, 2]:.2f}")

        print("\nDistortion Coefficients:")
        print(dist_coeffs.flatten())
        print(
            f"k1={dist_coeffs[0, 0]:.4f}, k2={dist_coeffs[0, 1]:.4f}, p1={dist_coeffs[0, 2]:.4f}, p2={dist_coeffs[0, 3]:.4f}, k3={dist_coeffs[0, 4]:.4f}")
        print(f"{'=' * 50}\n")

        return ret, camera_matrix, dist_coeffs, rvecs, tvecs

    def save_calibration(self, camera_matrix, dist_coeffs, filename="camera_calibration.npz"):
        """Save calibration to file"""
        np.savez(filename,
                 camera_matrix=camera_matrix,
                 dist_coeffs=dist_coeffs,
                 checkerboard_size=self.checkerboard_size,
                 square_size=self.square_size)
        print(f"Calibration saved to: {filename}")

        # Also save as JSON for easy viewing
        json_filename = filename.replace('.npz', '.json')
        calib_data = {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.flatten().tolist(),
            'checkerboard_size': self.checkerboard_size,
            'square_size': self.square_size
        }
        with open(json_filename, 'w') as f:
            json.dump(calib_data, f, indent=2)
        print(f"Calibration also saved to: {json_filename}")

    def test_calibration(self, camera_matrix, dist_coeffs, camera_source=0):
        """
        Test calibration by showing undistorted video

        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            camera_source: Camera device ID (int) or IP camera URL (str)
        """
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print(f"Error: Cannot open camera from {camera_source}")
            return

        print("\nTesting calibration - Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # Get optimal camera matrix
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )

            # Undistort
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

            # Show side by side
            combined = np.hstack([frame, undistorted])
            cv2.putText(combined, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Undistorted", (w + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Calibration Test', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def load_calibration(filename="camera_calibration.npz"):
    """Load saved calibration"""
    if not os.path.exists(filename):
        print(f"Error: Calibration file not found: {filename}")
        return None, None

    data = np.load(filename)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    print(f"Loaded calibration from: {filename}")
    print("Camera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)

    return camera_matrix, dist_coeffs


def print_checkerboard_guide():
    """Print instructions for creating a checkerboard"""
    print("\n" + "=" * 60)
    print("HOW TO CREATE A CHECKERBOARD PATTERN")
    print("=" * 60)
    print("\nOption 1: Print a standard checkerboard")
    print("  - Download: https://github.com/opencv/opencv/blob/master/doc/pattern.png")
    print("  - Or search: 'opencv checkerboard pattern 9x6'")
    print("  - Print on A4 paper")
    print("  - Glue to flat, rigid surface (cardboard/foam board)")
    print("\nOption 2: Use online generator")
    print("  - Visit: https://calib.io/pages/camera-calibration-pattern-generator")
    print("  - Select: Checkerboard")
    print("  - Rows: 7, Columns: 10 (gives 6x9 interior corners)")
    print("  - Size: 25mm per square")
    print("  - Download and print")
    print("\nIMPORTANT:")
    print("  - Must be perfectly flat!")
    print("  - All squares must be equal size")
    print("  - Measure actual square size after printing")
    print("=" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("CAMERA CALIBRATION FOR SDC26 ArUco POSITIONING")
    print("=" * 60)

    print("\nChoose an option:")
    print("1. Full calibration (capture + calibrate)")
    print("2. Capture images only")
    print("3. Calibrate from existing images")
    print("4. Test existing calibration")
    print("5. Load existing calibration")
    print("6. Show checkerboard guide")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == '6':
        print_checkerboard_guide()
        return

    # Get checkerboard parameters
    print("\nCheckerboard pattern (interior corners):")
    cols = int(input("  Columns (default 9): ") or "9")
    rows = int(input("  Rows (default 6): ") or "6")
    square_size = float(input("  Square size in meters (default 0.025): ") or "0.025")

    calibrator = CameraCalibrator((cols, rows), square_size)

    # Get camera source
    print("\nCamera source:")
    print("  Enter '0' for USB camera")
    print("  Or enter IP camera URL:")
    print("    RTSP: rtsp://192.168.1.100:554/stream")
    print("    HTTP: http://192.168.1.100:8080/video")
    camera_input = input("Camera source: ").strip() or "0"
    camera_source = int(camera_input) if camera_input.isdigit() else camera_input

    if choice == '1':
        # Full calibration
        num_images = int(input("\nNumber of images to capture (default 20): ") or "20")

        if calibrator.capture_images(num_images, camera_source):
            result = calibrator.calibrate_from_images()
            if result:
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = result
                calibrator.save_calibration(camera_matrix, dist_coeffs)

                test = input("\nTest calibration? (y/n): ").strip().lower()
                if test == 'y':
                    calibrator.test_calibration(camera_matrix, dist_coeffs, camera_source)

    elif choice == '2':
        # Capture only
        num_images = int(input("\nNumber of images to capture (default 20): ") or "20")
        calibrator.capture_images(num_images, camera_source)

    elif choice == '3':
        # Calibrate from existing images
        image_dir = input("\nImage directory (default 'calibration_images'): ").strip() or "calibration_images"
        result = calibrator.calibrate_from_images(image_dir)
        if result:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = result
            calibrator.save_calibration(camera_matrix, dist_coeffs)

    elif choice == '4':
        # Test calibration
        camera_matrix, dist_coeffs = load_calibration()
        if camera_matrix is not None:
            print("\nCamera source:")
            print("  Enter '0' for USB camera or IP camera URL")
            camera_input = input("Camera source: ").strip() or "0"
            camera_source = int(camera_input) if camera_input.isdigit() else camera_input
            calibrator.test_calibration(camera_matrix, dist_coeffs, camera_source)

    elif choice == '5':
        # Load calibration
        load_calibration()

    print("\nDone!")


if __name__ == "__main__":
    main()
