import cv2
import numpy as np
import time

# Capture video from camera (0 is default camera)
cap = cv2.VideoCapture(0)

# Create background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# Tracking variables
tracks = []  # List of {'id': int, 'positions': [(x, y, area, time), ...]}
next_id = 0
max_distance = 100  # Max distance to match tracks

# Certainty calculation parameters
ASPECT_RATIO_IDEAL = 1.5  # Ideal aspect ratio for drones (flattened shape)
SIZE_MULTIPLIER = 1  # Size score = min(100, area / SIZE_MULTIPLIER)
DURATION_MULTIPLIER = 100  # Duration score = min(100, duration * DURATION_MULTIPLIER)
SMOOTHNESS_MULTIPLIER = 5  # Smoothness penalty = (var_x + var_y) * SMOOTHNESS_MULTIPLIER
# Drone size parameters (in pixels)
MIN_DRONE_PIXELS = 100  # Minimum area for a drone (10x10)
MAX_DRONE_PIXELS = 50000  # Maximum area for a drone

# Drone movement parameters
MIN_DRONE_SPEED = 1  # Minimum speed (pixels per frame) to be detected as flying drone
BORDER_MARGIN = 30  # Margin in pixels from image border - drones touching border are ignored
prev_gray = None
camera_dx = 0
camera_dy = 0

# FPS tracking
fps_clock = time.time()
fps_counter = 0
fps_value = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Calculate FPS
    fps_counter += 1
    elapsed = current_time - fps_clock
    if elapsed >= 1.0:
        fps_value = fps_counter / elapsed
        fps_counter = 0
        fps_clock = current_time

    # Calculate camera motion using optical flow
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        # Find good features in previous frame
        prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        if prev_points is not None:
            # Calculate optical flow
            curr_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None)
            if curr_points is not None and len(curr_points) > 0:
                # Calculate motion vectors
                motion_vectors = curr_points - prev_points
                # Filter valid points
                valid = status.ravel() == 1
                motion_vectors = motion_vectors[valid]
                if len(motion_vectors) > 0:
                    # Compute median motion as camera motion
                    camera_dx = np.median(motion_vectors[:, 0, 0])
                    camera_dy = np.median(motion_vectors[:, 0, 1])
                else:
                    camera_dx = 0
                    camera_dy = 0
            else:
                camera_dx = 0
                camera_dy = 0
        else:
            camera_dx = 0
            camera_dy = 0
    else:
        camera_dx = 0
        camera_dy = 0

    # Update prev_gray
    prev_gray = curr_gray

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Apply background subtraction to detect moving objects
    fgmask = fgbg.apply(frame)

    # Apply threshold to get binary image
    thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate to fill holes in the detected objects
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the moving objects
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        # Filter out small contours (noise)
        if cv2.contourArea(contour) < 500:
            continue

        # Get bounding box
        (x, y, w, h) = cv2.boundingRect(contour)

        # Filter for potential drones based on size and aspect ratio
        aspect_ratio = float(w) / h if h != 0 else 0
        area = w * h

        if 0.5 < aspect_ratio < 2.0 and MIN_DRONE_PIXELS < area < MAX_DRONE_PIXELS:  # Within min/max size range
            # Check if centroid is within border margin
            cx_det = x + w//2
            cy_det = y + h//2
            if BORDER_MARGIN < cx_det < frame_width - BORDER_MARGIN and BORDER_MARGIN < cy_det < frame_height - BORDER_MARGIN:
                detections.append((cx_det, cy_det, area, current_time, aspect_ratio))  # centroid x,y, area, time, aspect_ratio

    # Match detections to tracks
    matched = set()
    for det in detections:
        cx, cy, carea, ctime, _ = det
        best_match = None
        best_dist = float('inf')
        for track in tracks:
            if track['id'] in matched:
                continue
            if track['positions']:
                px, py, _, _, _ = track['positions'][-1]
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < max_distance and dist < best_dist:
                    best_match = track
                    best_dist = dist
        if best_match:
            best_match['positions'].append(det)
            matched.add(best_match['id'])
        else:
            # New track
            tracks.append({'id': next_id, 'positions': [det]})
            next_id += 1

    # Remove old tracks (not updated for 1 second)
    tracks = [t for t in tracks if current_time - t['positions'][-1][3] < 1.0]

    # Annotate confirmed drones (visible for at least 0.2 seconds)
    for track in tracks:
        positions = track['positions']
        if len(positions) < 2:
            continue
        start_time = positions[0][3]
        duration = current_time - start_time
        if duration < 0.2:
            continue

        # Get current position
        cx, cy, carea, ctime, caspect = positions[-1]
        prev_cx, prev_cy, _, _, _ = positions[-2] if len(positions) > 1 else (positions[-1][0], positions[-1][1], 0, 0, 0)

        # Calculate direction vector (compensate for camera motion)
        dx = cx - prev_cx - camera_dx
        dy = cy - prev_cy - camera_dy
        speed = np.sqrt(dx**2 + dy**2)
        
        # Skip if movement is too minimal (not a flying drone)
        if speed < MIN_DRONE_SPEED:
            continue
        
        # Skip if drone is too close to image border
        if not (BORDER_MARGIN < cx < frame_width - BORDER_MARGIN and BORDER_MARGIN < cy < frame_height - BORDER_MARGIN):
            continue

        # Determine direction
        if speed > MIN_DRONE_SPEED:  # Moving
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if -45 <= angle < 45:
                direction = "Right"
            elif 45 <= angle < 135:
                direction = "Down"
            elif -135 <= angle < -45:
                direction = "Up"
            else:
                direction = "Left"
        else:
            direction = "Stationary"

        # Estimate approach/moving away based on area change
        areas = [p[2] for p in positions[-5:]]  # Last 5 positions
        if len(areas) > 1:
            area_trend = np.polyfit(range(len(areas)), areas, 1)[0]  # Linear trend
            if area_trend > 10:  # Area increasing
                motion = "Approaching"
            elif area_trend < -10:  # Area decreasing
                motion = "Moving Away"
            else:
                motion = "Sideways"
        else:
            motion = "Unknown"

        # Calculate certainty score (0-100)
        # Factors: aspect ratio closeness to 1, size, tracking duration, motion smoothness
        aspect_score = max(0, 100 - abs(caspect - ASPECT_RATIO_IDEAL) * 100)  # 100 if ratio=1, decreases as deviates
        size_score = min(100, carea / SIZE_MULTIPLIER)  # More points for larger objects, cap at 100
        duration_score = min(100, duration * DURATION_MULTIPLIER)  # 100 points for 0.2s, increases with time
        
        # Motion smoothness: lower variance in position
        if len(positions) > 2:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            var_x = np.var(xs)
            var_y = np.var(ys)
            smoothness = max(0, 100 - (var_x + var_y) * SMOOTHNESS_MULTIPLIER)  # Lower variance = higher score
        else:
            smoothness = 50  # Default for short tracks
        
        certainty = int((aspect_score + size_score + duration_score + smoothness) / 4)
        certainty = max(0, min(100, certainty))  # Clamp to 0-100

        # Draw bounding box (approximate size, since we have centroid)
        # For simplicity, use fixed size or estimate from area
        size = int(np.sqrt(carea) / 2)
        cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 255, 0), 2)
        cv2.putText(frame, f'Drone {track["id"]}', (cx - size, cy - size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'{direction}, {motion} ({certainty}%)', (cx - size, cy + size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Display FPS on main frame
    cv2.putText(frame, f'FPS: {fps_value:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Drone Detection', frame)
    
    # Create a debug/info frame showing processed data
    debug_frame = fgmask.copy()
    debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)  # Convert to BGR for display
    
    # Add text information to debug frame
    cv2.putText(debug_frame, f'FPS: {fps_value:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(debug_frame, f'Camera Motion: dx={camera_dx:.1f}, dy={camera_dy:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(debug_frame, f'Active Tracks: {len(tracks)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(debug_frame, f'Detections: {len(detections)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw detection contours on debug frame
    for det in detections:
        cx, cy, area, _, _ = det
        size = int(np.sqrt(area) / 2)
        cv2.circle(debug_frame, (int(cx), int(cy)), size, (0, 255, 255), 2)
    
    # Display debug frame
    cv2.imshow('Motion Detection (Debug)', debug_frame)

    # Check if either window is closed
    if cv2.getWindowProperty('Drone Detection', cv2.WND_PROP_VISIBLE) <= 0 or cv2.getWindowProperty('Motion Detection (Debug)', cv2.WND_PROP_VISIBLE) <= 0:
        break

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()