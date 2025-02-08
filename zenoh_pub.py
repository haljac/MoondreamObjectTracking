#!/usr/bin/env python3
"""
tracking_pub_zenoh.py

This script performs asynchronous tracking (using optical flow, periodic detections,
and a 2D Kalman filter) on webcam frames and computes a steering command. The command is published
over Zenoh (to the 'robot/cmd' topic) as a JSON message. Additionally, it logs the tracking
results to Rerun for visualization.

Usage:
    python tracking_pub_zenoh.py [--display-ui]
"""

import argparse
import cv2
import numpy as np
import moondream as md
from PIL import Image
import threading
import time
import collections
import json
import zenoh
import rerun as rr

# ----- Kalman Filter Classes -----
class KalmanFilter:
    def __init__(self, x=0.0, p=1.0, Q=0.1, R=2.0):
        self.x = x    # state estimate
        self.p = p    # estimation covariance
        self.Q = Q    # process noise covariance
        self.R = R    # measurement noise covariance

    def predict(self, u):
        # Add displacement u to state
        self.x += u
        self.p += self.Q

    def update(self, z):
        # Update state estimate with measurement z
        K = self.p / (self.p + self.R)
        self.x += K * (z - self.x)
        self.p = (1 - K) * self.p

class KalmanFilter2D:
    def __init__(self, center=(0, 0)):
        self.kf_x = KalmanFilter(x=center[0])
        self.kf_y = KalmanFilter(x=center[1])
    
    def predict(self, u):
        self.kf_x.predict(u[0])
        self.kf_y.predict(u[1])
    
    def update(self, z):
        self.kf_x.update(z[0])
        self.kf_y.update(z[1])
    
    def get_state(self):
        return (self.kf_x.x, self.kf_y.x)

# ----- Utility Functions -----
def cv2_to_pil(cv2_image):
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    # bbox format: (x_min, y_min, x_max, y_max)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return frame

# ----- Global Shared State -----
state = {
    'kalman': None,         # Kalman filter object for tracking center
    'bbox': None,           # Current bounding box (x_min, y_min, x_max, y_max)
    'features': None,       # Points to track (optical flow features)
    'lock': threading.Lock(),
    'frame_buffer': collections.deque(maxlen=200),
    'pending_detection': None,
    'requested_detection_frame': None
}

# ----- Detection Thread -----
def detection_loop(model, prompt, detection_interval=2.0):
    """
    Periodically performs object detection on the latest frame.
    The result (bounding box) is stored in state['pending_detection'] for later backfill.
    """
    global state
    while True:
        time.sleep(detection_interval)
        with state['lock']:
            if not state['frame_buffer']:
                continue
            last_frame_index, last_frame_bgr, last_frame_gray = state['frame_buffer'][-1]
            state['requested_detection_frame'] = last_frame_index

        pil_image = cv2_to_pil(last_frame_bgr)
        encoded = model.encode_image(pil_image)
        result = model.detect(encoded, prompt)
        objects = result.get("objects", [])
        if objects:
            obj = objects[0]
            height, width = last_frame_bgr.shape[:2]
            x_min = int(obj['x_min'] * width)
            y_min = int(obj['y_min'] * height)
            x_max = int(obj['x_max'] * width)
            y_max = int(obj['y_max'] * height)
            with state['lock']:
                state['pending_detection'] = (last_frame_index, (x_min, y_min, x_max, y_max))

# ----- Main Tracking and Publishing Loop -----
def main():
    parser = argparse.ArgumentParser(
        description="Track an object from the webcam and publish steering commands over Zenoh."
    )
    parser.add_argument("--display-ui", action="store_true",
                        help="Display the OpenCV window with tracking visualization.")
    args = parser.parse_args()

    # Initialize Zenoh session and publisher
    session = zenoh.open(zenoh.Config())
    publisher = session.declare_publisher('robot/cmd')

    # Initialize Rerun visualization
    rr.init("Tracking", spawn=True)

    # Read API key and create Moondream model
    with open('api_key.txt', 'r') as f:
        api_key = f.read().strip()
    model = md.vl(api_key=api_key)

    prompt = input("What object would you like to track? ")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Grab an initial frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame.")
        return

    # Initial detection to bootstrap tracking
    pil_image = cv2_to_pil(frame)
    encoded = model.encode_image(pil_image)
    result = model.detect(encoded, prompt)
    objects = result.get("objects", [])
    if not objects:
        print(f"No {prompt} detected initially.")
        return

    obj = objects[0]
    height, width = frame.shape[:2]
    x_min = int(obj['x_min'] * width)
    y_min = int(obj['y_min'] * height)
    x_max = int(obj['x_max'] * width)
    y_max = int(obj['y_max'] * height)
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    with state['lock']:
        state['bbox'] = (x_min, y_min, x_max, y_max)
        state['kalman'] = KalmanFilter2D(center=(center_x, center_y))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        mask[y_min:y_max, x_min:x_max] = 255
        features = cv2.goodFeaturesToTrack(
            gray, 
            mask=mask, 
            maxCorners=20,
            qualityLevel=0.1,
            minDistance=10
        )
        state['features'] = features

    # Start the detection thread (to update the target periodically)
    detection_thread = threading.Thread(
        target=detection_loop, 
        args=(model, prompt),
        daemon=True
    )
    detection_thread.start()

    frame_index = 0
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define control parameters for steering command
    linear_speed = 0.5        # Constant forward speed
    steering_gain = 0.005     # Gain for converting horizontal error to angular velocity

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Save the current frame into the buffer
        with state['lock']:
            state['frame_buffer'].append((frame_index, frame.copy(), curr_gray.copy()))
            features = state.get('features', None)

        # Optical flow tracking using Lucas-Kanade
        if features is not None:
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
            good_old = features[status.flatten() == 1]
            good_new = new_features[status.flatten() == 1]

            if len(good_old) > 0:
                displacement = np.mean(good_new - good_old, axis=0)
                displacement = np.squeeze(displacement)
                dx, dy = displacement.astype(int)

                with state['lock']:
                    if state['kalman'] is not None:
                        state['kalman'].predict((dx, dy))
                    if state['bbox'] is not None:
                        x_min, y_min, x_max, y_max = state['bbox']
                        state['bbox'] = (x_min + dx, y_min + dy, x_max + dx, y_max + dy)
                with state['lock']:
                    state['features'] = new_features

            if args.display_ui:
                for pt in good_new:
                    x, y = pt.ravel()
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Check for any pending detection update
        with state['lock']:
            if state['pending_detection'] is not None:
                det_frame_index, det_bbox = state['pending_detection']
                state['pending_detection'] = None

                # Reset state before reapplying detection
                state['kalman'] = KalmanFilter2D()
                state['bbox'] = None
                state['features'] = None

                detection_found = False
                # Find the detection frame in the buffer and reinitialize state
                for i, (f_idx, f_bgr, f_gray) in enumerate(state['frame_buffer']):
                    if f_idx == det_frame_index:
                        x_min, y_min, x_max, y_max = det_bbox
                        cx = (x_min + x_max) // 2
                        cy = (y_min + y_max) // 2
                        state['kalman'] = KalmanFilter2D(center=(cx, cy))
                        state['bbox'] = (x_min, y_min, x_max, y_max)
                        mask = np.zeros_like(f_gray)
                        mask[y_min:y_max, x_min:x_max] = 255
                        features = cv2.goodFeaturesToTrack(
                            f_gray,
                            mask=mask,
                            maxCorners=20,
                            qualityLevel=0.1,
                            minDistance=10
                        )
                        state['features'] = features
                        detection_found = True
                        detection_index_in_buffer = i
                        break

                if detection_found:
                    # Backfill tracking state for frames after the detection frame
                    for j in range(detection_index_in_buffer + 1, len(state['frame_buffer'])):
                        prev_idx, prev_bgr, prev_gray_buf = state['frame_buffer'][j - 1]
                        curr_idx, curr_bgr, curr_gray_buf = state['frame_buffer'][j]
                        if state['features'] is not None and len(state['features']) > 0:
                            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                                prev_gray_buf, curr_gray_buf, state['features'], None
                            )
                            good_old = state['features'][status.flatten() == 1]
                            good_new = new_features[status.flatten() == 1]
                            if len(good_old) > 0:
                                displacement = np.mean(good_new - good_old, axis=0)
                                displacement = np.squeeze(displacement)
                                dx, dy = displacement.astype(int)
                                if state['kalman']:
                                    state['kalman'].predict((dx, dy))
                                if state['bbox'] is not None:
                                    x_min, y_min, x_max, y_max = state['bbox']
                                    state['bbox'] = (x_min + dx, y_min + dy, x_max + dx, y_max + dy)
                                state['features'] = new_features
                            else:
                                break
                    print(f"Detection backfilled at frame {det_frame_index}; state recomputed.")

        # If tracking state is available, compute the steering command and draw the bounding box.
        with state['lock']:
            if state['kalman'] is not None and state['bbox'] is not None:
                center = state['kalman'].get_state()
                x_min, y_min, x_max, y_max = state['bbox']
                w = x_max - x_min
                h = y_max - y_min
                new_bbox = (
                    int(center[0] - w/2),
                    int(center[1] - h/2),
                    int(center[0] + w/2),
                    int(center[1] + h/2)
                )
                if args.display_ui:
                    draw_bbox(frame, new_bbox)
                # Compute horizontal error (object center vs. frame center)
                frame_center_x = frame.shape[1] // 2
                error_x = center[0] - frame_center_x
                # Compute angular velocity (steering) command based on error
                angular_velocity = -steering_gain * error_x
                cmd = {
                    'x': linear_speed,
                    'theta': angular_velocity
                }
                publisher.put(json.dumps(cmd))
                print(f"Published command: {cmd}")
            else:
                # If no valid tracking state, send a stop command.
                cmd = {'x': 0.0, 'theta': 0.0}
                publisher.put(json.dumps(cmd))
                print("No tracking state available, published stop command.")

        # Log the current frame with overlay to Rerun (convert BGR to RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log("camera", rr.Image(frame_rgb))

        if args.display_ui:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        prev_gray = curr_gray
        frame_index += 1

    cap.release()
    if args.display_ui:
        cv2.destroyAllWindows()
    session.close()
    rr.disconnect()

if __name__ == '__main__':
    main()
