#!/usr/bin/env python3
import cv2
import zenoh
import json
import time
import rerun as rr
import numpy as np
import argparse
import threading
import moondream as md
from async_tracking import AsyncTracker, draw_bbox


class ZenohPub:
    def __init__(self, prompt):
        # Initialize Zenoh session
        self.session = zenoh.open(zenoh.Config())
        # Publisher for twist (command) messages on topic "robot/cmd"
        self.cmd_pub = self.session.declare_publisher("robot/cmd")
        # Initialize a lock and storage for the latest camera frame.
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        # Subscribe to the camera feed published from the robot (e.g. from webcam_publisher)
        self.frame_sub = self.session.declare_subscriber("robot/camera/frame", self.on_frame)
        
        # Initialize Rerun visualization
        rr.init("Object Tracking", spawn=True)
        
        # Initialize Moondream model using an API key stored in api_key.txt
        try:
            with open('api_key.txt', 'r') as f:
                api_key = f.read().strip()
            self.model = md.vl(api_key=api_key)
        except Exception as e:
            print(f"Failed to initialize moondream model: {e}")
            exit(1)
            
        self.prompt = prompt
        # Create AsyncTracker; note we do NOT call tracker.start() so no detection thread is spawned –
        # instead, we drive the detection update manually via tick() on each frame.
        self.tracker = AsyncTracker(self.model, self.prompt, detection_interval=2.0, display_ui=False)

    def on_frame(self, sample):
        """Callback for Zenoh subscriber: decodes the frame and stores it."""
        try:
            np_arr = np.frombuffer(sample.payload.to_bytes(), np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            print(f"Error processing camera frame: {e}")

    def compute_twist(self, state, frame_width, loop_rate):
        """
        Compute a twist command based on tracking state.
        Here we compute the error between the target's x position (from the Kalman filter)
        and the center of the frame. We then compute angular velocity to steer toward the target.
        A constant linear velocity is used if the loop rate is above 10 Hz, else zero.
        """
        if state['kalman'] is None:
            # No valid tracking data – publish zero commands
            return {'x': 0.0, 'theta': 0.0}
        target_x = state['kalman'][0]
        center_x = frame_width / 2.0
        error_x = target_x - center_x
        # Gain factor for angular velocity (adjust as needed)
        gain = 0.005
        angular_z = -gain * error_x  # Negative sign to counter the error
        # Set linear speed based on loop rate
        if loop_rate > 10:
            linear_x = 0.3
        else:
            linear_x = 0.0
        return {'x': float(linear_x), 'theta': float(angular_z)}

    def run(self):
        """Main loop: processes incoming frames, updates tracking using the tick method, computes twist, and publishes it."""
        try:
            print("Running zenoh publisher for tracking and twist commands.")
            prev_loop_time = time.time()
            while True:
                # Compute loop rate
                current_time = time.time()
                dt = current_time - prev_loop_time
                prev_loop_time = current_time
                loop_rate = 1.0 / dt if dt > 0 else 0.0

                frame = None
                with self.frame_lock:
                    if self.latest_frame is not None:
                        # Copy latest frame and then reset it; this prevents processing the same frame twice.
                        frame = self.latest_frame.copy()
                        self.latest_frame = None
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Manually run detection update before processing the frame
                self.tracker.tick()
                self.tracker.process_frame(frame)
                state = self.tracker.get_state()

                # If tracking data is valid, compute twist command based on the target's position
                if state['bbox'] is not None and state['kalman'] is not None:
                    frame_width = frame.shape[1]
                    twist_cmd = self.compute_twist(state, frame_width, loop_rate)
                    print(f"Loop rate: {loop_rate:.2f} Hz")
                    print(f"Tracking: center={state['kalman']}, bbox={state['bbox']}")
                    print(f"Twist command: {twist_cmd}")
                    self.cmd_pub.put(json.dumps(twist_cmd))
                    
                    # Draw bounding box on frame for visualization
                    x_min, y_min, x_max, y_max = state['bbox']
                    w = x_max - x_min
                    h = y_max - y_min
                    center = state['kalman']
                    new_bbox = (int(center[0] - w/2), int(center[1] - h/2), int(center[0] + w/2), int(center[1] + h/2))
                    frame = draw_bbox(frame, new_bbox)
                else:
                    # If there's no valid tracking state, send a zero twist command
                    twist_cmd = {'x': 0.0, 'theta': 0.0}
                    print("No valid tracking, publishing zero twist command")
                    self.cmd_pub.put(json.dumps(twist_cmd))

                # Visualize the frame using Rerun (convert BGR to RGB for visualization)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log("camera/frame", rr.Image(frame_rgb))

                time.sleep(1.0 / 30.0)  # Run at ~30 Hz
        except KeyboardInterrupt:
            print("Shutting down zenoh publisher.")
        finally:
            self.session.close()
            rr.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Zenoh publisher for object tracking and twist command generation using Moondream and AsyncTracker"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Target object prompt for tracking")
    args = parser.parse_args()
    tracker = ZenohPub(args.prompt)
    tracker.run()


if __name__ == "__main__":
    main()
