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


class TrackingPublisher:
    def __init__(self, prompt):
        # Initialize Zenoh session
        self.session = zenoh.open(zenoh.Config())
        # Publisher for object position messages
        self.pos_pub = self.session.declare_publisher("tracking/position")
        # Initialize a lock and storage for the latest camera frame
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        # Subscribe to the camera feed
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
        # Create AsyncTracker with 2 second detection interval
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

    def run(self):
        """Main loop: processes incoming frames, updates tracking, and publishes position."""
        try:
            print("Running tracking publisher.")
            # Start the AsyncTracker's detection thread
            self.tracker.start()
            prev_loop_time = time.time()
            
            while True:
                current_time = time.time()
                dt = current_time - prev_loop_time
                prev_loop_time = current_time
                loop_rate = 1.0 / dt if dt > 0 else 0.0

                frame = None
                with self.frame_lock:
                    if self.latest_frame is not None:
                        frame = self.latest_frame.copy()
                        self.latest_frame = None
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Process frame; AsyncTracker's detection thread updates state concurrently
                self.tracker.process_frame(frame)
                state = self.tracker.get_state()

                if state['bbox'] is not None and state['kalman'] is not None:
                    # Get frame dimensions for normalized coordinates
                    frame_height, frame_width = frame.shape[:2]
                    
                    # Extract position information
                    center_x, center_y = state['kalman']
                    x_min, y_min, x_max, y_max = state['bbox']
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Create position message with normalized coordinates
                    pos_msg = {
                        'x': float(center_x / frame_width),
                        'y': float(center_y / frame_height),
                        'width': float(width / frame_width),
                        'height': float(height / frame_height),
                        'timestamp': time.time()
                    }
                    
                    print(f"Loop rate: {loop_rate:.2f} Hz")
                    print(f"Publishing position: {pos_msg}")
                    self.pos_pub.put(json.dumps(pos_msg))
                    
                    # Draw bounding box on frame for visualization
                    new_bbox = (int(center_x - width/2), int(center_y - height/2),
                              int(center_x + width/2), int(center_y + height/2))
                    frame = draw_bbox(frame, new_bbox)
                else:
                    print("No valid tracking detected")

                # Visualize the frame with Rerun
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log("camera/frame", rr.Image(frame_rgb))

                time.sleep(1.0 / 30.0)  # Cap at 30 FPS
                
        except KeyboardInterrupt:
            print("Shutting down tracking publisher.")
        finally:
            self.session.close()
            rr.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Zenoh publisher for object tracking using Moondream"
    )
    parser.add_argument("--prompt", type=str, required=True,
                      help="Target object prompt for tracking")
    args = parser.parse_args()
    
    tracker = TrackingPublisher(args.prompt)
    tracker.run()


if __name__ == "__main__":
    main()
