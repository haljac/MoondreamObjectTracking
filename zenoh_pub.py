#!/usr/bin/env python3
"""
tracking_pub_zenoh.py

This script performs asynchronous tracking on frames received over Zenoh and computes
a steering command to aim towards the tracked object. The command is published over 
Zenoh (to the 'robot/cmd' topic) as a JSON message. Visualization is done through Rerun.

Usage:
    python tracking_pub_zenoh.py [--headless]
"""

import argparse
import numpy as np
import moondream as md
from PIL import Image
import json
import zenoh
import rerun as rr
import cv2
import time
from async_tracking import AsyncTracker
import collections
import threading

FRAME_TIMEOUT = 0.2

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class TrackingController:
    def __init__(self, model, prompt, headless=False):
        # Initialize Rerun for visualization
        if not headless:
            rr.init("Object Tracking", spawn=True)
            print("Initialized Rerun visualization")
        self.headless = headless
        
        # Initialize Zenoh session and publisher/subscriber
        self.session = zenoh.open(zenoh.Config())
        self.publisher = self.session.declare_publisher('robot/cmd')
        self.subscriber = self.session.declare_subscriber(
            'robot/camera/frame',
            self._on_camera_frame
        )
        
        # Initialize tracker
        self.tracker = AsyncTracker(model, prompt)
        
        # Frame processing queue and thread
        self.frame_queue = collections.deque(maxlen=1)  # Only keep latest frame
        self.frame_lock = threading.Lock()
        self.running = False
        self.processing_thread = None
        
        # Control parameters
        self.steering_gain = 0.0035  # Gain for converting horizontal error to angular velocity
        self.last_valid_center = None
        self.smoothing_factor = 0.3  # For exponential smoothing of commands
        self.last_command = 0.0
        
    def _on_camera_frame(self, sample):
        """Handle incoming camera frames from Zenoh"""
        try:
            # Convert Zenoh bytes to numpy array
            np_arr = np.frombuffer(sample.payload.to_bytes(), np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Failed to decode frame")
                return
            
            # Store the latest frame
            with self.frame_lock:
                self.frame_queue.append(frame)
                
        except Exception as e:
            print(f"Error receiving camera frame: {e}")
            
    def _processing_loop(self):
        """Continuous processing loop for frames"""
        # Initialize visualization timer
        self._last_viz_time = time.time()
        viz_interval = 0.033  # update visualization at max ~30 FPS
        while self.running:
            # Get latest frame if available
            with self.frame_lock:
                if not self.frame_queue:
                    time.sleep(FRAME_TIMEOUT)  # Sleep longer if no frame is available
                    continue
                frame = self.frame_queue.pop()
            
            try:
                # Process frame through tracker
                self.tracker.process_frame(frame)
                
                # Get current tracking state
                state = self.tracker.get_state()
                
                # Always update control
                self._update_control(frame.shape[1], state)
                
                # Update visualization only if sufficient time has passed
                current_time = time.time()
                if current_time - self._last_viz_time >= viz_interval and not self.headless:
                    self._update_visualization(frame, state)
                    self._last_viz_time = current_time
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
            
            # Brief sleep to yield CPU
            time.sleep(0.005)
            
    def start(self):
        """Start the tracking system"""
        self.running = True
        self.tracker.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def _update_visualization(self, frame, state):
        """Update visualization in Rerun"""
        if self.headless:
            return
            
        # Convert BGR to RGB for Rerun
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Log the base frame
        rr.log("camera/frame", rr.Image(frame_rgb))
        
        if state['bbox'] is not None:
            center = state['kalman']
            bbox = state['bbox']
            if center and bbox:
                # Log tracking center point
                rr.log("camera/tracking/center", 
                      rr.Points2D(positions=np.array([center]), 
                                colors=np.array([[255, 0, 0]])))
                
                # Log bounding box
                bbox_points = np.array([
                    [bbox[0], bbox[1]],  # top-left
                    [bbox[2], bbox[1]],  # top-right
                    [bbox[2], bbox[3]],  # bottom-right
                    [bbox[0], bbox[3]],  # bottom-left
                    [bbox[0], bbox[1]]   # close the box
                ])
                rr.log("camera/tracking/bbox", 
                      rr.LineStrips2D(bbox_points, 
                                    colors=np.array([[0, 255, 0]])))
                
                # Log target line
                height = frame.shape[0]
                rr.log("camera/tracking/target_line",
                      rr.LineStrips2D(np.array([[center[0], 0], [center[0], height]]),
                                    colors=np.array([[255, 0, 0]])))
                
                # Log center line
                center_x = frame.shape[1] // 2
                rr.log("camera/tracking/center_line",
                      rr.LineStrips2D(np.array([[center_x, 0], [center_x, height]]),
                                    colors=np.array([[255, 255, 0]])))
            
    def _update_control(self, frame_width, state):
        """Update and publish control command with smoothing"""
        if state['bbox'] is not None and state['kalman'] is not None:
            center = state['kalman']
            self.last_valid_center = center
            
            # Compute horizontal error (object center vs. frame center)
            frame_center_x = frame_width // 2
            error_x = center[0] - frame_center_x
            
            # Compute angular velocity with smoothing
            raw_angular_velocity = float(-self.steering_gain * error_x)
            max_angular_velocity = 0.4
            raw_angular_velocity = max(min(raw_angular_velocity, max_angular_velocity), -max_angular_velocity)
            
            cmd = {
                'x': 0.0,  # No forward motion
                'theta': raw_angular_velocity
            }
            
            # Use custom encoder to handle numpy types
            self.publisher.put(json.dumps(cmd, cls=NumpyJSONEncoder))
            print(f"\rTracking at ({center[0]:.1f}, {center[1]:.1f}), "
                  f"command: {cmd}", end="", flush=True)
        else:
            # If no valid tracking state, gradually reduce command
            self.last_command *= 0.9  # Decay the last command
            cmd = {'x': 0.0, 'theta': self.last_command}
            self.publisher.put(json.dumps(cmd, cls=NumpyJSONEncoder))
            if abs(self.last_command) < 0.01:  # If command is very small, just stop
                self.last_command = 0.0
                cmd = {'x': 0.0, 'theta': 0.0}
                self.publisher.put(json.dumps(cmd))
            
    def stop(self):
        """Clean up resources"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        if self.tracker:
            self.tracker.stop()
        self.session.close()
        if not self.headless:
            rr.disconnect()

def main():
    parser = argparse.ArgumentParser(
        description="Track an object from Zenoh camera feed and publish steering commands."
    )
    parser.add_argument("--headless", action="store_true",
                      help="Run without visualization")
    args = parser.parse_args()

    try:
        # Initialize Moondream model
        with open('api_key.txt', 'r') as f:
            api_key = f.read().strip()
        model = md.vl(api_key=api_key)

        # Get tracking target
        prompt = input("What object would you like to track? ")

        # Create and start tracking controller
        controller = TrackingController(model, prompt, headless=args.headless)
        controller.start()  # Start the processing thread
        print("\nPress Ctrl+C to stop tracking")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        if 'controller' in locals():
            controller.stop()

if __name__ == '__main__':
    main()
