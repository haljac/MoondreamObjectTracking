import argparse
import cv2
import moondream as md
from async_tracking import AsyncTracker
import rerun as rr
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Object tracking demo with Rerun visualization")
    parser.add_argument("--headless", action="store_true",
                      help="Run without any visualization (default)")
    args = parser.parse_args()

    # Initialize Rerun for visualization
    if not args.headless:
        rr.init("Object Tracking", spawn=True)
        print("Initialized Rerun visualization")

    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        # Initialize Moondream model
        try:
            model = md.vl(endpoint="http://localhost:2020/v1")
        except Exception as e:
            print(f"Error initializing Moondream model: {e}")
            return

        # Get tracking target
        prompt = input("What object would you like to track? ")

        # Create and start tracker
        tracker = AsyncTracker(model, prompt)
        tracker.start()

        print("\nPress Ctrl+C to stop tracking")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process the frame through the tracker
            tracker.process_frame(frame)

            # Get current tracking state and visualize with Rerun
            if not args.headless:
                state = tracker.get_state()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Log the base frame
                rr.log("camera/frame", rr.Image(frame_rgb))
                
                if state['bbox'] is not None:
                    center = state['kalman']
                    bbox = state['bbox']
                    if center and bbox:
                        # Log tracking information
                        rr.log("camera/tracking/center", 
                              rr.Points2D(positions=np.array([center]), 
                                        colors=np.array([[255, 0, 0]])))
                        
                        # Log bounding box as points
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
                        
                        # Print tracking info to console
                        print(f"\rTracking at ({center[0]:.1f}, {center[1]:.1f})", 
                              end="", flush=True)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        # Clean up
        print("\nCleaning up...")
        if tracker:
            tracker.stop()
        if cap:
            cap.release()
        if not args.headless:
            rr.disconnect()

if __name__ == '__main__':
    main()
