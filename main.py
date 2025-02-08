import cv2
import numpy as np
import moondream as md
from PIL import Image
import threading
import time
import collections

# ----- Simple 1D Kalman Filter for one coordinate -----
class KalmanFilter:
    def __init__(self, x=0.0, p=1.0, Q=0.1, R=2.0):
        self.x = x    # state estimate
        self.p = p    # estimation covariance
        self.Q = Q    # process noise covariance
        self.R = R    # measurement noise covariance

    def predict(self, u):
        # Prediction step: add optical flow displacement u
        self.x += u
        self.p += self.Q

    def update(self, z):
        # Correction step: update with measurement z
        K = self.p / (self.p + self.R)
        self.x += K * (z - self.x)
        self.p = (1 - K) * self.p

# ----- 2D Kalman Filter using two 1D filters for center (x,y) -----
class KalmanFilter2D:
    def __init__(self, center=(0, 0)):
        self.kf_x = KalmanFilter(x=center[0])
        self.kf_y = KalmanFilter(x=center[1])
    
    def predict(self, u):
        # u is a tuple (dx, dy)
        self.kf_x.predict(u[0])
        self.kf_y.predict(u[1])
    
    def update(self, z):
        # z is a tuple (z_x, z_y)
        self.kf_x.update(z[0])
        self.kf_y.update(z[1])
    
    def get_state(self):
        # Return the current estimated center position
        return (self.kf_x.x, self.kf_y.x)  # note: kf_y.x holds the y state

# ----- Utility Functions -----
def cv2_to_pil(cv2_image):
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    # bbox is (x_min, y_min, x_max, y_max)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return frame

# ----- Global Shared State -----
state = {
    'kalman': None,    # our 2D Kalman filter for the center of the box
    'bbox': None,      # current bounding box (x_min, y_min, x_max, y_max)
    'features': None,  # optical flow feature points (numpy array)
    'lock': threading.Lock(),
    'frame_buffer': collections.deque(maxlen=200),  # store recent frames (you can pick a size)
    'pending_detection': None,  # will keep (frame_index, result) once detection finishes
    'requested_detection_frame': None  # store the index for which a detection was requested
}

# ----- Detection Thread -----
def detection_loop(model, prompt, detection_interval=2.0):
    """
    Periodically requests detection but doesn't update Kalman right away.
    Instead, it records the result, which the main loop will apply at the correct 'past' frame.
    
    Args:
        model: The moondream model instance
        prompt: String describing what object to detect (e.g. "pliers", "coffee mug")
        detection_interval: Seconds between detection attempts
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
        # Use the provided prompt instead of hardcoded "pliers"
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
                state['pending_detection'] = (
                    last_frame_index,
                    (x_min, y_min, x_max, y_max)
                )

# ----- Main Loop: Optical Flow Tracking and Display -----
def main_loop(model, prompt):
    """
    Main tracking loop that combines fast optical flow tracking with slower but accurate detections.
    
    Args:
        model: The moondream model instance
        prompt: String describing what object to detect (e.g. "pliers", "coffee mug")
    """
    global state

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Create a named window for sliders and output
    cv2.namedWindow("Asynchronous Tracking", cv2.WINDOW_NORMAL)

    # Create trackbars for maxCorners, qualityLevel (scaled by 100), and minDistance
    # We'll set some typical ranges. Feel free to adjust them:
    cv2.createTrackbar("maxCorners", "Asynchronous Tracking", 27, 300, lambda v: None)
    cv2.createTrackbar("qualityLevel x100", "Asynchronous Tracking", 10, 100, lambda v: None)
    cv2.createTrackbar("minDistance", "Asynchronous Tracking", 10, 50, lambda v: None)

    frame_index = 0  # keep an integer frame counter
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame.")
        return

    # Do an initial detection to bootstrap the state
    pil_image = cv2_to_pil(frame)
    encoded = model.encode_image(pil_image)
    # Use the provided prompt
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
        features = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=20,
                                           qualityLevel=0.1, minDistance=10)
        state['features'] = features

    # Start detection thread with the prompt
    detection_thread = threading.Thread(
        target=detection_loop, 
        args=(model, prompt), 
        daemon=True
    )
    detection_thread.start()

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Main loop: Failed to grab frame")
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        with state['lock']:
            # Store current frame in the buffer
            state['frame_buffer'].append(
                (frame_index, frame.copy(), curr_gray.copy())
            )
            features = state.get('features', None)

        if features is not None:
            # Use Lucas-Kanade optical flow to track features
            new_features, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
            good_old = features[status.flatten() == 1]
            good_new = new_features[status.flatten() == 1]

            if len(good_old) > 0:
                # Calculate the average displacement between tracked features
                displacement = np.mean(good_new - good_old, axis=0)
                # Remove any singleton dimensions so that displacement has shape (2,)
                displacement = np.squeeze(displacement)
                dx, dy = displacement.astype(int)

                with state['lock']:
                    if state['kalman'] is not None:
                        state['kalman'].predict((dx, dy))
                    if state['bbox'] is not None:
                        # Update bbox position by shifting it
                        x_min, y_min, x_max, y_max = state['bbox']
                        state['bbox'] = (x_min + dx, y_min + dy, x_max + dx, y_max + dy)
                # Update features for the next iteration
                with state['lock']:
                    state['features'] = new_features

            # Optionally, draw the tracked points for visualization:
            for pt in good_new:
                x, y = pt.ravel()
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        #
        # --- If there's a completed detection waiting, "rewind" and reapply it ---
        #
        with state['lock']:
            if state['pending_detection'] is not None:
                det_frame_index, det_bbox = state['pending_detection']
                state['pending_detection'] = None  

                # Rewind
                state['kalman'] = KalmanFilter2D()
                state['bbox'] = None
                state['features'] = None

                # Find the detection frame in the buffer
                detection_found = False
                for i, (f_idx, f_bgr, f_gray) in enumerate(state['frame_buffer']):
                    if f_idx == det_frame_index:
                        x_min, y_min, x_max, y_max = det_bbox
                        cx = (x_min + x_max) // 2
                        cy = (y_min + y_max) // 2
                        # Initialize Kalman & bounding box at detection
                        state['kalman'] = KalmanFilter2D(center=(cx, cy))
                        state['bbox'] = (x_min, y_min, x_max, y_max)
                        
                        # Re-init features at detection frame
                        mask = np.zeros_like(f_gray)
                        mask[y_min:y_max, x_min:x_max] = 255
                        maxCorners = cv2.getTrackbarPos("maxCorners", "Asynchronous Tracking")
                        ql_scaled = cv2.getTrackbarPos("qualityLevel x100", "Asynchronous Tracking")
                        qualityLevel = ql_scaled / 100.0
                        minDistance = cv2.getTrackbarPos("minDistance", "Asynchronous Tracking")
                        features = cv2.goodFeaturesToTrack(
                            f_gray,
                            mask=mask,
                            maxCorners=maxCorners,
                            qualityLevel=qualityLevel,
                            minDistance=minDistance
                        )
                        state['features'] = features

                        detection_found = True
                        detection_index_in_buffer = i
                        break

                # Make sure we actually found the detection frame in the buffer
                if detection_found:
                    # Replay from detection_index_in_buffer to the end
                    for j in range(detection_index_in_buffer + 1, len(state['frame_buffer'])):
                        prev_idx, prev_bgr, prev_gray = state['frame_buffer'][j - 1]
                        curr_idx, curr_bgr, curr_gray = state['frame_buffer'][j]

                        # Re-run optical flow from prev_gray -> curr_gray
                        if state['features'] is not None and len(state['features']) > 0:
                            new_features, status, err = cv2.calcOpticalFlowPyrLK(
                                prev_gray, curr_gray, state['features'], None
                            )
                            good_old = state['features'][status.flatten() == 1]
                            good_new = new_features[status.flatten() == 1]

                            if len(good_old) > 0:
                                displacement = np.mean(good_new - good_old, axis=0)
                                displacement = np.squeeze(displacement)
                                dx, dy = displacement.astype(int)

                                # Predict the Kalman
                                if state['kalman']:
                                    state['kalman'].predict((dx, dy))

                                # Also update bbox
                                if state['bbox'] is not None:
                                    x_min, y_min, x_max, y_max = state['bbox']
                                    state['bbox'] = (x_min + dx, y_min + dy,
                                                     x_max + dx, y_max + dy)

                                # Update features for the next iteration
                                state['features'] = new_features
                            else:
                                # If we lose all features, you may want to attempt reinit
                                # or just break. For now, let's just break out.
                                break

                    print(f"Detection backfilled at frame {det_frame_index}; "
                          f"state is recomputed up to frame {curr_idx}.")

        #
        # --- Continue normal display code with the updated bounding box ---
        #
        with state['lock']:
            if state['kalman'] is not None and state['bbox'] is not None:
                center = state['kalman'].get_state()
                x_min, y_min, x_max, y_max = state['bbox']
                w = x_max - x_min
                h = y_max - y_min
                # bounding box centered at the filtered position:
                new_bbox = (
                    int(center[0] - w/2),
                    int(center[1] - h/2),
                    int(center[0] + w/2),
                    int(center[1] + h/2)
                )
                draw_bbox(frame, new_bbox)

        cv2.imshow("Asynchronous Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = curr_gray
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

# ----- Entry Point -----
if __name__ == '__main__':
    with open('api_key.txt', 'r') as f:
        api_key = f.read().strip()
    model = md.vl(api_key=api_key)
    
    tracking_prompt = input("What object would you like to track? ")
    main_loop(model, tracking_prompt)
