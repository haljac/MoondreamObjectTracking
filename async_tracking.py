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
    # bbox is (x_min, y_min, x_max, y_max)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return frame


# ----- Global Shared State -----
state = {
    'kalman': None,    
    'bbox': None,      
    'features': None,  
    'lock': threading.Lock(),
    'frame_buffer': collections.deque(maxlen=200),
    'pending_detection': None,
    'requested_detection_frame': None
}


# ----- Detection Thread -----
def detection_loop(model, prompt, detection_interval=2.0):
    """
    Periodically requests detection but doesn't update Kalman right away.
    Instead, it records the result, which the main loop will apply at the correct 'past' frame.
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
                state['pending_detection'] = (
                    last_frame_index,
                    (x_min, y_min, x_max, y_max)
                )


def run_async_tracking(model, prompt, display_ui=True):
    """
    Main tracking logic that combines fast optical flow with slower but accurate detections.
    Optionally display a UI with trackbars and the bounding box.

    Args:
        model: moondream model instance
        prompt: object prompt / label (e.g. "pliers", "coffee mug")
        display_ui: if True, show the OpenCV window, trackbars, etc.
    """
    global state

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    if display_ui:
        cv2.namedWindow("Asynchronous Tracking", cv2.WINDOW_NORMAL)
        # Create trackbars for maxCorners, qualityLevel (scaled by 100), and minDistance
        cv2.createTrackbar("maxCorners", "Asynchronous Tracking", 27, 300, lambda v: None)
        cv2.createTrackbar("qualityLevel x100", "Asynchronous Tracking", 10, 100, lambda v: None)
        cv2.createTrackbar("minDistance", "Asynchronous Tracking", 10, 50, lambda v: None)

    frame_index = 0
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame.")
        return

    # Initial detection to bootstrap state
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

    # Start detection thread
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
            state['frame_buffer'].append((frame_index, frame.copy(), curr_gray.copy()))
            features = state.get('features', None)

        # Lucas-Kanade Optical Flow
        if features is not None:
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
            good_old = features[status.flatten() == 1]
            good_new = new_features[status.flatten() == 1]

            if len(good_old) > 0:
                displacement = np.mean(good_new - good_old, axis=0)
                displacement = np.squeeze(displacement)
                dx, dy = displacement
                with state['lock']:
                    if state['kalman'] is not None:
                        state['kalman'].predict((dx, dy))
                    if state['bbox'] is not None:
                        x_min, y_min, x_max, y_max = state['bbox']
                        state['bbox'] = (int(x_min + dx), int(y_min + dy), int(x_max + dx), int(y_max + dy))

                with state['lock']:
                    state['features'] = new_features

            if display_ui:
                # Draw tracked points
                for pt in good_new:
                    x, y = pt.ravel()
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        # If there's a completed detection waiting, "rewind" and reapply it
        with state['lock']:
            if state['pending_detection'] is not None:
                det_frame_index, det_bbox = state['pending_detection']
                state['pending_detection'] = None

                # Rewind
                state['kalman'] = KalmanFilter2D()
                state['bbox'] = None
                state['features'] = None

                detection_found = False
                # Search for detection frame in buffer
                for i, (f_idx, f_bgr, f_gray) in enumerate(state['frame_buffer']):
                    if f_idx == det_frame_index:
                        x_min, y_min, x_max, y_max = det_bbox
                        cx = (x_min + x_max) // 2
                        cy = (y_min + y_max) // 2
                        state['kalman'] = KalmanFilter2D(center=(cx, cy))
                        state['bbox'] = (x_min, y_min, x_max, y_max)

                        mask = np.zeros_like(f_gray)
                        mask[y_min:y_max, x_min:x_max] = 255
                        if display_ui:
                            maxCorners = cv2.getTrackbarPos("maxCorners", "Asynchronous Tracking")
                            ql_scaled = cv2.getTrackbarPos("qualityLevel x100", "Asynchronous Tracking")
                            minDistance = cv2.getTrackbarPos("minDistance", "Asynchronous Tracking")
                            qualityLevel = ql_scaled / 100.0
                        else:
                            # Fallback defaults if no UI is displayed
                            maxCorners = 20
                            qualityLevel = 0.1
                            minDistance = 10

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

                if detection_found:
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
                                dx, dy = displacement

                                if state['kalman']:
                                    state['kalman'].predict((dx, dy))

                                if state['bbox'] is not None:
                                    x_min, y_min, x_max, y_max = state['bbox']
                                    state['bbox'] = (int(x_min + dx), int(y_min + dy), int(x_max + dx), int(y_max + dy))

                                state['features'] = new_features
                            else:
                                break

                    print(f"Detection backfilled at frame {det_frame_index}; "
                          f"state is recomputed up to frame {curr_idx}.")

        # Display bounding box if we have one
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
                if display_ui:
                    draw_bbox(frame, new_bbox)

        if display_ui:
            cv2.imshow("Asynchronous Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        prev_gray = curr_gray
        frame_index += 1

    cap.release()
    if display_ui:
        cv2.destroyAllWindows()


class AsyncTracker:
    def __init__(self, model, prompt, detection_interval=2.0, display_ui=False):
        self.model = model
        self.prompt = prompt
        self.detection_interval = detection_interval
        self.display_ui = display_ui
        
        # Tracking state
        self.kalman = None
        self.bbox = None
        self.features = None
        self.frame_buffer = collections.deque(maxlen=200)
        self.pending_detection = None
        self.requested_detection_frame = None
        self.frame_index = 0
        
        # Threading
        self.lock = threading.Lock()
        self.running = False
        self.detection_thread = None
        self.last_detection_tick = time.time()  # Initialize last detection tick for tick-based updates
        
    def _cv2_to_pil(self, cv2_image):
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
        
    def _detection_loop(self):
        """Detection loop for thread-based operation. Calls tick() repeatedly."""
        while self.running:
            self.tick()
            time.sleep(0.01)
        
    def tick(self):
        """Perform a single detection update if the detection interval has elapsed.
        This function can be called externally to integrate the detection step with other event loops.
        """
        current_time = time.time()
        if current_time - self.last_detection_tick >= self.detection_interval:
            self.last_detection_tick = current_time
            with self.lock:
                if not self.frame_buffer:
                    print(f"[{{time.time()}}] [AsyncTracker] No frames available for detection in tick")
                    return
                last_frame_index, last_frame_bgr, last_frame_gray = self.frame_buffer[-1]
                self.requested_detection_frame = last_frame_index
            pil_image = self._cv2_to_pil(last_frame_bgr)
            encoded = self.model.encode_image(pil_image)
            result = self.model.detect(encoded, self.prompt)
            objects = result.get("objects", [])
            if objects:
                obj = objects[0]
                height, width = last_frame_bgr.shape[:2]
                x_min = int(obj['x_min'] * width)
                y_min = int(obj['y_min'] * height)
                x_max = int(obj['x_max'] * width)
                y_max = int(obj['y_max'] * height)
                with self.lock:
                    self.pending_detection = (last_frame_index, (x_min, y_min, x_max, y_max))
        
    def start(self):
        """Start the tracking system"""
        self.running = True
        
        # Start detection thread
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self.detection_thread.start()
        
    def stop(self):
        """Stop the tracking system"""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
            
    def get_state(self):
        """Get current tracking state"""
        with self.lock:
            return {
                'kalman': self.kalman.get_state() if self.kalman else None,
                'bbox': self.bbox,
                'frame_index': self.frame_index
            }
            
    def process_frame(self, frame):
        """Process a new frame and update tracking state"""
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"[{time.time()}] [AsyncTracker] Processing frame index: {self.frame_index}")
        with self.lock:
            print(f"[{time.time()}] [AsyncTracker] Acquired lock for appending frame; buffer length before append: {len(self.frame_buffer)}")
            self.frame_buffer.append((self.frame_index, frame.copy(), curr_gray.copy()))
            print(f"[{time.time()}] [AsyncTracker] Updated frame_buffer length: {len(self.frame_buffer)}")
            features = self.features
            print(f"[{time.time()}] [AsyncTracker] Features status: {'available' if features is not None else 'None'}")

        # Get previous frame's grayscale image
        if len(self.frame_buffer) > 1:
            prev_idx, prev_bgr, prev_gray = self.frame_buffer[-2]
        else:
            # Initialize tracking on first frame
            if not self.kalman:
                pil_image = self._cv2_to_pil(frame)
                encoded = self.model.encode_image(pil_image)
                result = self.model.detect(encoded, self.prompt)
                objects = result.get("objects", [])
                
                if objects:
                    obj = objects[0]
                    height, width = frame.shape[:2]
                    x_min = int(obj['x_min'] * width)
                    y_min = int(obj['y_min'] * height)
                    x_max = int(obj['x_max'] * width)
                    y_max = int(obj['y_max'] * height)
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                    
                    with self.lock:
                        self.bbox = (x_min, y_min, x_max, y_max)
                        self.kalman = KalmanFilter2D(center=(center_x, center_y))
                        mask = np.zeros_like(curr_gray)
                        mask[y_min:y_max, x_min:x_max] = 255
                        self.features = cv2.goodFeaturesToTrack(
                            curr_gray, 
                            mask=mask,
                            maxCorners=20,
                            qualityLevel=0.1,
                            minDistance=10
                        )
            self.frame_index += 1
            return

        # Optical flow tracking
        if features is not None:
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
            good_old = features[status.flatten() == 1]
            good_new = new_features[status.flatten() == 1]

            if len(good_old) > 0:
                displacement = np.mean(good_new - good_old, axis=0)
                displacement = np.squeeze(displacement)
                dx, dy = displacement
                print(f"[{time.time()}] [AsyncTracker] Optical flow displacement: dx={dx}, dy={dy}")
                with self.lock:
                    print(f"[{time.time()}] [AsyncTracker] Acquired lock for updating tracking state")
                    if self.kalman is not None:
                        self.kalman.predict((dx, dy))
                    if self.bbox is not None:
                        x_min, y_min, x_max, y_max = self.bbox
                        self.bbox = (int(x_min + dx), int(y_min + dy), int(x_max + dx), int(y_max + dy))
                    self.features = new_features

            if self.display_ui:
                # Draw tracked points
                for pt in good_new:
                    x, y = pt.ravel()
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Handle any pending detection updates
        with self.lock:
            if self.pending_detection is not None:
                print(f"[{time.time()}] [AsyncTracker] Pending detection found, processing detection update")
                det_frame_index, det_bbox = self.pending_detection
                self.pending_detection = None
                # Reset state
                self.kalman = KalmanFilter2D()
                self.bbox = None
                self.features = None
                detection_found = False
                for i, (f_idx, f_bgr, f_gray) in enumerate(self.frame_buffer):
                    if f_idx == det_frame_index:
                        print(f"[{time.time()}] [AsyncTracker] Found detection frame at index {f_idx}")
                        x_min, y_min, x_max, y_max = det_bbox
                        cx = (x_min + x_max) // 2
                        cy = (y_min + y_max) // 2
                        self.kalman = KalmanFilter2D(center=(cx, cy))
                        self.bbox = det_bbox
                        mask = np.zeros_like(f_gray)
                        mask[y_min:y_max, x_min:x_max] = 255
                        self.features = cv2.goodFeaturesToTrack(
                            f_gray,
                            mask=mask,
                            maxCorners=20,
                            qualityLevel=0.1,
                            minDistance=10
                        )
                        detection_found = True
                        detection_index = i
                        break
                if detection_found:
                    print(f"[{time.time()}] [AsyncTracker] Starting backfill from detection_index: {detection_index}")
                    self._backfill_tracking(detection_index)
                    
        self.frame_index += 1
        
    def _backfill_tracking(self, detection_index):
        """Backfill tracking state from detection frame to current frame"""
        print(f"[{time.time()}] [AsyncTracker] _backfill_tracking started from detection_index={detection_index}")
        for j in range(detection_index + 1, len(self.frame_buffer)):
            prev_idx, prev_bgr, prev_gray = self.frame_buffer[j - 1]
            curr_idx, curr_bgr, curr_gray = self.frame_buffer[j]
            if self.features is not None and len(self.features) > 0:
                new_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, self.features, None)
                good_old = self.features[status.flatten() == 1]
                good_new = new_features[status.flatten() == 1]
                if len(good_old) > 0:
                    displacement = np.mean(good_new - good_old, axis=0)
                    displacement = np.squeeze(displacement)
                    dx, dy = displacement
                    print(f"[{time.time()}] [AsyncTracker] Backfill displacement at j={j}: dx={dx}, dy={dy}")
                    if self.kalman:
                        self.kalman.predict((dx, dy))
                    if self.bbox is not None:
                        x_min, y_min, x_max, y_max = self.bbox
                        self.bbox = (int(x_min + dx), int(y_min + dy), int(x_max + dx), int(y_max + dy))
                    self.features = new_features
                else:
                    print(f"[{time.time()}] [AsyncTracker] No good features in backfill at j={j}, breaking")
                    break
            else:
                print(f"[{time.time()}] [AsyncTracker] No features to track in backfill at j={j}, breaking")
                break 