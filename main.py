# ===== STEP 1: Install Dependencies =====
# pip install moondream opencv-python  # Install dependencies in your project directory


# ===== STEP 2: Download Model =====
# Download model (1,733 MiB download size, 2,624 MiB memory usage)
# Use: wget (Linux and Mac) or curl.exe -O (Windows)
# wget https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz

import moondream as md
from PIL import Image
import cv2
import numpy as np

def cv2_to_pil(cv2_image):
    # Convert from BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    return Image.fromarray(rgb_image)

def draw_bounding_boxes(frame, objects):
    height, width = frame.shape[:2]
    for obj in objects:
        # Convert normalized coordinates to pixel coordinates
        x_min = int(obj['x_min'] * width)
        y_min = int(obj['y_min'] * height)
        x_max = int(obj['x_max'] * width)
        y_max = int(obj['y_max'] * height)
        
        # Draw rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return frame

# Initialize model
model = md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI5NGZhNGM2My1hZmU1LTRkYWItODY1OS1hNTZjZjliYTk4ODEiLCJpYXQiOjE3MzkwMzk1NDV9.sdg5lgCsG-WslptZA6T25PQphTz3euSQy9jJn3DEolg")

# Initialize webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert CV2 frame to PIL Image
        pil_image = cv2_to_pil(frame)
        
        # Encode image
        encoded_image = model.encode_image(pil_image)

        # Object Detection
        print("\nObject detection:")
        objects = model.detect(encoded_image, "pliers")["objects"]
        print(f"Found {objects}")

        # Draw bounding boxes on the frame
        frame = draw_bounding_boxes(frame, objects)

        # Display the frame
        cv2.imshow('Moondream Webcam', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()