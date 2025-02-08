# Realtime object detection & tracking with moondream.ai

A Python script that demonstrates asynchronous object detection and tracking using:
- [OpenCV](https://opencv.org/) for video capture, feature tracking (Lucas-Kanade optical flow), and drawing
- A [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) for smoothing tracked coordinates
- [Moondream](https://moondream.ai/) for VLM based object detection

<b>Disclaimer:</b> If not already obvious this document was written by a LLM but should be mostly correct.  

## Features

1. **Real-time Detection**  
   A detection thread periodically attempts to detect a specified object in the camera feed using the Moondream model.

2. **Fast Optical Flow Tracking**  
   The main loop uses lightweight Lucas-Kanade optical flow to track features within the bounding box, ensuring fast updates.

3. **Rewind and Reapply**  
   When a new detection arrives, it's applied retroactively (rewinds to the detection frame, replays up to the present), ensuring the bounding box and features remain up-to-date.

You may also need libraries for concurrency (threading) and data structures (collections), but these are typically part of Python's standard library.

## Setup

1. **Clone or Download** this repository.  
2. **Install dependencies** (using `pip`):

   ```bash
   pip install opencv-python Pillow numpy moondream
   ```

3. **API Key**  
   This script reads an API key from a file named `api_key.txt` in the same directory. Make sure you have a file `api_key.txt` containing your Moondream API key:
   
   ```
   your_moondream_api_key_goes_here
   ```

4. **Run**  
   Use Python to run the `main.py` file:
   ```bash
   python main.py
   ```
   The script will prompt you for the object you want to track (e.g., "pliers").

## Usage

1. **Program Flow**  
   - The script starts your webcam, captures frames, and looks for the specified object.  
   - Once the initial detection is found, it initializes a Kalman filter around the object's bounding box and starts tracking features via optical flow.
   - Meanwhile, a separate thread regularly performs new detections, in case the object changes position drastically or is lost.

2. **Controls**  
   - The program opens an OpenCV window with some sliders you can adjust:
     - **maxCorners** (defaults to 27): The maximum number of features to track.  
     - **qualityLevel x100** (defaults to 10): Determines the minimum quality (scaled by 100) of selected features.  
     - **minDistance** (defaults to 10): The minimum distance between tracked features.  
   - Press **'q'** in the OpenCV window to exit the program.

3. **Detection Interval**  
   - The detection thread runs every few seconds (default is 2.0). You can change it by editing the `detection_interval` argument in the `detection_loop` function.

4. **Debugging**  
   - If you lose the object or the bounding box drifts away, you can:
     - Wait for the next detection to correct it.
     - Or manually kill the program (`Ctrl + C` or `q` in the window), adjust parameters, and rerun.
