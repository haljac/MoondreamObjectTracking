# Realtime object detection & tracking with moondream.ai

A Python script that demonstrates asynchronous object detection and tracking using:
- [OpenCV](https://opencv.org/) for video capture, feature tracking (Lucas-Kanade optical flow), and drawing
- A [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) for smoothing tracked coordinates
- [Moondream](https://moondream.ai/) for VLM based object detection
- [Zenoh](https://zenoh.io/) for network communication between components

<b>Disclaimer:</b> If not already obvious this document was written by an LLM but should be mostly correct.  

## System Architecture

The system consists of three main components that can run on different machines as long as they're on the same network:

1. **Webcam Publisher (`webcam_publisher.py`)**
   - Captures frames from a webcam and publishes them to a Zenoh topic (`robot/camera/frame`)
   - Can run on either your laptop or directly on a robot
   - Configurable parameters for camera resolution, FPS, and frame processing
   - Supports both monocular and stereo camera setups with cropping options

2. **Object Tracking Publisher (`tracking_publisher.py`)**
   - Subscribes to camera frames from the webcam publisher
   - Uses Moondream for object detection and tracking
   - Publishes normalized object position and size information to `tracking/position`
   - Position messages are published in the following JSON format:
     ```json
     {
       "x": float,        // Normalized x position (0-1)
       "y": float,        // Normalized y position (0-1)
       "width": float,    // Normalized width (0-1)
       "height": float,   // Normalized height (0-1)
       "timestamp": float  // Unix timestamp
     }
     ```

3. **Visual Servoing Example (`servoing_example.py`)**
   - Demonstrates how to use tracking for robot control
   - Subscribes to camera frames and computes twist commands
   - Publishes robot control commands to `robot/cmd` topic
   - Twist messages are in the following format:
     ```json
     {
       "x": float,     // Linear velocity in m/s
       "theta": float  // Angular velocity in rad/s
     }
     ```

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
   pip install opencv-python Pillow numpy moondream eclipse-zenoh rerun-sdk
   ```

3. **API Key**  
   This script reads an API key from a file named `api_key.txt` in the same directory. Make sure you have a file `api_key.txt` containing your Moondream API key:
   
   ```
   your_moondream_api_key_goes_here
   ```

4. **Run Components**  
   First, start the webcam publisher:
   ```bash
   python webcam_publisher.py
   ```
   
   Then in a separate terminal, start object tracking:
   ```bash
   python tracking_publisher.py --prompt "object to track"
   ```

   Optionally, run the visual servoing example:
   ```bash
   python servoing_example.py --prompt "object to track"
   ```
   Replace "object to track" with your target object (e.g., "red ball", "person", etc.)

## Usage

1. **Network Setup**
   - Ensure all machines are on the same network
   - Zenoh will automatically discover peers on the network
   - No manual IP configuration needed in most cases

2. **Webcam Publisher**
   - Publishes camera frames at configurable resolution and FPS
   - Check `constants.py` for camera configuration options
   - Supports stereo camera setups with options to crop to left/right frame

3. **Object Tracking**
   - Subscribes to camera frames and tracks specified object
   - Publishes normalized position information (0-1 range)
   - Uses Rerun for real-time visualization of tracking
   - Position updates are smoothed using Kalman filtering

4. **Visual Servoing Example**
   - Demonstrates robot control using tracking data
   - Configurable gains in `servoing_example.py`:
     - `gain`: Controls how aggressively to turn (default: 0.005)
     - `max_angular_z`: Maximum turning speed in rad/s (default: 0.35)

5. **Controls**
   - Press **Ctrl+C** in any terminal to stop the respective component
