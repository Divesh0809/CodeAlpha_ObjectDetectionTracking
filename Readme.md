# AI-Based Object Detection and Tracking System

## Project Overview

This project is an AI-powered object detection and tracking system built using YOLO, OpenCV, ByteTrack, and Flask. It detects objects in uploaded videos and webcam streams, draws bounding boxes around detected objects, and tracks them across video frames.

## Features

* Object detection using YOLO
* Object tracking using ByteTrack
* Video upload and processing
* Live webcam detection
* Bounding boxes and object labels
* Detection statistics and dashboard
* Processed video output

## Technologies Used

* Python
* Flask
* Ultralytics YOLO
* OpenCV
* ByteTrack
* HTML and CSS

## Project Structure

```text
ObjectDetectionTracking/
├── app.py
├── requirements.txt
├── templates/
│   ├── index.html
│   └── webcam.html
└── static/
    └── results/
```

## Installation

1. Clone this repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```
2. Open the project directory:

   ```bash
   cd ObjectDetectionTracking
   ```
3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:

   ```bash
   python app.py
   ```
5. Open `http://127.0.0.1:5000/` in your browser.

## How It Works

1. Upload a video or open the webcam page.
2. YOLO detects objects in the frames.
3. ByteTrack assigns tracking IDs to detected objects.
4. OpenCV processes the frames and displays the results.
5. View detection statistics on the dashboard.

## Note

The YOLO model weights may be downloaded automatically by Ultralytics on the first run. Webcam access depends on camera permissions and device availability.

## Author

Sanjay Divesh V

## License

This project is intended for educational and portfolio purposes.
