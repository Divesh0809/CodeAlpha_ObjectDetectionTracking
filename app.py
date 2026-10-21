
from flask import Flask, render_template, request, url_for, Response
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# Load AI object detection model
model = YOLO("yolo11n.pt")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        return "Please select a video.", 400

    video = request.files["video"]

    if video.filename == "":
        return "No video selected.", 400

    filename = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, filename + ".mp4")
    output_path = os.path.join(OUTPUT_FOLDER, filename + ".avi")

    video.save(input_path)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return "Unable to open this video. Try another video.", 400

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        return "Invalid video format.", 400

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        return "Unable to create output video.", 500

    total_objects = 0
    tracked_ids = set()
    class_counts = {}

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            result = results[0]
            total_objects += len(result.boxes)
            if result.boxes.cls is not None:
                detected_classes = result.boxes.cls.int().cpu().tolist()

                for class_id in detected_classes:
                    class_name = model.names[class_id]

                    class_counts[class_name] = (
                        class_counts.get(class_name, 0) + 1
        )

            if result.boxes.id is not None:
                for track_id in result.boxes.id.int().cpu().tolist():
                    tracked_ids.add(track_id)

            annotated_frame = result.plot()
            writer.write(annotated_frame)

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    return render_template(
        "index.html",
        result_video=url_for(
            "static",
            filename="results/" + filename + ".avi"
        ),
        total_detections=total_objects,
        unique_objects=len(tracked_ids),
        class_counts=class_counts
    )


@app.errorhandler(413)
def file_too_large(error):
    return "Video size exceeds the 200 MB limit.", 413


def generate_webcam_frames():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        return

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            annotated_frame = results[0].plot()

            success, buffer = cv2.imencode(
                ".jpg",
                annotated_frame
            )

            if not success:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    finally:
        camera.release()


@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


@app.route("/webcam_feed")
def webcam_feed():
    return Response(
        generate_webcam_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
if __name__ == "__main__":
        app.run(debug=True)

