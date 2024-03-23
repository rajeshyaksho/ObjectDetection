from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Define paths to YOLO model files
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
names_path = "coco.names"

# Load the YOLO model and class names
net = cv2.dnn.readNet(weights_path, config_path)
with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

# Function to perform object detection
def detect_objects(image):
    # Preprocess the input image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Parse the output of the model
    detections = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                detections.append((x, y, w, h, class_id, confidence))

    # Sort detections by confidence score in descending order
    detections.sort(key=lambda x: x[5], reverse=True)

    # Keep only the highest confidence detection for each class
    unique_classes = set()
    filtered_detections = []
    for detection in detections:
        class_id = detection[4]
        if class_id not in unique_classes:
            unique_classes.add(class_id)
            filtered_detections.append(detection)

    return filtered_detections

# Function to draw bounding boxes around detected objects
def draw_boxes(image, detections):
    for detection in detections:
        x, y, w, h, class_id, confidence = detection
        color = (0, 255, 0)  # Green color for bounding box
        label = f"{classes[class_id]}: {confidence:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Function to capture frames from webcam and perform object detection
def webcam_object_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Perform object detection
        detections = detect_objects(frame)
        # Draw bounding boxes on the frame
        output_frame = draw_boxes(frame, detections)
        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', output_frame)
        # Convert the buffer to bytes
        frame_bytes = buffer.tobytes()
        # Yield the frame bytes for the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return Response(webcam_object_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
