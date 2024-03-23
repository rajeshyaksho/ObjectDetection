import cv2
import numpy as np

# Define the paths to the YOLO model files
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
names_path = "coco.names"

# Load the YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Load the class names
with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

# Function to perform object detection
def detect_objects(image):
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    return outs

# Function to draw bounding boxes around detected objects
def draw_boxes(image, outs):
    height, width, _ = image.shape
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 3, color, 3)
    return image

# Function to capture frames from webcam, resize, and perform object detection
def webcam_object_detection():
    cap = cv2.VideoCapture(0)
    # Set the desired frame width and height
    frame_width = 640
    frame_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame to a smaller resolution
        resized_frame = cv2.resize(frame, (416, 416))
        
        # Perform object detection on the resized frame
        outs = detect_objects(resized_frame)
        
        # Draw bounding boxes on the original frame
        output_image = draw_boxes(frame, outs)
        
        # Display the frame with bounding boxes
        cv2.imshow("Webcam Object Detection", output_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run object detection using webcam
webcam_object_detection()
