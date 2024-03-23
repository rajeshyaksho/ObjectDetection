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
    font = cv2.FONT_HERSHEY_COMPLEX  # Change font style here
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 1, color, 1)  # Reduce font size here
    return image

# Load an image
image_path = "person.jpg"
#image_path = "dog.jpg"
image = cv2.imread(image_path)

# Perform object detection
outs = detect_objects(image)

# Draw bounding boxes around detected objects
output_image = draw_boxes(image, outs)

# Display the output image
cv2.imshow("Object Detection", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
