import cv2
import numpy as np

# Load YOLO
yolo_cfg = r"C:\Users\theor\Downloads\archive (8)\yolov3.cfg"  # Path to YOLOv3 config file
yolo_weights = r"C:\Users\theor\Downloads\archive (8)\yolov3.weights"  # Path to YOLOv3 pre-trained weights
yolo_names = r"C:\Users\theor\Downloads\archive (8)\coco.names"  # Path to coco names file with class labels

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the COCO class labels
with open(yolo_names, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get the shape of the frame
    height, width, channels = frame.shape

    # Prepare the frame for YOLO model (resize and normalize)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the blob as input to the network
    net.setInput(blob)

    # Perform forward pass and get predictions
    detections = net.forward(output_layers)

#     # Initialize lists to hold detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop through all detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)  # Class ID with the highest score
            confidence = scores[class_id]

            # Only consider detections with confidence above a threshold (e.g., 0.5)
            if confidence > 0.5:
                # Get the bounding box coordinates
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Get the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append the box, confidence, and class ID to the lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the boxes and labels on the frame
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Get the class name
            confidence = confidences[i]

            # Draw the rectangle around the detected object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put the class name and confidence
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the detected objects
    cv2.imshow('Real-Time Object Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

