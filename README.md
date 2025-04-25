# WORKSHOP 2- Object detection using web camera

## AIM :
To Perform real-time object detection using a trained YOLO v4 model through your laptop camera

## REQUIREMENTS :

### 1. Python 3.x installed.
### 2. OpenCV library installed (pip install opencv-python).
### 3. NumPy library installed (pip install numpy).
### 4. YOLOv4 weights (yolov4.weights), configuration file (yolov4.cfg), and COCO names file (coco.names).
### 5. Webcam for real-time video capture.
### 6. CV2 and DNN module from OpenCV for object detection processing.

## PROGRAM :
```
Developed By : HARSHITHA V
Register No : 212223230074
```
```
!pip install opencv-python numpy

```
```
import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    height, width, channels = frame.shape

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show output
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

```
## OUTPUT :

![image](https://github.com/user-attachments/assets/05ca7f01-4192-46ce-8ebb-2ecdf5bc2b20)

## RESULT :
Thius, Successfully Performed real-time object detection using a trained YOLO v4 model through my laptop camera.
