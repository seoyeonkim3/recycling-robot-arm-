import torch
import cv2
import numpy as np

# Load YOLOv5 modelq
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
cap = cv2.VideoCapture(0)  # Access the USB camera (device 0)

if not cap.isOpened():
    print("Can't open camera.")
    exit()

# Frame resolution (640x480) - pixel center
frame_width = 640
frame_height = 480
frame_center = (frame_width // 2, frame_height // 2)

try:
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Perform object detection using YOLOv5
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        closest_center = None
        min_distance = float('inf')

        # Draw the image pixel center in RED
        cv2.circle(frame, frame_center, 5, (0, 0, 255), -1)
        cv2.putText(frame, "Image Center", (frame_center[0] + 10, frame_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Extract the center coordinates of detected objects
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            center_x = (x1 + x2) / 2  # Object center X-coordinate
            center_y = (y1 + y2) / 2  # Object center Y-coordinate

            # Calculate distance between object center and image center
            distance = np.sqrt((center_x - frame_center[0]) ** 2 + (center_y - frame_center[1]) ** 2)

            # Find the closest object
            if distance < min_distance:
                closest_center = (int(center_x), int(center_y))
                min_distance = distance

        # Draw the closest object's center in GREEN
        if closest_center is not None:
            cv2.circle(frame, closest_center, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Object Center: {closest_center}", (closest_center[0] + 10, closest_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the center coordinates
            np.save('pixel_coordinates.npy', np.array(closest_center))
            print(f"Pixel Coordinates Saved: {closest_center}")

        # Display the frame with annotations
        cv2.imshow('Camera Feed', frame)

        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Process interrupted.")
finally:
    cap.release()
    cv2.destroyAllWindows()
