from ultralytics import YOLO
import cv2
import time
import pyttsx3

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("YOLO model loaded successfully!")

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Store the last detection time
last_detection_time = time.time()

while True:
    # Capture one frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Current time
    current_time = time.time()

    # Detect objects every 10 seconds
    if current_time - last_detection_time >= 10:

        print("\n----- Detecting Objects -----")

        # Run YOLO on the current frame
        results = model(frame)

        # Print detected object names
        detected_objects = set()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                object_name = model.names[class_id]
                detected_objects.add(object_name)

        if detected_objects:
            print("Detected Objects:")
            for obj in detected_objects:
                print("-", obj)
        else:
            print("No objects detected.")

        # Update last detection time
        last_detection_time = current_time

    # Draw bounding boxes on every frame
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Display webcam
    cv2.imshow("BlindAssist - Object Detection", annotated_frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing webcam...")
        break

# Release webcam
cap.release()




engine = pyttsx3.init()
engine.say("Blind Assist started")
engine.runAndWait()


# Close all OpenCV windows
cv2.destroyAllWindows()
