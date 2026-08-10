import cv2

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    # Capture one frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Display webcam
    cv2.imshow("BlindAssist - Webcam Feed", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing webcam...")
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()
