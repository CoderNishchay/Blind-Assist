import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Face Recognition Test Started")
print("Press 'q' to stop.")

known_face = "known_faces/Md Eliyas.jpeg"

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    try:
        results = DeepFace.find(
            img_path=frame,
            db_path="known_faces",
            model_name="VGG-Face",
            enforce_detection=False,
            silent=True
        )

        name = "Unknown Person"

        if len(results) > 0 and len(results[0]) > 0:
            identity = results[0].iloc[0]["identity"]
            name = "Md Eliyas"

        cv2.putText(
            frame,
            name,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    except Exception as e:
        cv2.putText(
            frame,
            "Processing...",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    cv2.imshow("BlindAssist - Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Face Recognition Test Stopped")