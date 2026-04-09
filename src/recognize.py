import cv2
import os
import csv
from datetime import datetime

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("trainer.yml")

# Label mapping (sorted)
dataset_path = "dataset"
label_map = {}

for idx, name in enumerate(sorted(os.listdir(dataset_path))):
    label_map[idx] = name

# Create attendance folder
os.makedirs("attendance", exist_ok=True)

# File name
file_path = f"attendance/attendance_{datetime.now().date()}.csv"

# Create CSV file if not exists
if not os.path.exists(file_path):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

# 🔥 Track marked names (no duplicates)
marked_names = set()

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = model.predict(face)

        # 🔥 STRICT threshold for accuracy
        if confidence < 45:
            name = label_map[label]
        else:
            name = "Unknown"

        # 🔥 Mark attendance safely
        if name != "Unknown" and name not in marked_names:
            marked_names.add(name)

            print(f"Marked: {name}")

            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, datetime.now().strftime("%H:%M:%S")])

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show name + confidence
        cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()