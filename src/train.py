import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
label_map = {}

label_id = 0

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for person_name in sorted(os.listdir(dataset_path)):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_map[label_id] = person_name

    for image_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, image_name)

        if not img_path.endswith((".jpg", ".png")):
            continue

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))  # normalize size

            faces.append(face)
            labels.append(label_id)

    label_id += 1

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, np.array(labels))

model.save("trainer.yml")

print("Training complete")