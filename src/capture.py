import cv2
import os

name = "Pranathii"
folder = f"dataset/{name}"

os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)   # camera

count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):   # press S to save
        cv2.imwrite(f"{folder}/{count}.jpg", frame)
        count += 1
        print("Image saved")

    elif key == ord('q'):  # press Q to quit
        break

cap.release()
cv2.destroyAllWindows()