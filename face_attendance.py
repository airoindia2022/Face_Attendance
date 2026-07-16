"""
Face Recognition Attendance System using OpenCV
------------------------------------------------
1. Register students by capturing their face from the webcam.
2. Train the recognizer on registered faces.
3. Mark attendance automatically when a registered face is seen.

Attendance is saved to attendance.csv with name, date and time.

Requires a webcam.
Run:  python face_attendance.py
"""

import cv2
import os
import csv
import numpy as np
from datetime import datetime

FACES_DIR = "faces"
ATTENDANCE_FILE = "attendance.csv"
SAMPLES_PER_STUDENT = 30

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def register_student():
    """Capture face samples for a new student from the webcam."""
    name = input("Enter student name: ").strip().replace(" ", "_")
    if not name:
        print("Name cannot be empty.")
        return

    student_dir = os.path.join(FACES_DIR, name)
    os.makedirs(student_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    print(f"Capturing {SAMPLES_PER_STUDENT} face samples. Look at the camera...")

    while count < SAMPLES_PER_STUDENT:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            cv2.imwrite(os.path.join(student_dir, f"{count}.jpg"), face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {count}/{SAMPLES_PER_STUDENT}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Registering - Press Q to stop", frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Registered '{name}' with {count} samples.")


def train_recognizer():
    """Train LBPH face recognizer on all registered faces."""
    faces, labels, label_map = [], [], {}
    if not os.path.isdir(FACES_DIR):
        return None, None

    for label_id, name in enumerate(sorted(os.listdir(FACES_DIR))):
        label_map[label_id] = name
        student_dir = os.path.join(FACES_DIR, name)
        for img_file in os.listdir(student_dir):
            img = cv2.imread(os.path.join(student_dir, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label_id)

    if not faces:
        return None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    return recognizer, label_map


def mark_attendance(name):
    """Write attendance to CSV (once per person per day)."""
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")

    already_marked = False
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            for row in csv.reader(f):
                if row and row[0] == name and row[1] == today:
                    already_marked = True
                    break

    if not already_marked:
        file_exists = os.path.exists(ATTENDANCE_FILE)
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Date", "Time"])
            writer.writerow([name, today, now])
        print(f"Attendance marked for {name} at {now}")


def take_attendance():
    """Recognize faces from webcam and mark attendance."""
    recognizer, label_map = train_recognizer()
    if recognizer is None:
        print("No registered students found. Register students first (option 1).")
        return

    cap = cv2.VideoCapture(0)
    print("Taking attendance. Press Q to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            label_id, confidence = recognizer.predict(face_img)

            if confidence < 70:  # lower = better match
                name = label_map[label_id]
                mark_attendance(name)
                color, text = (0, 255, 0), name
            else:
                color, text = (0, 0, 255), "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Attendance - Press Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    while True:
        print("\n" + "=" * 50)
        print("  FACE RECOGNITION ATTENDANCE SYSTEM")
        print("=" * 50)
        print("1. Register a new student")
        print("2. Take attendance")
        print("3. View attendance file")
        print("4. Exit")
        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            register_student()
        elif choice == "2":
            take_attendance()
        elif choice == "3":
            if os.path.exists(ATTENDANCE_FILE):
                with open(ATTENDANCE_FILE) as f:
                    print("\n" + f.read())
            else:
                print("No attendance recorded yet.")
        elif choice == "4":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
