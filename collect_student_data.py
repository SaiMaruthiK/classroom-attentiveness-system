"""
=============================================================
STUDENT DATA COLLECTION SCRIPT
Captures images from webcam to build the student dataset.
=============================================================
Usage:
    python collect_student_data.py --name "John Doe" --count 30
"""

import cv2
import os
import argparse
import time

import backend.config as cfg


def collect_images(student_name: str, count: int = 30, camera: int = 0):
    safe_name = student_name.lower().replace(" ", "_")
    save_dir = os.path.join(cfg.DATASET_DIR, safe_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print(f"\n📸 Collecting {count} images for '{student_name}'")
    print("Press SPACE to capture, Q to quit\n")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    captured = 0

    while captured < count:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(display, f"Captured: {captured}/{count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=capture  Q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(f"Collecting: {student_name}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if len(faces) == 0:
                print("⚠️  No face detected — try again")
                continue
            img_path = os.path.join(save_dir, f"{captured+1:03d}.jpg")
            cv2.imwrite(img_path, frame)
            captured += 1
            print(f"  ✅ Saved {img_path}")
            time.sleep(0.2)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done! {captured} images saved to {save_dir}")
    print("Now run: python -m backend.recognition.dataset_encoder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Student full name")
    parser.add_argument("--count", type=int, default=30, help="Number of images")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    collect_images(args.name, args.count, args.camera)
