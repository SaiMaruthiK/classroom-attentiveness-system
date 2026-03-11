"""
Dataset Encoder — NO DLIB version
Uses DeepFace + Facenet to generate encodings.

Usage:
    python -m backend.recognition.dataset_encoder
"""

import os
import pickle
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_dataset(dataset_dir: str, output_path: str):
    try:
        import cv2
        from deepface import DeepFace
    except ImportError as e:
        logger.error("Missing package: %s  —  run: pip install deepface", e)
        sys.exit(1)

    known_encodings = []
    known_names = []
    known_ids = []
    student_id = 1

    if not os.path.isdir(dataset_dir):
        logger.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    for student_folder in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, student_folder)
        if not os.path.isdir(folder_path):
            continue

        student_name = student_folder.replace("_", " ").title()
        logger.info("Processing: %s", student_name)
        count = 0

        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            try:
                result = DeepFace.represent(
                    img, model_name="Facenet",
                    enforce_detection=False
                )
                if result:
                    enc = result[0]["embedding"]
                    known_encodings.append(enc)
                    known_names.append(student_name)
                    known_ids.append(str(student_id))
                    count += 1
            except Exception as e:
                logger.warning("  Skipped %s: %s", img_file, e)

        logger.info("  %d encodings added for %s", count, student_name)
        student_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names, "ids": known_ids}, f)
    logger.info("Saved %d encodings to %s", len(known_encodings), output_path)


if __name__ == "__main__":
    import backend.config as cfg
    encode_dataset(cfg.DATASET_DIR, cfg.FACE_ENCODINGS_PATH)
