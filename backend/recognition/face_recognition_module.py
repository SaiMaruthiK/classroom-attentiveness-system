"""
Face Recognition — OpenCV + DeepFace (NO dlib required)
Replaces the dlib-dependent face_recognition library entirely.
"""

import os
import pickle
import logging
import numpy as np
import cv2
from typing import Tuple, List

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Recognizes student identities using DeepFace (no dlib needed).
    Falls back to track-ID-based naming when unavailable.
    """

    def __init__(self, encodings_path: str, tolerance: float = 0.50,
                 model: str = "hog"):
        self.encodings_path = encodings_path
        self.tolerance = tolerance
        self._available = False
        self._known_encodings: List[np.ndarray] = []
        self._known_names: List[str] = []
        self._known_ids: List[str] = []
        self._load()

    def _load(self):
        try:
            from deepface import DeepFace
            self._DeepFace = DeepFace
            self._available = True
            logger.info("DeepFace face recognizer loaded (no dlib needed).")
        except ImportError:
            logger.warning("DeepFace not installed. Using track-ID naming.")
            return

        if os.path.exists(self.encodings_path):
            with open(self.encodings_path, "rb") as f:
                data = pickle.load(f)
            self._known_encodings = data.get("encodings", [])
            self._known_names = data.get("names", [])
            self._known_ids = data.get("ids", [])
            logger.info("Loaded %d face encodings.", len(self._known_encodings))

    def recognize(self, frame: np.ndarray, box: Tuple[int, int, int, int],
                  track_id: int) -> Tuple[str, str]:
        fallback_id = f"student_{int(track_id):03d}"
        fallback_name = f"Student {track_id}"

        if not self._available or not self._known_encodings:
            return fallback_id, fallback_name

        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return fallback_id, fallback_name

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return fallback_id, fallback_name

        try:
            embedding_obj = self._DeepFace.represent(
                face_crop, model_name="Facenet",
                enforce_detection=False, detector_backend="skip"
            )
            if not embedding_obj:
                return fallback_id, fallback_name

            query_enc = np.array(embedding_obj[0]["embedding"])

            best_dist = float("inf")
            best_idx = -1
            for i, enc in enumerate(self._known_encodings):
                dist = np.linalg.norm(query_enc - np.array(enc))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            threshold = 10.0  # Facenet L2 threshold
            if best_idx >= 0 and best_dist < threshold:
                return str(self._known_ids[best_idx]), self._known_names[best_idx]

        except Exception as e:
            logger.debug("Recognition error: %s", e)

        return fallback_id, fallback_name
