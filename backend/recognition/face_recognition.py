"""
backend/recognition/face_recognition.py
FaceNet-style recognition using the face_recognition library.
Loads pre-computed encodings and matches live faces.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from backend.config import (
    ENCODINGS_PATH, FACE_RECOGNITION_TOLERANCE,
    FACE_RECOGNITION_MODEL, UNKNOWN_LABEL
)

# Lazy import so the module can be imported even without the library installed
try:
    import face_recognition as fr
    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False
    logger.warning("face_recognition library not installed. Recognition disabled.")


class FaceRecognizer:
    """
    Matches face crops to registered student identities.

    Usage:
        recognizer = FaceRecognizer()
        name = recognizer.identify(face_crop_bgr)
    """

    def __init__(self) -> None:
        self._known_encodings: List[np.ndarray] = []
        self._known_names:     List[str]         = []
        self._id_map:          Dict[str, str]     = {}   # name -> student_id
        self._load_encodings()

    # ──────────────────────────────────────────
    # Encodings
    # ──────────────────────────────────────────

    def _load_encodings(self) -> None:
        if not ENCODINGS_PATH.exists():
            logger.warning(f"Encodings file not found at {ENCODINGS_PATH}. "
                           "Run dataset_encoder.py first.")
            return
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        self._known_encodings = data.get("encodings", [])
        self._known_names     = data.get("names", [])
        self._id_map          = data.get("id_map", {})
        logger.info(
            f"Loaded {len(self._known_encodings)} face encodings "
            f"for {len(set(self._known_names))} students."
        )

    def reload(self) -> None:
        """Hot-reload encodings without restarting."""
        self._known_encodings = []
        self._known_names     = []
        self._id_map          = {}
        self._load_encodings()

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def identify(self, face_bgr: np.ndarray) -> Tuple[str, str]:
        """
        Identify a face crop.

        Returns (student_name, student_id).
        Returns (UNKNOWN_LABEL, "unknown") if not recognised.
        """
        if not _FR_AVAILABLE or not self._known_encodings:
            return UNKNOWN_LABEL, "unknown"

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        encodings = fr.face_encodings(rgb, model=FACE_RECOGNITION_MODEL)

        if not encodings:
            return UNKNOWN_LABEL, "unknown"

        encoding = encodings[0]
        distances = fr.face_distance(self._known_encodings, encoding)
        best_idx  = int(np.argmin(distances))

        if distances[best_idx] <= FACE_RECOGNITION_TOLERANCE:
            name = self._known_names[best_idx]
            sid  = self._id_map.get(name, name.lower().replace(" ", "_"))
            return name, sid

        return UNKNOWN_LABEL, "unknown"

    def identify_batch(
        self, face_crops: List[np.ndarray]
    ) -> List[Tuple[str, str]]:
        """Identify multiple faces in one call."""
        return [self.identify(crop) for crop in face_crops]

    @property
    def registered_count(self) -> int:
        return len(set(self._known_names))
