"""
Eye Blink / Closure Detector — Eye Aspect Ratio (EAR)
Uses MediaPipe FaceMesh landmarks.
"""

import logging
import numpy as np
import cv2
from typing import Tuple, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)

# MediaPipe landmark indices for left / right eye
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


def _ear(eye_pts: np.ndarray) -> float:
    """Compute Eye Aspect Ratio from 6 landmark points."""
    # Vertical distances
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    # Horizontal distance
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


class BlinkDetector:
    """
    Detects eye blinks and prolonged eye closure per tracked face.
    """

    def __init__(self, ear_threshold: float = 0.25, consec_frames: int = 2):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self._face_mesh = None
        self._available = False
        self._frame_counts: Dict[int, int] = defaultdict(int)  # track_id → closed frames
        self._load()

    def _load(self):
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._available = True
            logger.info("BlinkDetector: MediaPipe FaceMesh loaded.")
        except Exception as e:
            logger.warning("BlinkDetector: MediaPipe unavailable (%s).", e)

    def detect(self, frame: np.ndarray, box: Tuple[int, int, int, int],
               track_id: int) -> Tuple[str, float]:
        """
        Returns (eye_state, ear_value).
        eye_state: 'open' | 'closed'
        """
        if not self._available:
            return "open", 0.3

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return "open", 0.3

        x1, y1, x2, y2 = box
        cx_box = (x1 + x2) / 2
        cy_box = (y1 + y2) / 2

        best_lm = None
        best_dist = float("inf")
        for face_lm in results.multi_face_landmarks:
            lm = face_lm.landmark[1]
            dist = (lm.x * w - cx_box) ** 2 + (lm.y * h - cy_box) ** 2
            if dist < best_dist:
                best_dist = dist
                best_lm = face_lm

        if best_lm is None:
            return "open", 0.3

        def get_pts(indices):
            return np.array([
                [best_lm.landmark[i].x * w, best_lm.landmark[i].y * h]
                for i in indices
            ])

        left_pts = get_pts(LEFT_EYE_IDX)
        right_pts = get_pts(RIGHT_EYE_IDX)
        ear = (_ear(left_pts) + _ear(right_pts)) / 2.0

        if ear < self.ear_threshold:
            self._frame_counts[track_id] += 1
        else:
            self._frame_counts[track_id] = 0

        closed = self._frame_counts[track_id] >= self.consec_frames
        return ("closed" if closed else "open"), round(ear, 3)

    @staticmethod
    def eye_state_to_score(eye_state: str) -> float:
        return 1.0 if eye_state == "open" else 0.0
