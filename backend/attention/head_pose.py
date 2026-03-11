"""
Head Pose Estimator — MediaPipe FaceMesh
Estimates yaw, pitch, roll from facial landmarks.
"""

import logging
import numpy as np
import cv2
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class HeadPoseEstimator:
    """
    Uses MediaPipe FaceMesh + solvePnP to estimate head orientation.
    Falls back to landmark-ratio heuristic if MediaPipe unavailable.
    """

    # 3D reference model points (nose, chin, L/R eye corners, L/R mouth corners)
    _MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),        # nose tip
        (0.0, -330.0, -65.0),   # chin
        (-225.0, 170.0, -135.0),# left eye corner
        (225.0, 170.0, -135.0), # right eye corner
        (-150.0, -150.0, -125.0),# left mouth corner
        (150.0, -150.0, -125.0), # right mouth corner
    ], dtype=np.float64)

    # MediaPipe landmark indices for the 6 reference points
    _MP_INDICES = [1, 152, 263, 33, 287, 57]

    def __init__(self):
        self._mp_face_mesh = None
        self._face_mesh = None
        self._available = False
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
            logger.info("MediaPipe FaceMesh loaded.")
        except Exception as e:
            logger.warning("MediaPipe unavailable (%s). Using heuristic fallback.", e)

    def estimate(self, frame: np.ndarray, box: Tuple[int, int, int, int]
                 ) -> Tuple[str, float, float, float]:
        """
        Returns (pose_label, yaw_deg, pitch_deg, roll_deg).
        pose_label: 'forward' | 'away'
        """
        if self._available:
            return self._estimate_mediapipe(frame, box)
        return self._estimate_heuristic(frame, box)

    def _estimate_mediapipe(self, frame, box):
        h_frame, w_frame = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return "forward", 0.0, 0.0, 0.0

        # Pick the face landmark set closest to our detection box
        x1, y1, x2, y2 = box
        cx_box = (x1 + x2) / 2
        cy_box = (y1 + y2) / 2
        best_lm = None
        best_dist = float("inf")

        for face_lm in results.multi_face_landmarks:
            lm0 = face_lm.landmark[1]
            cx = lm0.x * w_frame
            cy = lm0.y * h_frame
            dist = (cx - cx_box) ** 2 + (cy - cy_box) ** 2
            if dist < best_dist:
                best_dist = dist
                best_lm = face_lm

        if best_lm is None:
            return "forward", 0.0, 0.0, 0.0

        image_points = np.array([
            (best_lm.landmark[i].x * w_frame,
             best_lm.landmark[i].y * h_frame)
            for i in self._MP_INDICES
        ], dtype=np.float64)

        focal_length = w_frame
        cam_matrix = np.array([
            [focal_length, 0, w_frame / 2],
            [0, focal_length, h_frame / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            self._MODEL_POINTS, image_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return "forward", 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)
        pitch = float(euler[0])
        yaw = float(euler[1])
        roll = float(euler[2])

        import backend.config as cfg
        is_forward = (
            abs(yaw) < cfg.HEAD_POSE_YAW_THRESHOLD and
            abs(pitch) < cfg.HEAD_POSE_PITCH_THRESHOLD
        )
        label = "forward" if is_forward else "away"
        return label, round(yaw, 1), round(pitch, 1), round(roll, 1)

    def _estimate_heuristic(self, frame, box):
        """Fallback: use face box aspect ratio as crude proxy."""
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            return "forward", 0.0, 0.0, 0.0
        ratio = w / h
        # Narrower face = turned away
        label = "forward" if ratio > 0.6 else "away"
        return label, 0.0, 0.0, 0.0

    @staticmethod
    def pose_to_score(pose_label: str) -> float:
        return 1.0 if pose_label == "forward" else 0.0
