"""
Face Detector — YOLOv8
Detects faces / persons in each frame.
"""

import logging
import numpy as np
import cv2
from typing import List, Tuple

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    YOLOv8-based face/person detector.

    Returns bounding boxes as [x1, y1, x2, y2, confidence].
    Falls back to Haar Cascade if ultralytics is not installed.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.40,
                 device: str = "cpu"):
        self.conf = conf
        self.device = device
        self._model = None
        self._haar = None
        self._use_yolo = False
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._model.to(self.device)
            self._use_yolo = True
            logger.info("YOLOv8 loaded: %s on %s", model_path, self.device)
        except Exception as e:
            logger.warning("YOLOv8 unavailable (%s). Using Haar Cascade fallback.", e)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame: np.ndarray) -> List[List[float]]:
        """
        Returns list of [x1, y1, x2, y2, confidence].
        """
        if self._use_yolo:
            return self._detect_yolo(frame)
        return self._detect_haar(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[List[float]]:
        results = self._model(frame, conf=self.conf, classes=[0], verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])
        return detections

    def _detect_haar(self, frame: np.ndarray) -> List[List[float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        detections = []
        for (x, y, w, h) in faces:
            detections.append([float(x), float(y), float(x + w), float(y + h), 0.9])
        return detections
