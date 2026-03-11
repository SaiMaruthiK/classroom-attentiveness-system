"""
Emotion Detector — FER library (FER2013 CNN)
"""

import logging
import numpy as np
import cv2
from typing import Tuple

logger = logging.getLogger(__name__)

POSITIVE_EMOTIONS = {"happy", "neutral", "surprise"}
NEGATIVE_EMOTIONS = {"sad", "angry", "disgust", "fear"}


class EmotionDetector:
    """
    Uses FER library for emotion classification.
    Falls back to "neutral" if FER is unavailable.
    """

    def __init__(self):
        self._fer = None
        self._available = False
        self._load()

    def _load(self):
        try:
            from fer import FER
            self._fer = FER(mtcnn=False)
            self._available = True
            logger.info("FER emotion detector loaded.")
        except Exception as e:
            logger.warning("FER unavailable (%s). Defaulting to neutral.", e)

    def detect(self, face_crop: np.ndarray) -> Tuple[str, float]:
        """
        Returns (emotion_label, emotion_score 0-1).
        """
        if not self._available or face_crop is None or face_crop.size == 0:
            return "neutral", 1.0

        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            result = self._fer.detect_emotions(rgb)
            if not result:
                return "neutral", 1.0

            emotions = result[0].get("emotions", {})
            if not emotions:
                return "neutral", 1.0

            top_emotion = max(emotions, key=emotions.get)
            score = emotions[top_emotion]
            return top_emotion, round(score, 3)

        except Exception as e:
            logger.debug("Emotion detection error: %s", e)
            return "neutral", 1.0

    @staticmethod
    def emotion_to_score(emotion: str) -> float:
        """
        Maps emotion to attention relevance score (0.0 – 1.0).
        """
        mapping = {
            "happy": 1.0,
            "neutral": 1.0,
            "surprise": 0.8,
            "sad": 0.5,
            "fear": 0.5,
            "angry": 0.4,
            "disgust": 0.3,
        }
        return mapping.get(emotion.lower(), 0.5)
