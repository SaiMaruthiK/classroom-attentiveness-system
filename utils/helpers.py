"""
General helpers
"""

import time
import logging
import numpy as np
import cv2
from typing import List, Tuple

logger = logging.getLogger(__name__)


class FPSCounter:
    def __init__(self, avg_window: int = 30):
        self._timestamps: List[float] = []
        self._window = avg_window

    def tick(self) -> float:
        now = time.time()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height))


def crop_face(frame: np.ndarray, box: Tuple[int, int, int, int],
              padding: float = 0.10) -> np.ndarray:
    """Crop face region with optional padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return frame[y1:y2, x1:x2]


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
