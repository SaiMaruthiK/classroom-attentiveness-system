"""
Multi-Object Tracker — DeepSORT
Assigns consistent IDs to each student across frames.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class FaceTracker:
    """
    Wraps DeepSORT for multi-face tracking.
    Falls back to simple IoU tracker if deep_sort_realtime is unavailable.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3,
                 max_cosine_distance: float = 0.4):
        self._tracker = None
        self._use_deepsort = False
        self._simple_tracks: Dict[int, dict] = {}
        self._next_id = 1
        self._load_tracker(max_age, n_init, max_cosine_distance)

    def _load_tracker(self, max_age, n_init, max_cosine_distance):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_cosine_distance=max_cosine_distance,
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",
                half=False,
                bgr=True,
                embedder_gpu=False,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None,
            )
            self._use_deepsort = True
            logger.info("DeepSORT tracker initialized.")
        except Exception as e:
            logger.warning("DeepSORT unavailable (%s). Using IoU fallback tracker.", e)

    def update(self, detections: List[List[float]], frame: np.ndarray) -> List[Tuple]:
        """
        Args:
            detections: [[x1,y1,x2,y2,conf], ...]
            frame: BGR frame

        Returns:
            List of (track_id, x1, y1, x2, y2)
        """
        if not detections:
            return []

        if self._use_deepsort:
            return self._update_deepsort(detections, frame)
        return self._update_simple(detections)

    def _update_deepsort(self, detections, frame):
        # DeepSORT expects [[x1,y1,w,h], conf, class]
        ds_input = []
        for d in detections:
            x1, y1, x2, y2, conf = d
            w = x2 - x1
            h = y2 - y1
            ds_input.append(([x1, y1, w, h], conf, 0))

        tracks = self._tracker.update_tracks(ds_input, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            results.append((tid, x1, y1, x2, y2))
        return results

    def _update_simple(self, detections):
        """Basic greedy IoU matching tracker."""
        new_tracks = {}
        used_ids = set()

        for d in detections:
            x1, y1, x2, y2, conf = d
            best_id = None
            best_iou = 0.3  # minimum IoU threshold

            for tid, tr in self._simple_tracks.items():
                if tid in used_ids:
                    continue
                iou = self._iou([x1, y1, x2, y2], tr["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1

            new_tracks[best_id] = {"box": [x1, y1, x2, y2], "age": 0}
            used_ids.add(best_id)

        self._simple_tracks = new_tracks
        return [(tid, *map(int, tr["box"])) for tid, tr in new_tracks.items()]

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0
