"""
=============================================================
CLASSROOM STUDENT ATTENTIVENESS DETECTION SYSTEM
Main Detection Pipeline — Cloud-connected version
=============================================================
Run:  python backend/main.py
"""

import time
import logging
import threading
import queue
import os
from datetime import datetime
from typing import Dict, Any, List

import cv2
import numpy as np
import requests

import backend.config as cfg
from utils.helpers import setup_logging, FPSCounter, resize_frame, crop_face
from utils.draw import draw_student_overlay, draw_class_hud

from backend.detection.face_detector import FaceDetector
from backend.detection.face_tracker import FaceTracker
from backend.recognition.face_recognition_module import FaceRecognizer
from backend.attention.emotion_detector import EmotionDetector
from backend.attention.head_pose import HeadPoseEstimator
from backend.attention.blink_detection import BlinkDetector
from backend.attention.attention_score import compute_attention_score, classify_attention
from backend.database import db

setup_logging(cfg.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ── Cloud API URL ─────────────────────────────────────────
# Set this to your Railway URL or leave blank to use local DB only
CLOUD_API_URL = os.environ.get(
    "CLOUD_API_URL",
    "https://web-production-a77df.up.railway.app"
)
USE_CLOUD = bool(CLOUD_API_URL)


def push_to_cloud(records: List[Dict]):
    """Push attentiveness records to cloud API."""
    if not USE_CLOUD:
        return
    try:
        resp = requests.post(
            f"{CLOUD_API_URL}/save_records",
            json={"records": records},
            timeout=3,
        )
        if resp.status_code != 200:
            logger.debug("Cloud push status: %s", resp.status_code)
    except Exception as e:
        logger.debug("Cloud push failed: %s", e)


class AttentivenessSystem:
    """
    Orchestrates the full detection → recognition → analysis → storage pipeline.
    Saves to local DB and pushes to cloud API simultaneously.
    """

    def __init__(self):
        logger.info("Initializing Attentiveness System on device: %s", cfg.DEVICE)

        # ── Init DB ────────────────────────────────────────
        db.init_db()

        # ── Detection / Tracking ───────────────────────────
        self.detector = FaceDetector(
            model_path=cfg.YOLO_MODEL, conf=cfg.YOLO_CONF, device=cfg.DEVICE
        )
        self.tracker = FaceTracker(
            max_age=cfg.TRACKER_MAX_AGE,
            n_init=cfg.TRACKER_N_INIT,
            max_cosine_distance=cfg.TRACKER_MAX_COSINE_DISTANCE,
        )

        # ── Recognition ────────────────────────────────────
        self.recognizer = FaceRecognizer(
            encodings_path=cfg.FACE_ENCODINGS_PATH,
            tolerance=cfg.FACE_RECOGNITION_TOLERANCE,
            model=cfg.FACE_RECOGNITION_MODEL,
        )

        # ── Attention Modules ──────────────────────────────
        self.emotion_detector = EmotionDetector()
        self.head_pose = HeadPoseEstimator()
        self.blink_detector = BlinkDetector(
            ear_threshold=cfg.EAR_THRESHOLD,
            consec_frames=cfg.EAR_CONSEC_FRAMES,
        )

        # ── State ──────────────────────────────────────────
        self.student_cache: Dict[int, Dict[str, Any]] = {}
        self.db_queue: queue.Queue = queue.Queue(maxsize=500)
        self.cloud_queue: queue.Queue = queue.Queue(maxsize=500)
        self.fps_counter = FPSCounter()
        self._running = False
        self._db_thread = threading.Thread(target=self._db_writer, daemon=True)
        self._cloud_thread = threading.Thread(target=self._cloud_writer, daemon=True)

    # ── DB Writer Thread ───────────────────────────────────
    def _db_writer(self):
        batch: List[Dict] = []
        last_flush = time.time()
        while self._running or not self.db_queue.empty():
            try:
                record = self.db_queue.get(timeout=0.5)
                batch.append(record)
                if len(batch) >= cfg.DB_BATCH_SIZE or (time.time() - last_flush) >= cfg.DB_SAVE_INTERVAL:
                    db.bulk_save_records(batch)
                    batch.clear()
                    last_flush = time.time()
            except queue.Empty:
                if batch:
                    db.bulk_save_records(batch)
                    batch.clear()

    # ── Cloud Writer Thread ────────────────────────────────
    def _cloud_writer(self):
        batch: List[Dict] = []
        last_flush = time.time()
        while self._running or not self.cloud_queue.empty():
            try:
                record = self.cloud_queue.get(timeout=0.5)
                # Convert datetime to string for JSON
                r = dict(record)
                if isinstance(r.get("timestamp"), datetime):
                    r["timestamp"] = r["timestamp"].isoformat()
                batch.append(r)
                if len(batch) >= 10 or (time.time() - last_flush) >= 2.0:
                    push_to_cloud(batch)
                    batch.clear()
                    last_flush = time.time()
            except queue.Empty:
                if batch:
                    push_to_cloud(batch)
                    batch.clear()

    # ── Per-Frame Processing ───────────────────────────────
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)

        class_summary: Dict[str, int] = {"Attentive": 0, "Distracted": 0, "Sleeping": 0}

        for track in tracks:
            tid, x1, y1, x2, y2 = track
            box = (x1, y1, x2, y2)

            if tid not in self.student_cache:
                sid, sname = self.recognizer.recognize(frame, box, tid)
                self.student_cache[tid] = {"student_id": sid, "student_name": sname}
                db.upsert_student_profile(sid, sname)
            else:
                sid = self.student_cache[tid]["student_id"]
                sname = self.student_cache[tid]["student_name"]

            face_crop = crop_face(frame, box)
            emotion, _ = self.emotion_detector.detect(face_crop)
            head_pose_label, yaw, pitch, _ = self.head_pose.estimate(frame, box)
            eye_state, ear = self.blink_detector.detect(frame, box, int(tid))
            score = compute_attention_score(head_pose_label, eye_state, emotion)
            label = classify_attention(score)

            now = datetime.utcnow()
            record = {
                "student_id": sid,
                "student_name": sname,
                "emotion": emotion,
                "eye_state": eye_state,
                "head_pose": head_pose_label,
                "body_pose": "upright" if score > 0.5 else "slouched",
                "attention_score": score,
                "attention_label": label,
                "timestamp": now,
            }

            # Save locally
            try:
                self.db_queue.put_nowait(record)
            except queue.Full:
                pass

            # Push to cloud
            try:
                self.cloud_queue.put_nowait(record)
            except queue.Full:
                pass

            student_info = {
                "student_name": sname,
                "attention_score": score,
                "attention_label": label,
                "emotion": emotion,
                "eye_state": eye_state,
                "head_pose": head_pose_label,
            }
            draw_student_overlay(frame, box, student_info)
            class_summary[label] = class_summary.get(label, 0) + 1

        total = len(tracks)
        att = class_summary.get("Attentive", 0)
        summary = {
            "total_students": total,
            "attentive": att,
            "distracted": class_summary.get("Distracted", 0),
            "sleeping": class_summary.get("Sleeping", 0),
            "engagement_pct": (att / total * 100) if total else 0.0,
        }
        draw_class_hud(frame, summary)

        fps = self.fps_counter.tick()
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (frame.shape[1] - 90, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )
        return frame

    # ── Main Loop ─────────────────────────────────────────
    def run(self, source=None):
        if source is None:
            source = cfg.CAMERA_INDEX

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Cannot open video source: %s", source)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, cfg.TARGET_FPS)

        self._running = True
        self._db_thread.start()
        self._cloud_thread.start()

        if USE_CLOUD:
            logger.info("Cloud sync enabled → %s", CLOUD_API_URL)
        else:
            logger.info("Cloud sync disabled — local DB only")

        logger.info("Detection pipeline started. Press 'q' to quit.")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = resize_frame(frame, cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT)

                if frame_idx % cfg.FRAME_SKIP == 0:
                    frame = self.process_frame(frame, frame_idx)

                cv2.imshow("Classroom Attentiveness Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_idx += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._running = False
            cap.release()
            cv2.destroyAllWindows()
            logger.info("System shut down cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classroom Attentiveness System")
    parser.add_argument("--source", default=None,
                        help="Video source: camera index or file path")
    args = parser.parse_args()

    source = args.source
    if source and source.isdigit():
        source = int(source)

    system = AttentivenessSystem()
    system.run(source=source)
