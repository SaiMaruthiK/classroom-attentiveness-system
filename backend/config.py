"""
=============================================================
CLASSROOM STUDENT ATTENTIVENESS DETECTION SYSTEM
Configuration Module
=============================================================
"""

import os
import torch

# ─── Base Paths ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "students")
DATABASE_PATH = os.path.join(BASE_DIR, "attentiveness.db")

# ─── Device ───────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = torch.cuda.is_available()

# ─── Camera / Video ───────────────────────────────────────
CAMERA_INDEX = 0            # 0 = default webcam, change for IP cam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 20
FRAME_SKIP = 2              # process every Nth frame

# ─── YOLOv8 Face Detector ─────────────────────────────────
YOLO_MODEL = "yolov8n.pt"   # nano = fastest; swap for yolov8s/m for accuracy
YOLO_CONF = 0.40
YOLO_IOU = 0.45
YOLO_CLASSES = [0]          # person class; replace with face-specific model

# ─── Face Recognition ─────────────────────────────────────
FACE_ENCODINGS_PATH = os.path.join(MODELS_DIR, "face_encodings.pkl")
FACE_RECOGNITION_TOLERANCE = 0.50
FACE_RECOGNITION_MODEL = "hog"   # "hog" (CPU) or "cnn" (GPU)

# ─── Emotion Detection ────────────────────────────────────
EMOTION_DETECTION_ENABLED = True

# ─── Eye Blink ────────────────────────────────────────────
EAR_THRESHOLD = 0.25        # Eye Aspect Ratio below = closed
EAR_CONSEC_FRAMES = 2       # consecutive frames below threshold = blink

# ─── Head Pose ────────────────────────────────────────────
HEAD_POSE_YAW_THRESHOLD = 20    # degrees
HEAD_POSE_PITCH_THRESHOLD = 15  # degrees

# ─── Attention Score Weights ──────────────────────────────
WEIGHT_HEAD_POSE = 0.35
WEIGHT_EYE = 0.25
WEIGHT_EMOTION = 0.20
WEIGHT_BODY_POSE = 0.20

# ─── Attention Classification ─────────────────────────────
ATTENTION_SLEEPING = 0.3
ATTENTION_DISTRACTED = 0.6
# >= 0.6 → Attentive

# ─── API ──────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Tracking ─────────────────────────────────────────────
TRACKER_MAX_AGE = 30
TRACKER_N_INIT = 3
TRACKER_MAX_COSINE_DISTANCE = 0.4

# ─── Database ─────────────────────────────────────────────
DB_URL = f"sqlite:///{DATABASE_PATH}"
DB_BATCH_SIZE = 50
DB_SAVE_INTERVAL = 1.0      # seconds

# ─── Dashboard ────────────────────────────────────────────
DASHBOARD_REFRESH_RATE = 2  # seconds
HISTORY_WINDOW_MINUTES = 10

# ─── Logging ──────────────────────────────────────────────
LOG_LEVEL = "INFO"
