"""
=============================================================
STUDENT SIMULATION SCRIPT
Generates realistic fake attentiveness data for demo/testing
without a physical camera.
=============================================================
Usage:
    python simulate_students.py --students 8 --duration 120
"""

import time
import random
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

from backend.database import db
from backend.attention.attention_score import compute_attention_score, classify_attention

STUDENT_PROFILES = [
    {"id": "s001", "name": "Alice Johnson",   "base_attention": 0.85, "variability": 0.10},
    {"id": "s002", "name": "Bob Smith",       "base_attention": 0.60, "variability": 0.20},
    {"id": "s003", "name": "Carol White",     "base_attention": 0.75, "variability": 0.15},
    {"id": "s004", "name": "David Brown",     "base_attention": 0.45, "variability": 0.25},
    {"id": "s005", "name": "Eva Martinez",    "base_attention": 0.90, "variability": 0.08},
    {"id": "s006", "name": "Frank Lee",       "base_attention": 0.30, "variability": 0.20},
    {"id": "s007", "name": "Grace Kim",       "base_attention": 0.70, "variability": 0.15},
    {"id": "s008", "name": "Henry Zhang",     "base_attention": 0.55, "variability": 0.20},
    {"id": "s009", "name": "Isla Patel",      "base_attention": 0.80, "variability": 0.12},
    {"id": "s010", "name": "Jake Wilson",     "base_attention": 0.40, "variability": 0.22},
]

EMOTIONS = ["happy", "neutral", "neutral", "neutral", "sad", "angry", "surprise"]


def simulate(n_students: int = 8, duration: int = 120, interval: float = 1.0):
    db.init_db()
    students = STUDENT_PROFILES[:n_students]

    for s in students:
        db.upsert_student_profile(s["id"], s["name"])
    logger.info("Simulating %d students for %ds", n_students, duration)

    start = time.time()
    tick = 0

    while time.time() - start < duration:
        records = []
        for s in students:
            score = random.gauss(s["base_attention"], s["variability"])
            score = max(0.0, min(1.0, score))

            # Random signals consistent with score
            head_pose = "forward" if score > 0.4 else random.choice(["forward", "away"])
            eye_state = "open" if score > 0.25 else "closed"
            emotion = random.choice(EMOTIONS)

            label = classify_attention(score)
            records.append({
                "student_id": s["id"],
                "student_name": s["name"],
                "emotion": emotion,
                "eye_state": eye_state,
                "head_pose": head_pose,
                "body_pose": "upright" if score > 0.5 else "slouched",
                "attention_score": round(score, 3),
                "attention_label": label,
                "timestamp": datetime.utcnow(),
            })

        db.bulk_save_records(records)
        tick += 1
        if tick % 10 == 0:
            elapsed = time.time() - start
            logger.info("Tick %d | %.1fs elapsed | %d students simulated", tick, elapsed, n_students)

        time.sleep(interval)

    logger.info("Simulation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", type=int, default=8)
    parser.add_argument("--duration", type=int, default=120, help="seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between ticks")
    args = parser.parse_args()
    simulate(args.students, args.duration, args.interval)
