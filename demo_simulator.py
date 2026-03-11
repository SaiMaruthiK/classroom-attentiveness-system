"""
demo_simulator.py
Simulates 8 students with randomised attentiveness data and writes
records to the database so the API and dashboard can be tested
without a physical camera.

Run:
    python demo_simulator.py
"""

from __future__ import annotations

import os
import sys
import time
import random
import math
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from backend.database.db import init_db, bulk_insert_records, upsert_student_profile

# ─────────────────────────────────────────────
# Student roster
# ─────────────────────────────────────────────

STUDENTS = [
    ("alice_johnson",  "Alice Johnson"),
    ("bob_smith",      "Bob Smith"),
    ("carol_white",    "Carol White"),
    ("david_lee",      "David Lee"),
    ("emma_brown",     "Emma Brown"),
    ("frank_garcia",   "Frank Garcia"),
    ("grace_kim",      "Grace Kim"),
    ("henry_patel",    "Henry Patel"),
]

EMOTIONS   = ["happy", "neutral", "neutral", "sad", "surprise", "neutral"]
EYE_STATES = ["open", "open", "open", "closed"]
HEAD_POSES = ["forward", "forward", "forward", "away"]


def simulate_score(base: float, noise: float = 0.15) -> float:
    s = base + random.uniform(-noise, noise)
    return max(0.0, min(1.0, s))


def label(score: float) -> str:
    if score >= 0.6:
        return "Attentive"
    if score >= 0.3:
        return "Distracted"
    return "Sleeping"


def run(interval_secs: float = 2.0) -> None:
    init_db()
    print("Demo simulator started. Press Ctrl-C to stop.")

    # Register all students
    for sid, name in STUDENTS:
        upsert_student_profile(sid, name)

    # Per-student base attention (slow drift)
    bases = {sid: random.uniform(0.5, 0.95) for sid, _ in STUDENTS}

    tick = 0
    while True:
        tick += 1
        records = []
        for sid, name in STUDENTS:
            # Slowly drift the base attention up/down
            bases[sid] += random.uniform(-0.03, 0.03)
            bases[sid] = max(0.1, min(0.98, bases[sid]))

            score = simulate_score(bases[sid])
            emotion   = random.choice(EMOTIONS)
            eye_state = random.choices(EYE_STATES, weights=[8, 8, 8, 1])[0]
            head_pose = random.choices(HEAD_POSES, weights=[8, 8, 8, 1])[0]
            yaw   = random.uniform(-15, 15) if head_pose == "forward" else random.uniform(25, 45)
            pitch = random.uniform(-10, 10) if head_pose == "forward" else random.uniform(-30, -15)

            records.append({
                "student_id":      sid,
                "student_name":    name,
                "emotion":         emotion,
                "eye_state":       eye_state,
                "head_pose":       head_pose,
                "head_yaw":        round(yaw, 2),
                "head_pitch":      round(pitch, 2),
                "attention_score": round(score, 4),
                "attention_label": label(score),
                "body_pose":       "upright" if pitch > -20 else "slouched",
                "timestamp":       datetime.utcnow(),
            })

        bulk_insert_records(records)

        # Print summary
        attentive  = sum(1 for r in records if r["attention_label"] == "Attentive")
        distracted = sum(1 for r in records if r["attention_label"] == "Distracted")
        sleeping   = sum(1 for r in records if r["attention_label"] == "Sleeping")
        avg        = sum(r["attention_score"] for r in records) / len(records)
        print(f"[Tick {tick:4d}]  Attentive:{attentive}  Distracted:{distracted}  "
              f"Sleeping:{sleeping}  AvgScore:{avg:.3f}")

        time.sleep(interval_secs)


if __name__ == "__main__":
    run()
