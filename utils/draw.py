"""
Drawing Utilities — overlays on video frames
"""

import cv2
import numpy as np
from typing import Dict, Any


COLORS = {
    "Attentive":  (0, 200, 0),    # green
    "Distracted": (0, 165, 255),  # orange
    "Sleeping":   (0, 0, 220),    # red
    "Unknown":    (180, 180, 180),
}


def draw_student_overlay(
    frame: np.ndarray,
    box: tuple,
    student_info: Dict[str, Any],
) -> np.ndarray:
    """Draw bounding box + HUD over a single student."""
    x1, y1, x2, y2 = box
    label = student_info.get("attention_label", "Unknown")
    color = COLORS.get(label, COLORS["Unknown"])
    score = student_info.get("attention_score", 0.0)
    name = student_info.get("student_name", "?")
    emotion = student_info.get("emotion", "-")
    eye = student_info.get("eye_state", "-")
    pose = student_info.get("head_pose", "-")

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Top label bar
    bar_h = 22
    cv2.rectangle(frame, (x1, y1 - bar_h), (x2, y1), color, -1)
    cv2.putText(
        frame, f"{name}  {label} ({score:.2f})",
        (x1 + 4, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )

    # Info strip below face
    info = f"E:{emotion[:4]}  Eye:{eye[:3]}  H:{pose[:3]}"
    cv2.putText(
        frame, info,
        (x1 + 4, y2 + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA
    )

    # Attention bar
    bar_w = x2 - x1
    filled = int(bar_w * score)
    cv2.rectangle(frame, (x1, y2 + 18), (x2, y2 + 24), (50, 50, 50), -1)
    cv2.rectangle(frame, (x1, y2 + 18), (x1 + filled, y2 + 24), color, -1)

    return frame


def draw_class_hud(frame: np.ndarray, summary: Dict[str, Any]) -> np.ndarray:
    """Draw class-wide statistics in top-left corner."""
    total = summary.get("total_students", 0)
    attentive = summary.get("attentive", 0)
    distracted = summary.get("distracted", 0)
    sleeping = summary.get("sleeping", 0)
    eng = summary.get("engagement_pct", 0.0)

    lines = [
        f"Students: {total}",
        f"Attentive: {attentive}",
        f"Distracted: {distracted}",
        f"Sleeping: {sleeping}",
        f"Engagement: {eng:.1f}%",
    ]

    panel_h = len(lines) * 20 + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (180, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (8, 20 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
        )
    return frame
