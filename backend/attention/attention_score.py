"""
Attention Score Calculator
Weighted formula:
  AttentionScore = 0.35*HeadPose + 0.25*Eye + 0.20*Emotion + 0.20*BodyPose
"""

import backend.config as cfg
from backend.attention.emotion_detector import EmotionDetector
from backend.attention.head_pose import HeadPoseEstimator
from backend.attention.blink_detection import BlinkDetector


def classify_attention(score: float) -> str:
    """Map 0-1 score to label."""
    if score < cfg.ATTENTION_SLEEPING:
        return "Sleeping"
    elif score < cfg.ATTENTION_DISTRACTED:
        return "Distracted"
    return "Attentive"


def estimate_body_pose_score(head_pose_label: str, eye_state: str) -> float:
    """
    Proxy body pose score from available signals.
    In a full deployment, a skeleton pose model (e.g. YOLOv8-pose) would be used.
    """
    # Heuristic: if head is forward and eyes open → likely upright
    if head_pose_label == "forward" and eye_state == "open":
        return 1.0
    elif head_pose_label == "away":
        return 0.5
    return 0.7


def compute_attention_score(
    head_pose_label: str,
    eye_state: str,
    emotion: str,
    body_pose_score: float = None,
) -> float:
    """
    Returns attention score in [0, 1].
    """
    head_score = HeadPoseEstimator.pose_to_score(head_pose_label)
    eye_score = BlinkDetector.eye_state_to_score(eye_state)
    emotion_score = EmotionDetector.emotion_to_score(emotion)

    if body_pose_score is None:
        body_pose_score = estimate_body_pose_score(head_pose_label, eye_state)

    score = (
        cfg.WEIGHT_HEAD_POSE * head_score
        + cfg.WEIGHT_EYE * eye_score
        + cfg.WEIGHT_EMOTION * emotion_score
        + cfg.WEIGHT_BODY_POSE * body_pose_score
    )
    return round(min(max(score, 0.0), 1.0), 3)
