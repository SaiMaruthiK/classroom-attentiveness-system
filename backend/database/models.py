"""
Database ORM Models
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class AttentivenessRecord(Base):
    __tablename__ = "attentiveness_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), nullable=False, index=True)
    student_name = Column(String(100), nullable=False)
    emotion = Column(String(50), default="neutral")
    eye_state = Column(String(20), default="open")   # open / closed
    head_pose = Column(String(30), default="forward") # forward / away
    body_pose = Column(String(30), default="upright") # upright / slouched
    attention_score = Column(Float, default=0.0)
    attention_label = Column(String(20), default="Attentive")  # Attentive/Distracted/Sleeping
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_student_timestamp", "student_id", "timestamp"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "student_name": self.student_name,
            "emotion": self.emotion,
            "eye_state": self.eye_state,
            "head_pose": self.head_pose,
            "body_pose": self.body_pose,
            "attention_score": round(self.attention_score, 3),
            "attention_label": self.attention_label,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class StudentProfile(Base):
    __tablename__ = "student_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), unique=True, nullable=False)
    student_name = Column(String(100), nullable=False)
    registered_at = Column(DateTime, default=datetime.utcnow)
    total_sessions = Column(Integer, default=0)
    avg_attention = Column(Float, default=0.0)

    def to_dict(self):
        return {
            "student_id": self.student_id,
            "student_name": self.student_name,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
            "total_sessions": self.total_sessions,
            "avg_attention": round(self.avg_attention, 3),
        }
