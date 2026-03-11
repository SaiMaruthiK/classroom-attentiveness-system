"""
Database Layer — SQLAlchemy + SQLite
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from backend.database.models import Base, AttentivenessRecord, StudentProfile
import backend.config as cfg

logger = logging.getLogger(__name__)

# ─── Engine ───────────────────────────────────────────────
engine = create_engine(
    cfg.DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at %s", cfg.DATABASE_PATH)


def get_db() -> Session:
    """Dependency for FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Write ────────────────────────────────────────────────

def save_record(data: Dict[str, Any]) -> AttentivenessRecord:
    db = SessionLocal()
    try:
        record = AttentivenessRecord(**data)
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except Exception as e:
        db.rollback()
        logger.error("save_record error: %s", e)
    finally:
        db.close()


def bulk_save_records(records: List[Dict[str, Any]]):
    db = SessionLocal()
    try:
        db.bulk_insert_mappings(AttentivenessRecord, records)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("bulk_save_records error: %s", e)
    finally:
        db.close()


def upsert_student_profile(student_id: str, student_name: str):
    db = SessionLocal()
    try:
        profile = db.query(StudentProfile).filter_by(student_id=student_id).first()
        if not profile:
            profile = StudentProfile(student_id=student_id, student_name=student_name)
            db.add(profile)
        else:
            profile.total_sessions += 1
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("upsert_student_profile error: %s", e)
    finally:
        db.close()


# ─── Read ─────────────────────────────────────────────────

def get_latest_records(limit: int = 100) -> List[Dict]:
    db = SessionLocal()
    try:
        rows = (
            db.query(AttentivenessRecord)
            .order_by(desc(AttentivenessRecord.timestamp))
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]
    finally:
        db.close()


def get_current_attention_summary() -> Dict[str, Any]:
    """Aggregate last 30 seconds of data."""
    db = SessionLocal()
    cutoff = datetime.utcnow() - timedelta(seconds=30)
    try:
        rows = (
            db.query(
                AttentivenessRecord.student_id,
                AttentivenessRecord.student_name,
                func.avg(AttentivenessRecord.attention_score).label("avg_score"),
                func.max(AttentivenessRecord.emotion).label("emotion"),
                func.max(AttentivenessRecord.attention_label).label("label"),
            )
            .filter(AttentivenessRecord.timestamp >= cutoff)
            .group_by(AttentivenessRecord.student_id)
            .all()
        )

        students = [
            {
                "student_id": r.student_id,
                "student_name": r.student_name,
                "avg_attention_score": round(r.avg_score, 3),
                "emotion": r.emotion,
                "attention_label": r.label,
            }
            for r in rows
        ]

        total = len(students)
        attentive = sum(1 for s in students if s["attention_label"] == "Attentive")
        distracted = sum(1 for s in students if s["attention_label"] == "Distracted")
        sleeping = sum(1 for s in students if s["attention_label"] == "Sleeping")
        class_avg = (
            round(sum(s["avg_attention_score"] for s in students) / total, 3)
            if total else 0.0
        )

        return {
            "total_students": total,
            "attentive": attentive,
            "distracted": distracted,
            "sleeping": sleeping,
            "class_avg_attention": class_avg,
            "engagement_pct": round(attentive / total * 100, 1) if total else 0.0,
            "students": students,
        }
    finally:
        db.close()


def get_student_history(student_id: str, minutes: int = 10) -> List[Dict]:
    db = SessionLocal()
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    try:
        rows = (
            db.query(AttentivenessRecord)
            .filter(
                AttentivenessRecord.student_id == student_id,
                AttentivenessRecord.timestamp >= cutoff,
            )
            .order_by(AttentivenessRecord.timestamp)
            .all()
        )
        return [r.to_dict() for r in rows]
    finally:
        db.close()


def get_class_history(minutes: int = 10) -> List[Dict]:
    db = SessionLocal()
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    try:
        rows = (
            db.query(AttentivenessRecord)
            .filter(AttentivenessRecord.timestamp >= cutoff)
            .order_by(AttentivenessRecord.timestamp)
            .all()
        )
        return [r.to_dict() for r in rows]
    finally:
        db.close()
