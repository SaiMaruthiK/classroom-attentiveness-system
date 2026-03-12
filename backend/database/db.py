"""
Database module — SQLite via SQLAlchemy
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import backend.config as cfg
from backend.database.models import Base, AttentivenessRecord, StudentProfile

logger = logging.getLogger(__name__)

engine = None
SessionLocal = None


def init_db():
    global engine, SessionLocal
    engine = create_engine(
        cfg.DB_URL,
        connect_args={"check_same_thread": False} if "sqlite" in cfg.DB_URL else {},
        pool_pre_ping=True,
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at %s", cfg.DB_URL)


def get_session():
    if SessionLocal is None:
        init_db()
    return SessionLocal()


def upsert_student_profile(student_id: str, student_name: str):
    session = get_session()
    try:
        profile = session.query(StudentProfile).filter_by(student_id=student_id).first()
        if not profile:
            profile = StudentProfile(student_id=student_id, student_name=student_name)
            session.add(profile)
        else:
            profile.student_name = student_name
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("upsert_student_profile error: %s", e)
    finally:
        session.close()


def bulk_save_records(records: List[Dict[str, Any]]):
    if not records:
        return
    session = get_session()
    try:
        for r in records:
            ts = r.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            elif not isinstance(ts, datetime):
                ts = datetime.utcnow()
            record = AttentivenessRecord(
                student_id=r["student_id"],
                student_name=r.get("student_name", r["student_id"]),
                emotion=r.get("emotion", "neutral"),
                eye_state=r.get("eye_state", "open"),
                head_pose=r.get("head_pose", "forward"),
                body_pose=r.get("body_pose", "upright"),
                attention_score=float(r.get("attention_score", 0.5)),
                attention_label=r.get("attention_label", "Distracted"),
                timestamp=ts,
            )
            session.add(record)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("bulk_save_records error: %s", e)
    finally:
        session.close()


def get_latest_records(limit: int = 50) -> List[Dict]:
    session = get_session()
    try:
        rows = (
            session.query(AttentivenessRecord)
            .order_by(AttentivenessRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [_record_to_dict(r) for r in rows]
    except Exception as e:
        logger.error("get_latest_records error: %s", e)
        return []
    finally:
        session.close()


def get_class_summary(seconds: int = 30) -> Dict:
    session = get_session()
    try:
        since = datetime.utcnow() - timedelta(seconds=seconds)
        rows = (
            session.query(AttentivenessRecord)
            .filter(AttentivenessRecord.timestamp >= since)
            .all()
        )
        if not rows:
            return {
                "total_students": 0, "attentive": 0,
                "distracted": 0, "sleeping": 0,
                "engagement_pct": 0.0, "class_avg_attention": 0.0,
                "students": [],
            }

        # Group by student
        student_map: Dict[str, List] = {}
        for r in rows:
            student_map.setdefault(r.student_id, []).append(r)

        students = []
        attentive = distracted = sleeping = 0
        scores = []

        for sid, recs in student_map.items():
            avg_score = sum(r.attention_score for r in recs) / len(recs)
            latest = max(recs, key=lambda x: x.timestamp)
            label = latest.attention_label
            scores.append(avg_score)

            if label == "Attentive":
                attentive += 1
            elif label == "Distracted":
                distracted += 1
            else:
                sleeping += 1

            students.append({
                "student_id": sid,
                "student_name": latest.student_name,
                "avg_attention_score": round(avg_score, 3),
                "attention_label": label,
                "emotion": latest.emotion,
            })

        total = len(students)
        return {
            "total_students": total,
            "attentive": attentive,
            "distracted": distracted,
            "sleeping": sleeping,
            "engagement_pct": round(attentive / total * 100, 1) if total else 0.0,
            "class_avg_attention": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "students": students,
        }
    except Exception as e:
        logger.error("get_class_summary error: %s", e)
        return {}
    finally:
        session.close()


def get_student_history(student_id: str, minutes: int = 10) -> List[Dict]:
    session = get_session()
    try:
        since = datetime.utcnow() - timedelta(minutes=minutes)
        rows = (
            session.query(AttentivenessRecord)
            .filter(
                AttentivenessRecord.student_id == student_id,
                AttentivenessRecord.timestamp >= since,
            )
            .order_by(AttentivenessRecord.timestamp.asc())
            .all()
        )
        return [_record_to_dict(r) for r in rows]
    except Exception as e:
        logger.error("get_student_history error: %s", e)
        return []
    finally:
        session.close()


def get_history(minutes: int = 10) -> Dict:
    session = get_session()
    try:
        since = datetime.utcnow() - timedelta(minutes=minutes)
        rows = (
            session.query(AttentivenessRecord)
            .filter(AttentivenessRecord.timestamp >= since)
            .order_by(AttentivenessRecord.timestamp.asc())
            .all()
        )
        return {"records": [_record_to_dict(r) for r in rows]}
    except Exception as e:
        logger.error("get_history error: %s", e)
        return {"records": []}
    finally:
        session.close()


def _record_to_dict(r: AttentivenessRecord) -> Dict:
    return {
        "id": r.id,
        "student_id": r.student_id,
        "student_name": r.student_name,
        "emotion": r.emotion,
        "eye_state": r.eye_state,
        "head_pose": r.head_pose,
        "body_pose": r.body_pose,
        "attention_score": r.attention_score,
        "attention_label": r.attention_label,
        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
    }
