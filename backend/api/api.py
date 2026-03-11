"""
=============================================================
CLASSROOM STUDENT ATTENTIVENESS DETECTION SYSTEM
FastAPI REST Backend — with /save_records endpoint
=============================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import logging

from backend.database import db

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Student Attentiveness API",
    description="Real-time classroom attentiveness monitoring REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── On startup ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    db.init_db()
    logger.info("Database initialised.")


# ── Models ─────────────────────────────────────────────────
class AttentivenessRecord(BaseModel):
    student_id: str
    student_name: str
    emotion: Optional[str] = "neutral"
    eye_state: Optional[str] = "open"
    head_pose: Optional[str] = "forward"
    body_pose: Optional[str] = "upright"
    attention_score: float
    attention_label: str
    timestamp: Optional[str] = None


class RecordsBatch(BaseModel):
    records: List[AttentivenessRecord]


# ── Endpoints ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "attentiveness-api"}


@app.post("/save_records")
def save_records(batch: RecordsBatch):
    """Receive attentiveness records from the classroom detection PC."""
    try:
        rows = []
        for r in batch.records:
            rows.append({
                "student_id": r.student_id,
                "student_name": r.student_name,
                "emotion": r.emotion,
                "eye_state": r.eye_state,
                "head_pose": r.head_pose,
                "body_pose": r.body_pose,
                "attention_score": r.attention_score,
                "attention_label": r.attention_label,
                "timestamp": datetime.fromisoformat(r.timestamp)
                             if r.timestamp else datetime.now(timezone.utc),
            })
        db.bulk_save_records(rows)
        return {"status": "saved", "count": len(rows)}
    except Exception as e:
        logger.error("Error saving records: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/attention")
def get_attention(limit: int = 50):
    """Get latest attentiveness records."""
    try:
        return db.get_latest_records(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/class_attention")
def get_class_attention():
    """Get aggregated class attention for last 30 seconds."""
    try:
        return db.get_class_summary(seconds=30)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/student/{student_id}")
def get_student(student_id: str, minutes: int = 10):
    """Get attention history for a specific student."""
    try:
        return db.get_student_history(student_id=student_id, minutes=minutes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def get_history(minutes: int = 10):
    """Get full class history."""
    try:
        return db.get_history(minutes=minutes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
