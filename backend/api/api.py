"""
FastAPI REST Backend
Endpoints:
  GET /attention          → current snapshot
  GET /class_attention    → class summary
  GET /student/{id}       → student history
  GET /history            → full recent history
  GET /health             → health check
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from backend.database import db

app = FastAPI(
    title="Student Attentiveness API",
    description="Real-time classroom attentiveness monitoring REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
@app.on_event("startup")
def startup():
    db.init_db()


@app.get("/health")
def health():
    return {"status": "ok", "service": "attentiveness-api"}


@app.get("/attention")
def get_attention(limit: int = Query(default=50, le=500)):
    """Latest N attentiveness records."""
    records = db.get_latest_records(limit=limit)
    return {"count": len(records), "records": records}


@app.get("/class_attention")
def get_class_attention():
    """Aggregated class summary (last 30 seconds)."""
    summary = db.get_current_attention_summary()
    return summary


@app.get("/student/{student_id}")
def get_student(student_id: str, minutes: int = Query(default=10, le=60)):
    """History for a specific student."""
    history = db.get_student_history(student_id, minutes=minutes)
    if not history:
        return {"student_id": student_id, "records": [], "message": "No data found"}

    avg_score = sum(r["attention_score"] for r in history) / len(history)
    return {
        "student_id": student_id,
        "student_name": history[0]["student_name"] if history else student_id,
        "record_count": len(history),
        "avg_attention_score": round(avg_score, 3),
        "records": history,
    }


@app.get("/history")
def get_history(minutes: int = Query(default=10, le=60)):
    """Class history over the last N minutes."""
    records = db.get_class_history(minutes=minutes)
    return {"minutes": minutes, "count": len(records), "records": records}
