"""
=============================================================
CLASSROOM STUDENT ATTENTIVENESS DETECTION SYSTEM
Streamlit Real-Time Dashboard
=============================================================
Run:  streamlit run dashboard/dashboard.py
"""

import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Config ────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
REFRESH_RATE = 2  # seconds

st.set_page_config(
    page_title="Classroom Attentiveness Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ───────────────────────────────────────────────
def fetch(endpoint: str, params: dict = None):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return None


def status_color(label: str) -> str:
    return {"Attentive": "🟢", "Distracted": "🟠", "Sleeping": "🔴"}.get(label, "⚪")


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/classroom.png", width=80)
    st.title("Attentiveness\nMonitor")
    st.markdown("---")
    history_mins = st.slider("History window (mins)", 1, 30, 10)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_sec = st.slider("Refresh interval (s)", 1, 10, REFRESH_RATE)
    st.markdown("---")
    st.markdown("**API Status**")
    health = fetch("/health")
    if health:
        st.success("✅ API Online")
    else:
        st.error("❌ API Offline — start the backend first")

# ── Main Layout ───────────────────────────────────────────
st.title("🎓 Classroom Student Attentiveness Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ── Top KPI Cards ─────────────────────────────────────────
summary = fetch("/class_attention") or {}
total = summary.get("total_students", 0)
attentive = summary.get("attentive", 0)
distracted = summary.get("distracted", 0)
sleeping = summary.get("sleeping", 0)
engagement = summary.get("engagement_pct", 0.0)
class_avg = summary.get("class_avg_attention", 0.0)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("👥 Total Students", total)
col2.metric("🟢 Attentive", attentive)
col3.metric("🟠 Distracted", distracted)
col4.metric("🔴 Sleeping", sleeping)
col5.metric("📊 Engagement", f"{engagement:.1f}%")
col6.metric("🎯 Class Avg Score", f"{class_avg:.2f}")

st.markdown("---")

# ── Pie Chart + Bar Chart ─────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.subheader("📊 Attention Distribution")
    if total > 0:
        pie_data = {
            "Status": ["Attentive", "Distracted", "Sleeping"],
            "Count": [attentive, distracted, sleeping],
        }
        fig_pie = px.pie(
            pd.DataFrame(pie_data),
            names="Status", values="Count",
            color="Status",
            color_discrete_map={
                "Attentive": "#22c55e",
                "Distracted": "#f97316",
                "Sleeping": "#ef4444",
            },
            hole=0.45,
        )
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No live data yet.")

with right:
    st.subheader("📈 Attention Score per Student")
    students = summary.get("students", [])
    if students:
        df_bar = pd.DataFrame(students).sort_values("avg_attention_score", ascending=True)
        fig_bar = px.bar(
            df_bar,
            x="avg_attention_score",
            y="student_name",
            orientation="h",
            color="attention_label",
            color_discrete_map={
                "Attentive": "#22c55e",
                "Distracted": "#f97316",
                "Sleeping": "#ef4444",
            },
            range_x=[0, 1],
            labels={"avg_attention_score": "Score", "student_name": "Student"},
            text="avg_attention_score",
        )
        fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_bar.update_layout(
            showlegend=True, margin=dict(t=10, b=10), height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No student data yet.")

st.markdown("---")

# ── Time Series ───────────────────────────────────────────
st.subheader("📉 Attention Over Time")
hist_data = fetch("/history", params={"minutes": history_mins}) or {}
records = hist_data.get("records", [])

if records:
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig_line = px.line(
        df,
        x="timestamp",
        y="attention_score",
        color="student_name",
        markers=True,
        labels={"attention_score": "Score", "timestamp": "Time", "student_name": "Student"},
        range_y=[0, 1],
    )
    fig_line.add_hline(y=0.6, line_dash="dash", line_color="green",
                       annotation_text="Attentive threshold")
    fig_line.add_hline(y=0.3, line_dash="dash", line_color="red",
                       annotation_text="Sleeping threshold")
    fig_line.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("No historical data yet. Detection pipeline must be running.")

st.markdown("---")

# ── Live Student Table ────────────────────────────────────
st.subheader("🧑‍🎓 Live Student Status")
if students:
    table_rows = []
    for s in students:
        table_rows.append({
            "Status": status_color(s.get("attention_label", "?")),
            "Name": s.get("student_name", s["student_id"]),
            "Score": f"{s['avg_attention_score']:.2f}",
            "Label": s.get("attention_label", "?"),
            "Emotion": s.get("emotion", "-"),
        })
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

# ── Emotion Breakdown ─────────────────────────────────────
if records:
    st.markdown("---")
    st.subheader("😶 Emotion Breakdown")
    df_emo = pd.DataFrame(records)
    emo_counts = df_emo["emotion"].value_counts().reset_index()
    emo_counts.columns = ["Emotion", "Count"]
    fig_emo = px.bar(
        emo_counts, x="Emotion", y="Count",
        color="Emotion",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_emo.update_layout(height=280, margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig_emo, use_container_width=True)

# ── Auto Refresh ──────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
