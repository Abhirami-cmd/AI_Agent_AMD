import streamlit as st
import pandas as pd
from typing import List, Dict, Any


# -----------------------------
# Convert telemetry → story steps
# -----------------------------
def build_story_steps(telemetry: pd.DataFrame, incident_id: str) -> List[Dict[str, Any]]:
    df = telemetry.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df[df.get("incident_id", incident_id) == incident_id] if "incident_id" in df.columns else df

    df = df.sort_values("timestamp") if "timestamp" in df.columns else df

    steps = []
    for _, row in df.iterrows():
        baseline = float(row.get("baseline", 0) or 0)
        value = float(row.get("value", 0) or 0)

        anomaly_ratio = (
            abs(value - baseline) / (abs(baseline) + 1e-6)
            if baseline != 0
            else value
        )

        tower = row.get("tower", "unknown")
        signal = row.get("signal", "unknown")

        steps.append({
            "timestamp": str(row.get("timestamp", "")),
            "title": f"{tower} anomaly detected in {signal}",
            "description": (
                f"{row.get('component', 'component')} shows abnormal behavior. "
                f"Value={value}, Baseline={baseline}, Unit={row.get('unit','')}"
            ),
            "tower": tower,
            "component": row.get("component", ""),
            "signal": signal,
            "value": value,
            "baseline": baseline,
            "score": round(anomaly_ratio, 3),
        })

    return steps


# -----------------------------
# UI Renderer
# -----------------------------
def render_incident_story_tab(telemetry: pd.DataFrame, incident_id: str):
    st.header("📖 Incident Story Mode")

    if telemetry is None or telemetry.empty:
        st.warning("No telemetry available for story mode.")
        return

    steps = build_story_steps(telemetry, incident_id)

    if not steps:
        st.info("No story steps found for this incident.")
        return

    # -----------------------------
    # Controls
    # -----------------------------
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        step_index = st.slider(
            "Replay Timeline",
            0,
            len(steps) - 1,
            0,
            key="story_slider"
        )

    with col2:
        if st.button("▶ Play"):
            for i in range(len(steps)):
                st.session_state.story_slider = i
                st.rerun()

    with col3:
        if st.button("⏹ Reset"):
            st.session_state.story_slider = 0
            st.rerun()

    step = steps[step_index]

    # -----------------------------
    # Visual Summary
    # -----------------------------
    st.subheader(f"Step {step_index + 1} / {len(steps)}")

    colA, colB, colC = st.columns(3)

    tower_color = {
        "Network": "🔴",
        "Compute": "🟠",
        "Application": "🟡",
        "Storage": "🔵",
    }

    with colA:
        st.metric("Tower", f"{tower_color.get(step['tower'], '⚪')} {step['tower']}")

    with colB:
        st.metric("Signal", step["signal"])

    with colC:
        st.metric("Anomaly Score", step["score"])

    # -----------------------------
    # Narrative panel
    # -----------------------------
    st.markdown("### 📌 What is happening?")
    st.write(step["description"])

    st.markdown("### 🧩 Technical Details")
    st.json({
        "timestamp": step["timestamp"],
        "component": step["component"],
        "value": step["value"],
        "baseline": step["baseline"],
    })

    # -----------------------------
    # Simple timeline preview
    # -----------------------------
    st.markdown("### ⏱ Incident Flow")

    for i, s in enumerate(steps):
        prefix = "👉" if i == step_index else "•"
        st.write(
            f"{prefix} {s['timestamp']} | {s['tower']} | {s['signal']} | score={s['score']}"
        )