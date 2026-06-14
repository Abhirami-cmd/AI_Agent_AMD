from __future__ import annotations

import io
import re
from html import escape

import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.data_loader import load_dataset_reference_sources, load_incidents, load_telemetry
from src.models import FeedbackRequest, ReferenceSource
from src.rca_engine import filter_to_incident_window
from src.reference_loader import dynamic_reference_source, load_reference_sources
from src.rag import get_vector_store
from src.services.rca_service import RCAService
from src.vllm_client import generate_with_vllm, is_vllm_configured
from typing import Any


st.set_page_config(
    page_title="Unified Observability RCA Agent",
    page_icon=":bar_chart:",
    layout="wide",
)


def render_metric_card(label: str, value: str, help_text: str) -> None:
    st.metric(label=label, value=value, help=help_text)


def sanitize_description(description: str) -> str:
    cleaned = re.sub(r"Ground truth root cause:.*", "", description, flags=re.IGNORECASE)
    cleaned = re.sub(r"Source dataset:.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Source:.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"You are tasked with.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Identify .* root cause.*", "Investigate the observable failure and report the likely impact.", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def generate_incident_description(incident: dict[str, Any], use_vllm: bool = False) -> str:
    description = sanitize_description(str(incident.get("description", "")))
    if not use_vllm or not is_vllm_configured() or not description:
        return description

    prompt = (
        "Write a concise incident description for an operations dashboard. "
        "Use only the incident metadata and do not expose dataset instructions or ground-truth labels. "
        "Keep the description focused on the observable event and what the operator should investigate.\n\n"
        f"Incident title: {incident.get('title')}\n"
        f"Service: {incident.get('service')}\n"
        f"Severity: {incident.get('severity')}\n"
        f"Started at: {incident.get('started_at')}\n"
        f"Dependencies: {incident.get('dependencies')}\n"
        f"Description: {description}\n"
        "Return a single paragraph summary."
    )
    try:
        return generate_with_vllm(
            system_prompt="You are a dashboard writer for observability incidents.",
            user_prompt=prompt,
        ).strip()
    except Exception:
        return description


@st.cache_data
def cached_incidents() -> pd.DataFrame:
    return load_incidents(limit=10)


@st.cache_data
def cached_telemetry() -> pd.DataFrame:
    return load_telemetry()


@st.cache_data
def cached_reference_sources() -> list[dict[str, str]]:
    return load_reference_sources()


@st.cache_data
def cached_dataset_reference_sources() -> list[dict[str, str]]:
    return load_dataset_reference_sources()


@st.cache_resource
def get_cached_vector_store() -> Any:
    store = get_vector_store()
    reference_sources = cached_reference_sources() + cached_dataset_reference_sources()
    parsed_sources = [ReferenceSource(**source) for source in reference_sources]
    store.index_reference_sources(parsed_sources)
    return store


@st.cache_resource
def get_rca_service() -> RCAService:
    return RCAService()


def filtered_incident_telemetry(
    incident: dict[str, Any],
    telemetry_df: pd.DataFrame,
) -> pd.DataFrame:
    if telemetry_df is None or telemetry_df.empty:
        return pd.DataFrame()
    incident_tps = telemetry_df[telemetry_df["incident_id"] == incident["incident_id"]].copy()
    if incident_tps.empty:
        return incident_tps
    incident_tps, _ = filter_to_incident_window(incident, incident_tps)
    return incident_tps


def dependency_map_for_telemetry(
    incident: dict[str, Any],
    incident_tps: pd.DataFrame,
) -> pd.DataFrame:
    if incident_tps.empty:
        return pd.DataFrame()

    rows = []
    dependency_target = str(incident.get("incident_id") or incident.get("service") or "incident")
    for _, row in incident_tps.iterrows():
        timestamp = row.get("timestamp")
        if hasattr(timestamp, "strftime"):
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M")
        rows.append(
            {
                "source": str(row.get("component", "")),
                "dependency": dependency_target,
                "tower": str(row.get("tower", "unknown")),
                "signal": str(row.get("signal", "")),
                "timestamp": str(timestamp),
            }
        )

    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _telemetry_anomaly_score(row: pd.Series) -> float:
    gpu_score = pd.to_numeric(row.get("gpu_anomaly_score"), errors="coerce")
    if pd.notna(gpu_score):
        return float(gpu_score)
    baseline = pd.to_numeric(row.get("baseline"), errors="coerce")
    value = pd.to_numeric(row.get("value"), errors="coerce")
    if pd.isna(baseline) or pd.isna(value) or float(baseline) == 0:
        return 0.0
    return max(0.0, (float(value) - float(baseline)) / abs(float(baseline)))


def tower_summary(incident_tps: pd.DataFrame, analysis: Any | None = None) -> tuple[str, dict[str, list[str]]]:
    if analysis is not None and getattr(analysis, "evidence", None):
        tower_scores: dict[str, float] = {}
        towers: dict[str, list[str]] = {}
        for item in analysis.evidence:
            tower = str(item.tower)
            towers.setdefault(tower, []).append(str(item.signal))
            tower_scores[tower] = max(tower_scores.get(tower, 0.0), float(item.anomaly_score))
        primary = max(tower_scores, key=tower_scores.get) if tower_scores else "Pending"
        return primary, towers

    if incident_tps.empty:
        return "Pending", {}

    towers = {}
    tower_scores = {}
    for _, row in incident_tps.iterrows():
        tower = str(row.get("tower", "unknown"))
        signal = str(row.get("signal", ""))
        towers.setdefault(tower, []).append(signal)
        tower_scores[tower] = max(tower_scores.get(tower, 0.0), _telemetry_anomaly_score(row))

    primary = max(tower_scores, key=tower_scores.get) if tower_scores else "Pending"
    return primary, towers


def dataframe_to_excel_bytes(frame: pd.DataFrame, sheet_name: str) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()


def markdown_to_pdf_bytes(title: str, markdown_text: str) -> bytes:
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, title=title)
    styles = getSampleStyleSheet()
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 12)]

    for block in markdown_text.splitlines():
        text = block.strip()
        if not text:
            story.append(Spacer(1, 8))
            continue
        if text.startswith("**") and text.endswith("**"):
            story.append(Paragraph(escape(text.strip("*")), styles["Heading2"]))
        elif text.startswith("- "):
            story.append(Paragraph(f"- {escape(text[2:])}", styles["BodyText"]))
        else:
            cleaned = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", escape(text))
            story.append(Paragraph(cleaned, styles["BodyText"]))

    doc.build(story)
    return output.getvalue()


def main() -> None:
    incidents = cached_incidents()
    service = None

    if "closed_incidents" not in st.session_state:
        st.session_state.closed_incidents = []

    incident_options = incidents.drop_duplicates("incident_id").reset_index(drop=True)
    active_incidents = incident_options[~incident_options["incident_id"].isin(st.session_state.closed_incidents)].reset_index(drop=True)
    if active_incidents.empty:
        st.info("All incidents have been closed. No active incidents remain.")
        return

    active_ids = active_incidents["incident_id"].tolist()
    default_index = 0
    if st.session_state.get("selected_incident") in active_ids:
        default_index = active_ids.index(st.session_state.selected_incident)
    elif active_ids:
        st.session_state.selected_incident = active_ids[0]

    title_col, selector_col = st.columns([2.8, 1.2], vertical_alignment="bottom")
    with title_col:
        st.title("Unified Observability & RCA Agent")
        st.caption("Cross-tower incident workflow with explainable RCA and continuous learning.")
    with selector_col:
        incident_id = st.selectbox(
            "Active incident",
            active_ids,
            index=default_index,
        )
        run_rca_clicked = st.button("Run RCA", use_container_width=True)
    if incident_id != st.session_state.selected_incident:
        st.session_state.run_rca = False
        st.session_state.selected_incident = incident_id

    incident = incidents[incidents["incident_id"] == incident_id].iloc[0].to_dict()
    operator_notes = ""

    if "run_rca" not in st.session_state:
        st.session_state.run_rca = False
    if "rca_results" not in st.session_state:
        st.session_state.rca_results = {}
    if "selected_incident" not in st.session_state:
        st.session_state.selected_incident = incident_id
    if incident_id != st.session_state.selected_incident:
        st.session_state.run_rca = False
        st.session_state.selected_incident = incident_id

    telemetry = cached_telemetry()
    incident_tps = filtered_incident_telemetry(incident, telemetry)
    cache_key = incident["incident_id"]
    if run_rca_clicked:
        st.session_state.run_rca = True
        service = get_rca_service()
        with st.spinner("Running RCA..."):
            st.session_state.rca_results[cache_key] = service.investigate(
                incident=incident,
                operator_notes=operator_notes,
                telemetry=telemetry,
            )

    agent_result = st.session_state.rca_results.get(cache_key)
    analysis = agent_result.analysis if agent_result is not None else None

    workflow_tabs = st.tabs(
        [
            "1. Incident",
            "2. Correlated Evidence",
            "3. RCA",
            "4. Alternatives",
            "5. Feedback",
        ]
    )

    with workflow_tabs[0]:
        st.subheader(f"Incident {incident['incident_id']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Severity", incident["severity"])
        c2.metric("Service", incident["service"])
        if analysis is not None:
            c3.metric("RCA Confidence", f"{analysis.primary.confidence:.0%}")
        else:
            c3.metric("RCA Confidence", "Pending")

        st.write(generate_incident_description(incident, use_vllm=False))

        st.markdown("**Tower impact**")
        primary_tower, towers = tower_summary(incident_tps, analysis)
        if not towers:
            st.info("No tower data available for this incident")
        else:
            c1, c2 = st.columns([1, 2])
            c1.metric("Primary Tower", primary_tower)
            c2.metric("Affected Towers", ", ".join(sorted(towers)))
            tower_cols = st.columns(max(1, len(towers)))
            for col, (tower, signals) in zip(tower_cols, towers.items()):
                with col:
                    st.metric(tower, len(signals))
                    st.caption(f"Signals: {', '.join(list(dict.fromkeys(signals))[:3])}")

    with workflow_tabs[1]:
        st.subheader("Cross-Tower Correlated Evidence")
        st.caption("Anomalies are ranked by time proximity, tower severity, and service dependency relevance.")

        if analysis is None:
            st.info("Run RCA to compute correlated evidence.")
        else:
            evidence_df = pd.DataFrame([item.to_dict() for item in analysis.evidence])
            if not evidence_df.empty:
                st.dataframe(evidence_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Correlated Evidence",
                    dataframe_to_excel_bytes(evidence_df, "Correlated Evidence"),
                    file_name=f"correlated_evidence_{incident_id}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.info("No correlated evidence found for this incident.")

    with workflow_tabs[2]:
        st.subheader("Human-Readable RCA")
        if analysis is None:
            st.info("Run RCA to generate the report.")
        else:
            st.markdown(agent_result.report_markdown)
            st.download_button(
                "Download RCA PDF",
                markdown_to_pdf_bytes(f"RCA Report {incident_id}", agent_result.report_markdown),
                file_name=f"rca_{incident_id}.pdf",
                mime="application/pdf",
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                render_metric_card(
                    "Confidence",
                    f"{analysis.primary.confidence:.0%}",
                    "Weighted score from anomaly strength, tower relevance, timing, and memory boost.",
                )
            with c2:
                render_metric_card(
                    "Evidence Items",
                    str(len(analysis.evidence)),
                    "Structured observations used to support or reject hypotheses.",
                )
            with c3:
                render_metric_card(
                    "Similar Incidents",
                    str(len(analysis.similar_incidents)),
                    "Validated historical incidents retrieved from local memory.",
                )

            if analysis.similar_incidents:
                st.markdown("**Historical feedback that influenced this analysis**")
                feedback_records = pd.DataFrame(analysis.similar_incidents)
                feedback_cols = [col for col in ['incident_id', 'service', 'selected_root_cause', 'actual_root_cause', 'correctness', 'notes'] if col in feedback_records.columns]
                st.dataframe(
                    feedback_records[feedback_cols] if feedback_cols else feedback_records,
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("These historical incidents matched your current evidence terms and service. Their feedback helped boost or penalize hypotheses.")

    with workflow_tabs[3]:
        st.subheader("Alternative Hypotheses")
        st.caption("The agent shows plausible alternatives so correlation is not presented as certainty.")

        if analysis is None:
            st.info("Run RCA to see alternative hypotheses.")
        else:
            for rank, hypothesis in enumerate(analysis.alternatives, start=2):
                with st.expander(
                    f"Rank {rank}: {hypothesis.title} ({hypothesis.confidence:.0%})",
                    expanded=True,
                ):
                    st.write(hypothesis.summary)
                    st.markdown("**Why it ranked lower**")
                    for reason in hypothesis.rejection_reasons:
                        st.write(f"- {reason}")

    with workflow_tabs[4]:
        st.subheader("Feedback & Continuous Learning")
        st.caption("Engineer validation is stored locally and retrieved for future similar incidents.")

        if analysis is None:
            st.info("Run RCA first to enable feedback submission and incident memory retrieval.")
        else:
            service = service or get_rca_service()
            with st.form("feedback_form"):
                is_correct = st.radio(
                    "Was the primary RCA correct?",
                    ["Correct", "Partially correct", "Incorrect"],
                    horizontal=True,
                )
                actual_root_cause = st.selectbox(
                    "Actual root cause",
                    [analysis.primary.title]
                    + [hypothesis.title for hypothesis in analysis.alternatives]
                    + ["Other / unknown"],
                )
                other_root_cause = ""
                if actual_root_cause == "Other / unknown":
                    other_root_cause = st.text_input(
                        "Confirm actual root cause",
                        placeholder="Describe the confirmed root cause or failure mode...",
                    )

                notes = st.text_area("Engineer notes", placeholder="Resolution steps, missed evidence, ticket notes...")
                submitted = st.form_submit_button("Store feedback")

            if submitted and service is not None:
                final_root_cause = other_root_cause.strip() or actual_root_cause
                service.submit_feedback(
                    FeedbackRequest(
                        incident_id=incident["incident_id"],
                        service=incident["service"],
                        selected_root_cause=final_root_cause,
                        actual_root_cause=other_root_cause.strip() or None,
                        correctness=is_correct,
                        notes=notes,
                    )
                )
                st.session_state.closed_incidents.append(incident["incident_id"])
                st.success("Feedback stored. This incident has been removed from the active list.")

            if service is not None:
                st.markdown("**Known incident memory**")
                memory_records = service.get_memory()
                if memory_records:
                    memory_df = pd.DataFrame(memory_records)
                    preferred_cols = ['incident_id', 'service', 'selected_root_cause', 'actual_root_cause', 'agent_root_cause', 'correctness', 'notes']
                    cols_to_show = [c for c in preferred_cols if c in memory_df.columns]
                    st.dataframe(memory_df[cols_to_show], use_container_width=True, hide_index=True)
                else:
                    st.info("No feedback history yet. Submit feedback above to start building incident memory.")

            st.markdown("**How learning is applied**")
            st.markdown(
                "- **Correct feedback** boosts the selected root cause by +10% for similar incidents.\n"
                "- **Partial feedback** boosts the selected root cause by +5%.\n"
                "- **Incorrect feedback** penalizes the agent's predicted root cause by -8%.\n"
                "- Feedback is stored as incident memory and influences future RCA ranking for similar service incidents."
            )


if __name__ == "__main__":
    main()
