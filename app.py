from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_sample_incidents, load_sample_telemetry
from src.incident_memory import IncidentMemory
from src.reference_loader import load_reference_sources
from src.rca_engine import analyze_incident
from src.reporting import build_rca_markdown
from src.vllm_client import is_vllm_configured


st.set_page_config(
    page_title="Unified Observability RCA Agent",
    page_icon=":bar_chart:",
    layout="wide",
)


def render_metric_card(label: str, value: str, help_text: str) -> None:
    st.metric(label=label, value=value, help=help_text)


@st.cache_data
def cached_data() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    return load_sample_incidents(), load_sample_telemetry(), load_reference_sources()


def main() -> None:
    incidents, telemetry, reference_sources = cached_data()
    memory = IncidentMemory()

    title_col, selector_col = st.columns([2.8, 1.2], vertical_alignment="bottom")
    with title_col:
        st.title("Unified Observability & RCA Agent")
        st.caption("Cross-tower incident workflow with explainable RCA and continuous learning.")
    with selector_col:
        incident_id = st.selectbox(
            "Active incident",
            incidents["incident_id"].tolist(),
            format_func=lambda item: incidents.set_index("incident_id").loc[item, "title"],
        )

    incident = incidents[incidents["incident_id"] == incident_id].iloc[0].to_dict()
    analysis = analyze_incident(incident, telemetry, memory)

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
        st.subheader(incident["title"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Severity", incident["severity"])
        c2.metric("Service", incident["service"])
        c3.metric("Started", incident["started_at"])
        c4.metric("RCA Confidence", f"{analysis.primary.confidence:.0%}")

        st.write(incident["description"])
        st.info(
            "Workflow focus: this screen starts from the incident, then guides the operator "
            "toward evidence, RCA, alternatives, and feedback."
        )

        st.markdown("**Affected dependency map**")
        st.dataframe(
            pd.DataFrame(incident["dependencies"]),
            use_container_width=True,
            hide_index=True,
        )

    with workflow_tabs[1]:
        st.subheader("Cross-Tower Correlated Evidence")
        st.caption("Anomalies are ranked by time proximity, tower severity, and service dependency relevance.")

        evidence_df = pd.DataFrame([item.to_dict() for item in analysis.evidence])
        st.dataframe(evidence_df, use_container_width=True, hide_index=True)

        selected_towers = st.multiselect(
            "Tower filter",
            sorted(telemetry["tower"].unique()),
            default=sorted(telemetry["tower"].unique()),
        )
        chart_data = telemetry[
            (telemetry["incident_id"] == incident_id)
            & (telemetry["tower"].isin(selected_towers))
        ].copy()

        fig = px.line(
            chart_data,
            x="timestamp",
            y="value",
            color="signal",
            facet_row="tower",
            markers=True,
            title="Incident Timeline Across Towers",
            labels={"value": "Observed value", "timestamp": "Time"},
        )
        fig.update_layout(height=760, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with workflow_tabs[2]:
        st.subheader("Human-Readable RCA")
        st.markdown(build_rca_markdown(incident, analysis, reference_sources))

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

        st.markdown("**Confidence rationale**")
        for reason in analysis.primary.confidence_drivers:
            st.write(f"- {reason}")

        st.markdown("**Inference sources**")
        st.write(
            "vLLM enabled"
            if is_vllm_configured()
            else "vLLM not configured; deterministic fallback is displayed."
        )
        for source in reference_sources:
            st.write(f"- {source['name']} ({source['type'].upper()}): `{source['path']}`")

    with workflow_tabs[3]:
        st.subheader("Alternative Hypotheses")
        st.caption("The agent shows plausible alternatives so correlation is not presented as certainty.")

        for rank, hypothesis in enumerate(analysis.alternatives, start=2):
            with st.expander(
                f"Rank {rank}: {hypothesis.title} ({hypothesis.confidence:.0%})",
                expanded=True,
            ):
                st.write(hypothesis.summary)
                st.markdown("**Why it ranked lower**")
                for reason in hypothesis.rejection_reasons:
                    st.write(f"- {reason}")
                st.markdown("**Supporting evidence**")
                for item in hypothesis.evidence_refs:
                    st.write(f"- {item}")

    with workflow_tabs[4]:
        st.subheader("Feedback & Continuous Learning")
        st.caption("Engineer validation is stored locally and retrieved for future similar incidents.")

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
            notes = st.text_area("Engineer notes", placeholder="Resolution steps, missed evidence, ticket notes...")
            submitted = st.form_submit_button("Store feedback")

        if submitted:
            memory.save_feedback(
                incident_id=incident_id,
                service=incident["service"],
                selected_root_cause=actual_root_cause,
                agent_root_cause=analysis.primary.title,
                correctness=is_correct,
                notes=notes,
                evidence_summary=analysis.primary.summary,
            )
            st.success("Feedback stored. Future RCA scoring can now retrieve this incident as memory.")

        st.markdown("**Known incident memory**")
        st.dataframe(pd.DataFrame(memory.load_all()), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
