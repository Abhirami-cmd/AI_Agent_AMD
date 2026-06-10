from __future__ import annotations

from src.rca_engine import RCAAnalysis
from src.vllm_client import generate_with_vllm, is_vllm_configured


def build_rca_markdown(
    incident: dict,
    analysis: RCAAnalysis,
    reference_sources: list[dict[str, str]] | None = None,
) -> str:
    if is_vllm_configured():
        try:
            return _build_vllm_report(incident, analysis, reference_sources or [])
        except Exception as exc:
            return (
                _build_deterministic_report(incident, analysis, reference_sources or [])
                + f"\n\n**vLLM Status**\n\nConfigured, but request failed: `{exc}`. "
                + "Showing deterministic fallback RCA from structured evidence."
            )

    return (
        _build_deterministic_report(incident, analysis, reference_sources or [])
        + "\n\n**vLLM Status**\n\nSet `VLLM_BASE_URL` or `USE_VLLM=1` to generate this RCA through a vLLM OpenAI-compatible endpoint."
    )


def _build_deterministic_report(
    incident: dict,
    analysis: RCAAnalysis,
    reference_sources: list[dict[str, str]],
) -> str:
    primary = analysis.primary
    evidence_lines = "\n".join(
        f"- {item.explanation} Observed {item.observed:.2f} vs baseline {item.baseline:.2f} at {item.timestamp}."
        for item in analysis.evidence[:5]
    )
    similar = "None yet. Submit feedback to build incident memory."
    if analysis.similar_incidents:
        similar = "\n".join(
            f"- {item['incident_id']}: {item['selected_root_cause']} ({item['correctness']})"
            for item in analysis.similar_incidents
        )
    references = "\n".join(
        f"- {source['name']} ({source['type'].upper()}): `{source['path']}`"
        for source in reference_sources
    ) or "- No reference sources loaded."

    return f"""
**Executive Summary**

{incident["service"]} is experiencing a {incident["severity"].lower()} incident. The most likely root cause is **{primary.title}** with **{primary.confidence:.0%} confidence**.

**What Happened**

{primary.summary}

**Primary Evidence**

{evidence_lines}

**Recommended Remediation**

- Validate the top dependency named in the evidence table.
- Check recent platform events and deployment activity during the incident window.
- Mitigate the leading cause first, then watch application error rate and latency recover.
- Record the confirmed root cause in the feedback tab so the agent can improve future ranking.

**Similar Past Incidents**

{similar}

**Reference Sources Used for Inference**

{references}
"""


def _build_vllm_report(
    incident: dict,
    analysis: RCAAnalysis,
    reference_sources: list[dict[str, str]],
) -> str:
    evidence = "\n".join(
        f"- {item.explanation} Observed={item.observed:.2f}, baseline={item.baseline:.2f}, time={item.timestamp}"
        for item in analysis.evidence
    )
    alternatives = "\n".join(
        f"- {item.title}: confidence={item.confidence:.0%}; lower-rank reasons={'; '.join(item.rejection_reasons)}"
        for item in analysis.alternatives
    )
    references = "\n\n".join(
        f"Source: {source['name']} ({source['path']})\n{source['text'][:2500]}"
        for source in reference_sources
    )
    system_prompt = (
        "You are a vLLM-hosted RCA analyst for a unified observability platform. "
        "Write concise, evidence-grounded incident RCA. Use only the provided evidence and reference sources. "
        "Always include confidence, evidence, alternative hypotheses, and validation steps."
    )
    user_prompt = f"""
Incident:
{incident}

Primary hypothesis:
{analysis.primary.title}
Confidence: {analysis.primary.confidence:.0%}
Summary: {analysis.primary.summary}
Confidence drivers: {analysis.primary.confidence_drivers}

Evidence:
{evidence}

Alternative hypotheses:
{alternatives}

Reference PDF content:
{references}

Return Markdown with these sections:
Executive Summary, Most Likely Root Cause, Evidence, Confidence Rationale,
Alternative Hypotheses, Recommended Remediation, Reference Sources Used.
"""
    return generate_with_vllm(system_prompt=system_prompt, user_prompt=user_prompt)
