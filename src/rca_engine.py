from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.rag import retrieve_known_issues

SEVERITY_PROXIMITY_WINDOWS = {
    "critical": pd.Timedelta(minutes=30),
    "major": pd.Timedelta(minutes=15),
    "high": pd.Timedelta(minutes=15),
    "minor": pd.Timedelta(minutes=5),
    "medium": pd.Timedelta(minutes=5),
    "low": pd.Timedelta(minutes=5),
}
DEFAULT_PROXIMITY_WINDOW = pd.Timedelta(minutes=5)


@dataclass
class EvidenceItem:
    tower: str
    signal: str
    component: str
    observed: float
    baseline: float
    anomaly_score: float
    timestamp: str
    explanation: str
    causal_score: float = 0.0
    role: str = "symptom"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tower": self.tower,
            "signal": self.signal,
            "component": self.component,
            "observed": round(self.observed, 2),
            "baseline": round(self.baseline, 2),
            "anomaly_score": round(self.anomaly_score, 2),
            "timestamp": self.timestamp,
            "explanation": self.explanation,
            "causal_score": round(self.causal_score, 2),
            "role": self.role,
        }


@dataclass
class Hypothesis:
    title: str
    summary: str
    confidence: float
    confidence_drivers: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)


@dataclass
class RCAAnalysis:
    primary: Hypothesis
    alternatives: list[Hypothesis]
    evidence: list[EvidenceItem]
    similar_incidents: list[dict[str, Any]] = field(default_factory=list)
    causal_chain: list[EvidenceItem] = field(default_factory=list)


def analyze_incident(
    incident: dict[str, Any],
    telemetry: pd.DataFrame,
    memory: Any,
    reference_sources: list[dict[str, str]] | None = None,
) -> RCAAnalysis:
    incident_telemetry = telemetry[telemetry["incident_id"] == incident["incident_id"]].copy()
    incident_telemetry, anchor_time = filter_to_incident_window(incident, incident_telemetry)
    evidence = detect_anomalies(incident_telemetry, anchor_time)
    evidence = _calculate_causal_scores(evidence)
    evidence = _assign_roles(evidence)
    causal_chain = list(evidence)

    # Detect multi-tower scenarios for enhanced analysis
    towers_involved = set(item.tower for item in evidence)
    multi_tower_boost = 1.0 + (0.1 * (len(towers_involved) - 1)) if len(towers_involved) > 1 else 1.0
    
    similar_incidents = memory.find_similar(
        service=incident["service"],
        evidence_terms=[item.signal for item in evidence],
    )
    hypotheses = score_hypotheses(incident, evidence, similar_incidents, reference_sources or [])
    
    # Boost confidence for hypotheses when multiple towers corroborate
    if len(towers_involved) > 1:
        for hyp in hypotheses:
            hyp.confidence = min(0.99, hyp.confidence * multi_tower_boost)
            hyp.confidence_drivers.append(
                f"Cross-tower corroboration: anomalies detected in {len(towers_involved)} towers ({', '.join(sorted(towers_involved))})."
            )
    
    hypotheses = sorted(hypotheses, key=lambda item: item.confidence, reverse=True)
    return RCAAnalysis(
        primary=hypotheses[0],
        alternatives=hypotheses[1:],
        evidence=evidence,
        similar_incidents=similar_incidents,
        causal_chain=causal_chain,
    )


def detect_anomalies(
    frame: pd.DataFrame,
    anchor_time: pd.Timestamp | None = None,
) -> list[EvidenceItem]:
    evidence: list[tuple[float, EvidenceItem]] = []
    for _, row in frame.iterrows():
        baseline = float(row["baseline"])
        value = float(row["value"])
        if baseline == 0:
            if str(row.get("unit", "")).lower() != "anomaly" or value <= 0:
                continue
            anomaly_score = value
        else:
            anomaly_score = max(0.0, (value - baseline) / baseline)
        gpu_score = _gpu_anomaly_score(row)
        if gpu_score is not None:
            anomaly_score = max(anomaly_score, gpu_score)
        if anomaly_score < 0.35:
            continue

        direction = "above"
        gpu_context = ""
        if gpu_score is not None:
            gpu_context = f" GPU model score={gpu_score:.2f}."
        explanation = (
            f"{row['signal']} on {row['component']} is {anomaly_score:.1f}x "
            f"{direction} baseline in the {row['tower']} tower.{gpu_context}"
        )
        item = EvidenceItem(
            tower=str(row["tower"]),
            signal=str(row["signal"]),
            component=str(row["component"]),
            observed=value,
            baseline=baseline,
            anomaly_score=anomaly_score,
            timestamp=row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            explanation=explanation,
        )
        proximity_minutes = _proximity_minutes(row["timestamp"], anchor_time)
        evidence.append((proximity_minutes, item))

    ranked = sorted(evidence, key=lambda pair: (pair[0], -pair[1].anomaly_score))
    return [item for _, item in ranked[:10]]


def _calculate_causal_scores(
    evidence: list[EvidenceItem],
) -> list[EvidenceItem]:
    if not evidence:
        return evidence

    tower_count = max(1, len({item.tower for item in evidence}))
    max_anomaly_score = max(item.anomaly_score for item in evidence) or 1.0
    max_magnitude = max(
        abs(item.observed - item.baseline) / max(abs(item.baseline), 1.0)
        for item in evidence
    ) or 1.0
    timestamps = [pd.to_datetime(item.timestamp, errors="coerce") for item in evidence]
    min_timestamp = min(ts for ts in timestamps if not pd.isna(ts))
    max_timestamp = max(ts for ts in timestamps if not pd.isna(ts))
    time_span = max(1.0, (max_timestamp - min_timestamp).total_seconds())

    for item, timestamp in zip(evidence, timestamps):
        if pd.isna(timestamp):
            temporal_priority = 0.0
        else:
            age_seconds = (timestamp - min_timestamp).total_seconds()
            temporal_priority = 1.0 - min(age_seconds / time_span, 1.0)

        magnitude_priority = min(
            abs(item.observed - item.baseline) / max(abs(item.baseline), 1.0) / max_magnitude,
            1.0,
        )
        normalized_anomaly = min(item.anomaly_score / max_anomaly_score, 1.0)
        tower_factor = min(tower_count / 5.0, 1.0)

        item.causal_score = (
            normalized_anomaly * 0.55
            + tower_factor * 0.20
            + temporal_priority * 0.15
            + magnitude_priority * 0.10
        )

    return sorted(
        evidence,
        key=lambda item: (item.causal_score, -item.anomaly_score),
        reverse=True,
    )


def _assign_roles(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    if not evidence:
        return evidence

    top_score = evidence[0].causal_score
    for index, item in enumerate(evidence):
        if index == 0:
            item.role = "root_cause"
        elif item.causal_score >= top_score * 0.60:
            item.role = "propagation"
        else:
            item.role = "symptom"
    return evidence


def filter_to_incident_window(
    incident: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    if frame.empty or "timestamp" not in frame.columns:
        return frame, None

    filtered = frame.copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], errors="coerce")
    filtered = filtered[filtered["timestamp"].notna()]
    if filtered.empty:
        return filtered, None

    anchor_time = _affected_component_time(incident, filtered)
    if anchor_time is None:
        return filtered, None

    proximity_window = proximity_window_for_incident(incident)
    lower_bound = anchor_time - proximity_window
    upper_bound = anchor_time + proximity_window
    return filtered[
        (filtered["timestamp"] >= lower_bound)
        & (filtered["timestamp"] <= upper_bound)
    ].copy(), anchor_time


def _affected_component_time(
    incident: dict[str, Any],
    frame: pd.DataFrame,
) -> pd.Timestamp | None:
    affected_component = str(incident.get("service", "")).strip()
    started_at = pd.to_datetime(incident.get("started_at"), errors="coerce")
    component_rows = frame
    if affected_component:
        component_rows = frame[frame["component"].astype(str) == affected_component]

    if component_rows.empty:
        if pd.notna(started_at):
            return started_at
        return frame["timestamp"].min()

    if pd.notna(started_at):
        distances = (component_rows["timestamp"] - started_at).abs()
        return component_rows.loc[distances.idxmin(), "timestamp"]

    return component_rows["timestamp"].min()


def proximity_window_for_incident(incident: dict[str, Any]) -> pd.Timedelta:
    severity = str(incident.get("severity", "")).strip().lower()
    return SEVERITY_PROXIMITY_WINDOWS.get(severity, DEFAULT_PROXIMITY_WINDOW)


def _proximity_minutes(
    timestamp: Any,
    anchor_time: pd.Timestamp | None,
) -> float:
    if anchor_time is None:
        return 0.0
    timestamp = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(timestamp):
        return float("inf")
    return abs((timestamp - anchor_time).total_seconds()) / 60


def _gpu_anomaly_score(row: pd.Series) -> float | None:
    if "gpu_anomaly_score" not in row:
        return None
    score = pd.to_numeric(row.get("gpu_anomaly_score"), errors="coerce")
    if pd.isna(score):
        return None
    return float(score)


def score_hypotheses(
    incident: dict[str, Any],
    evidence: list[EvidenceItem],
    similar_incidents: list[dict[str, Any]],
    reference_sources: list[dict[str, str]] | None = None,
) -> list[Hypothesis]:
    rag_hypotheses = _score_reference_hypotheses(
        evidence,
        similar_incidents,
        reference_sources or [],
    )
    if rag_hypotheses:
        return rag_hypotheses

    memory_hypotheses = _score_memory_hypotheses(evidence, similar_incidents)
    if memory_hypotheses:
        return memory_hypotheses

    return _no_reference_hypothesis(evidence)


def _score_reference_hypotheses(
    evidence: list[EvidenceItem],
    similar_incidents: list[dict[str, Any]],
    reference_sources: list[dict[str, str]],
) -> list[Hypothesis]:
    query_terms = []
    for item in evidence:
        query_terms.extend([item.tower, item.signal, item.component, item.explanation])

    # Use Chroma vector DB for semantic search
    known_issues = retrieve_known_issues(reference_sources=reference_sources, query_terms=query_terms)
    if not known_issues:
        return []

    max_score = max(issue["score"] for issue in known_issues) or 1
    hypotheses = []
    for issue in known_issues:
        confidence = _bounded(0.35 + (issue["score"] / max_score) * 0.45)
        matched_evidence = _matching_evidence(issue["body"], evidence)
        hypotheses.append(
            Hypothesis(
                title=issue["title"],
                summary=f"{issue['body']} Source: {issue['source']}.",
                confidence=confidence,
                confidence_drivers=[
                    "Retrieved from Chroma vector database via semantic RAG.",
                    "Ranked by vector similarity with anomalous tower signals and components.",
                ],
                rejection_reasons=[
                    "Ranked lower when vector similarity or evidence match is weaker.",
                    "Requires operator validation before declaring causation.",
                ],
                evidence_refs=matched_evidence,
            )
        )

    return _apply_feedback_learning(hypotheses, similar_incidents)


def _matching_evidence(issue_text: str, evidence: list[EvidenceItem]) -> list[str]:
    issue_text = issue_text.lower()
    matches = [
        item.explanation
        for item in evidence
        if item.tower.lower() in issue_text
        or item.component.lower() in issue_text
        or any(token in issue_text for token in item.signal.lower().split("_"))
    ]
    return matches or [item.explanation for item in evidence[:3]]


def _score_memory_hypotheses(
    evidence: list[EvidenceItem],
    similar_incidents: list[dict[str, Any]],
) -> list[Hypothesis]:
    if not similar_incidents:
        return []

    counts: dict[str, dict[str, Any]] = {}
    for incident in similar_incidents:
        selected = incident.get("selected_root_cause", "Unknown")
        correctness = incident.get("correctness", "")
        if selected not in counts:
            counts[selected] = {"count": 0, "score": 0.0, "correctness": []}
        counts[selected]["count"] += 1
        if correctness == "Correct":
            counts[selected]["score"] += 1.0
        elif correctness == "Partially correct":
            counts[selected]["score"] += 0.5
        else:
            counts[selected]["score"] += 0.2
        counts[selected]["correctness"].append(correctness)

    sorted_roots = sorted(
        counts.items(),
        key=lambda item: (item[1]["score"], item[1]["count"]),
        reverse=True,
    )

    hypotheses: list[Hypothesis] = []
    for root_cause, stats in sorted_roots[:3]:
        confidence = _bounded(0.25 + min(stats["score"], 2.0) * 0.2)
        hypotheses.append(
            Hypothesis(
                title=root_cause,
                summary=(
                    "Historical incident memory suggests this root cause for similar service incidents. "
                    "Use operator validation and current evidence to confirm the candidate."
                ),
                confidence=confidence,
                confidence_drivers=[
                    "Derived from similar past incidents with operator feedback.",
                    f"{stats['count']} similar incident(s) matched based on service and evidence terms.",
                ],
                rejection_reasons=[
                    "This candidate comes from prior incident memory, not from a direct Chroma vector match.",
                    "Verify with current telemetry anomalies before treating it as the only cause.",
                ],
                evidence_refs=[item.explanation for item in evidence[:3]],
            )
        )

    return hypotheses


def _bounded(value: float) -> float:
    return max(0.05, min(value, 0.96))


def _no_reference_hypothesis(evidence: list[EvidenceItem]) -> list[Hypothesis]:
    return [
        Hypothesis(
            title="No matching known issue found in reference repository",
            summary="The agent found anomalies, but no vector match in Chroma DB matched strongly enough.",
            confidence=0.20,
            confidence_drivers=["RCA hypotheses should come from the Chroma vector database."],
            rejection_reasons=["Add or update known issues to improve Chroma vector DB coverage."],
            evidence_refs=[item.explanation for item in evidence[:3]],
        )
    ]


def _apply_feedback_learning(
    hypotheses: list[Hypothesis],
    similar_incidents: list[dict[str, Any]],
) -> list[Hypothesis]:
    for hypothesis in hypotheses:
        adjustment = 0.0
        for incident in similar_incidents:
            correctness = incident.get("correctness", "")
            selected = incident.get("selected_root_cause", "")
            agent_root = incident.get("agent_root_cause", "")
            if selected == hypothesis.title and correctness == "Correct":
                adjustment += 0.10
            elif selected == hypothesis.title and correctness == "Partially correct":
                adjustment += 0.05
            elif agent_root == hypothesis.title and correctness == "Incorrect":
                adjustment -= 0.08

        if adjustment:
            hypothesis.confidence = _bounded(hypothesis.confidence + adjustment)
            direction = "boosted" if adjustment > 0 else "reduced"
            hypothesis.confidence_drivers.append(
                f"Historical user feedback {direction} this hypothesis by {abs(adjustment):.0%}."
            )
    return hypotheses
