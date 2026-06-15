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
MAX_PRIMARY_CONFIDENCE = 0.92
MAX_ALTERNATIVE_CONFIDENCE = 0.84


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
    score_factors: dict[str, float] = field(default_factory=dict)


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

    similar_incidents = memory.find_similar(
        service=incident["service"],
        evidence_terms=[item.signal for item in evidence],
    )
    hypotheses = score_hypotheses(incident, evidence, similar_incidents, reference_sources or [])

    MAX_ALTERNATIVES = 4
    hypotheses = sorted(hypotheses, key=lambda item: item.confidence, reverse=True)
    return RCAAnalysis(
        primary=hypotheses[0],
        alternatives=hypotheses[1:MAX_ALTERNATIVES+1],
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
    memory_hypotheses = _score_memory_hypotheses(evidence, similar_incidents)

    all_hypotheses = _merge_duplicate_hypotheses(rag_hypotheses + memory_hypotheses)
    if not all_hypotheses:
        return _no_reference_hypothesis(evidence)

    all_hypotheses = _apply_weighted_confidence(
        incident,
        evidence,
        all_hypotheses,
        similar_incidents,
    )
    all_hypotheses = _apply_confidence_caps(all_hypotheses)
    return sorted(all_hypotheses, key=lambda h: h.confidence, reverse=True)


def _score_reference_hypotheses(
    evidence: list[EvidenceItem],
    similar_incidents: list[dict[str, Any]],
    reference_sources: list[dict[str, str]],
) -> list[Hypothesis]:
    query_terms = []
    for item in evidence:
        query_terms.extend([item.tower, item.signal, item.component])

    # Use Chroma vector DB for semantic search
    known_issues = retrieve_known_issues(reference_sources=reference_sources, query_terms=query_terms)
    if not known_issues:
        return []
    clusters = {}
    for issue in known_issues:
        key = issue["title"].lower().strip()
        #key = issue["title"].split(":")[0]  # simple semantic bucket
        clusters.setdefault(key, []).append(issue)

    #max_score = max(issue["score"] for issue in known_issues) or 1
    hypotheses = []
   
    for cluster_name, items in clusters.items():
        #best = max(items, key=lambda x: x["score"])
        best = max(items, key=lambda x: x.get("score", 0))

        rag_similarity = _bounded(float(best.get("score", 0.0)), floor=0.0)

        hypotheses.append(
            Hypothesis(
                title=best["title"],
                summary=best["body"],
                confidence=0.0,
                confidence_drivers=[
                    "Clustered RCA hypothesis from distinct failure group.",
                ],
                rejection_reasons=[
                    "Alternative hypothesis selected from different failure cluster.",
                ],
                evidence_refs=_matching_evidence(best["body"], evidence),
                score_factors={"rag_similarity": rag_similarity},
            )
        )
    return sorted(
        hypotheses,
        key=lambda h: h.confidence,
        reverse=True
    )[:4]

    


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


def _merge_duplicate_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    merged: dict[str, Hypothesis] = {}
    for hypothesis in hypotheses:
        key = hypothesis.title.strip().lower()
        if key not in merged:
            merged[key] = hypothesis
            continue

        existing = merged[key]
        existing.confidence = max(existing.confidence, hypothesis.confidence)
        existing.confidence_drivers = _unique(existing.confidence_drivers + hypothesis.confidence_drivers)
        existing.rejection_reasons = _unique(existing.rejection_reasons + hypothesis.rejection_reasons)
        existing.evidence_refs = _unique(existing.evidence_refs + hypothesis.evidence_refs)
        existing.score_factors = _merge_score_factors(existing.score_factors, hypothesis.score_factors)
        if len(hypothesis.summary) > len(existing.summary):
            existing.summary = hypothesis.summary
    return list(merged.values())


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _merge_score_factors(
    first: dict[str, float],
    second: dict[str, float],
) -> dict[str, float]:
    merged = dict(first)
    for key, value in second.items():
        merged[key] = max(float(merged.get(key, 0.0)), float(value))
    return merged


def _score_memory_hypotheses(
    evidence: list[EvidenceItem],
    similar_incidents: list[dict[str, Any]],
) -> list[Hypothesis]:
    if not similar_incidents:
        return []

    counts: dict[str, dict[str, Any]] = {}
    for incident in similar_incidents:
        selected = str(incident.get("selected_root_cause", "")).strip()
        actual = str(incident.get("actual_root_cause", "") or "").strip()
        agent_root = str(incident.get("agent_root_cause", "") or "").strip()
        correctness = str(incident.get("correctness", "")).strip().lower()
        if correctness == "correct":
            _add_memory_score(counts, selected, 1.0, incident.get("correctness", ""))
        elif correctness == "partially correct":
            _add_memory_score(counts, selected, 0.5, incident.get("correctness", ""))
        elif correctness == "incorrect":
            if actual and actual.lower() not in {"other / unknown", "unknown"}:
                _add_memory_score(counts, actual, 0.8, incident.get("correctness", ""))
            elif selected and selected.lower() not in {"other / unknown", "unknown"} and selected != agent_root:
                _add_memory_score(counts, selected, 0.5, incident.get("correctness", ""))

    sorted_roots = sorted(
        counts.items(),
        key=lambda item: (item[1]["score"], item[1]["count"]),
        reverse=True,
    )

    hypotheses: list[Hypothesis] = []
    for root_cause, stats in sorted_roots[:3]:
        memory_support = _bounded(float(stats["score"]) / 2.0, floor=0.0)
        hypotheses.append(
            Hypothesis(
                title=root_cause,
                summary=(
                    "Historical incident memory suggests this root cause for similar service incidents. "
                    "Use operator validation and current evidence to confirm the candidate."
                ),
                confidence=0.0,
                confidence_drivers=[
                    "Derived from similar past incidents with operator feedback.",
                    f"{stats['count']} similar incident(s) matched based on service and evidence terms.",
                ],
                rejection_reasons=[
                    "This candidate comes from prior incident memory, not from a direct Chroma vector match.",
                    "Verify with current telemetry anomalies before treating it as the only cause.",
                ],
                evidence_refs=[item.explanation for item in evidence[:3]],
                score_factors={"memory_support": memory_support},
            )
        )

    return hypotheses


def _add_memory_score(
    counts: dict[str, dict[str, Any]],
    root_cause: str,
    score: float,
    correctness: str,
) -> None:
    if not root_cause:
        return
    if root_cause not in counts:
        counts[root_cause] = {"count": 0, "score": 0.0, "correctness": []}
    counts[root_cause]["count"] += 1
    counts[root_cause]["score"] += score
    counts[root_cause]["correctness"].append(correctness)


def _bounded(value: float, floor: float = 0.05, ceiling: float = 0.96) -> float:
    return max(floor, min(value, ceiling))


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


def _apply_weighted_confidence(
    incident: dict[str, Any],
    evidence: list[EvidenceItem],
    hypotheses: list[Hypothesis],
    similar_incidents: list[dict[str, Any]],
) -> list[Hypothesis]:
    for hypothesis in hypotheses:
        factors = {
            "rag_similarity": float(hypothesis.score_factors.get("rag_similarity", 0.0)),
            "evidence_strength": _evidence_strength(hypothesis, evidence),
            "topology_support": _topology_support(incident, hypothesis, evidence),
            "memory_support": max(
                float(hypothesis.score_factors.get("memory_support", 0.0)),
                _memory_support(hypothesis, similar_incidents),
            ),
            "anomaly_severity": _anomaly_severity(evidence),
        }
        hypothesis.score_factors = {key: round(value, 4) for key, value in factors.items()}
        hypothesis.confidence = _bounded(
            0.35 * factors["rag_similarity"]
            + 0.25 * factors["evidence_strength"]
            + 0.20 * factors["topology_support"]
            + 0.10 * factors["memory_support"]
            + 0.10 * factors["anomaly_severity"],
            floor=0.0,
            ceiling=1.0,
        )
        hypothesis.confidence_drivers.append(
            "Confidence = 0.35*RAG similarity + 0.25*evidence strength + "
            "0.20*topology support + 0.10*memory support + 0.10*anomaly severity."
        )
        hypothesis.confidence_drivers.append(
            "Score factors: "
            + ", ".join(f"{key}={value:.2f}" for key, value in factors.items())
            + "."
        )
    return hypotheses


def _apply_confidence_caps(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    ranked = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
    for index, hypothesis in enumerate(ranked):
        cap = MAX_PRIMARY_CONFIDENCE if index == 0 else MAX_ALTERNATIVE_CONFIDENCE
        if hypothesis.confidence > cap:
            hypothesis.confidence = cap
            hypothesis.confidence_drivers.append(
                f"Confidence capped at {cap:.0%} to avoid over-certainty from synthetic or overlapping evidence."
            )
    return ranked


def _evidence_strength(hypothesis: Hypothesis, evidence: list[EvidenceItem]) -> float:
    if not evidence:
        return 0.0
    refs = {ref.lower() for ref in hypothesis.evidence_refs}
    matched = [
        item
        for item in evidence
        if not refs
        or item.explanation.lower() in refs
        or item.component.lower() in hypothesis.summary.lower()
        or item.tower.lower() in hypothesis.summary.lower()
    ]
    matched = matched or evidence[:3]
    avg_causal = sum(item.causal_score or item.anomaly_score for item in matched) / len(matched)
    coverage = len(matched) / len(evidence)
    return min((avg_causal * 0.65) + (coverage * 0.35), 1.0)


def _topology_support(
    incident: dict[str, Any],
    hypothesis: Hypothesis,
    evidence: list[EvidenceItem],
) -> float:
    dependencies = incident.get("dependencies") or []
    if not dependencies and not evidence:
        return 0.0

    dependency_text = " ".join(
        " ".join(str(value) for value in dependency.values())
        for dependency in dependencies
        if isinstance(dependency, dict)
    ).lower()
    hypothesis_text = " ".join([hypothesis.title, hypothesis.summary, *hypothesis.evidence_refs]).lower()

    evidence_towers = {item.tower.lower() for item in evidence}
    evidence_components = {item.component.lower() for item in evidence}
    topology_hits = 0
    for token in evidence_towers | evidence_components:
        if token and (token in dependency_text or token in hypothesis_text):
            topology_hits += 1

    coverage_denominator = max(1, len(evidence_towers | evidence_components))
    coverage = topology_hits / coverage_denominator
    cross_tower_bonus = min(len(evidence_towers) / 4.0, 1.0)
    return min((coverage * 0.70) + (cross_tower_bonus * 0.30), 1.0)


def _memory_support(
    hypothesis: Hypothesis,
    similar_incidents: list[dict[str, Any]],
) -> float:
    if not similar_incidents:
        return 0.0

    support = 0.0
    title = hypothesis.title.strip().lower()
    for incident in similar_incidents:
        correctness = str(incident.get("correctness", "")).strip().lower()
        selected = str(incident.get("selected_root_cause", "")).strip().lower()
        actual = str(incident.get("actual_root_cause", "") or "").strip().lower()
        agent_root = str(incident.get("agent_root_cause", "")).strip().lower()
        if selected == title and correctness == "correct":
            support += 1.0
        elif selected == title and correctness == "partially correct":
            support += 0.5
        elif actual == title and correctness == "incorrect":
            support += 0.8
        elif agent_root == title and correctness == "incorrect":
            support -= 0.8
    return max(0.0, min(support / 2.0, 1.0))


def _anomaly_severity(evidence: list[EvidenceItem]) -> float:
    if not evidence:
        return 0.0
    top_scores = sorted((item.anomaly_score for item in evidence), reverse=True)[:3]
    return min(sum(top_scores) / len(top_scores), 1.0)
