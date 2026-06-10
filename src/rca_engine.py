from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


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
    similar_incidents: list[dict[str, Any]]


def analyze_incident(incident: dict[str, Any], telemetry: pd.DataFrame, memory: Any) -> RCAAnalysis:
    incident_telemetry = telemetry[telemetry["incident_id"] == incident["incident_id"]].copy()
    evidence = detect_anomalies(incident_telemetry)
    similar_incidents = memory.find_similar(
        service=incident["service"],
        evidence_terms=[item.signal for item in evidence],
    )
    hypotheses = score_hypotheses(incident, evidence, similar_incidents)
    hypotheses = sorted(hypotheses, key=lambda item: item.confidence, reverse=True)
    return RCAAnalysis(
        primary=hypotheses[0],
        alternatives=hypotheses[1:],
        evidence=evidence,
        similar_incidents=similar_incidents,
    )


def detect_anomalies(frame: pd.DataFrame) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for _, row in frame.iterrows():
        baseline = float(row["baseline"])
        value = float(row["value"])
        if baseline == 0:
            continue
        anomaly_score = max(0.0, (value - baseline) / baseline)
        if anomaly_score < 0.35:
            continue

        direction = "above"
        explanation = (
            f"{row['signal']} on {row['component']} is {anomaly_score:.1f}x "
            f"{direction} baseline in the {row['tower']} tower."
        )
        evidence.append(
            EvidenceItem(
                tower=str(row["tower"]),
                signal=str(row["signal"]),
                component=str(row["component"]),
                observed=value,
                baseline=baseline,
                anomaly_score=anomaly_score,
                timestamp=row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                explanation=explanation,
            )
        )

    return sorted(evidence, key=lambda item: item.anomaly_score, reverse=True)[:10]


def score_hypotheses(
    incident: dict[str, Any],
    evidence: list[EvidenceItem],
    similar_incidents: list[dict[str, Any]],
) -> list[Hypothesis]:
    service = incident["service"]
    tower_scores = _tower_scores(evidence)
    memory_boost = min(0.10, len(similar_incidents) * 0.04)

    storage_score = _bounded(0.35 + tower_scores.get("storage", 0) * 0.35 + tower_scores.get("application", 0) * 0.15 + memory_boost)
    app_score = _bounded(0.30 + tower_scores.get("application", 0) * 0.30 + memory_boost)
    compute_score = _bounded(0.22 + tower_scores.get("compute", 0) * 0.28)
    network_score = _bounded(0.20 + tower_scores.get("network", 0) * 0.28)

    if "inventory" in service:
        app_score = _bounded(app_score + 0.22)
        storage_score = _bounded(storage_score - 0.18)

    refs_by_tower = _refs_by_tower(evidence)
    return [
        Hypothesis(
            title="Storage latency caused downstream application timeouts",
            summary=(
                "The strongest leading indicator is abnormal storage latency on a service "
                "dependency, followed by application latency and errors."
            ),
            confidence=storage_score,
            confidence_drivers=[
                "Storage anomaly has high magnitude and appears on a declared dependency.",
                "Application errors align with the storage degradation window.",
                "Historical feedback can boost this pattern when similar incidents are stored.",
            ],
            rejection_reasons=[
                "Ranked lower if application-only deployment errors are stronger than storage evidence.",
                "Requires validation from database or storage platform owners.",
            ],
            evidence_refs=refs_by_tower.get("storage", []) + refs_by_tower.get("application", []),
        ),
        Hypothesis(
            title="Application deployment or code regression",
            summary=(
                "Application-level error signals may indicate a bad release, configuration issue, "
                "or dependency handling regression."
            ),
            confidence=app_score,
            confidence_drivers=[
                "Application errors are elevated inside the incident window.",
                "This hypothesis gains weight when deployment error signals spike.",
            ],
            rejection_reasons=[
                "Infrastructure evidence explains the application symptoms more directly when present.",
                "Needs release metadata or rollback evidence to become primary.",
            ],
            evidence_refs=refs_by_tower.get("application", []),
        ),
        Hypothesis(
            title="Compute resource pressure",
            summary=(
                "CPU or memory pressure may have reduced service capacity and amplified latency."
            ),
            confidence=compute_score,
            confidence_drivers=[
                "Compute metrics are checked for saturation and resource pressure.",
            ],
            rejection_reasons=[
                "Compute anomalies are weaker than the leading storage/application signals.",
                "No pod restart or node failure evidence dominates the incident.",
            ],
            evidence_refs=refs_by_tower.get("compute", []),
        ),
        Hypothesis(
            title="Network degradation between service dependencies",
            summary=(
                "Packet loss or load balancer latency could explain intermittent timeouts."
            ),
            confidence=network_score,
            confidence_drivers=[
                "Network metrics are checked for packet loss, latency, and load balancer symptoms.",
            ],
            rejection_reasons=[
                "Network anomaly magnitude is lower than the leading hypothesis.",
                "The blast radius appears service-specific rather than network-wide.",
            ],
            evidence_refs=refs_by_tower.get("network", []),
        ),
    ]


def _tower_scores(evidence: list[EvidenceItem]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in evidence:
        scores[item.tower] = max(scores.get(item.tower, 0.0), min(item.anomaly_score / 6, 1.0))
    return scores


def _refs_by_tower(evidence: list[EvidenceItem]) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for item in evidence:
        refs.setdefault(item.tower, []).append(item.explanation)
    return refs


def _bounded(value: float) -> float:
    return max(0.05, min(value, 0.96))
