from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Dependency(BaseModel):
    source: str
    dependency: str
    tower: str


class Incident(BaseModel):
    incident_id: str
    title: str
    service: str
    severity: str
    started_at: str
    description: str
    dependencies: list[Dependency] = Field(default_factory=list)
    telemetry: list[TelemetryPoint] = Field(default_factory=list)  # All variants in aggregated incident
    variant_count: int = 1  # How many records aggregated into this incident


class TelemetryPoint(BaseModel):
    incident_id: str
    timestamp: datetime
    tower: str
    signal: str
    value: float
    baseline: float
    unit: str
    component: str
    variant_index: int = 0  # Which variation within task (0-indexed)
    root_component: str = ""  # Ground truth RCA component label
    root_reason: str = ""  # Ground truth RCA reason label


class EvidenceItem(BaseModel):
    tower: str
    signal: str
    component: str
    observed: float
    baseline: float
    anomaly_score: float
    timestamp: str
    explanation: str


class Hypothesis(BaseModel):
    title: str
    summary: str
    confidence: float
    confidence_drivers: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class RCAAnalysis(BaseModel):
    primary: Hypothesis
    alternatives: list[Hypothesis]
    evidence: list[EvidenceItem]
    similar_incidents: list[dict[str, Any]] = Field(default_factory=list)


class ReferenceSource(BaseModel):
    name: str
    path: str
    type: str
    text: str


class MemoryRecord(BaseModel):
    stored_at: str
    incident_id: str
    service: str
    selected_root_cause: str
    actual_root_cause: str | None = None
    agent_root_cause: str
    correctness: str
    notes: str
    evidence_summary: str


class FeedbackRequest(BaseModel):
    incident_id: str
    service: str
    selected_root_cause: str
    actual_root_cause: str | None = None
    correctness: str
    notes: str
