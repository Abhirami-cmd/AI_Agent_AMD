from __future__ import annotations

import dataclasses
import logging
from typing import Any

import pandas as pd

from src.agents import UnifiedRCAAgent
from src.data_loader import (
    load_dataset_reference_sources,
    load_incidents,
    load_incident_memory_records,
    load_telemetry,
)
from src.graph_service import TopologyGraphService
from src.incident_memory import IncidentMemory
from src.models import FeedbackRequest
from src.reference_loader import dynamic_reference_source, get_reference_source_used, load_reference_sources

logger = logging.getLogger(__name__)


class RCAService:
    def __init__(self, memory: IncidentMemory | None = None) -> None:
        self.memory = memory or IncidentMemory()
        self._incidents = load_incidents()
        self._telemetry = load_telemetry()
        self._memory_seeded = False
        self.graph_service = TopologyGraphService()

    def get_topology(self, incident_id: str) -> dict[str, Any]:
        incident = self.get_incident(incident_id)
        topology = self.graph_service.build_topology_payload(incident)
        logger.debug("Built topology for incident %s", incident_id)
        return topology

    def list_incidents(self) -> list[dict[str, Any]]:
        return self._incidents.to_dict(orient="records")

    def get_incident(self, incident_id: str) -> dict[str, Any]:
        rows = self._incidents[self._incidents["incident_id"] == incident_id]
        if rows.empty:
            raise ValueError(f"Incident not found: {incident_id}")
        return rows.iloc[0].to_dict()

    def get_telemetry(self, incident_id: str) -> list[dict[str, Any]]:
        filtered = self._telemetry[self._telemetry["incident_id"] == incident_id]
        return filtered.to_dict(orient="records")

    def investigate(
        self,
        incident_id: str | None = None,
        operator_notes: str | None = None,
        incident: dict[str, Any] | None = None,
        telemetry: Any | None = None,
    ):
        if incident is None:
            if incident_id is None:
                raise ValueError("Either incident_id or incident data must be provided.")
            incident = self.get_incident(incident_id)

        if telemetry is None or (isinstance(telemetry, pd.DataFrame) and telemetry.empty):
            telemetry = self._telemetry

        self._ensure_memory_seeded()
        reference_sources = self.get_reference_sources(operator_notes)
        return UnifiedRCAAgent(self.memory).investigate(incident, telemetry, reference_sources)

    def investigate_payload(
        self,
        incident_id: str | None = None,
        operator_notes: str | None = None,
        incident: dict[str, Any] | None = None,
        telemetry: Any | None = None,
    ) -> dict[str, Any]:
        agent_result = self.investigate(
            incident_id=incident_id,
            operator_notes=operator_notes,
            incident=incident,
            telemetry=telemetry,
        )
        incident = incident or self.get_incident(incident_id)  # type: ignore[arg-type]
        return {
            "incident": incident,
            "analysis": dataclasses.asdict(agent_result.analysis),
            "report_markdown": agent_result.report_markdown,
            "agent_trace": agent_result.agent_trace,
            "reference_sources": self.get_reference_sources(operator_notes),
            "reference_source_used": get_reference_source_used(),
        }

    def submit_feedback(self, feedback: FeedbackRequest) -> None:
        incident = self.get_incident(feedback.incident_id)
        agent_result = self.investigate(feedback.incident_id)
        agent_root_cause = agent_result.analysis.primary.title
        evidence_summary = agent_result.analysis.primary.summary
        self.memory.save_feedback(
            incident_id=feedback.incident_id,
            service=incident["service"],
            selected_root_cause=feedback.selected_root_cause,
            actual_root_cause=feedback.actual_root_cause,
            agent_root_cause=agent_root_cause,
            correctness=feedback.correctness,
            notes=feedback.notes,
            evidence_summary=evidence_summary,
        )

    def get_memory(self) -> list[dict[str, Any]]:
        self._ensure_memory_seeded()
        return self.memory.load_all()

    def get_reference_sources(self, operator_notes: str | None = None) -> list[dict[str, str]]:
        sources = load_reference_sources() + load_dataset_reference_sources()
        if operator_notes:
            dynamic_source = dynamic_reference_source(operator_notes)
            if dynamic_source:
                sources.append(dynamic_source)
        logger.debug("Loaded %s reference sources", len(sources))
        return sources

    def _ensure_memory_seeded(self) -> None:
        if self._memory_seeded:
            return
        self.memory.seed_records(load_incident_memory_records())
        self._memory_seeded = True
