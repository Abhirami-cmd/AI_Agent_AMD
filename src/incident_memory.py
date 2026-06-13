from __future__ import annotations

from typing import Any

from src.config import settings
from pathlib import Path
from src.models import MemoryRecord
from src.repos.memory_repo import (
    IncidentMemoryRepository,
    JsonIncidentMemoryRepository,
    SQLiteIncidentMemoryRepository,
)


class IncidentMemory:
    def __init__(self, repository: IncidentMemoryRepository | str | Path | None = None) -> None:
        if isinstance(repository, (str, Path)):
            self.repository = JsonIncidentMemoryRepository(repository)
        elif repository is not None:
            self.repository = repository
        elif settings.database_url.startswith("sqlite"):
            self.repository = SQLiteIncidentMemoryRepository("data/incident_memory.db")
        else:
            self.repository = JsonIncidentMemoryRepository()

    def load_all(self) -> list[dict[str, Any]]:
        return [record.dict() for record in self.repository.load_all()]

    def save_feedback(
        self,
        incident_id: str,
        service: str,
        selected_root_cause: str,
        actual_root_cause: str | None,
        agent_root_cause: str,
        correctness: str,
        notes: str,
        evidence_summary: str,
    ) -> None:
        self.repository.save_feedback(
            incident_id=incident_id,
            service=service,
            selected_root_cause=selected_root_cause,
            actual_root_cause=actual_root_cause,
            agent_root_cause=agent_root_cause,
            correctness=correctness,
            notes=notes,
            evidence_summary=evidence_summary,
        )

    def seed_records(self, records: list[MemoryRecord]) -> None:
        for record in records:
            self.repository.save_record(record)

    def find_similar(self, service: str, evidence_terms: list[str]) -> list[dict[str, Any]]:
        return [record.dict() for record in self.repository.find_similar(service, evidence_terms)]
