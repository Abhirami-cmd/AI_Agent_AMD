from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models import MemoryRecord


class IncidentMemoryRepository(ABC):
    @abstractmethod
    def load_all(self) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def save_record(self, record: MemoryRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def find_similar(self, service: str, evidence_terms: list[str]) -> list[MemoryRecord]:
        raise NotImplementedError


class JsonIncidentMemoryRepository(IncidentMemoryRepository):
    def __init__(self, path: str | Path = "data/incident_memory.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load_all(self) -> list[MemoryRecord]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return [MemoryRecord(**record) for record in raw]
        except json.JSONDecodeError:
            return []

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
        records = self.load_all()
        record = MemoryRecord(
            stored_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            incident_id=incident_id,
            service=service,
            selected_root_cause=selected_root_cause,
            actual_root_cause=actual_root_cause or "",
            agent_root_cause=agent_root_cause,
            correctness=correctness,
            notes=notes,
            evidence_summary=evidence_summary,
        )
        records.append(record)
        self.path.write_text(json.dumps([item.dict() for item in records], indent=2), encoding="utf-8")

    def save_record(self, record: MemoryRecord) -> None:
        records = self.load_all()
        if any(item.incident_id == record.incident_id for item in records):
            return
        records.append(record)
        self.path.write_text(json.dumps([item.dict() for item in records], indent=2), encoding="utf-8")

    def find_similar(self, service: str, evidence_terms: list[str]) -> list[MemoryRecord]:
        terms = {term.lower() for term in evidence_terms}
        matches: list[MemoryRecord] = []
        for record in self.load_all():
            haystack = " ".join(
                [
                    record.service,
                    record.selected_root_cause,
                    record.actual_root_cause,
                    record.agent_root_cause,
                    record.notes,
                    record.evidence_summary,
                ]
            ).lower()
            if record.service == service or any(term in haystack for term in terms):
                matches.append(record)
        return matches[:3]


class SQLiteIncidentMemoryRepository(IncidentMemoryRepository):
    def __init__(self, path: str | Path = "data/incident_memory.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_memory (
                stored_at TEXT,
                incident_id TEXT,
                service TEXT,
                selected_root_cause TEXT,
                actual_root_cause TEXT,
                agent_root_cause TEXT,
                correctness TEXT,
                notes TEXT,
                evidence_summary TEXT
            )
            """
        )
        existing_columns = {row[1] for row in self._connection.execute("PRAGMA table_info(incident_memory)").fetchall()}
        if "actual_root_cause" not in existing_columns:
            self._connection.execute(
                "ALTER TABLE incident_memory ADD COLUMN actual_root_cause TEXT"
            )
        self._connection.commit()

    def load_all(self) -> list[MemoryRecord]:
        cursor = self._connection.execute("SELECT * FROM incident_memory ORDER BY stored_at")
        rows = cursor.fetchall()
        return [MemoryRecord(**dict(row)) for row in rows]

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
        record = MemoryRecord(
            stored_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            incident_id=incident_id,
            service=service,
            selected_root_cause=selected_root_cause,
            actual_root_cause=actual_root_cause or "",
            agent_root_cause=agent_root_cause,
            correctness=correctness,
            notes=notes,
            evidence_summary=evidence_summary,
        )
        self._connection.execute(
            "INSERT INTO incident_memory (stored_at, incident_id, service, selected_root_cause, actual_root_cause, agent_root_cause, correctness, notes, evidence_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                record.stored_at,
                record.incident_id,
                record.service,
                record.selected_root_cause,
                record.actual_root_cause,
                record.agent_root_cause,
                record.correctness,
                record.notes,
                record.evidence_summary,
            ],
        )
        self._connection.commit()

    def save_record(self, record: MemoryRecord) -> None:
        existing = self._connection.execute(
            "SELECT 1 FROM incident_memory WHERE incident_id = ? LIMIT 1",
            [record.incident_id],
        ).fetchone()
        if existing:
            return
        self._connection.execute(
            "INSERT INTO incident_memory (stored_at, incident_id, service, selected_root_cause, actual_root_cause, agent_root_cause, correctness, notes, evidence_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                record.stored_at,
                record.incident_id,
                record.service,
                record.selected_root_cause,
                record.actual_root_cause,
                record.agent_root_cause,
                record.correctness,
                record.notes,
                record.evidence_summary,
            ],
        )
        self._connection.commit()

    def find_similar(self, service: str, evidence_terms: list[str]) -> list[MemoryRecord]:
        terms = {term.lower() for term in evidence_terms}
        results: list[MemoryRecord] = []
        for record in self.load_all():
            text = " ".join(
                [
                    record.service,
                    record.selected_root_cause,
                    record.agent_root_cause,
                    record.notes,
                    record.evidence_summary,
                ]
            ).lower()
            if record.service == service or any(term in text for term in terms):
                results.append(record)
        return results[:3]
