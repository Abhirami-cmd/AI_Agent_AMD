from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class IncidentMemory:
    def __init__(self, path: str | Path = "data/incident_memory.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load_all(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def save_feedback(
        self,
        incident_id: str,
        service: str,
        selected_root_cause: str,
        agent_root_cause: str,
        correctness: str,
        notes: str,
        evidence_summary: str,
    ) -> None:
        records = self.load_all()
        records.append(
            {
                "stored_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "incident_id": incident_id,
                "service": service,
                "selected_root_cause": selected_root_cause,
                "agent_root_cause": agent_root_cause,
                "correctness": correctness,
                "notes": notes,
                "evidence_summary": evidence_summary,
            }
        )
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def find_similar(self, service: str, evidence_terms: list[str]) -> list[dict[str, Any]]:
        terms = {term.lower() for term in evidence_terms}
        matches = []
        for record in self.load_all():
            haystack = " ".join(
                [
                    record.get("service", ""),
                    record.get("selected_root_cause", ""),
                    record.get("agent_root_cause", ""),
                    record.get("evidence_summary", ""),
                    record.get("notes", ""),
                ]
            ).lower()
            service_match = record.get("service") == service
            term_overlap = any(term in haystack for term in terms)
            if service_match or term_overlap:
                matches.append(record)
        return matches[:3]
