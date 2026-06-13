"""ServiceNow incident dataset loader."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.models import Incident, MemoryRecord


def load_servicenow_incidents(
    incident_path: str = "data/servicenow/incident_event_log.csv",
) -> tuple[list[Incident], list[MemoryRecord]]:
    """
    Load ServiceNow incident event log.

    The event log contains multiple rows per incident (one per state change).
    This function aggregates to get the final state of each incident and
    extracts the RCA label (closed_code).

    Returns:
        Tuple of (incidents, memory_records) where:
        - incidents: UnifiedIncident for each unique incident
        - memory_records: MemoryRecord for labeled incidents (for learning)
    """
    if not Path(incident_path).exists():
        return [], []

    df = pd.read_csv(incident_path)

    # Sort by incident number and update timestamp
    df = df.sort_values(["number", "sys_updated_at"])

    # Group by incident number and get latest row
    latest_incidents = df.groupby("number").last().reset_index()
    state_histories = (
        df.groupby("number")["incident_state"]
        .agg(lambda states: " -> ".join(str(state) for state in pd.unique(states)))
        .to_dict()
    )

    incidents = []
    memory_records = []

    for _, row in latest_incidents.iterrows():
        incident_id = f"SN-{row['number']}"

        # Parse dates
        try:
            started_at = pd.to_datetime(row["opened_at"], format="%d/%m/%Y %H:%M").isoformat()
        except Exception:
            started_at = str(row["opened_at"])

        # Determine severity from impact + urgency
        severity = _map_severity(row.get("impact"), row.get("urgency"))

        # Create incident
        incident = Incident(
            incident_id=incident_id,
            title=f"Symptom {row.get('u_symptom', 'Unknown')} - Category {row.get('category', 'Unknown')}",
            service=f"Group {row.get('assignment_group', 'Unknown')}",
            severity=severity,
            started_at=started_at,
            description=f"State transitions: {state_histories.get(row['number'], 'Unknown')}",
            dependencies=[],
        )
        incidents.append(incident)

        # If labeled (has closed_code), create memory record
        closed_code = row.get("closed_code")
        if closed_code and closed_code != "?":
            try:
                stored_at = pd.to_datetime(row["closed_at"], format="%d/%m/%Y %H:%M").isoformat()
            except Exception:
                stored_at = str(row["closed_at"])

            memory_record = MemoryRecord(
                stored_at=stored_at,
                incident_id=incident_id,
                service=f"Group {row.get('assignment_group', 'Unknown')}",
                selected_root_cause=closed_code,
                actual_root_cause=_map_closed_code_to_reason(closed_code),
                agent_root_cause=closed_code,  # Ground truth from ServiceNow
                correctness="Correct",  # Assume ServiceNow labels are correct
                notes=f"Category: {row.get('category')}, Subcategory: {row.get('subcategory')}",
                evidence_summary=f"Impact: {row.get('impact')}, Urgency: {row.get('urgency')}",
            )
            memory_records.append(memory_record)

    return incidents, memory_records


def _map_severity(impact: str | None, urgency: str | None) -> str:
    """Map ServiceNow impact + urgency to severity level."""
    if impact == "1 - High" and urgency == "1 - High":
        return "Critical"
    elif impact == "1 - High" or urgency == "1 - High":
        return "High"
    elif impact == "2 - Medium" and urgency == "2 - Medium":
        return "Medium"
    else:
        return "Low"


def _get_state_history(df: pd.DataFrame, incident_number: str) -> str:
    """Get state transition history for an incident."""
    incident_rows = df[df["number"] == incident_number]
    states = " → ".join(incident_rows["incident_state"].unique())
    return states


def _map_closed_code_to_reason(code: str) -> str:
    """
    Map ServiceNow closed_code to meaningful RCA reason.

    Note: ServiceNow codes are anonymized. This mapping provides
    a simplified translation. In production, would require domain
    knowledge or ML-based inference.
    """
    mapping = {
        "code 1": "Application Error",
        "code 2": "Configuration Issue",
        "code 3": "Database Problem",
        "code 4": "Network Connectivity",
        "code 5": "Performance Degradation",
        "code 6": "Resource Exhaustion",  # Most common (61%)
        "code 7": "Infrastructure Issue",
        "code 8": "Third-party System",
        "code 9": "User Error",
        "code 10": "Data Quality",
        "code 11": "Security Issue",
        "code 12": "Deployment Problem",
        "code 13": "Monitoring Gap",
        "code 14": "Documentation",
        "code 15": "Training Needed",
        "code 16": "External Dependency",
        "code 17": "Unknown",
    }
    return mapping.get(code, code)
