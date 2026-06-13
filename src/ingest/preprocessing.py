"""Data preprocessing pipeline for unified incident ingestion."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.models import Incident, TelemetryPoint


def preprocess_incidents(incidents: list[Incident]) -> list[Incident]:
    """
    Preprocess incidents by:
    1. Normalizing timestamps to ISO format
    2. Standardizing severity values
    3. Removing duplicates
    4. Handling nulls
    5. Adding tags

    Returns:
        Cleaned list of incidents.
    """
    # Remove duplicates by incident_id (keep first)
    seen = set()
    unique_incidents = []
    for incident in incidents:
        if incident.incident_id not in seen:
            unique_incidents.append(incident)
            seen.add(incident.incident_id)

    # Standardize and validate
    for incident in unique_incidents:
        # Normalize timestamps
        incident.started_at = _normalize_timestamp(incident.started_at)

        # Standardize severity
        incident.severity = _normalize_severity(incident.severity)

        # Handle nulls
        if incident.dependencies is None:
            incident.dependencies = []
        if incident.description is None:
            incident.description = incident.title

    return unique_incidents


def preprocess_telemetry(telemetry_points: list[TelemetryPoint]) -> list[TelemetryPoint]:
    """
    Preprocess telemetry points by:
    1. Normalizing timestamps to UTC
    2. Handling null values
    3. Validating numeric fields
    4. Removing duplicates

    Returns:
        Cleaned list of telemetry points.
    """
    # Remove duplicates while preserving task-aggregated variants. Aggregated
    # OpenRCA rows can share incident/timestamp/component but differ by signal,
    # tower, or variant metadata.
    seen = set()
    unique_points = []
    for point in telemetry_points:
        key = _telemetry_dedupe_key(point)
        if key not in seen:
            unique_points.append(point)
            seen.add(key)

    # Validate and normalize
    for point in unique_points:
        # Ensure timestamp is datetime
        if isinstance(point.timestamp, str):
            point.timestamp = datetime.fromisoformat(point.timestamp)

        # Validate numeric values
        if not isinstance(point.value, (int, float)):
            point.value = 0.0
        if not isinstance(point.baseline, (int, float)):
            point.baseline = 0.0

        # Ensure tower is valid
        valid_towers = {"application", "storage", "compute", "network", "unknown"}
        if point.tower not in valid_towers:
            point.tower = "unknown"

    return unique_points


def _normalize_timestamp(timestamp_str: str) -> str:
    """
    Normalize timestamp string to ISO format (YYYY-MM-DD HH:MM:SS).

    Handles multiple input formats:
    - ISO format: already valid
    - DD/MM/YYYY HH:MM: ServiceNow format
    - Other formats: attempt parsing
    """
    if not timestamp_str or timestamp_str == "?":
        return datetime.now().isoformat()

    # Try common formats
    formats = [
        "%Y-%m-%d %H:%M:%S",  # ISO
        "%Y-%m-%d %H:%M",  # ISO short
        "%d/%m/%Y %H:%M",  # ServiceNow
        "%Y-%m-%dT%H:%M:%S",  # ISO with T
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(str(timestamp_str).strip(), fmt)
            return dt.isoformat()
        except ValueError:
            continue

    # Fallback: return as-is
    return str(timestamp_str)


def _normalize_severity(severity_str: str | None) -> str:
    """
    Normalize severity value to standard levels:
    - "Critical", "CRITICAL", "1" → "Critical"
    - "High", "HIGH", "2" → "High"
    - "Medium", "MEDIUM", "3" → "Medium"
    - "Low", "LOW", "4" → "Low"
    """
    if not severity_str:
        return "Medium"

    s = str(severity_str).strip().upper()

    mapping = {
        "CRITICAL": "Critical",
        "1": "Critical",
        "HIGH": "High",
        "2": "High",
        "MEDIUM": "Medium",
        "3": "Medium",
        "LOW": "Low",
        "4": "Low",
    }

    return mapping.get(s, severity_str.capitalize())


def deduplicate_incidents(incidents: list[Incident]) -> list[Incident]:
    """Remove duplicate incidents by incident_id (keep first)."""
    seen = set()
    unique = []
    for inc in incidents:
        if inc.incident_id not in seen:
            unique.append(inc)
            seen.add(inc.incident_id)
    return unique


def deduplicate_telemetry(telemetry: list[TelemetryPoint]) -> list[TelemetryPoint]:
    """Remove duplicate telemetry points while preserving aggregated variants."""
    seen = set()
    unique = []
    for point in telemetry:
        key = _telemetry_dedupe_key(point)
        if key not in seen:
            unique.append(point)
            seen.add(key)
    return unique


def _telemetry_dedupe_key(point: TelemetryPoint) -> tuple:
    return (
        point.incident_id,
        str(point.timestamp),
        point.component,
        point.tower,
        point.signal,
        point.variant_index,
        point.root_component,
        point.root_reason,
    )


def merge_datasets(
    openrca_incidents: list[Incident],
    openrca_telemetry: list[TelemetryPoint],
    servicenow_incidents: list[Incident],
) -> tuple[list[Incident], list[TelemetryPoint]]:
    """
    Merge incidents and telemetry from multiple sources.

    Returns:
        Tuple of (all_incidents, all_telemetry) with duplicates removed.
    """
    all_incidents = openrca_incidents + servicenow_incidents
    all_incidents = deduplicate_incidents(all_incidents)
    all_incidents = preprocess_incidents(all_incidents)

    all_telemetry = openrca_telemetry
    all_telemetry = deduplicate_telemetry(all_telemetry)
    all_telemetry = preprocess_telemetry(all_telemetry)

    return all_incidents, all_telemetry
