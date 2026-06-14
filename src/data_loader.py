from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from src.config import settings
from src.ingest.openrca_loader import (
    load_openrca_cloudbed1,
    load_openrca_cloudbed2,
    load_openrca_telecom,
    load_openrca_telemetry,
    load_openrca_cloudbed1_aggregated,
    load_openrca_telemetry_aggregated,
)
from src.ingest.preprocessing import merge_datasets
from src.ingest.servicenow_loader import load_servicenow_incidents
from src.ingest.synthetic_loader import (
    load_synthetic_incidents,
    load_synthetic_telemetry,
    synthetic_dataset_statistics,
)
from src.models import Incident, MemoryRecord, TelemetryPoint

logger = logging.getLogger(__name__)


def load_incidents(limit: int | None = None) -> pd.DataFrame:
    frame = load_synthetic_incidents(settings.synthetic_metadata_path)
    if frame.empty:
        incidents, _, _ = load_real_datasets()
        frame = _incidents_to_frame(incidents)
    return frame.head(limit) if limit else frame


def load_telemetry() -> pd.DataFrame:
    frame = load_synthetic_telemetry(settings.synthetic_live_path)
    if not frame.empty:
        return frame
    _, telemetry, _ = load_real_datasets()
    return _telemetry_to_frame(telemetry)


def load_training_telemetry() -> pd.DataFrame:
    return load_synthetic_telemetry(settings.synthetic_train_path)


def load_telemetry_dataset_statistics(metadata: dict[str, object] | None = None) -> dict[str, object]:
    return synthetic_dataset_statistics(load_telemetry(), metadata)


def load_servicenow_memory_records() -> list[MemoryRecord]:
    return _load_servicenow_memory_records()


def load_dataset_reference_sources() -> list[dict[str, str]]:
    incidents, telemetry, _ = load_real_datasets()
    telemetry_by_incident: dict[str, list[TelemetryPoint]] = {}
    for point in telemetry:
        telemetry_by_incident.setdefault(point.incident_id, []).append(point)

    lines = []
    for incident in incidents:
        points = telemetry_by_incident.get(incident.incident_id, [])
        for point in points:
            lines.append(
                f"{point.signal} on {point.component}: "
                f"{incident.service} had this failure "
                f"in the {point.tower} tower at {point.timestamp.isoformat()}. "
                f"Incident {incident.incident_id}. {incident.description}"
            )

    if not lines:
        return []

    return [
        {
            "name": "OpenRCA Ground Truth Patterns",
            "path": "data/openrca",
            "type": "csv",
            "text": "\n".join(lines),
        }
    ]


@lru_cache(maxsize=1)
def load_real_datasets() -> tuple[list[Incident], list[TelemetryPoint], list[MemoryRecord]]:
    """
    Load real datasets with task-based aggregation for improved RCA grouping.
    
    Aggregation reduces incident count from 121 to 14 (7 tasks each for CloudBed-1 & Telemetry)
    while preserving all variants as embedded telemetry points for richer analysis.
    """
    openrca_incidents: list[Incident] = []
    openrca_telemetry: list[TelemetryPoint] = []

    # Use aggregated loaders for main datasets (CloudBed-1, Telemetry)
    for loader, query_path, record_path in [
        (
            load_openrca_cloudbed1_aggregated,
            settings.openrca_cloudbed1_query_path,
            settings.openrca_cloudbed1_record_path,
        ),
        (
            load_openrca_telemetry_aggregated,
            settings.openrca_telemetry_query_path,
            settings.openrca_telemetry_record_path,
        ),
    ]:
        incidents, telemetry = loader(query_path, record_path)
        openrca_incidents.extend(incidents)
        openrca_telemetry.extend(telemetry)

    # Keep original loaders for secondary datasets (CB-2, Telecom)
    for loader, query_path, record_path in [
        (
            load_openrca_cloudbed2,
            settings.openrca_cloudbed2_query_path,
            settings.openrca_cloudbed2_record_path,
        ),
        (
            load_openrca_telecom,
            settings.openrca_telecom_query_path,
            settings.openrca_telecom_record_path,
        ),
    ]:
        incidents, telemetry = loader(query_path, record_path)
        openrca_incidents.extend(incidents)
        openrca_telemetry.extend(telemetry)

    incidents, telemetry = merge_datasets(openrca_incidents, openrca_telemetry, [])
    logger.info(
        "Loaded %s OpenRCA incidents and %s telemetry points (aggregated)",
        len(incidents),
        len(telemetry),
    )
    if not incidents:
        raise FileNotFoundError(
            "No OpenRCA incidents were loaded. Check data/openrca CSV paths in configuration."
        )
    return incidents, telemetry, []


@lru_cache(maxsize=1)
def _load_servicenow_memory_records() -> list[MemoryRecord]:
    servicenow_incidents, servicenow_memory = load_servicenow_incidents(settings.servicenow_path)
    if not servicenow_incidents:
        logger.warning("No ServiceNow incidents were loaded from %s", settings.servicenow_path)
    return servicenow_memory


def _incidents_to_frame(incidents: list[Incident]) -> pd.DataFrame:
    frame = pd.DataFrame([incident.dict() for incident in incidents])
    if frame.empty:
        return frame
    frame["dependencies"] = frame["dependencies"].apply(
        lambda items: [
            item.dict() if hasattr(item, "dict") else dict(item)
            for item in (items or [])
        ]
    )
    return frame


def _telemetry_to_frame(telemetry: list[TelemetryPoint]) -> pd.DataFrame:
    frame = pd.DataFrame([point.dict() for point in telemetry])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "timestamp",
                "tower",
                "signal",
                "value",
                "baseline",
                "unit",
                "component",
            ]
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["value"] = pd.to_numeric(frame["value"])
    frame["baseline"] = pd.to_numeric(frame["baseline"])
    return frame
