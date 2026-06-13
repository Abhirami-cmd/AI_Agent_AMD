"""OpenRCA dataset loaders."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.models import (
    Dependency,
    Incident,
    TelemetryPoint,
)


def load_openrca_dataset(
    query_path: str,
    record_path: str,
    source_prefix: str,
    source_name: str,
) -> tuple[list[Incident], list[TelemetryPoint]]:
    query_file = Path(query_path)
    record_file = Path(record_path)
    if not query_file.exists() or not record_file.exists():
        return [], []

    query_df = pd.read_csv(query_file)
    record_df = pd.read_csv(record_file)

    incidents: list[Incident] = []
    telemetry_points: list[TelemetryPoint] = []

    for row_index, query_row in query_df.reset_index(drop=True).iterrows():
        if row_index >= len(record_df):
            break

        record_row = record_df.iloc[row_index]
        task_index = str(query_row["task_index"])
        instruction = str(query_row["instruction"])
        scoring_points = str(query_row["scoring_points"])
        rca_labels = _parse_scoring_points(scoring_points)

        timestamp = str(record_row["datetime"])
        component = str(record_row["component"])
        level = str(record_row["level"])
        reason = str(record_row["reason"])
        incident_id = _incident_id(source_prefix, timestamp, task_index, row_index)
        root_component = rca_labels.get("component", component)
        root_reason = rca_labels.get("reason", reason)

        incident = Incident(
            incident_id=incident_id,
            title=_extract_title_from_instruction(instruction),
            service=root_component,
            severity=_severity_from_task(task_index),
            started_at=timestamp,
            description=_build_incident_description(instruction, root_component, source_name),
            dependencies=_build_dependencies_from_component(root_component, level, source_name),
        )
        incidents.append(incident)

        telemetry_points.append(
            TelemetryPoint(
                incident_id=incident_id,
                timestamp=datetime.fromisoformat(timestamp),
                tower=_tower_from_level(level, root_reason),
                signal=root_reason,
                value=1.0,
                baseline=0.0,
                unit="anomaly",
                component=root_component,
            )
        )

    return incidents, telemetry_points


def load_openrca_cloudbed1(
    query_path: str = "data/openrca/market_cloudbed_1/query.csv",
    record_path: str = "data/openrca/market_cloudbed_1/record.csv",
) -> tuple[list[Incident], list[TelemetryPoint]]:
    return load_openrca_dataset(query_path, record_path, "CB1", "OpenRCA Cloudbed-1")


def load_openrca_cloudbed2(
    query_path: str = "data/openrca/market_cloudbed_2/query.csv",
    record_path: str = "data/openrca/market_cloudbed_2/record.csv",
) -> tuple[list[Incident], list[TelemetryPoint]]:
    return load_openrca_dataset(query_path, record_path, "CB2", "OpenRCA Cloudbed-2")


def load_openrca_telecom(
    query_path: str = "data/openrca/telecom/query.csv",
    record_path: str = "data/openrca/telecom/record.csv",
) -> tuple[list[Incident], list[TelemetryPoint]]:
    return load_openrca_dataset(query_path, record_path, "TEL", "OpenRCA Telecom")


def load_openrca_telemetry(
    query_path: str = "data/openrca/telemetry/query.csv",
    record_path: str = "data/openrca/telemetry/record.csv",
) -> tuple[list[Incident], list[TelemetryPoint]]:
    return load_openrca_dataset(query_path, record_path, "TM", "OpenRCA Telemetry")


def load_openrca_dataset_aggregated(
    query_path: str,
    record_path: str,
    source_prefix: str,
    source_name: str,
) -> tuple[list[Incident], list[TelemetryPoint]]:
    """
    Load OpenRCA dataset with task-based aggregation.
    
    Groups all records by task_index into a single incident with multiple
    telemetry variants. Reduces 121 incidents to 14 (7 tasks each for CB1 & TM).
    
    Returns:
        (incidents, telemetry_flat) where:
        - incidents: List of Incident objects with embedded telemetry variants
        - telemetry_flat: Flat list of all telemetry points (for backward compat)
    """
    query_file = Path(query_path)
    record_file = Path(record_path)
    if not query_file.exists() or not record_file.exists():
        return [], []

    query_df = pd.read_csv(query_file)
    record_df = pd.read_csv(record_file)

    # Group by task_index
    incidents_by_task: dict[str, dict] = {}
    telemetry_by_task: dict[str, list[TelemetryPoint]] = {}

    for row_index, query_row in query_df.reset_index(drop=True).iterrows():
        if row_index >= len(record_df):
            break

        record_row = record_df.iloc[row_index]
        task_index = str(query_row["task_index"])
        instruction = str(query_row["instruction"])
        scoring_points = str(query_row["scoring_points"])
        rca_labels = _parse_scoring_points(scoring_points)

        timestamp = str(record_row["datetime"])
        component = str(record_row["component"])
        level = str(record_row["level"])
        reason = str(record_row["reason"])
        
        root_component = rca_labels.get("component", component)
        root_reason = rca_labels.get("reason", reason)

        # Create telemetry point with variant metadata
        telemetry_point = TelemetryPoint(
            incident_id=task_index,  # Keyed by task
            timestamp=datetime.fromisoformat(timestamp),
            tower=_tower_from_level(level, root_reason),
            signal=root_reason,
            value=1.0,
            baseline=0.0,
            unit="anomaly",
            component=root_component,
            variant_index=row_index,  # Which variant within task
            root_component=root_component,  # Ground truth labels
            root_reason=root_reason,
        )

        # Aggregate telemetry by task
        if task_index not in telemetry_by_task:
            telemetry_by_task[task_index] = []
        telemetry_by_task[task_index].append(telemetry_point)

        # Store incident metadata on first occurrence
        if task_index not in incidents_by_task:
            incidents_by_task[task_index] = {
                "title": _extract_title_from_instruction(instruction),
                "service": root_component,
                "severity": _severity_from_task(task_index),
                "started_at": timestamp,
                "description": _build_incident_description(instruction, root_component, source_name),
                "first_component": root_component,
                "first_level": level,
            }

    # Build incidents with aggregated telemetry
    incidents: list[Incident] = []
    telemetry_flat: list[TelemetryPoint] = []

    for task_index, telemetry_list in telemetry_by_task.items():
        meta = incidents_by_task[task_index]
        
        # De-duplicate dependencies across variants
        dependencies_by_key: dict[tuple, Dependency] = {}
        for telemetry in telemetry_list:
            level = _level_from_tower_reverse(telemetry.tower, telemetry.signal)
            deps = _build_dependencies_from_component(telemetry.component, level, source_name)
            for dep in deps:
                key = (dep.source, dep.dependency, dep.tower)
                dependencies_by_key[key] = dep

        incident_id = _incident_id_aggregated(source_prefix, task_index)
        
        # Update incident_id in all telemetry points to match aggregated incident
        updated_telemetry = []
        for tp in telemetry_list:
            tp_dict = tp.dict()
            tp_dict['incident_id'] = incident_id
            updated_telemetry.append(TelemetryPoint(**tp_dict))
        
        incident = Incident(
            incident_id=incident_id,
            title=meta["title"],
            service=meta["service"],
            severity=meta["severity"],
            started_at=meta["started_at"],
            description=meta["description"],
            dependencies=list(dependencies_by_key.values()),
            telemetry=updated_telemetry,
            variant_count=len(updated_telemetry),
        )
        incidents.append(incident)
        telemetry_flat.extend(updated_telemetry)

    return incidents, telemetry_flat


def load_openrca_cloudbed1_aggregated(
    query_path: str = "data/openrca/market_cloudbed_1/query.csv",
    record_path: str = "data/openrca/market_cloudbed_1/record.csv",
) -> tuple[list[Incident], list[TelemetryPoint]]:
    return load_openrca_dataset_aggregated(query_path, record_path, "CB1", "OpenRCA Cloudbed-1")


def load_openrca_telemetry_aggregated(
    query_path: str = "data/openrca/telemetry/query.csv",
    record_path: str = "data/openrca/telemetry/record.csv",
) -> tuple[list[Incident], list[TelemetryPoint]]:
    return load_openrca_dataset_aggregated(query_path, record_path, "TM", "OpenRCA Telemetry")


def _parse_scoring_points(text: str) -> dict[str, str]:
    """
    Extract RCA labels from scoring_points field.

    Example:
        "The only predicted root cause component is node-1
         The only predicted root cause reason is node memory consumption"

    Returns:
        Dict with 'component' and 'reason' keys.
    """
    result = {}

    lines = text.split("\n")
    for line in lines:
        if "predicted root cause component" in line:
            # Extract component (last word after 'is')
            result["component"] = line.split("is")[-1].strip()
        elif "predicted root cause reason" in line:
            # Extract reason (last part after 'is')
            result["reason"] = line.split("is")[-1].strip()

    return result


def _build_incident_description(instruction: str, component: str, source_name: str) -> str:
    sentences = [sentence.strip() for sentence in instruction.split(".") if sentence.strip()]
    summary = sentences[0] if sentences else instruction
    summary = re.sub(r"You are tasked with.*", "", summary, flags=re.IGNORECASE).strip()
    if not summary.endswith('.'):
        summary += '.'
    return f"{summary} Affected component: {component}. Source dataset: {source_name}."


def _extract_title_from_instruction(instruction: str) -> str:
    """Extract a short title from instruction."""
    sentences = instruction.split(".")
    return sentences[0][:100] if sentences else "RCA Task"


def _incident_id(source_prefix: str, timestamp: str, task_index: str, row_index: int) -> str:
    normalized_time = timestamp.replace(" ", "T").replace(":", "")
    return f"{source_prefix}-{normalized_time}-{task_index}-{row_index + 1:03d}"


def _incident_id_aggregated(source_prefix: str, task_index: str) -> str:
    """Simpler incident ID for aggregated (task-based) incidents."""
    return f"{source_prefix}-{task_index}"


def _level_from_tower_reverse(tower: str, reason: str) -> str:
    """
    Reverse-infer level from tower and reason.
    Used in aggregated loader to reconstruct level for dependency building.
    """
    reason_lower = reason.lower()
    if tower == "storage":
        return "pod"
    if tower == "network":
        return "node"
    if tower == "application":
        return "service"
    # Default to pod for compute
    return "pod"


def _severity_from_task(task_index: str) -> str:
    return "Critical" if task_index in {"task_6", "task_7"} else "High"


def _tower_from_level(level: str, reason: str) -> str:
    reason_lower = reason.lower()
    if "db" in reason_lower or "disk" in reason_lower or "i/o" in reason_lower:
        return "storage"
    if "network" in reason_lower or "packet" in reason_lower or "delay" in reason_lower:
        return "network"
    if level == "service":
        return "application"
    return "compute"


def _build_dependencies_from_component(
    component: str,
    level: str,
    source_name: str,
) -> list[Dependency]:
    cluster = source_name.replace("OpenRCA ", "").lower().replace(" ", "-")
    if level == "service":
        return [
            Dependency(source=component, dependency=cluster, tower="application"),
        ]
    if level == "node":
        return [
            Dependency(source=cluster, dependency=component, tower="compute"),
        ]
    return [
        Dependency(source=component, dependency=cluster, tower="compute"),
    ]
