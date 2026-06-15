from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models import MemoryRecord


SYNTHETIC_COLUMNS = [
    "incident_id",
    "timestamp",
    "tower",
    "component",
    "signal",
    "value",
    "baseline",
    "unit",
]


def load_synthetic_telemetry(path: str) -> pd.DataFrame:
    telemetry_path = Path(path)
    if not telemetry_path.exists():
        return pd.DataFrame(columns=SYNTHETIC_COLUMNS)

    frame = pd.read_csv(telemetry_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").fillna(0.0)
    frame["baseline"] = pd.to_numeric(frame["baseline"], errors="coerce").fillna(0.0)
    return frame[SYNTHETIC_COLUMNS]


def load_synthetic_incidents(metadata_path: str) -> pd.DataFrame:
    path = Path(metadata_path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "incident_id",
                "title",
                "service",
                "severity",
                "started_at",
                "description",
                "dependencies",
                "variant_count",
            ]
        )

    metadata = pd.read_csv(path)
    metadata = metadata[metadata["split"] == "live"].copy()
    rows = []
    for _, row in metadata.iterrows():
        towers = str(row["towers"]).split("|")
        components = str(row["components"]).split("|")
        signals = str(row["signals"]).split("|")
        service = components[0] if components else "synthetic-service"
        dependencies = [
            {"source": service, "dependency": tower, "tower": tower}
            for tower in towers
        ]
        rows.append(
            {
                "incident_id": row["incident_id"],
                "title": f"Synthetic {row['anomaly_type']}",
                "service": service,
                "severity": "Critical",
                "started_at": row["start"],
                "description": (
                    f"Synthetic live incident for {row['anomaly_type']} affecting "
                    f"{len(towers)} towers and signals: {', '.join(signals)}."
                ),
                "dependencies": dependencies,
                "variant_count": len(towers) * len(signals),
            }
        )
    return pd.DataFrame(rows)


def load_synthetic_memory_records(memory_path: str) -> list[MemoryRecord]:
    path = Path(memory_path)
    if not path.exists():
        return []

    frame = pd.read_csv(path).fillna("")
    records: list[MemoryRecord] = []
    for _, row in frame.iterrows():
        evidence_context = (
            f"{row.get('evidence_summary', '')} "
            f"Tower: {row.get('tower', '')}. "
            f"Component: {row.get('component', '')}. "
            f"Signal: {row.get('signal', '')}."
        ).strip()
        records.append(
            MemoryRecord(
                stored_at=str(row.get("stored_at", "")),
                incident_id=str(row.get("incident_id", "")),
                service=str(row.get("service", "")),
                selected_root_cause=str(row.get("selected_root_cause", "")),
                actual_root_cause=str(row.get("actual_root_cause", "")),
                agent_root_cause=str(row.get("agent_root_cause", "")),
                correctness=str(row.get("correctness", "")),
                notes=str(row.get("notes", "")),
                evidence_summary=evidence_context,
            )
        )
    return records


def synthetic_dataset_statistics(
    telemetry: pd.DataFrame,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    if telemetry.empty:
        return {
            "row_count": 0,
            "tower_count": 0,
            "component_count": 0,
            "signal_count": 0,
            "anomaly_count": 0,
            "anomaly_percentage": 0.0,
            "sequence_count_generated": 0,
            "gpu_device_used": "none",
            "training_duration_seconds": 0.0,
        }

    anomaly_count = int((telemetry["incident_id"].astype(str) != "normal").sum())
    row_count = int(len(telemetry))
    metadata = metadata or {}
    return {
        "row_count": row_count,
        "tower_count": int(telemetry["tower"].nunique()),
        "component_count": int(telemetry["component"].nunique()),
        "signal_count": int(telemetry["signal"].nunique()),
        "anomaly_count": anomaly_count,
        "anomaly_percentage": round((anomaly_count / row_count) * 100, 3),
        "sequence_count_generated": int(metadata.get("sequence_count", 0) or 0),
        "gpu_device_used": str(metadata.get("device", "unknown")),
        "training_duration_seconds": round(
            float(metadata.get("training_duration_seconds", 0.0) or 0.0),
            3,
        ),
    }
