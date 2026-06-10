from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


EXCEL_DATA_PATH = Path("data/observability_sample.xlsx")


def load_sample_incidents() -> pd.DataFrame:
    excel_data = _load_from_excel()
    if excel_data is not None:
        incidents, _ = excel_data
        return incidents

    return pd.DataFrame(
        [
            {
                "incident_id": "INC-001",
                "title": "Checkout latency and payment failures",
                "service": "checkout-service",
                "severity": "Critical",
                "started_at": "2026-06-10 10:15",
                "description": (
                    "Checkout requests are timing out and payment authorization failures "
                    "started after database write latency increased."
                ),
                "dependencies": [
                    {"source": "checkout-service", "dependency": "payment-api", "tower": "application"},
                    {"source": "checkout-service", "dependency": "payment-db", "tower": "storage"},
                    {"source": "payment-api", "dependency": "node-pool-a", "tower": "compute"},
                    {"source": "payment-api", "dependency": "east-lb", "tower": "network"},
                ],
            },
            {
                "incident_id": "INC-002",
                "title": "Inventory API elevated 5xx errors",
                "service": "inventory-api",
                "severity": "High",
                "started_at": "2026-06-10 11:35",
                "description": (
                    "Inventory reads are failing after a deployment introduced a higher error rate "
                    "while infrastructure signals stayed mostly stable."
                ),
                "dependencies": [
                    {"source": "inventory-api", "dependency": "catalog-db", "tower": "storage"},
                    {"source": "inventory-api", "dependency": "node-pool-b", "tower": "compute"},
                    {"source": "inventory-api", "dependency": "west-lb", "tower": "network"},
                ],
            },
        ]
    )


def load_sample_telemetry() -> pd.DataFrame:
    excel_data = _load_from_excel()
    if excel_data is not None:
        _, telemetry = excel_data
        return telemetry

    rows = []
    rows.extend(_incident_one())
    rows.extend(_incident_two())
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def _load_from_excel() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not EXCEL_DATA_PATH.exists():
        return None

    try:
        sheets = pd.read_excel(
            EXCEL_DATA_PATH,
            sheet_name=["incidents", "dependencies", "telemetry"],
            engine="openpyxl",
        )
    except Exception:
        return None

    incidents = sheets["incidents"]
    dependencies = sheets["dependencies"]
    telemetry = sheets["telemetry"]

    incidents = incidents.copy()
    incidents["dependencies"] = incidents["incident_id"].apply(
        lambda incident_id: dependencies[
            dependencies["incident_id"] == incident_id
        ][["source", "dependency", "tower"]].to_dict("records")
    )

    telemetry = telemetry.copy()
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"])
    telemetry["value"] = pd.to_numeric(telemetry["value"])
    telemetry["baseline"] = pd.to_numeric(telemetry["baseline"])
    return incidents, telemetry


def _points(
    incident_id: str,
    base_time: datetime,
    tower: str,
    signal: str,
    values: list[float],
    baseline: float,
    unit: str,
    component: str,
) -> list[dict]:
    return [
        {
            "incident_id": incident_id,
            "timestamp": base_time + timedelta(minutes=5 * index),
            "tower": tower,
            "signal": signal,
            "value": value,
            "baseline": baseline,
            "unit": unit,
            "component": component,
        }
        for index, value in enumerate(values)
    ]


def _incident_one() -> list[dict]:
    start = datetime(2026, 6, 10, 9, 55)
    rows = []
    rows += _points("INC-001", start, "application", "checkout_p95_latency_ms", [180, 195, 210, 420, 950, 1200, 980], 220, "ms", "checkout-service")
    rows += _points("INC-001", start, "application", "checkout_error_rate_pct", [0.4, 0.5, 0.7, 2.2, 8.9, 12.5, 9.6], 1.0, "%", "checkout-service")
    rows += _points("INC-001", start, "storage", "payment_db_write_latency_ms", [8, 10, 12, 130, 260, 310, 190], 15, "ms", "payment-db")
    rows += _points("INC-001", start, "storage", "payment_db_iops", [1100, 1180, 1250, 2600, 3300, 3400, 2900], 1400, "iops", "payment-db")
    rows += _points("INC-001", start, "compute", "node_pool_cpu_pct", [52, 55, 61, 70, 74, 78, 73], 65, "%", "node-pool-a")
    rows += _points("INC-001", start, "network", "east_lb_packet_loss_pct", [0.1, 0.1, 0.2, 0.5, 0.7, 0.6, 0.4], 0.3, "%", "east-lb")
    return rows


def _incident_two() -> list[dict]:
    start = datetime(2026, 6, 10, 11, 15)
    rows = []
    rows += _points("INC-002", start, "application", "inventory_5xx_rate_pct", [0.3, 0.4, 1.2, 7.8, 10.4, 9.1, 6.2], 1.0, "%", "inventory-api")
    rows += _points("INC-002", start, "application", "deployment_errors", [0, 0, 1, 9, 15, 12, 8], 1, "count", "inventory-api")
    rows += _points("INC-002", start, "storage", "catalog_db_read_latency_ms", [12, 14, 17, 22, 24, 25, 20], 18, "ms", "catalog-db")
    rows += _points("INC-002", start, "compute", "node_pool_memory_pct", [61, 63, 65, 67, 70, 72, 69], 75, "%", "node-pool-b")
    rows += _points("INC-002", start, "network", "west_lb_latency_ms", [18, 19, 21, 26, 28, 27, 23], 25, "ms", "west-lb")
    return rows
