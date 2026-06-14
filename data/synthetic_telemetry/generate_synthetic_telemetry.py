from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from tower_label_mapping import TowerLabelMapper


TOWERS = [f"tower_{index:02d}" for index in range(1, 21)]
COMPONENTS = [
    "shippingservice",
    "emailservice",
    "cartservice",
    "paymentservice",
    "recommendationservice",
    "checkoutservice",
    "frontend",
    "node",
    "gpu-worker",
]
SIGNALS = [
    "cpu_utilization",
    "memory_usage",
    "disk_io",
    "network_latency",
    "error_rate",
    "gpu_utilization",
    "gpu_temperature",
    "request_latency",
]

UNITS = {
    "cpu_utilization": "percent",
    "memory_usage": "percent",
    "disk_io": "MBps",
    "network_latency": "ms",
    "error_rate": "percent",
    "gpu_utilization": "percent",
    "gpu_temperature": "celsius",
    "request_latency": "ms",
}

BASE_LEVELS = {
    "cpu_utilization": 44.0,
    "memory_usage": 52.0,
    "disk_io": 95.0,
    "network_latency": 38.0,
    "error_rate": 0.8,
    "gpu_utilization": 48.0,
    "gpu_temperature": 58.0,
    "request_latency": 145.0,
}

NOISE = {
    "cpu_utilization": 4.0,
    "memory_usage": 2.8,
    "disk_io": 15.0,
    "network_latency": 5.0,
    "error_rate": 0.18,
    "gpu_utilization": 6.0,
    "gpu_temperature": 2.2,
    "request_latency": 18.0,
}


@dataclass(frozen=True)
class IncidentSpec:
    incident_id: str
    anomaly_type: str
    start: datetime
    end: datetime
    towers: tuple[str, ...]
    components: tuple[str, ...]
    signals: tuple[str, ...]
    split: str


def _component_for_tower(tower: str) -> str:
    tower_index = int(tower.split("_")[1]) - 1
    return COMPONENTS[tower_index % len(COMPONENTS)]


def _clip(signal: str, value: float) -> float:
    if signal in {
        "cpu_utilization",
        "memory_usage",
        "error_rate",
        "gpu_utilization",
    }:
        return max(0.0, min(100.0, value))
    if signal == "gpu_temperature":
        return max(20.0, min(105.0, value))
    return max(0.0, value)


def _baseline(signal: str, tower: str, timestamp: datetime) -> float:
    tower_index = int(tower.split("_")[1])
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    day_phase = 2 * math.pi * minute_of_day / 1440
    week_phase = 2 * math.pi * timestamp.weekday() / 7
    tower_bias = ((tower_index % 5) - 2) * 1.7

    daily = math.sin(day_phase - math.pi / 3)
    business_hours = 1.0 if 8 <= timestamp.hour <= 20 else -0.35
    weekly = math.sin(week_phase)
    drift = (timestamp - START_TIME).total_seconds() / (30 * 24 * 3600)

    base = BASE_LEVELS[signal]
    if signal in {"cpu_utilization", "gpu_utilization"}:
        value = base + 12 * daily + 5 * business_hours + tower_bias + 5 * drift
    elif signal == "memory_usage":
        value = base + 4 * daily + tower_bias + 8 * drift
    elif signal == "disk_io":
        value = base + 28 * max(0.0, daily) + 10 * weekly + tower_bias * 3
    elif signal == "network_latency":
        value = base + 9 * max(0.0, daily) + 3 * weekly + tower_bias
    elif signal == "error_rate":
        value = base + 0.4 * max(0.0, daily) + 0.15 * weekly
    elif signal == "gpu_temperature":
        value = base + 0.18 * _baseline("gpu_utilization", tower, timestamp)
    else:
        value = base + 34 * max(0.0, daily) + 8 * business_hours + tower_bias * 2

    return round(_clip(signal, value), 3)


def _build_incidents() -> list[IncidentSpec]:
    return [
        IncidentSpec(
            "SYN-TRAIN-CPU-001",
            "CPU saturation",
            START_TIME + timedelta(days=5, hours=10),
            START_TIME + timedelta(days=5, hours=16),
            ("tower_03", "tower_04", "tower_05", "tower_06", "tower_07"),
            ("checkoutservice", "frontend", "paymentservice"),
            ("cpu_utilization", "memory_usage", "request_latency", "error_rate"),
            "train",
        ),
        IncidentSpec(
            "SYN-TRAIN-MEM-001",
            "Memory leak",
            START_TIME + timedelta(days=11, hours=8),
            START_TIME + timedelta(days=11, hours=16),
            ("tower_07", "tower_08", "tower_09", "tower_10", "tower_11"),
            ("cartservice", "paymentservice"),
            ("memory_usage", "cpu_utilization", "request_latency", "error_rate"),
            "train",
        ),
        IncidentSpec(
            "SYN-TRAIN-DISK-001",
            "Disk I/O bottleneck",
            START_TIME + timedelta(days=17, hours=15),
            START_TIME + timedelta(days=17, hours=21),
            ("tower_10", "tower_11", "tower_12", "tower_13"),
            ("shippingservice", "node"),
            ("disk_io", "cpu_utilization", "request_latency", "error_rate"),
            "train",
        ),
        IncidentSpec(
            "SYN-TRAIN-NET-001",
            "Network corruption",
            START_TIME + timedelta(days=3, hours=9),
            START_TIME + timedelta(days=3, hours=14),
            ("tower_01", "tower_02", "tower_03", "tower_04", "tower_05"),
            ("frontend", "checkoutservice"),
            ("network_latency", "request_latency", "error_rate"),
            "train",
        ),
        IncidentSpec(
            "SYN-TRAIN-GPU-001",
            "GPU overload",
            START_TIME + timedelta(days=7, hours=12),
            START_TIME + timedelta(days=7, hours=19),
            ("tower_14", "tower_15", "tower_16", "tower_17", "tower_18"),
            ("gpu-worker", "recommendationservice"),
            ("gpu_utilization", "gpu_temperature", "request_latency", "error_rate"),
            "train",
        ),
        IncidentSpec(
            "SYN-TRAIN-CPU-002",
            "CPU saturation",
            START_TIME + timedelta(days=9, hours=6),
            START_TIME + timedelta(days=9, hours=14),
            ("tower_12", "tower_13", "tower_14", "tower_15"),
            ("emailservice", "cartservice"),
            ("cpu_utilization", "memory_usage", "request_latency", "error_rate"),
            "train",
        ),
        IncidentSpec(
            "SYN-TRAIN-MEM-002",
            "Memory leak",
            START_TIME + timedelta(days=13, hours=10),
            START_TIME + timedelta(days=13, hours=17),
            ("tower_16", "tower_17", "tower_18", "tower_19"),
            ("frontend", "shippingservice"),
            ("memory_usage", "request_latency", "error_rate"),
            "train",
        ),
        IncidentSpec(
            "SYN-TRAIN-DISK-002",
            "Disk I/O bottleneck",
            START_TIME + timedelta(days=15, hours=7),
            START_TIME + timedelta(days=15, hours=15),
            ("tower_02", "tower_06", "tower_10", "tower_14", "tower_18"),
            ("node", "paymentservice"),
            ("disk_io", "request_latency", "error_rate"),
            "train",
        ),
        IncidentSpec(
            "SYN-TRAIN-NET-002",
            "Network corruption",
            START_TIME + timedelta(days=19, hours=11),
            START_TIME + timedelta(days=19, hours=17),
            ("tower_06", "tower_07", "tower_08", "tower_09"),
            ("checkoutservice", "recommendationservice"),
            ("network_latency", "request_latency", "error_rate"),
            "train",
        ),
        IncidentSpec(
            "SYN-TRAIN-GPU-002",
            "GPU overload",
            START_TIME + timedelta(days=22, hours=13),
            START_TIME + timedelta(days=22, hours=19),
            ("tower_04", "tower_08", "tower_20"),
            ("gpu-worker", "frontend"),
            ("gpu_utilization", "gpu_temperature", "request_latency", "error_rate"),
            "train",
        ),
        IncidentSpec(
            "SYN-LIVE-NET-001",
            "Network corruption",
            START_TIME + timedelta(days=24, hours=9),
            START_TIME + timedelta(days=24, hours=15),
            ("tower_02", "tower_05", "tower_06", "tower_13", "tower_19"),
            ("frontend", "checkoutservice", "paymentservice"),
            ("network_latency", "error_rate", "request_latency"),
            "live",
        ),
        IncidentSpec(
            "SYN-LIVE-GPU-001",
            "GPU overload",
            START_TIME + timedelta(days=27, hours=13),
            START_TIME + timedelta(days=27, hours=21),
            ("tower_09", "tower_14", "tower_15", "tower_16", "tower_20"),
            ("gpu-worker", "recommendationservice"),
            ("gpu_utilization", "gpu_temperature", "request_latency", "error_rate"),
            "live",
        ),
        IncidentSpec(
            "SYN-LIVE-CPU-002",
            "CPU saturation",
            START_TIME + timedelta(days=29, hours=7),
            START_TIME + timedelta(days=29, hours=13),
            ("tower_01", "tower_11", "tower_17", "tower_18"),
            ("frontend", "emailservice"),
            ("cpu_utilization", "memory_usage", "request_latency", "error_rate"),
            "live",
        ),
        IncidentSpec(
            "SYN-LIVE-MEM-001",
            "Memory leak",
            START_TIME + timedelta(days=25, hours=16),
            START_TIME + timedelta(days=25, hours=21),
            ("tower_03", "tower_12", "tower_20"),
            ("cartservice", "shippingservice"),
            ("memory_usage", "request_latency", "error_rate"),
            "live",
        ),
    ]


def _incident_multiplier(signal: str, timestamp: datetime, spec: IncidentSpec) -> float:
    duration = max(1.0, (spec.end - spec.start).total_seconds())
    progress = (timestamp - spec.start).total_seconds() / duration
    pulse = math.sin(math.pi * min(1.0, max(0.0, progress)))

    if spec.anomaly_type == "CPU saturation" and signal == "cpu_utilization":
        return 35 + 18 * pulse
    if spec.anomaly_type == "CPU saturation" and signal in {"request_latency", "error_rate"}:
        return 28 * pulse if signal == "request_latency" else 5 * pulse
    if spec.anomaly_type == "Memory leak" and signal == "memory_usage":
        return 45 * progress
    if spec.anomaly_type == "Memory leak" and signal in {"request_latency", "error_rate"}:
        return 22 * progress if signal == "request_latency" else 3.5 * progress
    if spec.anomaly_type == "Disk I/O bottleneck" and signal == "disk_io":
        return 120 * pulse
    if spec.anomaly_type == "Disk I/O bottleneck" and signal in {"request_latency", "error_rate"}:
        return 45 * pulse if signal == "request_latency" else 2.8 * pulse
    if spec.anomaly_type == "Network corruption" and signal == "network_latency":
        return 95 * pulse
    if spec.anomaly_type == "Network corruption" and signal in {"request_latency", "error_rate"}:
        return 55 * pulse if signal == "request_latency" else 7 * pulse
    if spec.anomaly_type == "GPU overload" and signal == "gpu_utilization":
        return 40 + 15 * pulse
    if spec.anomaly_type == "GPU overload" and signal == "gpu_temperature":
        return 18 + 8 * pulse
    if spec.anomaly_type == "GPU overload" and signal in {"request_latency", "error_rate"}:
        return 38 * pulse if signal == "request_latency" else 4.5 * pulse
    return 0.0


def _active_incidents(
    timestamp: datetime,
    tower: str,
    component: str,
    signal: str,
    incidents: list[IncidentSpec],
) -> list[IncidentSpec]:
    return [
        spec
        for spec in incidents
        if spec.start <= timestamp <= spec.end
        and tower in spec.towers
        and signal in spec.signals
    ]


def _write_metadata(
    path: Path,
    incidents: list[IncidentSpec],
    tower_label_mapper: TowerLabelMapper,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "incident_id",
                "split",
                "anomaly_type",
                "start",
                "end",
                "towers",
                "components",
                "signals",
            ],
        )
        writer.writeheader()
        for spec in incidents:
            component_labels = [tower_label_mapper.label_for(tower) for tower in spec.towers]
            writer.writerow(
                {
                    "incident_id": spec.incident_id,
                    "split": spec.split,
                    "anomaly_type": spec.anomaly_type,
                    "start": spec.start.isoformat(sep=" "),
                    "end": spec.end.isoformat(sep=" "),
                    "towers": "|".join(
                        dict.fromkeys(tower_label_mapper.layer_for(label) for label in component_labels)
                    ),
                    "components": "|".join(component_labels),
                    "signals": "|".join(spec.signals),
                }
            )


def generate(
    output_dir: Path,
    seed: int = 42,
    mapping_spec_path: Path | None = None,
) -> None:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    incidents = _build_incidents()
    mapping_spec_path = mapping_spec_path or Path(__file__).with_name("mapping_spec.json")
    tower_label_mapper = TowerLabelMapper.from_spec(mapping_spec_path)

    train_path = output_dir / "synthetic_train.csv"
    live_path = output_dir / "synthetic_live.csv"
    metadata_path = output_dir / "anomaly_metadata.csv"

    fieldnames = [
        "incident_id",
        "timestamp",
        "tower",
        "component",
        "signal",
        "value",
        "baseline",
        "unit",
    ]

    train_handle = train_path.open("w", newline="", encoding="utf-8")
    live_handle = live_path.open("w", newline="", encoding="utf-8")
    try:
        writers = {
            "train": csv.DictWriter(train_handle, fieldnames=fieldnames),
            "live": csv.DictWriter(live_handle, fieldnames=fieldnames),
        }
        for writer in writers.values():
            writer.writeheader()

        timestamp = START_TIME
        end_time = START_TIME + timedelta(days=30)
        while timestamp < end_time:
            split = "train" if timestamp < START_TIME + timedelta(days=24) else "live"
            for tower in TOWERS:
                component = _component_for_tower(tower)
                for signal in SIGNALS:
                    baseline = _baseline(signal, tower, timestamp)
                    value = baseline + rng.gauss(0.0, NOISE[signal])
                    active = _active_incidents(timestamp, tower, component, signal, incidents)
                    incident_id = "normal"
                    if active:
                        incident_id = active[0].incident_id
                        value += sum(_incident_multiplier(signal, timestamp, spec) for spec in active)
                    component_label = tower_label_mapper.label_for(tower)
                    writers[split].writerow(
                        {
                            "incident_id": incident_id,
                            "timestamp": timestamp.isoformat(sep=" "),
                            "tower": tower_label_mapper.layer_for(component_label),
                            "component": component_label,
                            "signal": signal,
                            "value": round(_clip(signal, value), 3),
                            "baseline": baseline,
                            "unit": UNITS[signal],
                        }
                    )
            timestamp += timedelta(minutes=5)
    finally:
        train_handle.close()
        live_handle.close()

    _write_metadata(metadata_path, incidents, tower_label_mapper)


START_TIME = datetime(2026, 1, 1, 0, 0, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic RCA telemetry.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where synthetic CSV files will be written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mapping-spec",
        type=Path,
        default=Path(__file__).resolve().with_name("mapping_spec.json"),
        help="JSON spec defining deterministic synthetic tower label remapping.",
    )
    args = parser.parse_args()
    generate(args.output_dir, seed=args.seed, mapping_spec_path=args.mapping_spec)


if __name__ == "__main__":
    main()
