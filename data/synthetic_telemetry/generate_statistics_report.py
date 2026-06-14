from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import (
    load_training_telemetry,
    load_telemetry,
)
from src.gpu_anomaly import enrich_with_gpu_anomaly_scores
from src.ingest.synthetic_loader import synthetic_dataset_statistics


def main() -> None:
    output_path = Path(__file__).resolve().parent / "telemetry_dataset_statistics.md"
    train = load_training_telemetry()
    live = load_telemetry()
    _, metadata = enrich_with_gpu_anomaly_scores(
        live,
        training_telemetry=train,
        epochs=6,
        max_train_sequences=12000,
    )
    model_stats = {
        "sequence_count": metadata.sequence_count,
        "device": metadata.device,
        "training_duration_seconds": metadata.training_duration_seconds,
    }
    train_stats = synthetic_dataset_statistics(train)
    live_stats = synthetic_dataset_statistics(live)
    combined_stats = synthetic_dataset_statistics(
        pd.concat([train, live], ignore_index=True)
    )

    output_path.write_text(
        "\n".join(
            [
                "# Synthetic Telemetry Dataset Statistics",
                "",
                "| Split | Row count | Tower count | Component count | Signal count | Anomaly count | Anomaly percentage |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                _stats_row("train", train_stats),
                _stats_row("live", live_stats),
                _stats_row("combined", combined_stats),
                "",
                "| Model metric | Value |",
                "| --- | ---: |",
                f"| Sequence count generated | {model_stats['sequence_count']} |",
                f"| GPU device used | {model_stats['device']} |",
                f"| Training duration | {metadata.training_duration_seconds:.3f} seconds |",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _stats_row(split: str, stats: dict[str, object]) -> str:
    return (
        f"| {split} | {stats['row_count']} | {stats['tower_count']} | "
        f"{stats['component_count']} | {stats['signal_count']} | "
        f"{stats['anomaly_count']} | {stats['anomaly_percentage']}% |"
    )


if __name__ == "__main__":
    main()
