import unittest

import numpy as np
import pandas as pd

from src.gpu_anomaly import (
    _build_feature_frame,
    _build_sliding_windows,
    _sequence_errors_to_row_scores,
    enrich_with_gpu_anomaly_scores,
)


def _telemetry_frame(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "incident_id": ["INC-1"] * rows,
            "timestamp": pd.date_range("2026-06-13 10:00", periods=rows, freq="min"),
            "tower": ["compute", "storage", "network", "application", "compute", "storage"][:rows],
            "signal": ["cpu", "latency", "loss", "errors", "memory", "iops"][:rows],
            "component": ["node-1", "disk-1", "link-1", "api-1", "node-1", "disk-1"][:rows],
            "value": [10, 12, 14, 80, 18, 20][:rows],
            "baseline": [10, 10, 10, 10, 10, 10][:rows],
            "unit": ["percent"] * rows,
        }
    )


class GPUAnomalyTests(unittest.TestCase):
    def test_sequence_creation_orders_by_timestamp(self) -> None:
        telemetry = _telemetry_frame(5).sample(frac=1, random_state=4)
        features = _build_feature_frame(telemetry)

        sequences, row_windows = _build_sliding_windows(
            features,
            telemetry["timestamp"],
            sequence_length=3,
        )

        self.assertEqual(sequences.shape, (3, 3, features.shape[1]))
        expected_order = telemetry.sort_values("timestamp").index.tolist()
        self.assertEqual(row_windows[0], expected_order[:3])
        self.assertEqual(row_windows[-1], expected_order[-3:])

    def test_sequence_errors_map_back_to_row_scores(self) -> None:
        output_index = pd.Index(["a", "b", "c", "d"])
        scores = _sequence_errors_to_row_scores(
            np.asarray([0.0, 2.0]),
            [["a", "b", "c"], ["b", "c", "d"]],
            output_index,
        )

        self.assertEqual(list(scores.index), list(output_index))
        self.assertTrue(scores.between(0.0, 1.0).all())
        self.assertGreater(scores.loc["d"], scores.loc["a"])

    def test_fallback_for_insufficient_sequence_length(self) -> None:
        enriched, metadata = enrich_with_gpu_anomaly_scores(
            _telemetry_frame(2),
            sequence_length=4,
        )

        self.assertFalse(metadata.enabled)
        self.assertEqual(metadata.model, "rule-fallback")
        self.assertIn("gpu_anomaly_score", enriched.columns)
        self.assertEqual(len(enriched), 2)

    def test_output_schema_compatibility(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")

        telemetry = _telemetry_frame(6)
        enriched, metadata = enrich_with_gpu_anomaly_scores(
            telemetry,
            epochs=1,
            sequence_length=3,
        )

        self.assertTrue(metadata.enabled)
        self.assertEqual(metadata.model, "lstm-autoencoder")
        self.assertEqual(list(enriched.columns[:-1]), list(telemetry.columns))
        self.assertIn("gpu_anomaly_score", enriched.columns)
        self.assertEqual(len(enriched), len(telemetry))
        self.assertTrue(enriched["gpu_anomaly_score"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
