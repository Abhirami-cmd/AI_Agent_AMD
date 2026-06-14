import unittest

import pandas as pd

from src.cross_tower_detector import CrossTowerAnomalyDetector, CrossTowerCorrelationConfig
from src.rca_engine import RCAAnalysis, analyze_incident


class _Memory:
    def find_similar(self, service, evidence_terms):
        return []


def _telemetry(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "incident_id": "INC-1",
        "value": 95.0,
        "baseline": 50.0,
        "unit": "percent",
        "gpu_anomaly_score": 0.9,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


class CrossTowerDetectorTests(unittest.TestCase):
    def test_correlated_anomalies_group_into_single_candidate(self) -> None:
        frame = _telemetry(
            [
                {"timestamp": "2026-06-13 10:00", "tower": "Compute", "component": "Cluster-A", "signal": "cpu"},
                {"timestamp": "2026-06-13 10:04", "tower": "Network", "component": "Edge-Router-2", "signal": "latency"},
                {"timestamp": "2026-06-13 10:08", "tower": "Storage", "component": "SAN-1", "signal": "iops"},
            ]
        )

        candidates = CrossTowerAnomalyDetector().detect(frame)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].incident_id, "INC-1")
        self.assertEqual(candidates[0].towers, ["Compute", "Network", "Storage"])
        self.assertEqual(candidates[0].variant_count, 3)

    def test_incident_candidate_shape_is_rca_compatible(self) -> None:
        frame = _telemetry(
            [
                {"timestamp": "2026-06-13 10:00", "tower": "Compute", "component": "Cluster-A", "signal": "cpu"},
                {"timestamp": "2026-06-13 10:01", "tower": "Application", "component": "Payments", "signal": "errors"},
            ]
        )

        candidate = CrossTowerAnomalyDetector().detect(frame)[0].to_incident()

        for key in [
            "incident_id",
            "title",
            "service",
            "severity",
            "started_at",
            "description",
            "dependencies",
            "variant_count",
        ]:
            self.assertIn(key, candidate)
        self.assertEqual(candidate["incident_id"], "INC-1")
        self.assertGreaterEqual(len(candidate["dependencies"]), 2)

    def test_duplicate_suppression_keeps_stronger_candidate(self) -> None:
        frame = _telemetry(
            [
                {
                    "timestamp": "2026-06-13 10:00",
                    "tower": "Compute",
                    "component": "Cluster-A",
                    "signal": "cpu",
                    "gpu_anomaly_score": 0.7,
                },
                {
                    "timestamp": "2026-06-13 10:01",
                    "tower": "Network",
                    "component": "Edge-Router-2",
                    "signal": "latency",
                    "gpu_anomaly_score": 0.7,
                },
                {
                    "timestamp": "2026-06-13 10:06",
                    "tower": "Compute",
                    "component": "Cluster-A",
                    "signal": "memory",
                    "gpu_anomaly_score": 0.95,
                },
                {
                    "timestamp": "2026-06-13 10:07",
                    "tower": "Network",
                    "component": "Edge-Router-2",
                    "signal": "loss",
                    "gpu_anomaly_score": 0.95,
                },
            ]
        )
        detector = CrossTowerAnomalyDetector(
            CrossTowerCorrelationConfig(
                time_window_minutes=3,
                anomaly_score_threshold=0.65,
                duplicate_suppression_minutes=10,
            )
        )

        candidates = detector.detect(frame)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].max_anomaly_score, 0.95)
        self.assertEqual(candidates[0].window_start, "2026-06-13 10:06:00")

    def test_candidate_can_flow_into_rca_engine(self) -> None:
        frame = _telemetry(
            [
                {"timestamp": "2026-06-13 10:00", "tower": "Compute", "component": "Cluster-A", "signal": "cpu"},
                {"timestamp": "2026-06-13 10:03", "tower": "Application", "component": "Payments", "signal": "errors"},
                {"timestamp": "2026-06-13 10:05", "tower": "Network", "component": "LoadBalancer-3", "signal": "latency"},
            ]
        )
        candidate = CrossTowerAnomalyDetector().detect(frame)[0].to_incident()

        analysis = analyze_incident(candidate, frame, _Memory(), reference_sources=[])

        self.assertIsInstance(analysis, RCAAnalysis)
        self.assertGreaterEqual(len(analysis.evidence), 2)
        self.assertIn("Cross-tower corroboration", " ".join(analysis.primary.confidence_drivers))


if __name__ == "__main__":
    unittest.main()
