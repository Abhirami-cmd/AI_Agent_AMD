import unittest

from src.rca_engine import (
    EvidenceItem,
    Hypothesis,
    _apply_confidence_caps,
    _apply_weighted_confidence,
    _score_memory_hypotheses,
    score_hypotheses,
)


class FeedbackConfidenceTests(unittest.TestCase):
    def test_correct_feedback_boosts_matching_hypothesis(self) -> None:
        hypotheses = [
            Hypothesis(
                title="Storage latency",
                summary="candidate",
                confidence=0.5,
            )
        ]
        memory = [
            {
                "selected_root_cause": "Storage latency",
                "actual_root_cause": "",
                "agent_root_cause": "Storage latency",
                "correctness": "Correct",
            }
        ]

        adjusted = _apply_weighted_confidence({}, [], hypotheses, memory)

        self.assertAlmostEqual(adjusted[0].score_factors["memory_support"], 0.5)
        self.assertAlmostEqual(adjusted[0].confidence, 0.05)

    def test_partially_correct_feedback_boosts_matching_hypothesis_less(self) -> None:
        hypotheses = [
            Hypothesis(
                title="Storage latency",
                summary="candidate",
                confidence=0.5,
            )
        ]
        memory = [
            {
                "selected_root_cause": "Storage latency",
                "actual_root_cause": "",
                "agent_root_cause": "Storage latency",
                "correctness": "Partially correct",
            }
        ]

        adjusted = _apply_weighted_confidence({}, [], hypotheses, memory)

        self.assertAlmostEqual(adjusted[0].score_factors["memory_support"], 0.25)
        self.assertAlmostEqual(adjusted[0].confidence, 0.025)

    def test_incorrect_feedback_does_not_boost_wrong_agent_root(self) -> None:
        evidence = [
            EvidenceItem(
                tower="Application",
                signal="error_rate",
                component="Payments",
                observed=10.0,
                baseline=1.0,
                anomaly_score=0.9,
                timestamp="2026-06-15 10:00",
                explanation="errors spiked",
            )
        ]
        memory = [
            {
                "selected_root_cause": "Wrong Root",
                "actual_root_cause": "Storage latency",
                "agent_root_cause": "Wrong Root",
                "correctness": "Incorrect",
            }
        ]

        hypotheses = _score_memory_hypotheses(evidence, memory)

        self.assertEqual(hypotheses[0].title, "Storage latency")
        self.assertNotIn("Wrong Root", [item.title for item in hypotheses])

    def test_feedback_learning_penalizes_incorrect_agent_root(self) -> None:
        hypotheses = [
            Hypothesis(
                title="Wrong Root",
                summary="candidate",
                confidence=0.5,
            )
        ]
        memory = [
            {
                "selected_root_cause": "Wrong Root",
                "actual_root_cause": "Storage latency",
                "agent_root_cause": "Wrong Root",
                "correctness": "Incorrect",
            }
        ]

        adjusted = _apply_weighted_confidence({}, [], hypotheses, memory)

        self.assertEqual(adjusted[0].score_factors["memory_support"], 0.0)
        self.assertEqual(adjusted[0].confidence, 0.0)

    def test_score_hypotheses_applies_correct_feedback_to_memory_candidate(self) -> None:
        evidence = [
            EvidenceItem(
                tower="Application",
                signal="error_rate",
                component="Payments",
                observed=10.0,
                baseline=1.0,
                anomaly_score=0.9,
                timestamp="2026-06-15 10:00",
                explanation="errors spiked",
            )
        ]
        memory = [
            {
                "selected_root_cause": "Storage latency",
                "actual_root_cause": "",
                "agent_root_cause": "Storage latency",
                "correctness": "Correct",
            }
        ]

        hypotheses = score_hypotheses({}, evidence, memory, reference_sources=[])

        self.assertEqual(hypotheses[0].title, "Storage latency")
        self.assertEqual(hypotheses[0].score_factors["memory_support"], 0.5)
        self.assertGreater(hypotheses[0].confidence, 0.0)

    def test_confidence_caps_prevent_multiple_near_certain_hypotheses(self) -> None:
        hypotheses = [
            Hypothesis(title="Primary", summary="candidate", confidence=0.99),
            Hypothesis(title="Alternative", summary="candidate", confidence=0.98),
        ]

        capped = _apply_confidence_caps(hypotheses)

        self.assertEqual(capped[0].confidence, 0.92)
        self.assertEqual(capped[1].confidence, 0.84)
        self.assertIn("Confidence capped", capped[0].confidence_drivers[-1])


if __name__ == "__main__":
    unittest.main()
