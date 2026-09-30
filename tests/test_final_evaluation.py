import unittest

from euchre_ml.final_evaluation import evaluate_final_comparison
from euchre_ml.policies import simple_heuristic_policy


class FinalEvaluationTests(unittest.TestCase):
    def test_same_policy_has_balanced_head_to_head_report(self) -> None:
        result = evaluate_final_comparison(
            simple_heuristic_policy,
            simple_heuristic_policy,
            "same",
            [9_000_001],
            bootstrap_samples=20,
        )
        self.assertEqual(result.matches, 8)
        self.assertEqual(result.wins, 4)
        self.assertEqual(result.win_rate, 0.5)
        self.assertEqual(result.mean_capped_margin, 0.0)
        self.assertEqual(result.win_confidence_interval, (0.5, 0.5))
        self.assertEqual(result.margin_confidence_interval, (0.0, 0.0))
        self.assertEqual(
            result.diagnostics.successful_calls
            + result.diagnostics.euchred_calls,
            result.diagnostics.calls,
        )


if __name__ == "__main__":
    unittest.main()
