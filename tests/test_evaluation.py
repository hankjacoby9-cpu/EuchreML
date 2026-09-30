import unittest

from euchre_ml.evaluation import evaluate_paired, run_episode
from euchre_ml.policies import (
    first_legal_policy,
    random_legal_policy,
    simple_heuristic_policy,
)


class EvaluationTests(unittest.TestCase):
    def test_identical_policy_has_zero_paired_advantage(self) -> None:
        evaluation = evaluate_paired(
            random_legal_policy,
            random_legal_policy,
            range(1, 21),
            policy_a_name="random_a",
            policy_b_name="random_b",
            bootstrap_samples=50,
        )
        self.assertEqual(evaluation.pairs, 80)
        self.assertEqual(evaluation.mean_advantage, 0.0)
        self.assertEqual(evaluation.confidence_interval, (0.0, 0.0))
        self.assertEqual(evaluation.paired_ties, 80)
        self.assertEqual(evaluation.advantage_by_seat, (0.0, 0.0, 0.0, 0.0))

    def test_reference_policies_complete_legal_episodes(self) -> None:
        for policy in (
            first_legal_policy,
            random_legal_policy,
            simple_heuristic_policy,
        ):
            for seat in range(4):
                with self.subTest(policy=policy.__name__, seat=seat):
                    episode = run_episode(policy, seed=42, seat=seat)
                    self.assertIn(episode.reward, (-4, -2, -1, 1, 2, 4))
                    self.assertGreater(sum(episode.decisions.values()), 0)


if __name__ == "__main__":
    unittest.main()
