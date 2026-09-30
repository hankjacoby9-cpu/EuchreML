import tempfile
import unittest
from pathlib import Path

from euchre_ml.bidding_policy import BiddingPolicy, initialize_bidding_genome
from euchre_ml.card_play_evolution import (
    CardEvolutionConfig,
    MatchFitness,
    build_match_cases,
    evaluate_card_genome,
    train_card_evolution,
)
from euchre_ml.card_play_policy import HybridCardPolicy, initialize_card_genome
from euchre_ml.evaluation import run_team_episode


class CardPlayEvolutionTests(unittest.TestCase):
    def test_hybrid_policy_completes_team_episode(self) -> None:
        policy = HybridCardPolicy(
            BiddingPolicy(initialize_bidding_genome()), initialize_card_genome()
        )
        for dealer in range(4):
            result = run_team_episode(policy, 42, dealer % 2, dealer)
            self.assertIn(result.reward, (-4, -2, -1, 1, 2, 4))

    def test_tiny_card_evolution_saves_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "card.npz"
            history = train_card_evolution(
                BiddingPolicy(initialize_bidding_genome()),
                CardEvolutionConfig(
                    generations=1,
                    mutation_pairs=1,
                    screening_seeds=1,
                    training_seeds=1,
                    validation_seeds=1,
                    finalists=2,
                    checkpoint_path=str(checkpoint),
                ),
            )
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0].matches_evaluated, 48)
            HybridCardPolicy.load(checkpoint)

    def test_match_fitness_is_lexicographic(self) -> None:
        self.assertGreater(MatchFitness(0.1, -9.0), MatchFitness(0.0, 9.0))
        self.assertGreater(MatchFitness(0.1, 1.0), MatchFitness(0.1, 0.0))

    def test_match_fitness_has_valid_win_range(self) -> None:
        bidding = BiddingPolicy(initialize_bidding_genome())
        cases = build_match_cases([91], [bidding])
        fitness = evaluate_card_genome(
            initialize_card_genome(), bidding, cases, [bidding]
        )
        self.assertTrue(0.0 <= fitness.win_rate <= 1.0)


if __name__ == "__main__":
    unittest.main()
