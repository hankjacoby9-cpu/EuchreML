import tempfile
import unittest
from pathlib import Path

import numpy as np

from euchre_ml.bidding_evolution import (
    BiddingEvolutionConfig,
    train_bidding_evolution,
)
from euchre_ml.bidding_policy import (
    GENOME_SIZE,
    PARAMETER_NAMES,
    BiddingPolicy,
    initialize_bidding_genome,
    upcard_trump_strength,
)
from euchre_ml.evaluation import run_team_episode


class BiddingEvolutionTests(unittest.TestCase):
    def test_upcard_strength_uses_euchre_trump_order(self) -> None:
        strengths = [upcard_trump_strength(rank) for rank in range(6)]
        ordered_ranks = (0, 1, 3, 4, 5, 2)
        self.assertEqual(
            sorted(range(6), key=lambda rank: strengths[rank]),
            list(ordered_ranks),
        )
        self.assertEqual(strengths[2], 1.0)

    def test_bidding_policy_completes_private_team_episodes(self) -> None:
        policy = BiddingPolicy(initialize_bidding_genome())
        self.assertEqual(len(policy.named_parameters()), GENOME_SIZE)
        self.assertEqual(tuple(policy.named_parameters()), PARAMETER_NAMES)
        for team in (0, 1):
            episode = run_team_episode(policy, seed=42, team=team)
            self.assertIn(episode.reward, (-4, -2, -1, 1, 2, 4))

    def test_tiny_bidding_evolution_saves_named_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bidding.npz"
            history = train_bidding_evolution(
                BiddingEvolutionConfig(
                    generations=1,
                    mutation_pairs=1,
                    training_seeds=2,
                    validation_seeds=2,
                    checkpoint_path=str(checkpoint),
                    log_path=str(Path(directory) / "history.jsonl"),
                )
            )
            self.assertEqual(len(history), 1)
            self.assertTrue(np.isfinite(history[0].validation_advantage))
            loaded = BiddingPolicy.load(checkpoint)
            self.assertEqual(loaded.genome.shape, (GENOME_SIZE,))

    def test_unversioned_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.npz"
            np.savez(checkpoint, genome=initialize_bidding_genome())
            with self.assertRaisesRegex(ValueError, "unversioned"):
                BiddingPolicy.load(checkpoint)


if __name__ == "__main__":
    unittest.main()
