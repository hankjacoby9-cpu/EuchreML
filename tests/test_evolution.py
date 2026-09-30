import tempfile
import unittest
from pathlib import Path

import numpy as np

from euchre_ml.env import EuchreEnv
from euchre_ml.evolution import EvolutionConfig, train_evolution
from euchre_ml.features import FEATURE_SIZE, encode_features
from euchre_ml.neural_policy import NeuralPolicy, NetworkShape, initialize_genome


class EvolutionTests(unittest.TestCase):
    def test_feature_encoder_and_policy_use_only_legal_actions(self) -> None:
        with EuchreEnv() as env:
            result = env.reset(seed=42, learning_seat=0)
            features = encode_features(result.observation)
            self.assertEqual(features.shape, (FEATURE_SIZE,))
            self.assertEqual(features.dtype, np.float32)

            shape = NetworkShape(hidden_size=4)
            policy = NeuralPolicy(initialize_genome(np.random.default_rng(1), shape), shape)
            action = policy(result, None)
            self.assertTrue(result.action_mask[action])

    def test_tiny_evolution_run_saves_loadable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.npz"
            history = train_evolution(
                EvolutionConfig(
                    generations=1,
                    mutation_pairs=1,
                    training_seeds=2,
                    validation_seeds=2,
                    hidden_size=4,
                    checkpoint_path=str(checkpoint),
                )
            )
            self.assertEqual(len(history), 1)
            self.assertTrue(np.isfinite(history[0].validation_advantage))
            policy = NeuralPolicy.load(checkpoint)
            self.assertEqual(policy.shape.hidden_size, 4)


if __name__ == "__main__":
    unittest.main()
