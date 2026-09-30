import unittest

from euchre_ml import ACTION_COUNT, LAYOUT_VERSION, OBSERVATION_SIZE, EuchreEnv


class BridgeIntegrationTests(unittest.TestCase):
    def test_complete_episodes_for_every_seat(self) -> None:
        self.assertEqual(LAYOUT_VERSION, 1)
        self.assertEqual(OBSERVATION_SIZE, 138)
        self.assertEqual(ACTION_COUNT, 35)

        for seat in range(4):
            for seed in (1, 42, 2026):
                with self.subTest(seat=seat, seed=seed), EuchreEnv() as env:
                    result = env.reset(seed=seed, learning_seat=seat)
                    while not result.done:
                        self.assertEqual(len(result.observation), OBSERVATION_SIZE)
                        self.assertEqual(len(result.action_mask), ACTION_COUNT)
                        self.assertEqual(result.observation[0], LAYOUT_VERSION)
                        legal = [
                            action
                            for action, enabled in enumerate(result.action_mask)
                            if enabled
                        ]
                        self.assertGreater(len(legal), 1)
                        result = env.step(legal[0])

                    self.assertEqual(sum(result.action_mask), 0)
                    self.assertIn(result.reward, (-4, -2, -1, 1, 2, 4))

    def test_illegal_action_is_rejected(self) -> None:
        with EuchreEnv() as env:
            result = env.reset(seed=99, learning_seat=1)
            illegal = next(
                action
                for action, enabled in enumerate(result.action_mask)
                if not enabled
            )
            with self.assertRaises(ValueError):
                env.step(illegal)


if __name__ == "__main__":
    unittest.main()
