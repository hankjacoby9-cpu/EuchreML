import unittest

from euchre_ml import (
    ACTION_COUNT,
    LAYOUT_VERSION,
    OBSERVATION_SIZE,
    EuchreBatchEnv,
    EuchreEnv,
)


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

    def test_native_batch_completes_all_environments(self) -> None:
        environment_count = 8
        with EuchreBatchEnv(environment_count) as batch:
            results = batch.reset(
                seeds=range(1, environment_count + 1),
                learning_seats=[index % 4 for index in range(environment_count)],
            )
            while not all(result.done for result in results):
                actions = [
                    0 if result.done else next(
                        action
                        for action, enabled in enumerate(result.action_mask)
                        if enabled
                    )
                    for result in results
                ]
                results = batch.step(actions)

            self.assertTrue(all(sum(result.action_mask) == 0 for result in results))
            self.assertTrue(
                all(result.reward in (-4, -2, -1, 1, 2, 4) for result in results)
            )

    def test_native_batch_validates_lengths_and_actions(self) -> None:
        with EuchreBatchEnv(2) as batch:
            with self.assertRaises(ValueError):
                batch.reset([1], [0, 1])
            results = batch.reset([1, 2], [0, 1])
            actions = [
                next(
                    action
                    for action, enabled in enumerate(result.action_mask)
                    if not enabled
                )
                for result in results
            ]
            with self.assertRaises(ValueError):
                batch.step(actions)

    def test_zero_copy_batch_can_replace_terminal_slots(self) -> None:
        environment_count = 4
        target_episodes = 20
        next_episode = environment_count
        completed = 0
        active = [True] * environment_count
        with EuchreBatchEnv(environment_count) as batch:
            buffers = batch.reset_buffers(range(1, 5), range(4))
            observation_view = buffers.observations

            while completed < target_episodes:
                for environment in range(environment_count):
                    buffers.reset_flags[environment] = 0
                    if not active[environment]:
                        continue
                    if buffers.statuses[environment] == 1:
                        completed += 1
                        if next_episode < target_episodes:
                            buffers.reset_flags[environment] = 1
                            buffers.seeds[environment] = next_episode + 1
                            buffers.learning_seats[environment] = next_episode % 4
                            next_episode += 1
                        else:
                            active[environment] = False
                        continue

                    mask_start = environment * ACTION_COUNT
                    buffers.actions[environment] = next(
                        action
                        for action in range(ACTION_COUNT)
                        if buffers.action_masks[mask_start + action]
                    )

                if completed < target_episodes:
                    returned = batch.advance_buffers()
                    self.assertIs(returned.observations, observation_view)

            self.assertEqual(completed, target_episodes)


if __name__ == "__main__":
    unittest.main()
