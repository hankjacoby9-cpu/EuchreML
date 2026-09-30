"""Measure current CFFI throughput and establish a pre-batch baseline."""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from euchre_ml import ACTION_COUNT, EuchreBatchEnv, EuchreEnv, StepResult


@dataclass(frozen=True)
class BenchmarkResult:
    episodes: int
    decisions: int
    seconds: float

    @property
    def episodes_per_second(self) -> float:
        return self.episodes / self.seconds

    @property
    def decisions_per_second(self) -> float:
        return self.decisions / self.seconds

    @property
    def decisions_per_episode(self) -> float:
        return self.decisions / self.episodes


def first_legal(result: StepResult) -> int:
    return next(action for action, legal in enumerate(result.action_mask) if legal)


def run_single_environment(episodes: int) -> BenchmarkResult:
    decisions = 0
    with EuchreEnv() as env:
        start = time.perf_counter()
        for episode in range(episodes):
            result = env.reset(seed=episode + 1, learning_seat=episode % 4)
            while not result.done:
                result = env.step(first_legal(result))
                decisions += 1
        seconds = time.perf_counter() - start
    return BenchmarkResult(episodes, decisions, seconds)


def run_interleaved_environments(episodes: int, environment_count: int) -> BenchmarkResult:
    """Round-robin environments using individual CFFI calls, not native batching."""
    envs = [EuchreEnv() for _ in range(environment_count)]
    states: List[Optional[Tuple[StepResult, int]]] = []
    next_episode = 0
    decisions = 0

    try:
        start = time.perf_counter()
        for index, env in enumerate(envs):
            if next_episode >= episodes:
                break
            states.append(
                (
                    env.reset(seed=next_episode + 1,
                              learning_seat=next_episode % 4),
                    next_episode,
                )
            )
            next_episode += 1

        completed = 0
        while completed < episodes:
            for index in range(len(states)):
                slot = states[index]
                if slot is None:
                    continue

                result, episode = slot
                if result.done:
                    completed += 1
                    if next_episode < episodes:
                        result = envs[index].reset(
                            seed=next_episode + 1,
                            learning_seat=next_episode % 4,
                        )
                        states[index] = (result, next_episode)
                        next_episode += 1
                    else:
                        states[index] = None
                    continue

                result = envs[index].step(first_legal(result))
                states[index] = (result, episode)
                decisions += 1
            # Terminal results are counted on the following round-robin pass.
        seconds = time.perf_counter() - start
    finally:
        for env in envs:
            env.close()

    return BenchmarkResult(episodes, decisions, seconds)


def run_native_batch(episodes: int, environment_count: int) -> BenchmarkResult:
    """Advance each complete environment wave through batched C calls."""
    measured_episodes = (episodes // environment_count) * environment_count
    if measured_episodes == 0:
        raise ValueError("episodes must be at least the native batch size")

    decisions = 0
    with EuchreBatchEnv(environment_count) as batch:
        start = time.perf_counter()
        for episode_start in range(0, measured_episodes, environment_count):
            episode_numbers = range(
                episode_start, episode_start + environment_count
            )
            results = batch.reset(
                seeds=[episode + 1 for episode in episode_numbers],
                learning_seats=[episode % 4 for episode in episode_numbers],
            )
            while not all(result.done for result in results):
                actions = []
                for result in results:
                    if result.done:
                        actions.append(0)
                    else:
                        actions.append(first_legal(result))
                        decisions += 1
                results = batch.step(actions)
        seconds = time.perf_counter() - start
    return BenchmarkResult(measured_episodes, decisions, seconds)


def run_async_zero_copy_batch(
    episodes: int, environment_count: int
) -> BenchmarkResult:
    """Keep slots occupied and read/write the C-owned buffers directly."""
    if episodes < environment_count:
        raise ValueError("episodes must be at least the native batch size")

    decisions = 0
    completed = 0
    next_episode = environment_count
    active = [True] * environment_count
    with EuchreBatchEnv(environment_count) as batch:
        buffers = batch.reset_buffers(
            seeds=range(1, environment_count + 1),
            learning_seats=[index % 4 for index in range(environment_count)],
        )
        start = time.perf_counter()
        while completed < episodes:
            for environment in range(environment_count):
                buffers.reset_flags[environment] = 0
                if not active[environment]:
                    continue

                if buffers.statuses[environment] == 1:
                    completed += 1
                    if next_episode < episodes:
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
                decisions += 1

            if completed < episodes:
                batch.advance_buffers()
        seconds = time.perf_counter() - start
    return BenchmarkResult(episodes, decisions, seconds)


def print_result(label: str, result: BenchmarkResult) -> None:
    print(
        f"{label:<24} "
        f"{result.episodes_per_second:>10,.0f} episodes/s  "
        f"{result.decisions_per_second:>10,.0f} decisions/s  "
        f"{result.decisions_per_episode:>5.2f} decisions/episode  "
        f"({result.seconds:.3f}s)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64])
    args = parser.parse_args()
    if args.episodes <= 0 or any(size <= 0 for size in args.batch_sizes):
        parser.error("episode and batch counts must be positive")
    return args


def main() -> None:
    args = parse_args()

    # Warm the extension and CPU caches outside the measured runs.
    run_single_environment(min(500, args.episodes))

    print_result("single environment", run_single_environment(args.episodes))
    for batch_size in args.batch_sizes:
        print_result(
            f"interleaved x{batch_size}",
            run_interleaved_environments(args.episodes, batch_size),
        )
        print_result(
            f"native batch x{batch_size}",
            run_native_batch(args.episodes, batch_size),
        )
        print_result(
            f"async zero-copy x{batch_size}",
            run_async_zero_copy_batch(args.episodes, batch_size),
        )
    print("Native modes cross CFFI once per batch step.")


if __name__ == "__main__":
    main()
