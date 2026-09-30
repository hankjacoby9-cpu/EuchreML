#!/usr/bin/env python3
"""Evolve an interpretable Euchre calling score and its thresholds."""

import argparse

from euchre_ml.bidding_evolution import (
    BiddingEvolutionConfig,
    train_bidding_evolution,
)
from euchre_ml.bidding_policy import BiddingPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--mutation-pairs", type=int, default=8)
    parser.add_argument("--mutation-scale", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--training-seeds", type=int, default=256)
    parser.add_argument("--validation-seeds", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint", default="checkpoints/bidding_best.npz")
    parser.add_argument("--log", default="checkpoints/bidding_history.jsonl")
    parser.add_argument("--disable", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BiddingEvolutionConfig(
        generations=args.generations,
        mutation_pairs=args.mutation_pairs,
        mutation_scale=args.mutation_scale,
        learning_rate=args.learning_rate,
        training_seeds=args.training_seeds,
        validation_seeds=args.validation_seeds,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        log_path=args.log,
        disabled_parameters=tuple(args.disable),
    )
    print("Objective: paired team score margin; card play fixed to heuristic")
    for result in train_bidding_evolution(config):
        print(
            f"generation {result.generation:3d} | "
            f"train {result.training_advantage:+.4f} | "
            f"validation {result.validation_advantage:+.4f} | "
            f"best mutation {result.best_mutation_advantage:+.4f} | "
            f"{result.hands_evaluated:,} hands in {result.elapsed_seconds:.2f}s"
        )

    policy = BiddingPolicy.load(config.checkpoint_path)
    print("\nBest evolved bidding parameters:")
    for name, value in policy.named_parameters().items():
        print(f"  {name:<36} {value:+.4f}")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Generation log: {config.log_path}")


if __name__ == "__main__":
    main()
