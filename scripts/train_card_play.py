#!/usr/bin/env python3
"""Evolve interpretable card-play weights on a fixed bidding checkpoint."""

import argparse

from euchre_ml.bidding_policy import BiddingPolicy
from euchre_ml.card_play_evolution import CardEvolutionConfig, train_card_evolution
from euchre_ml.card_play_policy import HybridCardPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bidding_checkpoint")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--mutation-pairs", type=int, default=8)
    parser.add_argument("--training-seeds", type=int, default=128)
    parser.add_argument("--validation-seeds", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint", default="checkpoints/card_play_best.npz")
    args = parser.parse_args()
    bidding = BiddingPolicy.load(args.bidding_checkpoint)
    config = CardEvolutionConfig(
        generations=args.generations,
        mutation_pairs=args.mutation_pairs,
        training_seeds=args.training_seeds,
        validation_seeds=args.validation_seeds,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )
    print("Objective: paired team score margin; bidding checkpoint held fixed")
    for result in train_card_evolution(bidding, config):
        print(
            f"generation {result.generation:3d} | "
            f"train {result.training_advantage:+.4f} | "
            f"validation {result.validation_advantage:+.4f} | "
            f"{result.elapsed_seconds:.2f}s"
        )
    policy = HybridCardPolicy.load(config.checkpoint_path)
    print("\nBest evolved card parameters:")
    for name, value in policy.named_parameters().items():
        print(f"  {name:<34} {value:+.4f}")


if __name__ == "__main__":
    main()
