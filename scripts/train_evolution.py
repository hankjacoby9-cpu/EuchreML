#!/usr/bin/env python3
"""Train a small Euchre policy with mirrored evolutionary mutations."""

import argparse

from euchre_ml.evolution import EvolutionConfig, train_evolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--mutation-pairs", type=int, default=4)
    parser.add_argument("--mutation-scale", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--training-seeds", type=int, default=64)
    parser.add_argument("--validation-seeds", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--checkpoint", default="checkpoints/evolution_best.npz"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvolutionConfig(
        generations=args.generations,
        mutation_pairs=args.mutation_pairs,
        mutation_scale=args.mutation_scale,
        learning_rate=args.learning_rate,
        training_seeds=args.training_seeds,
        validation_seeds=args.validation_seeds,
        seed=args.seed,
        hidden_size=args.hidden_size,
        checkpoint_path=args.checkpoint,
    )
    print("Objective: paired team score margin against heuristic baseline")
    print(
        f"Genome: hidden={config.hidden_size}, mutations/generation="
        f"{2 * config.mutation_pairs}"
    )
    for result in train_evolution(config):
        print(
            f"generation {result.generation:3d} | "
            f"train {result.training_advantage:+.4f} | "
            f"validation {result.validation_advantage:+.4f} | "
            f"best mutation {result.best_mutation_advantage:+.4f}"
        )
    print(f"Best validation checkpoint: {config.checkpoint_path}")


if __name__ == "__main__":
    main()
