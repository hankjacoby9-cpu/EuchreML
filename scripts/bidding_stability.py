#!/usr/bin/env python3
"""Repeat dealer-balanced bidding evolution and summarize weight stability."""

import argparse
from pathlib import Path

import numpy as np

from euchre_ml.bidding_evolution import (
    BiddingEvolutionConfig,
    train_bidding_evolution,
)
from euchre_ml.bidding_policy import BiddingPolicy, PARAMETER_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--training-seeds", type=int, default=128)
    parser.add_argument("--validation-seeds", type=int, default=256)
    parser.add_argument("--mutation-pairs", type=int, default=8)
    parser.add_argument("--output", default="checkpoints/bidding_stability")
    parser.add_argument("--disable", nargs="*", default=[])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    genomes = []
    validations = []
    for run in range(args.runs):
        checkpoint = output / f"run_{run}.npz"
        history = train_bidding_evolution(
            BiddingEvolutionConfig(
                generations=args.generations,
                mutation_pairs=args.mutation_pairs,
                training_seeds=args.training_seeds,
                validation_seeds=args.validation_seeds,
                seed=2026 + run,
                checkpoint_path=str(checkpoint),
                log_path=str(output / f"run_{run}.jsonl"),
                disabled_parameters=tuple(args.disable),
            )
        )
        policy = BiddingPolicy.load(checkpoint)
        genomes.append(policy.genome)
        validations.append(max(item.validation_advantage for item in history))
        print(f"run {run}: best validation {validations[-1]:+.4f}")

    matrix = np.stack(genomes)
    print("\nParameter stability (mean ± population standard deviation):")
    for index, name in enumerate(PARAMETER_NAMES):
        print(
            f"  {name:<42} {matrix[:, index].mean():+7.3f} ± "
            f"{matrix[:, index].std():.3f}"
        )
    print(
        f"\nValidation advantage: {np.mean(validations):+.4f} ± "
        f"{np.std(validations):.4f}"
    )


if __name__ == "__main__":
    main()
