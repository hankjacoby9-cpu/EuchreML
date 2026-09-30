#!/usr/bin/env python3
"""Run a duplicate-deal comparison from the command line."""

import argparse

from euchre_ml.evaluation import PairedEvaluation, PolicySummary, evaluate_paired
from euchre_ml.policies import POLICIES


def percentage(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{100.0 * numerator / denominator:.1f}%"


def print_summary(name: str, summary: PolicySummary) -> None:
    print(f"\n{name}")
    print(f"  mean reward/hand: {summary.mean_reward:+.4f}")
    print(f"  hand win rate:    {100.0 * summary.hand_win_rate:.1f}%")
    print(f"  own calls:        {summary.own_calls}")
    print(
        "  euchred when calling: "
        f"{percentage(summary.own_call_euchres, summary.own_calls)}"
    )
    print(
        "  sweeps when calling:  "
        f"{percentage(summary.own_call_sweeps, summary.own_calls)}"
    )
    print(f"  lone calls:       {summary.own_lone_calls}")
    print(
        "  lone sweep rate:  "
        f"{percentage(summary.own_lone_sweeps, summary.own_lone_calls)}"
    )
    decisions = ", ".join(
        f"{category}={count}"
        for category, count in sorted(summary.decisions.items())
    )
    print(f"  decisions:        {decisions}")


def print_evaluation(evaluation: PairedEvaluation) -> None:
    low, high = evaluation.confidence_interval
    print(
        f"Paired duplicate-deal evaluation: {evaluation.policy_a} vs "
        f"{evaluation.policy_b}"
    )
    print(f"Pairs: {evaluation.pairs} (every seed rotated through all 4 seats)")
    print(f"Mean paired advantage: {evaluation.mean_advantage:+.4f} points/hand")
    print(f"95% bootstrap CI:      [{low:+.4f}, {high:+.4f}]")
    print(
        "Paired W/T/L:           "
        f"{evaluation.paired_wins}/{evaluation.paired_ties}/"
        f"{evaluation.paired_losses}"
    )
    print(
        "Advantage by seat:      "
        + ", ".join(
            f"seat {seat}={advantage:+.4f}"
            for seat, advantage in enumerate(evaluation.advantage_by_seat)
        )
    )
    print_summary(evaluation.policy_a, evaluation.summary_a)
    print_summary(evaluation.policy_b, evaluation.summary_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-a", choices=POLICIES, default="heuristic")
    parser.add_argument("--policy-b", choices=POLICIES, default="random")
    parser.add_argument(
        "--seeds",
        type=int,
        default=2000,
        help="number of deal seeds; each is evaluated from all four seats",
    )
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.seeds <= 0 or args.bootstrap_samples <= 0:
        parser.error("seed and bootstrap counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    evaluation = evaluate_paired(
        POLICIES[args.policy_a],
        POLICIES[args.policy_b],
        range(args.seed_start, args.seed_start + args.seeds),
        policy_a_name=args.policy_a,
        policy_b_name=args.policy_b,
        bootstrap_samples=args.bootstrap_samples,
    )
    print_evaluation(evaluation)


if __name__ == "__main__":
    main()
