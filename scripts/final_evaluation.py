#!/usr/bin/env python3
"""Run the locked final test against the heuristic and previous champion."""

import argparse

from euchre_ml.card_play_policy import HybridCardPolicy
from euchre_ml.final_evaluation import evaluate_final_test
from euchre_ml.policies import simple_heuristic_policy
from euchre_ml.seed_ranges import FINAL_TEST_SEED_START


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_checkpoint")
    parser.add_argument("previous_checkpoint")
    parser.add_argument("--seed-start", type=int, default=FINAL_TEST_SEED_START)
    parser.add_argument(
        "--allow-custom-seeds", action="store_true",
        help="allow a non-reserved seed range for exploratory reports",
    )
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.seed_start != FINAL_TEST_SEED_START and not args.allow_custom_seeds:
        raise SystemExit(
            "The locked report must use seed 9000001; pass --allow-custom-seeds "
            "for an explicitly exploratory run"
        )
    candidate = HybridCardPolicy.load(args.candidate_checkpoint)
    opponents = {
        "heuristic": simple_heuristic_policy,
        "previous_champion": HybridCardPolicy.load(args.previous_checkpoint),
    }
    results = evaluate_final_test(
        candidate, opponents,
        range(args.seed_start, args.seed_start + args.seeds),
        args.bootstrap_samples,
    )
    print(f"Locked final seeds: {args.seed_start}..{args.seed_start + args.seeds - 1}")
    for result in results:
        d = result.diagnostics
        print(f"\nOpponent: {result.opponent}")
        print(f"  matches: {result.matches:,}")
        print(
            f"  win rate: {result.win_rate:.4f} "
            f"[{result.win_confidence_interval[0]:.4f}, "
            f"{result.win_confidence_interval[1]:.4f}]"
        )
        print(
            f"  capped margin: {result.mean_capped_margin:+.4f} "
            f"[{result.margin_confidence_interval[0]:+.4f}, "
            f"{result.margin_confidence_interval[1]:+.4f}]"
        )
        print(
            f"  calls: {d.calls}/{d.hands} | success {d.calling_success_rate:.4f} "
            f"| euchred {d.euchre_rate:.4f}"
        )
        print(
            f"  lone calls: {d.lone_calls} | success {d.lone_success_rate:.4f} "
            f"| sweeps {d.lone_sweeps} ({d.lone_sweep_rate:.4f})"
        )


if __name__ == "__main__":
    main()
