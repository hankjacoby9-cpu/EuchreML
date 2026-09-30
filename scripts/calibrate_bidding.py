#!/usr/bin/env python3
"""Summarize evolved calling decisions and outcomes by predicted hand score."""

import argparse
import math
from collections import defaultdict

from euchre_ml.bidding_policy import BiddingPolicy
from euchre_ml.evaluation import run_team_episode
from euchre_ml.policies import CALL_FIRST, NO_SUIT, PASS


class RecordingPolicy:
    def __init__(self, policy: BiddingPolicy):
        self.policy = policy
        self.records = []

    def __call__(self, result, rng):
        action = self.policy(result, rng)
        phase = result.observation[8]
        if phase == 0 and result.observation[9] == NO_SUIT:
            suit = result.observation[10]
            score = self.policy.suit_score(result, suit)
            threshold = float(self.policy.genome[13])
            self.records.append((phase, score, threshold, action != PASS))
        elif phase == 1:
            suits = [
                suit for suit in range(4)
                if result.action_mask[CALL_FIRST + suit]
            ]
            score = max(self.policy.suit_score(result, suit) for suit in suits)
            threshold = float(self.policy.genome[14])
            self.records.append((phase, score, threshold, action != PASS))
        return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--seed-start", type=int, default=3_000_001)
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--bin-width", type=float, default=0.5)
    args = parser.parse_args()
    policy = BiddingPolicy.load(args.checkpoint)
    buckets = defaultdict(lambda: [0, 0, 0.0])

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for dealer in range(4):
            for team in range(2):
                recorder = RecordingPolicy(policy)
                episode = run_team_episode(recorder, seed, team, dealer)
                for phase, score, _threshold, called in recorder.records:
                    lower = math.floor(score / args.bin_width) * args.bin_width
                    bucket = buckets[(phase, lower)]
                    bucket[0] += 1
                    bucket[1] += called
                    bucket[2] += episode.reward

    print("Descriptive calibration; outcome is not a call/pass counterfactual.")
    print("round  score bin       chances  call rate  mean team margin")
    for (phase, lower), (count, calls, reward) in sorted(buckets.items()):
        print(
            f"  {phase + 1}    [{lower:+4.1f}, {lower + args.bin_width:+4.1f}) "
            f"{count:8d}  {calls / count:8.1%}  {reward / count:+8.3f}"
        )


if __name__ == "__main__":
    main()
