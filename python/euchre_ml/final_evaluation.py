"""Locked, head-to-head final evaluation on never-trained seed ranges."""

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from .env import EuchreEnv
from .evaluation import _bootstrap_interval, run_head_to_head_match
from .policies import Policy


@dataclass(frozen=True)
class CallingDiagnostics:
    hands: int
    calls: int
    successful_calls: int
    euchred_calls: int
    lone_calls: int
    successful_lone_calls: int
    lone_sweeps: int

    @property
    def calling_success_rate(self) -> float:
        return self.successful_calls / self.calls if self.calls else 0.0

    @property
    def euchre_rate(self) -> float:
        return self.euchred_calls / self.calls if self.calls else 0.0

    @property
    def lone_success_rate(self) -> float:
        return self.successful_lone_calls / self.lone_calls if self.lone_calls else 0.0

    @property
    def lone_sweep_rate(self) -> float:
        return self.lone_sweeps / self.lone_calls if self.lone_calls else 0.0


@dataclass(frozen=True)
class FinalComparison:
    opponent: str
    matches: int
    wins: int
    win_rate: float
    win_confidence_interval: Tuple[float, float]
    mean_capped_margin: float
    margin_confidence_interval: Tuple[float, float]
    diagnostics: CallingDiagnostics


def _run_diagnostic_hand(
    policy: Policy, opponent: Policy, seed: int, policy_team: int, dealer: int,
) -> Tuple[bool, bool, bool, bool]:
    """Return candidate call, success, lone call, and lone sweep flags."""
    policies = (policy, opponent) if policy_team == 0 else (opponent, policy)
    rngs = {
        seat: random.Random((seed << 4) ^ seat ^ 0x6A09E667)
        for seat in range(4)
    }
    with EuchreEnv() as env:
        result = env.reset_head_to_head_match(seed, policy_team, dealer, 1)
        while not result.done:
            seat = result.observation[1]
            action = policies[seat % 2](result, rngs[seat])
            if action < 0 or action >= len(result.action_mask) or not result.action_mask[action]:
                raise ValueError(f"Policy selected illegal action {action}")
            result = env.step(action)
    caller = result.observation[5]
    makers_team = caller % 2
    maker_tricks = result.observation[16 + makers_team]
    called = makers_team == policy_team
    alone = bool(result.observation[7])
    return (called, called and maker_tricks >= 3, called and alone,
            called and alone and maker_tricks == 5)


def evaluate_final_comparison(
    policy: Policy, opponent: Policy, opponent_name: str,
    seeds: Iterable[int], bootstrap_samples: int = 1000,
) -> FinalComparison:
    """Evaluate matches and one-hand calling diagnostics on locked seeds."""
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("At least one final-test seed is required")
    wins = 0
    margins = []
    paired_wins = []
    paired_margins = []
    calls = successes = euchres = lone_calls = lone_successes = lone_sweeps = 0
    for seed in seed_values:
        for dealer in range(4):
            pair_wins = 0
            pair_margin = 0
            for team in range(2):
                match = run_head_to_head_match(policy, opponent, seed, team, dealer)
                wins += match.won
                pair_wins += match.won
                margins.append(match.capped_margin)
                pair_margin += match.capped_margin
                called, succeeded, alone, swept = _run_diagnostic_hand(
                    policy, opponent, seed, team, dealer
                )
                calls += called
                successes += succeeded
                euchres += called and not succeeded
                lone_calls += called and alone
                lone_successes += called and alone and succeeded
                lone_sweeps += swept
            paired_wins.append(pair_wins / 2.0)
            paired_margins.append(pair_margin / 2.0)
    match_count = len(seed_values) * 8
    return FinalComparison(
        opponent=opponent_name,
        matches=match_count,
        wins=wins,
        win_rate=wins / match_count,
        win_confidence_interval=_bootstrap_interval(
            paired_wins, bootstrap_samples, 0.95, 0xF1A1
        ),
        mean_capped_margin=sum(margins) / match_count,
        margin_confidence_interval=_bootstrap_interval(
            paired_margins, bootstrap_samples, 0.95, 0xF1A2
        ),
        diagnostics=CallingDiagnostics(
            hands=match_count,
            calls=calls,
            successful_calls=successes,
            euchred_calls=euchres,
            lone_calls=lone_calls,
            successful_lone_calls=lone_successes,
            lone_sweeps=lone_sweeps,
        ),
    )


def evaluate_final_test(
    policy: Policy, opponents: Dict[str, Policy], seeds: Iterable[int],
    bootstrap_samples: int = 1000,
) -> Tuple[FinalComparison, ...]:
    seed_values = tuple(seeds)
    return tuple(
        evaluate_final_comparison(
            policy, opponent, name, seed_values, bootstrap_samples
        )
        for name, opponent in opponents.items()
    )
