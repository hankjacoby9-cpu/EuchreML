"""Duplicate-deal evaluation that separates policy skill from card variance."""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from .env import EuchreEnv, StepResult
from .policies import Policy


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    seat: int
    reward: int
    decisions: Dict[str, int]
    calls: int
    own_call: bool
    own_call_euchred: bool
    own_call_sweep: bool
    own_lone_call: bool
    own_lone_sweep: bool


@dataclass
class PolicySummary:
    hands: int = 0
    reward_total: int = 0
    hand_wins: int = 0
    decisions: Dict[str, int] = field(default_factory=dict)
    calls: int = 0
    own_calls: int = 0
    own_call_euchres: int = 0
    own_call_sweeps: int = 0
    own_lone_calls: int = 0
    own_lone_sweeps: int = 0

    def add(self, episode: EpisodeResult) -> None:
        self.hands += 1
        self.reward_total += episode.reward
        self.hand_wins += episode.reward > 0
        self.calls += episode.calls
        self.own_calls += episode.own_call
        self.own_call_euchres += episode.own_call_euchred
        self.own_call_sweeps += episode.own_call_sweep
        self.own_lone_calls += episode.own_lone_call
        self.own_lone_sweeps += episode.own_lone_sweep
        for category, count in episode.decisions.items():
            self.decisions[category] = self.decisions.get(category, 0) + count

    @property
    def mean_reward(self) -> float:
        return self.reward_total / self.hands

    @property
    def hand_win_rate(self) -> float:
        return self.hand_wins / self.hands


@dataclass(frozen=True)
class PairedEvaluation:
    policy_a: str
    policy_b: str
    pairs: int
    summary_a: PolicySummary
    summary_b: PolicySummary
    mean_advantage: float
    confidence_interval: Tuple[float, float]
    paired_wins: int
    paired_ties: int
    paired_losses: int
    advantage_by_seat: Tuple[float, ...]


@dataclass(frozen=True)
class MatchResult:
    seed: int
    team: int
    starting_dealer: int
    won: bool
    capped_margin: int
    raw_score: Tuple[int, int]


@dataclass(frozen=True)
class PairedMatchEvaluation:
    policy_a: str
    policy_b: str
    pairs: int
    wins_a: int
    wins_b: int
    paired_win_advantage: float
    win_confidence_interval: Tuple[float, float]
    capped_margin_advantage: float
    margin_confidence_interval: Tuple[float, float]


def _decision_category(result: StepResult) -> str:
    phase = result.observation[8]
    if phase == 0:
        return "discard" if result.observation[9] != 4 else "round_one_bid"
    if phase == 1:
        return "round_two_bid"
    if phase == 2:
        return "card_play"
    return "unknown"


def run_episode(policy: Policy, seed: int, seat: int) -> EpisodeResult:
    """Run one policy in one seat against deterministic heuristic opponents."""
    policy_rng = random.Random((seed << 3) ^ seat ^ 0xA5A5A5A5)
    decisions: Dict[str, int] = {}
    calls = 0

    with EuchreEnv() as env:
        result = env.reset(seed=seed, learning_seat=seat)
        while not result.done:
            category = _decision_category(result)
            decisions[category] = decisions.get(category, 0) + 1
            action = policy(result, policy_rng)
            if action < 0 or action >= len(result.action_mask) or not result.action_mask[action]:
                raise ValueError(f"Policy selected illegal action {action}")
            if action in (25, 26) or 27 <= action <= 34:
                calls += 1
            result = env.step(action)

    caller = result.observation[5]
    going_alone = bool(result.observation[7])
    makers_team = caller % 2
    maker_tricks = result.observation[16 + makers_team]
    own_call = caller == seat
    return EpisodeResult(
        seed=seed,
        seat=seat,
        reward=result.reward,
        decisions=decisions,
        calls=calls,
        own_call=own_call,
        own_call_euchred=own_call and maker_tricks < 3,
        own_call_sweep=own_call and maker_tricks == 5,
        own_lone_call=own_call and going_alone,
        own_lone_sweep=own_call and going_alone and maker_tricks == 5,
    )


def run_team_episode(
    policy: Policy, seed: int, team: int, dealer: int = 0
) -> EpisodeResult:
    """Run one shared policy for a partnership with separate per-seat RNGs."""
    if team not in (0, 1):
        raise ValueError("team must be 0 or 1")
    policy_rngs = {
        seat: random.Random((seed << 3) ^ seat ^ 0xA5A5A5A5)
        for seat in (team, team + 2)
    }
    decisions: Dict[str, int] = {}
    calls = 0

    with EuchreEnv() as env:
        result = env.reset_team(seed=seed, controlled_team=team, dealer=dealer)
        while not result.done:
            acting_seat = result.observation[1]
            if acting_seat not in policy_rngs:
                raise RuntimeError("Environment exposed a non-partner decision")
            category = _decision_category(result)
            decisions[category] = decisions.get(category, 0) + 1
            action = policy(result, policy_rngs[acting_seat])
            if action < 0 or action >= len(result.action_mask) or not result.action_mask[action]:
                raise ValueError(f"Policy selected illegal action {action}")
            if action in (25, 26) or 27 <= action <= 34:
                calls += 1
            result = env.step(action)

    caller = result.observation[5]
    going_alone = bool(result.observation[7])
    makers_team = caller % 2
    maker_tricks = result.observation[16 + makers_team]
    team_called = makers_team == team
    return EpisodeResult(
        seed=seed,
        seat=team,
        reward=result.reward,
        decisions=decisions,
        calls=calls,
        own_call=team_called,
        own_call_euchred=team_called and maker_tricks < 3,
        own_call_sweep=team_called and maker_tricks == 5,
        own_lone_call=team_called and going_alone,
        own_lone_sweep=team_called and going_alone and maker_tricks == 5,
    )


def run_team_match(
    policy: Policy, seed: int, team: int, starting_dealer: int,
    target_score: int = 10,
) -> MatchResult:
    """Run one private partnership policy through a complete match."""
    policy_rngs = {
        seat: random.Random((seed << 3) ^ seat ^ 0xA5A5A5A5)
        for seat in (team, team + 2)
    }
    with EuchreEnv() as env:
        result = env.reset_match(seed, team, starting_dealer, target_score)
        while not result.done:
            seat = result.observation[1]
            if seat not in policy_rngs:
                raise RuntimeError("Match environment exposed an opponent view")
            action = policy(result, policy_rngs[seat])
            if not result.action_mask[action]:
                raise ValueError(f"Policy selected illegal action {action}")
            result = env.step(action)
    raw_score = (result.observation[18], result.observation[19])
    return MatchResult(
        seed=seed,
        team=team,
        starting_dealer=starting_dealer,
        won=raw_score[team] >= target_score,
        capped_margin=result.reward,
        raw_score=raw_score,
    )


def _bootstrap_interval(
    differences: Sequence[int], samples: int, confidence: float, seed: int
) -> Tuple[float, float]:
    if not differences:
        raise ValueError("At least one paired difference is required")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if len(set(differences)) == 1:
        value = float(differences[0])
        return value, value

    rng = random.Random(seed)
    count = len(differences)
    means = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, math.floor(tail * samples))
    high_index = min(samples - 1, math.ceil((1.0 - tail) * samples) - 1)
    return means[low_index], means[high_index]


def evaluate_paired(
    policy_a: Policy,
    policy_b: Policy,
    seeds: Iterable[int],
    policy_a_name: str = "policy_a",
    policy_b_name: str = "policy_b",
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
) -> PairedEvaluation:
    """Compare policies on every identical seed and seat combination."""
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("At least one seed is required")

    summary_a = PolicySummary()
    summary_b = PolicySummary()
    differences: List[int] = []
    seat_differences: List[List[int]] = [[], [], [], []]

    for seed in seed_values:
        for seat in range(4):
            result_a = run_episode(policy_a, seed, seat)
            result_b = run_episode(policy_b, seed, seat)
            summary_a.add(result_a)
            summary_b.add(result_b)
            difference = result_a.reward - result_b.reward
            differences.append(difference)
            seat_differences[seat].append(difference)

    interval = _bootstrap_interval(
        differences, bootstrap_samples, confidence, seed=0xE0C4E
    )
    return PairedEvaluation(
        policy_a=policy_a_name,
        policy_b=policy_b_name,
        pairs=len(differences),
        summary_a=summary_a,
        summary_b=summary_b,
        mean_advantage=sum(differences) / len(differences),
        confidence_interval=interval,
        paired_wins=sum(difference > 0 for difference in differences),
        paired_ties=sum(difference == 0 for difference in differences),
        paired_losses=sum(difference < 0 for difference in differences),
        advantage_by_seat=tuple(
            sum(values) / len(values) for values in seat_differences
        ),
    )


def evaluate_paired_teams(
    policy_a: Policy,
    policy_b: Policy,
    seeds: Iterable[int],
    policy_a_name: str = "policy_a",
    policy_b_name: str = "policy_b",
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
) -> PairedEvaluation:
    """Compare shared policies on identical deals as both partnerships."""
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("At least one seed is required")

    summary_a = PolicySummary()
    summary_b = PolicySummary()
    differences: List[int] = []
    team_differences: List[List[int]] = [[], []]
    for seed in seed_values:
        for dealer in range(4):
            for team in range(2):
                result_a = run_team_episode(policy_a, seed, team, dealer)
                result_b = run_team_episode(policy_b, seed, team, dealer)
                summary_a.add(result_a)
                summary_b.add(result_b)
                difference = result_a.reward - result_b.reward
                differences.append(difference)
                team_differences[team].append(difference)

    interval = _bootstrap_interval(
        differences, bootstrap_samples, confidence, seed=0xE0C4E
    )
    return PairedEvaluation(
        policy_a=policy_a_name,
        policy_b=policy_b_name,
        pairs=len(differences),
        summary_a=summary_a,
        summary_b=summary_b,
        mean_advantage=sum(differences) / len(differences),
        confidence_interval=interval,
        paired_wins=sum(difference > 0 for difference in differences),
        paired_ties=sum(difference == 0 for difference in differences),
        paired_losses=sum(difference < 0 for difference in differences),
        advantage_by_seat=tuple(
            sum(values) / len(values) for values in team_differences
        ),
    )


def evaluate_paired_matches(
    policy_a: Policy,
    policy_b: Policy,
    seeds: Iterable[int],
    policy_a_name: str = "policy_a",
    policy_b_name: str = "policy_b",
    bootstrap_samples: int = 1000,
) -> PairedMatchEvaluation:
    """Compare match wins first and capped score margin second."""
    win_differences = []
    margin_differences = []
    wins_a = 0
    wins_b = 0
    for seed in seeds:
        for dealer in range(4):
            for team in range(2):
                result_a = run_team_match(policy_a, seed, team, dealer)
                result_b = run_team_match(policy_b, seed, team, dealer)
                wins_a += result_a.won
                wins_b += result_b.won
                win_differences.append(int(result_a.won) - int(result_b.won))
                margin_differences.append(
                    result_a.capped_margin - result_b.capped_margin
                )
    if not win_differences:
        raise ValueError("At least one seed is required")
    return PairedMatchEvaluation(
        policy_a=policy_a_name,
        policy_b=policy_b_name,
        pairs=len(win_differences),
        wins_a=wins_a,
        wins_b=wins_b,
        paired_win_advantage=sum(win_differences) / len(win_differences),
        win_confidence_interval=_bootstrap_interval(
            win_differences, bootstrap_samples, 0.95, 0xA11CE
        ),
        capped_margin_advantage=(
            sum(margin_differences) / len(margin_differences)
        ),
        margin_confidence_interval=_bootstrap_interval(
            margin_differences, bootstrap_samples, 0.95, 0xC4FFED
        ),
    )
