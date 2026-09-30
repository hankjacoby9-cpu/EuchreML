"""Evolve card-selection weights from complete matches to ten points."""

import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .bidding_policy import BiddingPolicy
from .card_play_policy import HybridCardPolicy, initialize_card_genome
from .evaluation import run_team_match
from .policies import Policy


@dataclass(frozen=True)
class CardEvolutionConfig:
    generations: int = 20
    mutation_pairs: int = 8
    mutation_scale: float = 0.15
    screening_seeds: int = 32
    training_seeds: int = 128
    validation_seeds: int = 256
    finalists: int = 3
    seed: int = 2026
    initial_checkpoint_path: str = ""
    checkpoint_path: str = "checkpoints/card_play_best.npz"


@dataclass(frozen=True)
class CardGenerationResult:
    generation: int
    training_advantage: float
    validation_advantage: float
    training_margin_advantage: float
    validation_margin_advantage: float
    matches_evaluated: int
    elapsed_seconds: float


@dataclass(frozen=True, order=True)
class MatchFitness:
    """Lexicographic fitness: win advantage first, capped margin second."""

    win_advantage: float
    margin_advantage: float


@dataclass(frozen=True)
class MatchCase:
    seed: int
    team: int
    dealer: int
    baseline_won: bool
    baseline_margin: int


def build_match_cases(
    seeds: Iterable[int], baseline: Policy,
) -> Tuple[MatchCase, ...]:
    """Cache baseline outcomes so every candidate receives paired matches."""
    cases = []
    for seed in seeds:
        for dealer in range(4):
            for team in range(2):
                result = run_team_match(baseline, seed, team, dealer)
                cases.append(MatchCase(
                    seed, team, dealer, result.won, result.capped_margin
                ))
    return tuple(cases)


def evaluate_card_genome(
    genome: np.ndarray, bidding: BiddingPolicy, cases: Sequence[MatchCase],
) -> MatchFitness:
    if not cases:
        raise ValueError("At least one match case is required")
    policy = HybridCardPolicy(bidding, genome)
    win_difference = 0
    margin_difference = 0
    for case in cases:
        result = run_team_match(policy, case.seed, case.team, case.dealer)
        win_difference += int(result.won) - int(case.baseline_won)
        margin_difference += result.capped_margin - case.baseline_margin
    return MatchFitness(
        win_difference / len(cases), margin_difference / len(cases)
    )


def train_card_evolution(
    bidding: BiddingPolicy, config: CardEvolutionConfig,
) -> List[CardGenerationResult]:
    if config.generations <= 0 or config.mutation_pairs <= 0:
        raise ValueError("generations and mutation_pairs must be positive")
    if min(config.screening_seeds, config.training_seeds,
           config.validation_seeds, config.finalists) <= 0:
        raise ValueError("seed counts and finalists must be positive")
    rng = np.random.default_rng(config.seed)
    genome = (
        HybridCardPolicy.load(config.initial_checkpoint_path).card_genome.copy()
        if config.initial_checkpoint_path else initialize_card_genome()
    )
    validation_cases = build_match_cases(
        range(1_000_001, 1_000_001 + config.validation_seeds), baseline=bidding
    )
    best_validation = MatchFitness(float("-inf"), float("-inf"))
    history = []
    for generation in range(1, config.generations + 1):
        started = time.perf_counter()
        first_seed = 1 + (generation - 1) * config.training_seeds
        screening_count = min(config.screening_seeds, config.training_seeds)
        screening_cases = build_match_cases(
            range(first_seed, first_seed + screening_count), baseline=bidding,
        )
        training_cases = build_match_cases(
            range(first_seed, first_seed + config.training_seeds), baseline=bidding,
        )
        candidates = [genome]
        for _ in range(config.mutation_pairs):
            noise = rng.standard_normal(genome.shape, dtype=np.float32)
            for direction in (1.0, -1.0):
                candidates.append(
                    genome + direction * config.mutation_scale * noise
                )
        screening_scores = [
            evaluate_card_genome(candidate, bidding, screening_cases)
            for candidate in candidates
        ]
        finalist_count = min(config.finalists, len(candidates))
        finalist_indices = sorted(
            range(len(candidates)), key=screening_scores.__getitem__, reverse=True
        )[:finalist_count]
        finalist_scores = [
            evaluate_card_genome(candidates[index], bidding, training_cases)
            for index in finalist_indices
        ]
        winner = max(range(finalist_count), key=finalist_scores.__getitem__)
        genome = candidates[finalist_indices[winner]]
        training = finalist_scores[winner]
        validation = evaluate_card_genome(genome, bidding, validation_cases)
        result = CardGenerationResult(
            generation=generation,
            training_advantage=training.win_advantage,
            validation_advantage=validation.win_advantage,
            training_margin_advantage=training.margin_advantage,
            validation_margin_advantage=validation.margin_advantage,
            matches_evaluated=(len(candidates) * len(screening_cases)
                               + finalist_count * len(training_cases)
                               + len(validation_cases)),
            elapsed_seconds=time.perf_counter() - started,
        )
        history.append(result)
        if validation > best_validation:
            best_validation = validation
            HybridCardPolicy(bidding, genome.copy()).save(
                config.checkpoint_path,
                generation=generation,
                validation_win_advantage=validation.win_advantage,
                validation_margin_advantage=validation.margin_advantage,
                objective="paired_match_wins_then_capped_margin",
            )
    return history
