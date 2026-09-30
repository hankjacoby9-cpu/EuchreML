"""Evolve card-selection weights from complete matches to ten points."""

import time
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .bidding_policy import BiddingPolicy
from .card_play_policy import HybridCardPolicy, initialize_card_genome
from .evaluation import run_head_to_head_match
from .policies import Policy, simple_heuristic_policy
from .seed_ranges import VALIDATION_SEED_START, validate_reserved_ranges


@dataclass(frozen=True)
class CardEvolutionConfig:
    generations: int = 20
    mutation_pairs: int = 8
    mutation_scale: float = 0.15
    screening_seeds: int = 32
    training_seeds: int = 128
    validation_seeds: int = 256
    finalists: int = 3
    max_opponents: int = 6
    seed: int = 2026
    initial_checkpoint_path: str = ""
    checkpoint_path: str = "checkpoints/card_play_best.npz"


@dataclass(frozen=True)
class CardGenerationResult:
    generation: int
    training_win_rate: float
    validation_win_rate: float
    training_mean_margin: float
    validation_mean_margin: float
    matches_evaluated: int
    elapsed_seconds: float


@dataclass(frozen=True, order=True)
class MatchFitness:
    """Lexicographic fitness: match win rate first, capped margin second."""

    win_rate: float
    mean_margin: float


@dataclass(frozen=True)
class MatchCase:
    seed: int
    team: int
    dealer: int
    opponent_index: int


def build_match_cases(
    seeds: Iterable[int], opponents: Sequence[Policy], assignment_seed: int = 0,
) -> Tuple[MatchCase, ...]:
    """Assign a reproducible random pool opponent to each balanced matchup."""
    if not opponents:
        raise ValueError("At least one opponent is required")
    rng = random.Random(assignment_seed)
    cases = []
    for seed in seeds:
        for dealer in range(4):
            for team in range(2):
                cases.append(MatchCase(seed, team, dealer,
                                       rng.randrange(len(opponents))))
    return tuple(cases)


def evaluate_card_genome(
    genome: np.ndarray, bidding: BiddingPolicy, cases: Sequence[MatchCase],
    opponents: Sequence[Policy],
) -> MatchFitness:
    if not cases:
        raise ValueError("At least one match case is required")
    policy = HybridCardPolicy(bidding, genome)
    wins = 0
    margin = 0
    for case in cases:
        result = run_head_to_head_match(
            policy, opponents[case.opponent_index], case.seed,
            case.team, case.dealer,
        )
        wins += result.won
        margin += result.capped_margin
    return MatchFitness(wins / len(cases), margin / len(cases))


def train_card_evolution(
    bidding: BiddingPolicy, config: CardEvolutionConfig,
    opponent_pool: Sequence[Policy] = (),
) -> List[CardGenerationResult]:
    if config.generations <= 0 or config.mutation_pairs <= 0:
        raise ValueError("generations and mutation_pairs must be positive")
    if min(config.screening_seeds, config.training_seeds,
           config.validation_seeds, config.finalists,
           config.max_opponents) <= 0:
        raise ValueError("seed counts and finalists must be positive")
    validate_reserved_ranges(
        config.generations, config.training_seeds, config.validation_seeds
    )
    rng = np.random.default_rng(config.seed)
    initial_policy = (
        HybridCardPolicy.load(config.initial_checkpoint_path)
        if config.initial_checkpoint_path
        else HybridCardPolicy(bidding, initialize_card_genome())
    )
    genome = initial_policy.card_genome.copy()
    opponents = [simple_heuristic_policy, bidding, *opponent_pool]
    if config.initial_checkpoint_path:
        opponents.append(initial_policy)
    validation_opponents = tuple(opponents)
    validation_cases = build_match_cases(
        range(VALIDATION_SEED_START,
              VALIDATION_SEED_START + config.validation_seeds),
        validation_opponents, assignment_seed=0xF1A1,
    )
    best_validation = MatchFitness(float("-inf"), float("-inf"))
    history = []
    for generation in range(1, config.generations + 1):
        started = time.perf_counter()
        first_seed = 1 + (generation - 1) * config.training_seeds
        screening_count = min(config.screening_seeds, config.training_seeds)
        screening_cases = build_match_cases(
            range(first_seed, first_seed + screening_count), opponents,
            assignment_seed=config.seed ^ generation,
        )
        training_cases = build_match_cases(
            range(first_seed, first_seed + config.training_seeds), opponents,
            assignment_seed=config.seed ^ generation ^ 0xBAD5EED,
        )
        candidates = [genome]
        for _ in range(config.mutation_pairs):
            noise = rng.standard_normal(genome.shape, dtype=np.float32)
            for direction in (1.0, -1.0):
                candidates.append(
                    genome + direction * config.mutation_scale * noise
                )
        screening_scores = [
            evaluate_card_genome(candidate, bidding, screening_cases, opponents)
            for candidate in candidates
        ]
        finalist_count = min(config.finalists, len(candidates))
        finalist_indices = sorted(
            range(len(candidates)), key=screening_scores.__getitem__, reverse=True
        )[:finalist_count]
        finalist_scores = [
            evaluate_card_genome(
                candidates[index], bidding, training_cases, opponents
            )
            for index in finalist_indices
        ]
        winner = max(range(finalist_count), key=finalist_scores.__getitem__)
        genome = candidates[finalist_indices[winner]]
        training = finalist_scores[winner]
        validation = evaluate_card_genome(
            genome, bidding, validation_cases, validation_opponents
        )
        result = CardGenerationResult(
            generation=generation,
            training_win_rate=training.win_rate,
            validation_win_rate=validation.win_rate,
            training_mean_margin=training.mean_margin,
            validation_mean_margin=validation.mean_margin,
            matches_evaluated=(len(candidates) * len(screening_cases)
                               + finalist_count * len(training_cases)
                               + len(validation_cases)),
            elapsed_seconds=time.perf_counter() - started,
        )
        history.append(result)
        if validation > best_validation:
            best_validation = validation
            champion = HybridCardPolicy(bidding, genome.copy())
            champion.save(
                config.checkpoint_path,
                generation=generation,
                validation_win_rate=validation.win_rate,
                validation_mean_margin=validation.mean_margin,
                objective="opponent_pool_match_wins_then_capped_margin",
            )
            opponents.append(champion)
            if len(opponents) > config.max_opponents:
                del opponents[2]
    return history
