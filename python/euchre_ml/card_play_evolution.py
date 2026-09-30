"""Evolve card-selection weights while holding bidding fixed."""

import time
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .bidding_policy import BiddingPolicy
from .card_play_policy import HybridCardPolicy, initialize_card_genome
from .evaluation import run_team_episode
from .evolution import EvaluationCase, build_cases


@dataclass(frozen=True)
class CardEvolutionConfig:
    generations: int = 20
    mutation_pairs: int = 8
    mutation_scale: float = 0.15
    training_seeds: int = 128
    validation_seeds: int = 256
    seed: int = 2026
    checkpoint_path: str = "checkpoints/card_play_best.npz"


@dataclass(frozen=True)
class CardGenerationResult:
    generation: int
    training_advantage: float
    validation_advantage: float
    elapsed_seconds: float


def evaluate_card_genome(
    genome: np.ndarray, bidding: BiddingPolicy,
    cases: Sequence[EvaluationCase],
) -> float:
    policy = HybridCardPolicy(bidding, genome)
    return float(np.mean([
        run_team_episode(policy, case.seed, case.seat, case.dealer).reward
        - case.baseline_reward
        for case in cases
    ]))


def train_card_evolution(
    bidding: BiddingPolicy, config: CardEvolutionConfig,
) -> List[CardGenerationResult]:
    rng = np.random.default_rng(config.seed)
    genome = initialize_card_genome()
    validation_cases = build_cases(
        range(1_000_001, 1_000_001 + config.validation_seeds), baseline=bidding
    )
    best_validation = float("-inf")
    history = []
    for generation in range(1, config.generations + 1):
        started = time.perf_counter()
        first_seed = 1 + (generation - 1) * config.training_seeds
        cases = build_cases(
            range(first_seed, first_seed + config.training_seeds),
            baseline=bidding,
        )
        parent_score = evaluate_card_genome(genome, bidding, cases)
        best_score = parent_score
        best_genome = genome
        for _ in range(config.mutation_pairs):
            noise = rng.standard_normal(genome.shape, dtype=np.float32)
            for direction in (1.0, -1.0):
                candidate = genome + direction * config.mutation_scale * noise
                score = evaluate_card_genome(candidate, bidding, cases)
                if score > best_score:
                    best_score = score
                    best_genome = candidate
        genome = best_genome
        validation = evaluate_card_genome(genome, bidding, validation_cases)
        result = CardGenerationResult(
            generation, best_score, validation, time.perf_counter() - started
        )
        history.append(result)
        if validation > best_validation:
            best_validation = validation
            HybridCardPolicy(bidding, genome.copy()).save(
                config.checkpoint_path,
                generation=generation,
                validation_advantage=validation,
                objective="paired_team_score_margin",
            )
    return history
