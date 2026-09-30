"""Evolutionary search over the interpretable bidding policy parameters."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .bidding_policy import BiddingPolicy, initialize_bidding_genome
from .evaluation import run_team_episode
from .evolution import EvaluationCase, build_cases


@dataclass(frozen=True)
class BiddingEvolutionConfig:
    generations: int = 25
    mutation_pairs: int = 8
    mutation_scale: float = 0.15
    learning_rate: float = 0.05
    training_seeds: int = 256
    validation_seeds: int = 512
    seed: int = 2026
    checkpoint_path: str = "checkpoints/bidding_best.npz"
    log_path: str = "checkpoints/bidding_history.jsonl"


@dataclass(frozen=True)
class BiddingGenerationResult:
    generation: int
    training_advantage: float
    validation_advantage: float
    best_mutation_advantage: float
    hands_evaluated: int
    elapsed_seconds: float


def evaluate_bidding_genome(
    genome: np.ndarray, cases: Sequence[EvaluationCase]
) -> float:
    policy = BiddingPolicy(genome)
    differences = [
        run_team_episode(policy, case.seed, case.seat).reward
        - case.baseline_reward
        for case in cases
    ]
    return float(np.mean(differences))


def train_bidding_evolution(
    config: BiddingEvolutionConfig,
) -> List[BiddingGenerationResult]:
    """Evolve bidding weights while card play remains at the reference policy."""
    if config.generations <= 0 or config.mutation_pairs <= 0:
        raise ValueError("generations and mutation_pairs must be positive")
    if config.mutation_scale <= 0.0 or config.learning_rate <= 0.0:
        raise ValueError("mutation_scale and learning_rate must be positive")
    if config.training_seeds <= 0 or config.validation_seeds <= 0:
        raise ValueError("training and validation seed counts must be positive")

    rng = np.random.default_rng(config.seed)
    genome = initialize_bidding_genome()
    validation_start = 1_000_001
    validation_cases = build_cases(
        range(validation_start, validation_start + config.validation_seeds)
    )
    best_validation = float("-inf")
    history: List[BiddingGenerationResult] = []
    log_path = Path(config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    for generation in range(1, config.generations + 1):
        generation_start = time.perf_counter()
        seed_start = 1 + (generation - 1) * config.training_seeds
        training_cases = build_cases(
            range(seed_start, seed_start + config.training_seeds)
        )
        parent_score = evaluate_bidding_genome(genome, training_cases)
        best_score = parent_score
        best_genome = None
        gradient = np.zeros_like(genome)
        mutation_scores = []

        for _ in range(config.mutation_pairs):
            noise = rng.standard_normal(genome.shape, dtype=np.float32)
            positive_genome = genome + config.mutation_scale * noise
            negative_genome = genome - config.mutation_scale * noise
            positive = evaluate_bidding_genome(positive_genome, training_cases)
            negative = evaluate_bidding_genome(negative_genome, training_cases)
            mutation_scores.extend((positive, negative))
            gradient += (positive - negative) * noise
            if positive > best_score:
                best_score = positive
                best_genome = positive_genome
            if negative > best_score:
                best_score = negative
                best_genome = negative_genome

        if best_genome is not None:
            genome = best_genome
            training_advantage = best_score
        else:
            gradient /= 2.0 * config.mutation_pairs * config.mutation_scale
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm > 1.0:
                gradient /= gradient_norm
            genome += config.learning_rate * gradient
            training_advantage = evaluate_bidding_genome(genome, training_cases)

        validation_advantage = evaluate_bidding_genome(genome, validation_cases)
        result = BiddingGenerationResult(
            generation=generation,
            training_advantage=training_advantage,
            validation_advantage=validation_advantage,
            best_mutation_advantage=max(mutation_scores),
            hands_evaluated=(1 + 2 * config.mutation_pairs)
                * len(training_cases) + len(validation_cases),
            elapsed_seconds=time.perf_counter() - generation_start,
        )
        history.append(result)
        with log_path.open("a") as log_file:
            log_file.write(json.dumps(result.__dict__) + "\n")
        if validation_advantage > best_validation:
            best_validation = validation_advantage
            BiddingPolicy(genome.copy()).save(
                config.checkpoint_path,
                generation=generation,
                training_advantage=training_advantage,
                validation_advantage=validation_advantage,
                objective="paired_team_score_margin",
            )

    return history
