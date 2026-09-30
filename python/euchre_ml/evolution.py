"""Mirrored evolutionary search using paired Euchre deal fitness."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .evaluation import run_episode
from .neural_policy import NeuralPolicy, NetworkShape, initialize_genome
from .policies import Policy, simple_heuristic_policy


@dataclass(frozen=True)
class EvaluationCase:
    seed: int
    seat: int
    baseline_reward: int


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int = 5
    mutation_pairs: int = 4
    mutation_scale: float = 0.05
    learning_rate: float = 0.03
    training_seeds: int = 64
    validation_seeds: int = 128
    seed: int = 2026
    hidden_size: int = 16
    checkpoint_path: str = "checkpoints/evolution_best.npz"


@dataclass(frozen=True)
class GenerationResult:
    generation: int
    training_advantage: float
    validation_advantage: float
    best_mutation_advantage: float


def build_cases(
    seeds: Iterable[int], baseline: Policy = simple_heuristic_policy
) -> Tuple[EvaluationCase, ...]:
    """Evaluate the control policy once so every candidate uses paired rewards."""
    return tuple(
        EvaluationCase(seed, seat, run_episode(baseline, seed, seat).reward)
        for seed in seeds
        for seat in range(4)
    )


def evaluate_genome(genome: np.ndarray, cases: Sequence[EvaluationCase],
                    shape: NetworkShape) -> float:
    policy = NeuralPolicy(genome, shape)
    differences = [
        run_episode(policy, case.seed, case.seat).reward - case.baseline_reward
        for case in cases
    ]
    return float(np.mean(differences))


def train_evolution(config: EvolutionConfig) -> List[GenerationResult]:
    """Optimize terminal paired score margin with antithetic mutations."""
    if config.generations <= 0 or config.mutation_pairs <= 0:
        raise ValueError("generations and mutation_pairs must be positive")
    if config.mutation_scale <= 0.0 or config.learning_rate <= 0.0:
        raise ValueError("mutation_scale and learning_rate must be positive")

    rng = np.random.default_rng(config.seed)
    shape = NetworkShape(hidden_size=config.hidden_size)
    genome = initialize_genome(rng, shape)
    training_cases = build_cases(range(1, config.training_seeds + 1))
    validation_start = 1_000_001
    validation_cases = build_cases(
        range(validation_start, validation_start + config.validation_seeds)
    )
    best_validation = float("-inf")
    history: List[GenerationResult] = []

    for generation in range(1, config.generations + 1):
        gradient = np.zeros_like(genome)
        mutation_scores: List[float] = []
        parent_score = evaluate_genome(genome, training_cases, shape)
        best_candidate_score = parent_score
        best_candidate = None
        for _ in range(config.mutation_pairs):
            noise = rng.standard_normal(genome.shape, dtype=np.float32)
            positive_genome = genome + config.mutation_scale * noise
            negative_genome = genome - config.mutation_scale * noise
            positive = evaluate_genome(positive_genome, training_cases, shape)
            negative = evaluate_genome(negative_genome, training_cases, shape)
            gradient += (positive - negative) * noise
            mutation_scores.extend((positive, negative))
            if positive > best_candidate_score:
                best_candidate_score = positive
                best_candidate = positive_genome
            if negative > best_candidate_score:
                best_candidate_score = negative
                best_candidate = negative_genome

        if best_candidate is not None:
            genome = best_candidate
        else:
            gradient /= 2.0 * config.mutation_pairs * config.mutation_scale
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm > 1.0:
                gradient /= gradient_norm
            genome += config.learning_rate * gradient

        training_advantage = evaluate_genome(genome, training_cases, shape)
        validation_advantage = evaluate_genome(genome, validation_cases, shape)
        result = GenerationResult(
            generation=generation,
            training_advantage=training_advantage,
            validation_advantage=validation_advantage,
            best_mutation_advantage=max(mutation_scores),
        )
        history.append(result)
        if validation_advantage > best_validation:
            best_validation = validation_advantage
            NeuralPolicy(genome.copy(), shape).save(
                config.checkpoint_path,
                generation=generation,
                validation_advantage=validation_advantage,
                training_advantage=training_advantage,
                objective="paired_team_score_margin",
            )

    return history
