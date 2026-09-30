"""Mirrored evolutionary search using paired, batched Euchre fitness."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .env import ACTION_COUNT, OBSERVATION_SIZE, EuchreBatchEnv
from .evaluation import run_team_episode
from .features import encode_feature_batch
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
    rollouts_per_candidate: int = 16
    checkpoint_path: str = "checkpoints/evolution_best.npz"
    log_path: str = "checkpoints/evolution_history.jsonl"


@dataclass(frozen=True)
class GenerationResult:
    generation: int
    training_advantage: float
    validation_advantage: float
    best_mutation_advantage: float
    hands_evaluated: int
    elapsed_seconds: float


def build_cases(
    seeds: Iterable[int], baseline: Policy = simple_heuristic_policy
) -> Tuple[EvaluationCase, ...]:
    """Evaluate the control policy once so every candidate uses paired rewards."""
    return tuple(
        EvaluationCase(seed, team, run_team_episode(baseline, seed, team).reward)
        for seed in seeds
        for team in range(2)
    )


def evaluate_genomes_batched(
    genomes: Sequence[np.ndarray], cases: Sequence[EvaluationCase],
    shape: NetworkShape, rollouts_per_candidate: int,
) -> np.ndarray:
    """Evaluate all genomes concurrently with asynchronous per-slot resets."""
    if not genomes or not cases or rollouts_per_candidate <= 0:
        raise ValueError("genomes, cases, and rollouts_per_candidate are required")

    lanes = min(rollouts_per_candidate, len(cases))
    environment_count = len(genomes) * lanes
    policies = [NeuralPolicy(genome, shape) for genome in genomes]
    slot_candidates = np.repeat(np.arange(len(genomes)), lanes)
    slot_cases = np.tile(np.arange(lanes), len(genomes))
    next_cases = np.full(len(genomes), lanes, dtype=np.int32)
    completed = np.zeros(len(genomes), dtype=np.int32)
    reward_differences = np.zeros(len(genomes), dtype=np.float64)
    active = np.ones(environment_count, dtype=bool)

    initial_cases = [cases[index] for index in slot_cases]
    with EuchreBatchEnv(environment_count) as batch:
        buffers = batch.reset_team_buffers(
            [case.seed for case in initial_cases],
            [case.seat for case in initial_cases],
        )
        observations = np.frombuffer(buffers.observations, dtype=np.int16).reshape(
            environment_count, OBSERVATION_SIZE
        )
        masks = np.frombuffer(buffers.action_masks, dtype=np.uint8).reshape(
            environment_count, ACTION_COUNT
        )
        statuses = np.frombuffer(buffers.statuses, dtype=np.int32)
        rewards = np.frombuffer(buffers.rewards, dtype=np.int32)
        actions = np.frombuffer(buffers.actions, dtype=np.int32)
        reset_flags = np.frombuffer(buffers.reset_flags, dtype=np.uint8)
        seeds = np.frombuffer(buffers.seeds, dtype=np.uint64)
        learning_seats = np.frombuffer(buffers.learning_seats, dtype=np.int32)

        while np.any(completed < len(cases)):
            reset_flags.fill(0)
            decision_slots: List[int] = []
            for slot in np.flatnonzero(active):
                candidate = slot_candidates[slot]
                if statuses[slot] == 1:
                    case = cases[slot_cases[slot]]
                    reward_differences[candidate] += (
                        int(rewards[slot]) - case.baseline_reward
                    )
                    completed[candidate] += 1
                    next_case = int(next_cases[candidate])
                    if next_case < len(cases):
                        slot_cases[slot] = next_case
                        next_cases[candidate] += 1
                        next_case_data = cases[next_case]
                        seeds[slot] = next_case_data.seed
                        learning_seats[slot] = next_case_data.seat
                        reset_flags[slot] = 1
                    else:
                        active[slot] = False
                else:
                    decision_slots.append(int(slot))

            if np.all(completed >= len(cases)):
                break

            if decision_slots:
                decision_array = np.asarray(decision_slots, dtype=np.intp)
                features = encode_feature_batch(observations[decision_array])
                for candidate, policy in enumerate(policies):
                    local = np.flatnonzero(
                        slot_candidates[decision_array] == candidate
                    )
                    if local.size == 0:
                        continue
                    hidden = np.tanh(
                        np.dot(features[local], policy.w1.T) + policy.b1
                    )
                    logits = np.dot(hidden, policy.w2.T) + policy.b2
                    legal = masks[decision_array[local]].astype(bool)
                    selected = np.argmax(np.where(legal, logits, -np.inf), axis=1)
                    actions[decision_array[local]] = selected
            batch.advance_team_buffers()

    return reward_differences / len(cases)


def evaluate_genome(
    genome: np.ndarray, cases: Sequence[EvaluationCase], shape: NetworkShape,
    rollouts_per_candidate: int = 16,
) -> float:
    return float(
        evaluate_genomes_batched(
            [genome], cases, shape, rollouts_per_candidate
        )[0]
    )


def train_evolution(config: EvolutionConfig) -> List[GenerationResult]:
    """Optimize terminal paired score margin with antithetic mutations."""
    if config.generations <= 0 or config.mutation_pairs <= 0:
        raise ValueError("generations and mutation_pairs must be positive")
    if config.mutation_scale <= 0.0 or config.learning_rate <= 0.0:
        raise ValueError("mutation_scale and learning_rate must be positive")
    if config.training_seeds <= 0 or config.validation_seeds <= 0:
        raise ValueError("training and validation seed counts must be positive")
    if config.rollouts_per_candidate <= 0:
        raise ValueError("rollouts_per_candidate must be positive")

    rng = np.random.default_rng(config.seed)
    shape = NetworkShape(hidden_size=config.hidden_size)
    genome = initialize_genome(rng, shape)
    validation_start = 1_000_001
    validation_cases = build_cases(
        range(validation_start, validation_start + config.validation_seeds)
    )
    best_validation = float("-inf")
    history: List[GenerationResult] = []
    log_path = Path(config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    for generation in range(1, config.generations + 1):
        generation_start = time.perf_counter()
        seed_start = 1 + (generation - 1) * config.training_seeds
        training_cases = build_cases(
            range(seed_start, seed_start + config.training_seeds)
        )
        noises: List[np.ndarray] = []
        candidate_genomes: List[np.ndarray] = [genome]
        for _ in range(config.mutation_pairs):
            noise = rng.standard_normal(genome.shape, dtype=np.float32)
            noises.append(noise)
            candidate_genomes.append(genome + config.mutation_scale * noise)
            candidate_genomes.append(genome - config.mutation_scale * noise)

        scores = evaluate_genomes_batched(
            candidate_genomes, training_cases, shape,
            config.rollouts_per_candidate,
        )
        parent_score = float(scores[0])
        mutation_scores = scores[1:]
        best_index = int(np.argmax(scores))
        if best_index > 0:
            genome = candidate_genomes[best_index]
        else:
            gradient = np.zeros_like(genome)
            for pair, noise in enumerate(noises):
                positive = mutation_scores[2 * pair]
                negative = mutation_scores[2 * pair + 1]
                gradient += (positive - negative) * noise
            gradient /= 2.0 * config.mutation_pairs * config.mutation_scale
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm > 1.0:
                gradient /= gradient_norm
            genome += config.learning_rate * gradient

        training_advantage = (
            float(scores[best_index]) if best_index > 0 else
            evaluate_genome(
                genome, training_cases, shape, config.rollouts_per_candidate
            )
        )
        validation_advantage = evaluate_genome(
            genome, validation_cases, shape, config.rollouts_per_candidate
        )
        elapsed_seconds = time.perf_counter() - generation_start
        result = GenerationResult(
            generation=generation,
            training_advantage=training_advantage,
            validation_advantage=validation_advantage,
            best_mutation_advantage=float(np.max(mutation_scores)),
            hands_evaluated=(1 + 2 * config.mutation_pairs) * len(training_cases)
                + len(validation_cases),
            elapsed_seconds=elapsed_seconds,
        )
        history.append(result)
        with log_path.open("a") as log_file:
            log_file.write(json.dumps(result.__dict__) + "\n")
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
