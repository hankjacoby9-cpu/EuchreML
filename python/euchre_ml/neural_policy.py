"""A small masked neural policy stored as one evolvable weight vector."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from .env import ACTION_COUNT, StepResult
from .features import FEATURE_SIZE, encode_features


@dataclass(frozen=True)
class NetworkShape:
    input_size: int = FEATURE_SIZE
    hidden_size: int = 16
    output_size: int = ACTION_COUNT

    @property
    def genome_size(self) -> int:
        return (
            self.input_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.output_size
            + self.output_size
        )


class NeuralPolicy:
    """Deterministic MLP whose illegal action logits are never selectable."""

    def __init__(self, genome: np.ndarray, shape: NetworkShape = NetworkShape()):
        genome = np.asarray(genome, dtype=np.float32)
        if genome.shape != (shape.genome_size,):
            raise ValueError(f"Expected genome shape ({shape.genome_size},)")
        self.genome = genome
        self.shape = shape
        offset = 0
        end = shape.input_size * shape.hidden_size
        self.w1 = genome[offset:end].reshape(shape.hidden_size, shape.input_size)
        offset = end
        end += shape.hidden_size
        self.b1 = genome[offset:end]
        offset = end
        end += shape.hidden_size * shape.output_size
        self.w2 = genome[offset:end].reshape(shape.output_size, shape.hidden_size)
        self.b2 = genome[end:]

    def __call__(self, result: StepResult, _rng: random.Random) -> int:
        features = encode_features(result.observation)
        # np.dot avoids spurious floating-point warnings emitted by NumPy 2.0's
        # macOS matmul path for finite float32 matrix-vector products.
        hidden = np.tanh(np.dot(self.w1, features) + self.b1)
        logits = np.dot(self.w2, hidden) + self.b2
        legal = np.asarray(result.action_mask, dtype=bool)
        if not legal.any():
            raise ValueError("Neural policy received no legal actions")
        return int(np.argmax(np.where(legal, logits, -np.inf)))

    def save(self, path: Union[str, Path], **metadata: object) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            genome=self.genome,
            input_size=self.shape.input_size,
            hidden_size=self.shape.hidden_size,
            output_size=self.shape.output_size,
            **metadata,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NeuralPolicy":
        with np.load(path) as checkpoint:
            shape = NetworkShape(
                input_size=int(checkpoint["input_size"]),
                hidden_size=int(checkpoint["hidden_size"]),
                output_size=int(checkpoint["output_size"]),
            )
            return cls(checkpoint["genome"].copy(), shape)


def initialize_genome(
    rng: np.random.Generator, shape: NetworkShape = NetworkShape()
) -> np.ndarray:
    """Initialize both layers at fan-in-scaled magnitudes with zero biases."""
    w1 = rng.normal(
        0.0, np.sqrt(2.0 / shape.input_size),
        size=(shape.hidden_size, shape.input_size),
    )
    b1 = np.zeros(shape.hidden_size)
    w2 = rng.normal(
        0.0, np.sqrt(2.0 / shape.hidden_size),
        size=(shape.output_size, shape.hidden_size),
    )
    b2 = np.zeros(shape.output_size)
    return np.concatenate((w1.ravel(), b1, w2.ravel(), b2)).astype(np.float32)
