"""Convert frozen public observations into neural-network input features."""

from typing import Iterable, List, Sequence

import numpy as np

from .env import LAYOUT_VERSION, OBSERVATION_SIZE


FEATURE_SIZE = 841


def _one_hot(features: List[float], value: int, values: Iterable[int]) -> None:
    choices = tuple(values)
    if value not in choices:
        raise ValueError(f"Unexpected categorical value {value}; expected {choices}")
    features.extend(1.0 if value == choice else 0.0 for choice in choices)


def encode_features(observation: Sequence[int]) -> np.ndarray:
    """Encode one version-1 public observation without adding private state."""
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(f"Expected {OBSERVATION_SIZE} observation values")
    if observation[0] != LAYOUT_VERSION:
        raise ValueError(f"Unsupported observation layout {observation[0]}")

    features: List[float] = []
    for offset in (1, 2, 3):
        _one_hot(features, observation[offset], range(4))
    for offset in (4, 5, 6):
        _one_hot(features, observation[offset], (-1, 0, 1, 2, 3))
    features.append(float(observation[7]))
    _one_hot(features, observation[8], range(4))
    _one_hot(features, observation[9], range(5))
    _one_hot(features, observation[10], range(5))
    _one_hot(features, observation[11], (-1, *range(24)))

    scales = (4.0, 8.0, 5.0, 4.0, 5.0, 5.0, 10.0, 10.0)
    features.extend(
        float(value) / scale
        for value, scale in zip(observation[12:20], scales)
    )
    features.extend(float(value) for value in observation[20:68])

    for record in range(8):
        start = 68 + record * 5
        _one_hot(features, observation[start], (-1, 0, 1, 2, 3))
        _one_hot(features, observation[start + 1], (-1, 1, 2))
        _one_hot(features, observation[start + 2], (-1, 0, 1, 2))
        _one_hot(features, observation[start + 3], (-1, 0, 1, 2, 3, 4))
        _one_hot(features, observation[start + 4], (-1, 0, 1))

    for card_id in observation[108:128]:
        _one_hot(features, card_id, (-1, *range(24)))
    for seat in observation[128:138]:
        _one_hot(features, seat, (-1, 0, 1, 2, 3))

    if len(features) != FEATURE_SIZE:
        raise RuntimeError(f"Feature layout produced {len(features)} values")
    return np.asarray(features, dtype=np.float32)
