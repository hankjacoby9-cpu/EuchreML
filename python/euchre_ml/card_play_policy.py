"""Interpretable card-choice policy layered on a fixed bidding policy."""

import random
from pathlib import Path
from typing import Dict, Union

import numpy as np

from .bidding_policy import BIDDING_FEATURE_VERSION, BiddingPolicy
from .env import StepResult
from .policies import _effective_suit


CARD_PARAMETER_NAMES = (
    "is_trump",
    "right_bower",
    "left_bower",
    "rank_strength",
    "is_ace",
    "lead_card_strength",
    "lead_trump",
    "overtakes_winning_partner",
    "wins_current_trick",
    "conserve_strength",
)
CARD_GENOME_SIZE = len(CARD_PARAMETER_NAMES)
CARD_FEATURE_VERSION = 2
INITIAL_CARD_GENOME = np.asarray(
    [0.2, 0.2, 0.2, 0.1, 0.1, 0.0, -0.1, -0.3, 1.0, -0.4],
    dtype=np.float32,
)


def _card_power(card: int, trump: int, led_suit: int) -> int:
    suit, rank = divmod(card, 6)
    effective = _effective_suit(card, trump)
    if effective == trump:
        if suit == trump and rank == 2:
            return 200
        if suit == 3 - trump and rank == 2:
            return 199
        return 180 + rank
    if effective == led_suit:
        return 100 + rank
    return rank


def _current_trick(result: StepResult):
    trick = result.observation[14]
    start = 108 + trick * 4
    return list(result.observation[start:start + 4])


def card_features(result: StepResult, card: int) -> np.ndarray:
    observation = result.observation
    player = observation[1]
    trump = observation[9]
    suit, rank = divmod(card, 6)
    effective = _effective_suit(card, trump)
    cards = _current_trick(result)
    played = [(seat, value) for seat, value in enumerate(cards) if value >= 0]
    leading = not played
    led_suit = effective if leading else _effective_suit(played[0][1], trump)
    current_winner = None
    if played:
        current_winner = max(
            played, key=lambda item: _card_power(item[1], trump, led_suit)
        )[0]
    partner_winning = current_winner == (player + 2) % 4
    proposed_power = _card_power(card, trump, led_suit)
    wins = not played or proposed_power > max(
        _card_power(value, trump, led_suit) for _seat, value in played
    )
    right = suit == trump and rank == 2
    left = suit == 3 - trump and rank == 2
    rank_strength = proposed_power / 200.0
    return np.asarray(
        [
            effective == trump,
            right,
            left,
            rank_strength,
            rank == 5,
            leading * rank_strength,
            leading and effective == trump,
            partner_winning and wins,
            wins,
            -rank_strength,
        ],
        dtype=np.float32,
    )


class HybridCardPolicy:
    """Use a fixed bidding model and evolved public-information card scores."""

    def __init__(self, bidding: BiddingPolicy, card_genome: np.ndarray):
        card_genome = np.asarray(card_genome, dtype=np.float32)
        if card_genome.shape != (CARD_GENOME_SIZE,):
            raise ValueError(f"Expected card genome shape ({CARD_GENOME_SIZE},)")
        self.bidding = bidding
        self.card_genome = card_genome

    def __call__(self, result: StepResult, rng: random.Random) -> int:
        if result.observation[8] != 2:
            return self.bidding(result, rng)
        legal_cards = [
            card for card in range(24) if result.action_mask[card]
        ]
        return max(
            legal_cards,
            key=lambda card: float(np.dot(self.card_genome, card_features(result, card))),
        )

    def named_parameters(self) -> Dict[str, float]:
        return dict(zip(CARD_PARAMETER_NAMES, map(float, self.card_genome)))

    def save(self, path: Union[str, Path], **metadata: object) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, card_genome=self.card_genome,
                 bidding_genome=self.bidding.genome,
                 feature_version=CARD_FEATURE_VERSION,
                 bidding_feature_version=BIDDING_FEATURE_VERSION, **metadata)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HybridCardPolicy":
        with np.load(path) as checkpoint:
            if "feature_version" not in checkpoint.files:
                raise ValueError("Legacy card checkpoint has unversioned features")
            if int(checkpoint["feature_version"]) != CARD_FEATURE_VERSION:
                raise ValueError("Unsupported card feature version")
            if "bidding_feature_version" not in checkpoint.files:
                raise ValueError("Card checkpoint has unversioned bidding features")
            if int(checkpoint["bidding_feature_version"]) != BIDDING_FEATURE_VERSION:
                raise ValueError("Unsupported embedded bidding feature version")
            return cls(
                BiddingPolicy(checkpoint["bidding_genome"].copy()),
                checkpoint["card_genome"].copy(),
            )


def initialize_card_genome() -> np.ndarray:
    return INITIAL_CARD_GENOME.copy()
