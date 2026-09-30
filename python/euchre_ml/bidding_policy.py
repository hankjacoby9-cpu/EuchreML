"""Interpretable candidate-suit bidding model with evolvable weights."""

import random
from pathlib import Path
from typing import Dict, Union

import numpy as np

from .env import StepResult
from .policies import (
    CALL_ALONE_FIRST,
    CALL_FIRST,
    NO_SUIT,
    ORDER_UP,
    ORDER_UP_ALONE,
    PASS,
    _effective_suit,
)


PARAMETER_NAMES = (
    "right_bower",
    "left_bower",
    "trump_count",
    "high_trump",
    "off_suit_aces",
    "void_suits",
    "represented_suits",
    "upcard_trump_strength_self_dealer",
    "upcard_trump_strength_partner_dealer",
    "upcard_trump_strength_opponent_dealer",
    "self_is_dealer",
    "partner_is_dealer",
    "opponent_is_dealer",
    "round_one_threshold",
    "round_two_threshold",
    "alone_threshold",
)
GENOME_SIZE = len(PARAMETER_NAMES)
BIDDING_FEATURE_VERSION = 2

INITIAL_GENOME = np.asarray(
    [2.5, 2.0, 0.9, 0.4, 0.6, 0.3, -0.1,
     0.5, 0.3, -0.5, 0.2, 0.1, -0.2, 3.0, 3.0, 6.0],
    dtype=np.float32,
)


def _cards_in_hand(result: StepResult):
    return [
        card_id for card_id, present in enumerate(result.observation[20:44])
        if present
    ]


def upcard_trump_strength(rank: int) -> float:
    """Rank a turned-suit upcard from nine through right bower."""
    if rank < 0 or rank > 5:
        raise ValueError("card rank must be between 0 and 5")
    # Printed rank order is 9, 10, J, Q, K, A. The jack becomes the right
    # bower, so strategic trump order is 9, 10, Q, K, A, J.
    strength_order = (1, 2, 6, 3, 4, 5)
    return strength_order[rank] / 6.0


def candidate_features(result: StepResult, trump: int) -> np.ndarray:
    """Measure one hand for one possible trump suit from public seat data."""
    observation = result.observation
    cards = _cards_in_hand(result)
    player = observation[1]
    dealer = observation[2]
    partner = (player + 2) % 4
    same_color = 3 - trump

    right = int(any(card // 6 == trump and card % 6 == 2 for card in cards))
    left = int(any(card // 6 == same_color and card % 6 == 2 for card in cards))
    effective_suits = [_effective_suit(card, trump) for card in cards]
    trump_cards = [
        card for card, suit in zip(cards, effective_suits) if suit == trump
    ]
    high_trump = sum(card % 6 >= 3 for card in trump_cards)
    off_suit_aces = sum(
        card % 6 == 5 and suit != trump
        for card, suit in zip(cards, effective_suits)
    )
    represented = len(set(effective_suits))
    voids = 4 - represented

    upcard_strength = 0.0
    if observation[8] == 0 and trump == observation[10]:
        upcard_strength = upcard_trump_strength(observation[11] % 6)

    dealer_self = int(dealer == player)
    dealer_partner = int(dealer == partner)
    dealer_opponent = int(not dealer_self and not dealer_partner)
    return np.asarray(
        [
            right,
            left,
            len(trump_cards),
            high_trump,
            off_suit_aces,
            voids,
            represented,
            upcard_strength * dealer_self,
            upcard_strength * dealer_partner,
            upcard_strength * dealer_opponent,
            dealer_self,
            dealer_partner,
            dealer_opponent,
        ],
        dtype=np.float32,
    )


class BiddingPolicy:
    """Use evolved bidding scores and the existing random legal card strategy."""

    def __init__(self, genome: np.ndarray):
        genome = np.asarray(genome, dtype=np.float32)
        if genome.shape != (GENOME_SIZE,):
            raise ValueError(f"Expected genome shape ({GENOME_SIZE},)")
        self.genome = genome

    def suit_score(self, result: StepResult, trump: int) -> float:
        return float(np.dot(self.genome[:13], candidate_features(result, trump)))

    def __call__(self, result: StepResult, rng: random.Random) -> int:
        phase = result.observation[8]
        trump = result.observation[9]
        if phase == 0:
            if trump != NO_SUIT:
                return next(
                    action for action in range(24) if result.action_mask[action]
                )
            candidate = result.observation[10]
            score = self.suit_score(result, candidate)
            if score >= self.genome[15]:
                return ORDER_UP_ALONE
            if score >= self.genome[13]:
                return ORDER_UP
            return PASS

        if phase == 1:
            candidates = [
                suit for suit in range(4) if result.action_mask[CALL_FIRST + suit]
            ]
            scored = [(self.suit_score(result, suit), suit) for suit in candidates]
            score, candidate = max(scored)
            if score >= self.genome[15]:
                return CALL_ALONE_FIRST + candidate
            if score >= self.genome[14] or not result.action_mask[PASS]:
                return CALL_FIRST + candidate
            return PASS

        if phase == 2:
            legal_cards = [
                action for action in range(24) if result.action_mask[action]
            ]
            return rng.choice(legal_cards)
        raise ValueError(f"Bidding policy received unsupported phase {phase}")

    def named_parameters(self) -> Dict[str, float]:
        return dict(zip(PARAMETER_NAMES, map(float, self.genome)))

    def save(self, path: Union[str, Path], **metadata: object) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, genome=self.genome, parameter_names=PARAMETER_NAMES,
                 feature_version=BIDDING_FEATURE_VERSION, **metadata)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BiddingPolicy":
        with np.load(path) as checkpoint:
            if "feature_version" not in checkpoint.files:
                raise ValueError(
                    "Legacy bidding checkpoint has unversioned upcard semantics"
                )
            version = int(checkpoint["feature_version"])
            if version != BIDDING_FEATURE_VERSION:
                raise ValueError(
                    f"Unsupported bidding feature version {version}"
                )
            return cls(checkpoint["genome"].copy())


def initialize_bidding_genome() -> np.ndarray:
    return INITIAL_GENOME.copy()
