"""Small reference policies for evaluation and future model comparisons."""

import random
from typing import Callable

from .env import StepResult


Policy = Callable[[StepResult, random.Random], int]

PASS = 24
ORDER_UP = 25
ORDER_UP_ALONE = 26
CALL_FIRST = 27
CALL_ALONE_FIRST = 31
NO_SUIT = 4
JACK = 2


def legal_actions(result: StepResult):
    return [action for action, enabled in enumerate(result.action_mask) if enabled]


def first_legal_policy(result: StepResult, _rng: random.Random) -> int:
    """Choose the lowest numbered legal action as a deterministic floor."""
    return next(action for action, enabled in enumerate(result.action_mask) if enabled)


def random_legal_policy(result: StepResult, rng: random.Random) -> int:
    """Sample uniformly from legal actions using the evaluator's seeded RNG."""
    return rng.choice(legal_actions(result))


def _effective_suit(card_id: int, trump: int) -> int:
    suit, rank = divmod(card_id, 6)
    same_color_suit = 3 - trump
    if rank == JACK and suit == same_color_suit:
        return trump
    return suit


def _count_suit(result: StepResult, suit: int) -> int:
    hand_mask = result.observation[20:44]
    return sum(
        present and _effective_suit(card_id, suit) == suit
        for card_id, present in enumerate(hand_mask)
    )


def simple_heuristic_policy(result: StepResult, rng: random.Random) -> int:
    """Match the C baseline's suit-count bidding and random legal card play."""
    observation = result.observation
    phase = observation[8]
    trump = observation[9]
    turned_suit = observation[10]

    if phase == 0:
        if trump != NO_SUIT:
            return first_legal_policy(result, rng)
        trump_count = _count_suit(result, turned_suit)
        if trump_count >= 2:
            return ORDER_UP_ALONE if trump_count >= 4 else ORDER_UP
        return PASS

    if phase == 1:
        best_suit = NO_SUIT
        best_count = -1
        for suit in range(4):
            if not result.action_mask[CALL_FIRST + suit]:
                continue
            count = _count_suit(result, suit)
            if count > best_count:
                best_suit = suit
                best_count = count
        if best_count >= 2 or not result.action_mask[PASS]:
            first_call = CALL_ALONE_FIRST if best_count >= 4 else CALL_FIRST
            return first_call + best_suit
        return PASS

    if phase == 2:
        return rng.choice(
            [action for action in range(24) if result.action_mask[action]]
        )

    raise ValueError(f"Policy received unsupported phase {phase}")


POLICIES = {
    "first": first_legal_policy,
    "random": random_legal_policy,
    "heuristic": simple_heuristic_policy,
}
