"""Disjoint seed namespaces for selection, validation, and final reporting."""

TRAINING_SEED_START = 1
VALIDATION_SEED_START = 1_000_001
FINAL_TEST_SEED_START = 9_000_001


def validate_reserved_ranges(
    generations: int, training_seeds: int, validation_seeds: int,
) -> None:
    training_end = TRAINING_SEED_START + generations * training_seeds - 1
    validation_end = VALIDATION_SEED_START + validation_seeds - 1
    if training_end >= VALIDATION_SEED_START:
        raise ValueError("Training seeds overlap the reserved validation range")
    if validation_end >= FINAL_TEST_SEED_START:
        raise ValueError("Validation seeds overlap the reserved final-test range")
