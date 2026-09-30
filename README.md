# EuchreML

A C11 Euchre engine intended for fast simulation, with a future Python layer for
training and evaluating learned players.

## Current milestone

The engine currently implements:

- a 24-card Euchre deck and four five-card hands
- two-round bidding with stick-the-dealer
- dealer pickup and discard after an order-up
- going alone, partner sit-out behavior, and lone-hand scoring
- right and left bower suit behavior
- follow-suit validation and legal-move generation
- trick resolution and standard four-player scoring
- complete matches to 10 points using intentionally simple players
- deterministic simulations for comparing the same seeds
- engine tests for bidding restrictions, lone hands, scoring, and full matches
- interchangeable per-seat policies with stable actions and legal-action masks
- independent per-game random state for parallel simulations
- public bidding and per-seat trick history for strategic inference

Match play to 10 points, interactive input, and the Python bridge are planned
after the core hand logic is stable.

## Build and run

```sh
make run
```

CMake is also supported with `cmake -S . -B build` followed by
`cmake --build build`.

## Test

```sh
make test
```

The test executable runs complete matches for several fixed seeds and checks
the engine's state after every hand. It also directly checks illegal bidding,
including attempts to call the turned-down suit during round two.

## Benchmark

Run 100,000 headless hands with:

```sh
make benchmark
```

Pass a different hand count directly to `./build/euchre_benchmark` when needed.

## Design boundary

`EuchreGame` owns the rules and complete state. Each `EuchrePolicy` receives an
`EuchreObservation` containing only information available to its seat, plus a
fixed-size legal-action mask. Card actions use permanent deck IDs rather than
mutable hand positions. This gives heuristic, human, and learned players the
same interface without exposing opponents' cards or the kitty.
