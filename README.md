# Euchre AI

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

## Design boundary

`EuchreGame` owns the rules and state. A player—heuristic, human, or learned—only
needs to choose among the indices returned by `euchre_legal_moves`. This keeps
illegal actions out of the ML layer and gives every future player the same API.
