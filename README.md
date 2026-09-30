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
- opaque decision-point environments that skip opponents and forced actions
- a frozen, versioned numeric observation and action-mask layout

The command-line demo supports match play to 10 points, and the Python bridge
exposes the engine at learning decision points.

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

Measure the current one-environment CFFI path and interleaved multi-environment
baseline with:

```sh
make benchmark-python
```

The interleaved modes still perform one CFFI call per environment decision.
They establish the baseline that a future native batch call must improve.

## Design boundary

`EuchreGame` owns the rules and complete state. Each `EuchrePolicy` receives an
`EuchreObservation` containing only information available to its seat, plus a
fixed-size legal-action mask. Card actions use permanent deck IDs rather than
mutable hand positions. This gives heuristic, human, and learned players the
same interface without exposing opponents' cards or the kitty.

The observation includes public bidding and trick history. It deliberately has
no fields for opponent hands, hidden kitty contents, the dealer's discard, or
internal random-generator state. A noninterference test mutates those hidden
values and requires the encoded observation to remain byte-for-byte identical.

The foreign-language buffer contract is documented in
[`docs/numeric-buffer-layout.md`](docs/numeric-buffer-layout.md). It uses a
fixed 138-value `int16_t` observation and a 35-value `uint8_t` action mask.

## Python CFFI wrapper

Install the package into an isolated Python environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e .
make test-python
```

The build compiles the C engine into `euchre_ml._native`. Python owns only an
opaque environment pointer and receives copies of the frozen numeric buffers.

```python
from euchre_ml import EuchreEnv

with EuchreEnv() as env:
    result = env.reset(seed=42, learning_seat=0)
    while not result.done:
        action = next(i for i, legal in enumerate(result.action_mask) if legal)
        result = env.step(action)
    print(result.reward)
```
