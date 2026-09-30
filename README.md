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

The benchmark compares a single environment, Python-interleaved environments,
and native batches. Interleaved modes perform one CFFI call per environment
decision; native modes cross CFFI once for the whole batch. The current native
batch uses synchronized episode waves and materializes every result as Python
tuples. This intentionally provides a first implementation to measure before
using the optimized asynchronous path. `reset_buffers()` and
`advance_buffers()` expose flat, writable `memoryview` objects over the C-owned
arrays and allow completed slots to be replaced while other slots continue.
NumPy can wrap a view without copying, for example:

```python
observations = np.frombuffer(batch.buffers.observations, dtype=np.int16)
observations = observations.reshape(batch.environment_count, 138)
```

## Duplicate-deal evaluation

Compare two policies on identical deals, rotating each deal through all four
seats:

```sh
make evaluate
.venv/bin/python scripts/evaluate_policies.py --policy-a heuristic \
    --policy-b random --seeds 10000
```

The report emphasizes paired reward advantage and a bootstrap confidence
interval. It also breaks advantage down by seat and reports bidding, euchre,
sweep, lone-hand, and decision-category statistics. Available reference
policies are `first`, `random`, and `heuristic`.

## Evolutionary training

Train a small masked neural policy with mirrored positive and negative weight
mutations:

```sh
make train-evolution
```

The only fitness signal is the candidate's paired team score margin relative
to the heuristic on identical deals. Training and validation use disjoint seed
ranges, and the best validation policy is saved under `checkpoints/`.

The observation encoder expands public categorical values such as seats,
suits, phases, cards, and history entries into one-hot features. The network
produces all 35 action scores, after which illegal actions are masked before
selection. Customize small experiments with:

```sh
.venv/bin/python scripts/train_evolution.py --generations 10 \
    --mutation-pairs 8 --training-seeds 256 --validation-seeds 512
```

Evaluate a saved policy on a separate, locked seed range:

```sh
.venv/bin/python scripts/evaluate_policies.py --policy-a neural \
    --checkpoint-a checkpoints/evolution_best.npz --policy-b heuristic \
    --seed-start 2000001 --seeds 10000
```

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
