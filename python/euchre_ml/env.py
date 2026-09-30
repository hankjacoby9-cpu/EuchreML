from dataclasses import dataclass
from typing import Sequence, Tuple

from ._native import ffi, lib


LAYOUT_VERSION = lib.euchre_bridge_layout_version()
OBSERVATION_SIZE = lib.euchre_bridge_observation_size()
ACTION_COUNT = lib.euchre_bridge_action_count()

_STATUS_ERROR = -1
_STATUS_DECISION = 0
_STATUS_TERMINAL = 1


@dataclass(frozen=True)
class StepResult:
    observation: Tuple[int, ...]
    action_mask: Tuple[int, ...]
    reward: int
    done: bool


@dataclass(frozen=True)
class BatchBuffers:
    """Zero-copy flat views over the arrays owned by a EuchreBatchEnv."""

    environment_count: int
    observations: memoryview
    action_masks: memoryview
    rewards: memoryview
    statuses: memoryview
    actions: memoryview
    reset_flags: memoryview
    seeds: memoryview
    learning_seats: memoryview


class EuchreEnv:
    """Thin Python owner for one opaque C decision-point environment."""

    def __init__(self) -> None:
        self._env = ffi.NULL
        pointer = lib.euchre_bridge_create()
        if pointer == ffi.NULL:
            raise MemoryError("Unable to allocate Euchre environment")
        self._env = pointer
        self._observation = ffi.new("int16_t[]", OBSERVATION_SIZE)
        self._action_mask = ffi.new("uint8_t[]", ACTION_COUNT)
        self._reward = ffi.new("int *")

    def close(self) -> None:
        if self._env != ffi.NULL:
            lib.euchre_bridge_destroy(self._env)
            self._env = ffi.NULL

    def __enter__(self) -> "EuchreEnv":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_env"):
            self.close()

    def reset(self, seed: int, learning_seat: int) -> StepResult:
        self._require_open()
        status = lib.euchre_bridge_reset(
            self._env,
            seed,
            learning_seat,
            self._observation,
            self._action_mask,
            self._reward,
        )
        return self._read_result(status)

    def step(self, action: int) -> StepResult:
        self._require_open()
        status = lib.euchre_bridge_step(
            self._env,
            action,
            self._observation,
            self._action_mask,
            self._reward,
        )
        return self._read_result(status)

    def _require_open(self) -> None:
        if self._env == ffi.NULL:
            raise RuntimeError("Euchre environment is closed")

    def _read_result(self, status: int) -> StepResult:
        if status == _STATUS_ERROR:
            raise ValueError("The C environment rejected the requested action")
        if status not in (_STATUS_DECISION, _STATUS_TERMINAL):
            raise RuntimeError(f"Unknown C environment status: {status}")
        observation = tuple(self._observation[index]
                            for index in range(OBSERVATION_SIZE))
        action_mask = tuple(self._action_mask[index]
                            for index in range(ACTION_COUNT))
        return StepResult(
            observation=observation,
            action_mask=action_mask,
            reward=self._reward[0],
            done=status == _STATUS_TERMINAL,
        )


class EuchreBatchEnv:
    """Opaque C-owned environment batch advanced through one CFFI call."""

    def __init__(self, environment_count: int) -> None:
        if environment_count <= 0:
            raise ValueError("environment_count must be positive")
        self._batch = ffi.NULL
        pointer = lib.euchre_bridge_batch_create(environment_count)
        if pointer == ffi.NULL:
            raise MemoryError("Unable to allocate Euchre environment batch")
        self._batch = pointer
        self.environment_count = environment_count
        self._seeds = ffi.new("uint64_t[]", environment_count)
        self._learning_seats = ffi.new("int[]", environment_count)
        self._actions = ffi.new("int[]", environment_count)
        self._reset_flags = ffi.new("uint8_t[]", environment_count)
        self._observations = ffi.new(
            "int16_t[]", environment_count * OBSERVATION_SIZE
        )
        self._action_masks = ffi.new(
            "uint8_t[]", environment_count * ACTION_COUNT
        )
        self._rewards = ffi.new("int[]", environment_count)
        self._statuses = ffi.new("int[]", environment_count)
        self.buffers = BatchBuffers(
            environment_count=environment_count,
            observations=self._view(
                self._observations, "int16_t", "h",
                environment_count * OBSERVATION_SIZE,
            ),
            action_masks=self._view(
                self._action_masks, "uint8_t", "B",
                environment_count * ACTION_COUNT,
            ),
            rewards=self._view(
                self._rewards, "int", "i", environment_count
            ),
            statuses=self._view(
                self._statuses, "int", "i", environment_count
            ),
            actions=self._view(
                self._actions, "int", "i", environment_count
            ),
            reset_flags=self._view(
                self._reset_flags, "uint8_t", "B", environment_count
            ),
            seeds=self._view(
                self._seeds, "uint64_t", "Q", environment_count
            ),
            learning_seats=self._view(
                self._learning_seats, "int", "i", environment_count
            ),
        )

    def close(self) -> None:
        if self._batch != ffi.NULL:
            lib.euchre_bridge_batch_destroy(self._batch)
            self._batch = ffi.NULL

    def __enter__(self) -> "EuchreBatchEnv":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_batch"):
            self.close()

    def reset(
        self, seeds: Sequence[int], learning_seats: Sequence[int]
    ) -> Tuple[StepResult, ...]:
        self._require_open()
        self._require_count(seeds, "seeds")
        self._require_count(learning_seats, "learning_seats")
        for index in range(self.environment_count):
            self._seeds[index] = seeds[index]
            self._learning_seats[index] = learning_seats[index]
        success = lib.euchre_bridge_batch_reset(
            self._batch,
            self._seeds,
            self._learning_seats,
            self._observations,
            self._action_masks,
            self._rewards,
            self._statuses,
        )
        return self._read_results(success)

    def step(self, actions: Sequence[int]) -> Tuple[StepResult, ...]:
        self._require_open()
        self._require_count(actions, "actions")
        for index, action in enumerate(actions):
            self._actions[index] = action
        success = lib.euchre_bridge_batch_step(
            self._batch,
            self._actions,
            self._observations,
            self._action_masks,
            self._rewards,
            self._statuses,
        )
        return self._read_results(success)

    def reset_buffers(
        self, seeds: Sequence[int], learning_seats: Sequence[int]
    ) -> BatchBuffers:
        """Reset every slot and return views without creating Python tuples."""
        self._require_open()
        self._require_count(seeds, "seeds")
        self._require_count(learning_seats, "learning_seats")
        for index in range(self.environment_count):
            self._seeds[index] = seeds[index]
            self._learning_seats[index] = learning_seats[index]
        success = lib.euchre_bridge_batch_reset(
            self._batch,
            self._seeds,
            self._learning_seats,
            self._observations,
            self._action_masks,
            self._rewards,
            self._statuses,
        )
        self._check_success(success)
        return self.buffers

    def advance_buffers(self) -> BatchBuffers:
        """Step or reset every slot using values in the writable input views."""
        self._require_open()
        success = lib.euchre_bridge_batch_advance(
            self._batch,
            self._actions,
            self._reset_flags,
            self._seeds,
            self._learning_seats,
            self._observations,
            self._action_masks,
            self._rewards,
            self._statuses,
        )
        self._check_success(success)
        return self.buffers

    def _require_open(self) -> None:
        if self._batch == ffi.NULL:
            raise RuntimeError("Euchre environment batch is closed")

    def _require_count(self, values: Sequence[int], name: str) -> None:
        if len(values) != self.environment_count:
            raise ValueError(
                f"{name} must contain {self.environment_count} values"
            )

    def _read_results(self, success: int) -> Tuple[StepResult, ...]:
        self._check_success(success)

        results = []
        for environment in range(self.environment_count):
            status = self._statuses[environment]
            if status not in (_STATUS_DECISION, _STATUS_TERMINAL):
                raise RuntimeError(f"Unknown C environment status: {status}")
            observation_start = environment * OBSERVATION_SIZE
            mask_start = environment * ACTION_COUNT
            results.append(
                StepResult(
                    observation=tuple(
                        self._observations[observation_start + offset]
                        for offset in range(OBSERVATION_SIZE)
                    ),
                    action_mask=tuple(
                        self._action_masks[mask_start + offset]
                        for offset in range(ACTION_COUNT)
                    ),
                    reward=self._rewards[environment],
                    done=status == _STATUS_TERMINAL,
                )
            )
        return tuple(results)

    @staticmethod
    def _view(cdata, c_type: str, format_code: str, length: int) -> memoryview:
        byte_view = memoryview(ffi.buffer(cdata, ffi.sizeof(c_type) * length))
        return byte_view if format_code == "B" else byte_view.cast(format_code)

    @staticmethod
    def _check_success(success: int) -> None:
        if not success:
            raise ValueError("The C environment batch rejected an action")
