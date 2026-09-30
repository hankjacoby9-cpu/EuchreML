from dataclasses import dataclass
from typing import Tuple

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
