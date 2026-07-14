from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

from .state_machine import MotionCommand, RaceState


class ControlOwner(str, Enum):
    NONE = "none"
    CRUISE = "cruise"
    CORNER_EXECUTOR = "corner_executor"
    RING_EXECUTOR = "ring_executor"
    FORK_EXECUTOR = "fork_executor"
    TERMINAL_EXECUTOR = "terminal_executor"
    MANUAL = "manual"


class SafetyState(str, Enum):
    CLEAR = "clear"
    SLOW = "slow"
    STOP_LATCHED = "stop_latched"


class StopCause(str, Enum):
    OBSTACLE = "obstacle"
    SEARCH_TIMEOUT = "search_timeout"
    ROUTE_LOST = "route_lost"
    EXECUTOR_FAILED = "executor_failed"
    MISSION_FINISHED = "mission_finished"
    OPERATOR_STOP = "operator_stop"
    PROCESS_LEASE_LOST = "process_lease_lost"
    CAMERA_FAILURE = "camera_failure"
    INVALID_COMMAND = "invalid_command"


class TransitionEvent(str, Enum):
    NONE = "none"
    TAKEOVER_ACCEPTED = "takeover_accepted"
    HANDOFF_READY = "handoff_ready"
    EXECUTOR_HELD = "executor_held"
    EXECUTOR_ABORTED = "executor_aborted"
    EXECUTOR_COMPLETED = "executor_completed"
    ROUTE_LOST = "route_lost"
    STOP_LATCHED = "stop_latched"
    SLOW_REQUESTED = "slow_requested"


@dataclass(frozen=True, slots=True)
class ArbitrationResult:
    owner: ControlOwner
    owner_epoch: int
    candidate: MotionCommand
    final: MotionCommand
    safety_state: SafetyState
    stop_cause: StopCause | None
    safety_override: str | None
    transition_event: TransitionEvent


class CommandArbiter:
    """The only place where an accepted controller command is safety-overridden.

    Mission and executor state remain untouched: safety may clamp or stop the
    command, but it never pretends that another controller won ownership.
    """

    def __init__(self, *, max_v: float, max_w: float) -> None:
        self.max_v = abs(float(max_v))
        self.max_w = abs(float(max_w))
        self.owner = ControlOwner.NONE
        self.owner_epoch = 0
        self._latched_cause: StopCause | None = None

    _AUTO_RECOVERABLE = frozenset({
        StopCause.OBSTACLE,
        StopCause.SEARCH_TIMEOUT,
        StopCause.CAMERA_FAILURE,
    })

    @property
    def latched_cause(self) -> StopCause | None:
        return self._latched_cause

    def clear_stop(self, expected_cause: StopCause) -> None:
        """Explicit clear authority for non-auto-recoverable stop causes."""
        if self._latched_cause != expected_cause:
            raise ValueError(
                f"cannot clear {expected_cause.value}; "
                f"latched={None if self._latched_cause is None else self._latched_cause.value}"
            )
        self._latched_cause = None

    def resolve(
        self,
        candidate: MotionCommand,
        *,
        owner: ControlOwner,
        stop_cause: StopCause | None = None,
        slow_v_limit: float | None = None,
        transition_event: TransitionEvent = TransitionEvent.NONE,
    ) -> ArbitrationResult:
        if owner != self.owner:
            self.owner = owner
            self.owner_epoch += 1
            if transition_event == TransitionEvent.NONE:
                transition_event = TransitionEvent.TAKEOVER_ACCEPTED

        invalid = not all(math.isfinite(value) for value in (candidate.v, candidate.w))
        invalid = invalid or abs(candidate.v) > self.max_v + 1e-9
        invalid = invalid or abs(candidate.w) > self.max_w + 1e-9
        if invalid:
            stop_cause = StopCause.INVALID_COMMAND

        if stop_cause is not None:
            if (
                self._latched_cause is None
                or self._latched_cause in self._AUTO_RECOVERABLE
            ):
                self._latched_cause = stop_cause
        elif self._latched_cause in self._AUTO_RECOVERABLE:
            self._latched_cause = None
        stop_cause = self._latched_cause

        if stop_cause is not None:
            final = replace(candidate, v=0.0, w=0.0, state=RaceState.STOPPED)
            event = (
                transition_event
                if transition_event != TransitionEvent.NONE
                else TransitionEvent.STOP_LATCHED
            )
            return ArbitrationResult(
                owner,
                self.owner_epoch,
                candidate,
                final,
                SafetyState.STOP_LATCHED,
                stop_cause,
                f"stop:{stop_cause.value}",
                event,
            )

        if slow_v_limit is not None and candidate.v > slow_v_limit:
            final = replace(candidate, v=max(0.0, float(slow_v_limit)))
            return ArbitrationResult(
                owner,
                self.owner_epoch,
                candidate,
                final,
                SafetyState.SLOW,
                None,
                f"v_clamped:{candidate.v:.4f}->{final.v:.4f}",
                TransitionEvent.SLOW_REQUESTED,
            )

        return ArbitrationResult(
            owner,
            self.owner_epoch,
            candidate,
            candidate,
            SafetyState.CLEAR,
            None,
            None,
            transition_event,
        )
