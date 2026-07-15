from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

from .control import ControlOwner
from .mission import CourseSession
from .path_memory import MotionSample
from .state_machine import MotionCommand, RaceState


class CandidateProducer(str, Enum):
    NONE = "none"
    CRUISE = "cruise"
    CORNER_EXECUTOR = "corner_executor"
    RING_EXECUTOR = "ring_executor"
    FORK_EXECUTOR = "fork_executor"
    TERMINAL_EXECUTOR = "terminal_executor"


class TransitionBarrierState(str, Enum):
    ACTIVE = "active"
    ZERO_COMMAND = "zero_command"
    AWAIT_STATIONARY = "await_stationary"
    DISPOSE_OUTGOING = "dispose_outgoing"
    INITIALIZE_INCOMING = "initialize_incoming"
    SWITCH_OWNER = "switch_owner"
    RAMP = "ramp"


class StationaryMethod(str, Enum):
    NONE = "none"
    MEASURED = "measured"
    TIMED_FALLBACK = "timed_fallback"


@dataclass(frozen=True, slots=True)
class TransitionBarrierConfig:
    linear_threshold: float = 0.008
    angular_threshold: float = 0.03
    measured_confirm_samples: int = 3
    timed_fallback_sec: float = 0.50
    max_v_slew_rate: float = 0.12
    max_w_slew_rate: float = 0.45


@dataclass(frozen=True, slots=True)
class TransitionBarrierStep:
    state: TransitionBarrierState
    command: MotionCommand
    owner: ControlOwner
    producer: CandidateProducer
    dispose_outgoing: bool = False
    initialize_incoming: bool = False
    owner_switched: bool = False
    transition_completed: bool = False
    stationary_method: StationaryMethod = StationaryMethod.NONE
    stationary_samples: int = 0
    stationary_elapsed_sec: float = 0.0
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StageFrameResult:
    stage: CourseSession
    executor_phase: str
    candidate_producer: CandidateProducer
    control_owner: ControlOwner
    candidate: MotionCommand
    transition_requested: bool
    barrier_state: TransitionBarrierState
    next_stage: CourseSession | None
    stationary_method: StationaryMethod = StationaryMethod.NONE
    stationary_samples: int = 0
    stationary_elapsed_sec: float = 0.0
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        expected_owner = owner_for_producer(self.candidate_producer)
        if expected_owner != self.control_owner:
            raise ValueError(
                "candidate producer/owner mismatch: "
                f"{self.candidate_producer.value} != {self.control_owner.value}"
            )


class StageComponent(Protocol):
    def dispose(self, next_stage: CourseSession) -> Any: ...

    def initialize(self, transfer: Any) -> None: ...


StageFactory = Callable[[], StageComponent]


def owner_for_producer(producer: CandidateProducer) -> ControlOwner:
    return ControlOwner(producer.value)


def producer_for_owner(owner: ControlOwner) -> CandidateProducer:
    return CandidateProducer(owner.value)


def zero_command(reason: str) -> MotionCommand:
    return MotionCommand(0.0, 0.0, reason, RaceState.STOPPED, None)


class TransitionBarrier:
    """Explicit stop-dispose-initialize-switch-ramp stage boundary."""

    def __init__(self, cfg: TransitionBarrierConfig | None = None) -> None:
        self.cfg = cfg or TransitionBarrierConfig()
        self.state = TransitionBarrierState.ACTIVE
        self.outgoing_owner = ControlOwner.NONE
        self.incoming_owner = ControlOwner.NONE
        self.started_at = 0.0
        self.last_now = 0.0
        self.stationary_samples = 0
        self.stationary_method = StationaryMethod.NONE
        self.fallback_reason: str | None = None
        self.ramp_v = 0.0
        self.ramp_w = 0.0

    def begin(
        self,
        *,
        outgoing_owner: ControlOwner,
        incoming_owner: ControlOwner,
        now: float,
    ) -> None:
        if self.state != TransitionBarrierState.ACTIVE:
            raise RuntimeError(f"transition already active: {self.state.value}")
        self.state = TransitionBarrierState.ZERO_COMMAND
        self.outgoing_owner = outgoing_owner
        self.incoming_owner = incoming_owner
        self.started_at = float(now)
        self.last_now = float(now)
        self.stationary_samples = 0
        self.stationary_method = StationaryMethod.NONE
        self.fallback_reason = None
        self.ramp_v = 0.0
        self.ramp_w = 0.0

    def step(
        self,
        *,
        now: float,
        motion: MotionSample,
        desired: MotionCommand,
    ) -> TransitionBarrierStep:
        if self.state == TransitionBarrierState.ACTIVE:
            raise RuntimeError("transition barrier is not active")

        now = float(now)
        elapsed = max(0.0, now - self.started_at)
        current_state = self.state

        if current_state == TransitionBarrierState.ZERO_COMMAND:
            self.state = TransitionBarrierState.AWAIT_STATIONARY
            self.last_now = now
            return self._zero_step(current_state, self.outgoing_owner, elapsed)

        if current_state == TransitionBarrierState.AWAIT_STATIONARY:
            measured = motion.source == "measured" and motion.fresh
            stationary = bool(
                measured
                and abs(motion.linear) <= self.cfg.linear_threshold
                and abs(motion.angular) <= self.cfg.angular_threshold
            )
            self.stationary_samples = self.stationary_samples + 1 if stationary else 0
            if self.stationary_samples >= self.cfg.measured_confirm_samples:
                self.stationary_method = StationaryMethod.MEASURED
                self.state = TransitionBarrierState.DISPOSE_OUTGOING
            elif not measured and elapsed >= self.cfg.timed_fallback_sec:
                self.stationary_method = StationaryMethod.TIMED_FALLBACK
                self.fallback_reason = (
                    "motion_feedback_unverified"
                    if motion.source == "measured_unverified"
                    else "motion_feedback_unavailable"
                )
                self.state = TransitionBarrierState.DISPOSE_OUTGOING
            self.last_now = now
            return self._zero_step(current_state, self.outgoing_owner, elapsed)

        if current_state == TransitionBarrierState.DISPOSE_OUTGOING:
            self.state = TransitionBarrierState.INITIALIZE_INCOMING
            self.last_now = now
            return self._zero_step(
                current_state,
                ControlOwner.NONE,
                elapsed,
                dispose_outgoing=True,
            )

        if current_state == TransitionBarrierState.INITIALIZE_INCOMING:
            self.state = TransitionBarrierState.SWITCH_OWNER
            self.last_now = now
            return self._zero_step(
                current_state,
                ControlOwner.NONE,
                elapsed,
                initialize_incoming=True,
            )

        if current_state == TransitionBarrierState.SWITCH_OWNER:
            self.state = TransitionBarrierState.RAMP
            self.last_now = now
            self.ramp_v = 0.0
            self.ramp_w = 0.0
            return self._zero_step(
                current_state,
                self.incoming_owner,
                elapsed,
                owner_switched=True,
            )

        if owner_for_producer(producer_for_owner(self.incoming_owner)) != self.incoming_owner:
            raise AssertionError("incoming owner has no command producer")
        dt = max(0.0, min(0.5, now - self.last_now))
        self.last_now = now
        self.ramp_v = self._slew(
            self.ramp_v,
            desired.v,
            self.cfg.max_v_slew_rate * dt,
        )
        self.ramp_w = self._slew(
            self.ramp_w,
            desired.w,
            self.cfg.max_w_slew_rate * dt,
        )
        complete = bool(
            abs(self.ramp_v - desired.v) <= 1e-9
            and abs(self.ramp_w - desired.w) <= 1e-9
        )
        command = MotionCommand(
            self.ramp_v,
            self.ramp_w,
            f"transition_ramp:{desired.reason}",
            desired.state,
            desired.mode,
        )
        result = TransitionBarrierStep(
            current_state,
            command,
            self.incoming_owner,
            producer_for_owner(self.incoming_owner),
            transition_completed=complete,
            stationary_method=self.stationary_method,
            stationary_samples=self.stationary_samples,
            stationary_elapsed_sec=elapsed,
            fallback_reason=self.fallback_reason,
        )
        if complete:
            self.state = TransitionBarrierState.ACTIVE
        return result

    def _zero_step(
        self,
        state: TransitionBarrierState,
        owner: ControlOwner,
        elapsed: float,
        *,
        dispose_outgoing: bool = False,
        initialize_incoming: bool = False,
        owner_switched: bool = False,
    ) -> TransitionBarrierStep:
        return TransitionBarrierStep(
            state,
            zero_command(f"transition_{state.value}"),
            owner,
            producer_for_owner(owner),
            dispose_outgoing=dispose_outgoing,
            initialize_incoming=initialize_incoming,
            owner_switched=owner_switched,
            stationary_method=self.stationary_method,
            stationary_samples=self.stationary_samples,
            stationary_elapsed_sec=elapsed,
            fallback_reason=self.fallback_reason,
        )

    @staticmethod
    def _slew(current: float, target: float, max_delta: float) -> float:
        delta = max(-max_delta, min(max_delta, target - current))
        return current + delta


class AtomicStageRuntime:
    """Own stage instances and apply one explicit transition contract."""

    def __init__(
        self,
        initial_stage: CourseSession,
        factories: Mapping[CourseSession, StageFactory],
        *,
        barrier_cfg: TransitionBarrierConfig | None = None,
    ) -> None:
        self.current_stage = initial_stage
        self.factories = dict(factories)
        self.component = self._create(initial_stage)
        self.component.initialize(None)
        self.next_stage: CourseSession | None = None
        self.transfer: Any = None
        self.barrier = TransitionBarrier(barrier_cfg)

    def request_transition(
        self,
        next_stage: CourseSession,
        *,
        outgoing_owner: ControlOwner,
        incoming_owner: ControlOwner,
        now: float,
    ) -> None:
        self.next_stage = next_stage
        self.barrier.begin(
            outgoing_owner=outgoing_owner,
            incoming_owner=incoming_owner,
            now=now,
        )

    def resolve(
        self,
        *,
        candidate: MotionCommand,
        producer: CandidateProducer,
        executor_phase: str,
        now: float,
        motion: MotionSample,
    ) -> StageFrameResult:
        if self.barrier.state == TransitionBarrierState.ACTIVE:
            owner = owner_for_producer(producer)
            return StageFrameResult(
                self.current_stage,
                executor_phase,
                producer,
                owner,
                candidate,
                False,
                TransitionBarrierState.ACTIVE,
                None,
            )

        step = self.barrier.step(now=now, motion=motion, desired=candidate)
        if step.dispose_outgoing:
            if self.next_stage is None:
                raise RuntimeError("transition has no incoming stage")
            self.transfer = self.component.dispose(self.next_stage)
            self.component = None
        if step.initialize_incoming:
            if self.next_stage is None:
                raise RuntimeError("transition has no incoming stage")
            incoming = self._create(self.next_stage)
            incoming.initialize(self.transfer)
            self.component = incoming
        if step.owner_switched:
            if self.next_stage is None:
                raise RuntimeError("transition has no incoming stage")
            self.current_stage = self.next_stage
        if step.transition_completed:
            self.next_stage = None
            self.transfer = None

        return StageFrameResult(
            self.current_stage,
            executor_phase,
            step.producer,
            step.owner,
            step.command,
            True,
            step.state,
            self.next_stage,
            step.stationary_method,
            step.stationary_samples,
            step.stationary_elapsed_sec,
            step.fallback_reason,
        )

    def _create(self, stage: CourseSession) -> StageComponent:
        try:
            factory = self.factories[stage]
        except KeyError as exc:
            raise ValueError(f"no stage factory for {stage.value}") from exc
        return factory()
