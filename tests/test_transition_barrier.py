import pytest

from transbot_race.control import ControlOwner
from transbot_race.mission import CourseSession
from transbot_race.path_memory import MotionSample
from transbot_race.stage_runtime import (
    AtomicStageRuntime,
    CandidateProducer,
    StageFrameResult,
    StationaryMethod,
    TransitionBarrier,
    TransitionBarrierConfig,
    TransitionBarrierState,
)
from transbot_race.state_machine import MotionCommand, RaceState


def command(v=0.06, w=-0.20, reason="target"):
    return MotionCommand(v, w, reason, RaceState.TRACK)


class Component:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.events.append(f"create:{name}")

    def initialize(self, transfer):
        self.events.append(f"initialize:{self.name}:{transfer}")

    def dispose(self, next_stage):
        self.events.append(f"dispose:{self.name}:{next_stage.value}")
        return f"{self.name}_transfer"


def measured(v=0.0, w=0.0):
    return MotionSample(v, w, "measured")


def fallback(v=0.0, w=0.0):
    return MotionSample(v, w, "command_fallback")


def test_frame_result_rejects_producer_owner_mismatch():
    with pytest.raises(ValueError, match="producer/owner mismatch"):
        StageFrameResult(
            CourseSession.CORNER,
            "turning",
            CandidateProducer.CORNER_EXECUTOR,
            ControlOwner.CRUISE,
            command(),
            False,
            TransitionBarrierState.ACTIVE,
            None,
        )


def test_barrier_requires_consecutive_measured_stationary_samples():
    barrier = TransitionBarrier(TransitionBarrierConfig(
        measured_confirm_samples=2,
        timed_fallback_sec=2.0,
    ))
    barrier.begin(
        outgoing_owner=ControlOwner.CORNER_EXECUTOR,
        incoming_owner=ControlOwner.RING_EXECUTOR,
        now=0.0,
    )

    first = barrier.step(now=0.0, motion=measured(0.02, 0.0), desired=command())
    assert first.state == TransitionBarrierState.ZERO_COMMAND
    assert first.command.v == first.command.w == 0.0
    assert first.owner == ControlOwner.CORNER_EXECUTOR

    moving = barrier.step(now=0.1, motion=measured(0.02, 0.0), desired=command())
    still_1 = barrier.step(now=0.2, motion=measured(), desired=command())
    still_2 = barrier.step(now=0.3, motion=measured(), desired=command())
    assert moving.stationary_samples == 0
    assert still_1.stationary_samples == 1
    assert still_2.stationary_samples == 2
    assert barrier.state == TransitionBarrierState.DISPOSE_OUTGOING

    dispose = barrier.step(now=0.4, motion=measured(), desired=command())
    initialize = barrier.step(now=0.5, motion=measured(), desired=command())
    switch = barrier.step(now=0.6, motion=measured(), desired=command())
    assert dispose.dispose_outgoing and dispose.owner == ControlOwner.NONE
    assert initialize.initialize_incoming and initialize.owner == ControlOwner.NONE
    assert switch.owner_switched and switch.owner == ControlOwner.RING_EXECUTOR
    assert switch.command.v == switch.command.w == 0.0
    assert switch.stationary_method == StationaryMethod.MEASURED


def test_command_fallback_uses_timed_stationary_method_not_measured():
    barrier = TransitionBarrier(TransitionBarrierConfig(
        measured_confirm_samples=2,
        timed_fallback_sec=0.5,
    ))
    barrier.begin(
        outgoing_owner=ControlOwner.CORNER_EXECUTOR,
        incoming_owner=ControlOwner.RING_EXECUTOR,
        now=1.0,
    )
    barrier.step(now=1.0, motion=fallback(), desired=command())
    waiting = barrier.step(now=1.2, motion=fallback(), desired=command())
    elapsed = barrier.step(now=1.5, motion=fallback(), desired=command())
    assert waiting.stationary_method == StationaryMethod.NONE
    assert elapsed.stationary_method == StationaryMethod.TIMED_FALLBACK
    assert elapsed.fallback_reason == "motion_feedback_unavailable"
    assert barrier.state == TransitionBarrierState.DISPOSE_OUTGOING


def test_unverified_measurement_uses_timed_fallback():
    barrier = TransitionBarrier(TransitionBarrierConfig(
        measured_confirm_samples=2,
        timed_fallback_sec=0.5,
    ))
    barrier.begin(
        outgoing_owner=ControlOwner.CORNER_EXECUTOR,
        incoming_owner=ControlOwner.RING_EXECUTOR,
        now=1.0,
    )
    barrier.step(
        now=1.0,
        motion=MotionSample(0.0, 0.0, "measured_unverified", False),
        desired=command(),
    )
    elapsed = barrier.step(
        now=1.5,
        motion=MotionSample(0.0, 0.0, "measured_unverified", False),
        desired=command(),
    )

    assert elapsed.stationary_method == StationaryMethod.TIMED_FALLBACK
    assert elapsed.fallback_reason == "motion_feedback_unverified"
    assert barrier.state == TransitionBarrierState.DISPOSE_OUTGOING


def test_measured_motion_never_uses_timed_fallback():
    barrier = TransitionBarrier(TransitionBarrierConfig(
        measured_confirm_samples=2,
        timed_fallback_sec=0.5,
    ))
    barrier.begin(
        outgoing_owner=ControlOwner.CORNER_EXECUTOR,
        incoming_owner=ControlOwner.RING_EXECUTOR,
        now=1.0,
    )
    barrier.step(now=1.0, motion=measured(0.03), desired=command())
    elapsed = barrier.step(now=2.0, motion=measured(0.03), desired=command())

    assert elapsed.state == TransitionBarrierState.AWAIT_STATIONARY
    assert elapsed.stationary_method == StationaryMethod.NONE
    assert elapsed.fallback_reason is None
    assert barrier.state == TransitionBarrierState.AWAIT_STATIONARY


def test_incoming_command_starts_at_zero_and_slew_limits_each_axis():
    barrier = TransitionBarrier(TransitionBarrierConfig(
        measured_confirm_samples=1,
        timed_fallback_sec=2.0,
        max_v_slew_rate=0.10,
        max_w_slew_rate=0.40,
    ))
    barrier.begin(
        outgoing_owner=ControlOwner.CORNER_EXECUTOR,
        incoming_owner=ControlOwner.RING_EXECUTOR,
        now=0.0,
    )
    barrier.step(now=0.0, motion=measured(), desired=command())
    barrier.step(now=0.1, motion=measured(), desired=command())
    barrier.step(now=0.2, motion=measured(), desired=command())
    barrier.step(now=0.3, motion=measured(), desired=command())
    switched = barrier.step(now=0.4, motion=measured(), desired=command())
    assert switched.owner_switched
    assert switched.command.v == switched.command.w == 0.0

    ramp_1 = barrier.step(now=0.5, motion=measured(), desired=command())
    ramp_2 = barrier.step(now=0.6, motion=measured(), desired=command())
    assert ramp_1.command.v == pytest.approx(0.01)
    assert ramp_1.command.w == pytest.approx(-0.04)
    assert ramp_2.command.v == pytest.approx(0.02)
    assert ramp_2.command.w == pytest.approx(-0.08)
    assert ramp_1.owner == ControlOwner.RING_EXECUTOR
    assert ramp_1.producer == CandidateProducer.RING_EXECUTOR


def test_atomic_runtime_disposes_before_creating_and_switching_stage():
    events = []
    runtime = AtomicStageRuntime(
        CourseSession.CORNER,
        {
            CourseSession.CORNER: lambda: Component("corner", events),
            CourseSession.RING_ENTRY: lambda: Component("ring_entry", events),
        },
        barrier_cfg=TransitionBarrierConfig(measured_confirm_samples=1),
    )
    runtime.request_transition(
        CourseSession.RING_ENTRY,
        outgoing_owner=ControlOwner.CORNER_EXECUTOR,
        incoming_owner=ControlOwner.RING_EXECUTOR,
        now=0.0,
    )

    runtime.resolve(
        candidate=command(),
        producer=CandidateProducer.RING_EXECUTOR,
        executor_phase="handoff",
        now=0.0,
        motion=measured(),
    )
    runtime.resolve(
        candidate=command(),
        producer=CandidateProducer.RING_EXECUTOR,
        executor_phase="handoff",
        now=0.1,
        motion=measured(),
    )
    runtime.resolve(
        candidate=command(),
        producer=CandidateProducer.RING_EXECUTOR,
        executor_phase="handoff",
        now=0.2,
        motion=measured(),
    )
    runtime.resolve(
        candidate=command(),
        producer=CandidateProducer.RING_EXECUTOR,
        executor_phase="handoff",
        now=0.3,
        motion=measured(),
    )
    switched = runtime.resolve(
        candidate=command(),
        producer=CandidateProducer.RING_EXECUTOR,
        executor_phase="waiting",
        now=0.4,
        motion=measured(),
    )

    assert events == [
        "create:corner",
        "initialize:corner:None",
        "dispose:corner:ring_entry",
        "create:ring_entry",
        "initialize:ring_entry:corner_transfer",
    ]
    assert runtime.current_stage == CourseSession.RING_ENTRY
    assert switched.control_owner == ControlOwner.RING_EXECUTOR
    assert switched.candidate.v == switched.candidate.w == 0.0
