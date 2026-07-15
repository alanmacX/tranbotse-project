import numpy as np

from transbot_race.config import RaceConfig
from transbot_race.control import ControlOwner, StopCause
from transbot_race.fixed_course import (
    ComponentStep,
    FixedCourseController,
    FixedCourseFrameContext,
    _RingEntryStage,
)
from transbot_race.mission import CourseSession, DetectorKind, FixedSessionMission
from transbot_race.path_memory import MotionSample, PathStrategyStatus
from transbot_race.ring_entry import RingEntryResult, RingStageTransfer
from transbot_race.stage_runtime import CandidateProducer, TransitionBarrierState
from transbot_race.state_machine import MotionCommand, RaceState
from transbot_race.vision import LineFeatures, TrajectoryFit


def context(now=0.0, motion=None):
    fit = TrajectoryFit(
        found=True,
        e0=0.0,
        e_look=0.0,
        theta=0.0,
        conf=0.95,
        n_bands=6,
        disconnected=False,
    )
    return FixedCourseFrameContext(
        np.full((480, 640, 3), 230, dtype=np.uint8),
        now,
        fit,
        LineFeatures(found=True),
        motion or MotionSample(0.0, 0.0, "measured"),
        MotionCommand(0.04, 0.0, "previous", RaceState.TRACK),
        365.0,
        290.0,
    )


def component_step(transition_to=None, producer=CandidateProducer.RING_EXECUTOR):
    return ComponentStep(
        MotionCommand(0.04, 0.12, "stage", RaceState.TRACK),
        producer,
        context().visual_fit,
        "ready",
        PathStrategyStatus(True, "fixed_sessions", "ready"),
        DetectorKind.RING_ENTRY,
        transition_to=transition_to,
    )


def test_controller_constructs_only_initial_stage_component():
    cfg = RaceConfig()
    controller = FixedCourseController(cfg)

    assert controller.runtime.current_stage == CourseSession.CORNER
    assert type(controller.component).__name__ == "_CornerStage"
    assert controller.component.detector == DetectorKind.CORNER


def test_ring_entry_to_exit_same_owner_uses_full_barrier(monkeypatch):
    cfg = RaceConfig()
    cfg.mission.initial_session = "ring_entry"
    controller = FixedCourseController.__new__(FixedCourseController)
    controller.cfg = cfg
    from transbot_race.mission import FixedSessionMission
    from transbot_race.stage_runtime import AtomicStageRuntime, TransitionBarrierConfig

    controller.mission = FixedSessionMission(cfg.mission)
    events = []

    class Entry:
        detector = DetectorKind.RING_ENTRY

        def initialize(self, transfer):
            events.append(("initialize_entry", transfer))

        def dispose(self, next_stage):
            events.append(("dispose_entry", next_stage))
            return RingStageTransfer(1, None, None, 0, context().visual_fit, 0, 0.0)

        def step(self, frame):
            return component_step(CourseSession.RING_EXIT)

    class Exit:
        detector = DetectorKind.RING_EXIT

        def initialize(self, transfer):
            events.append(("initialize_exit", transfer))

        def dispose(self, next_stage):
            return None

        def step(self, frame):
            return component_step(None)

    controller.runtime = AtomicStageRuntime(
        CourseSession.RING_ENTRY,
        {
            CourseSession.RING_ENTRY: Entry,
            CourseSession.RING_EXIT: Exit,
        },
        barrier_cfg=TransitionBarrierConfig(
            measured_confirm_samples=1,
            max_v_slew_rate=1.0,
            max_w_slew_rate=1.0,
        ),
    )
    controller.last_component_step = None

    states = []
    stages = []
    for index in range(6):
        result = controller.step(context(index * 0.1))
        states.append(result.barrier.barrier_state)
        stages.append((result.stage, controller.mission.session))
        assert result.candidate_producer.value == result.control_owner.value

    assert states[:5] == [
        TransitionBarrierState.ZERO_COMMAND,
        TransitionBarrierState.AWAIT_STATIONARY,
        TransitionBarrierState.DISPOSE_OUTGOING,
        TransitionBarrierState.INITIALIZE_INCOMING,
        TransitionBarrierState.SWITCH_OWNER,
    ]
    assert events[0] == ("initialize_entry", None)
    assert events[1] == ("dispose_entry", CourseSession.RING_EXIT)
    assert events[2][0] == "initialize_exit"
    assert stages[:4] == [
        (CourseSession.RING_ENTRY, CourseSession.RING_ENTRY),
    ] * 4
    assert stages[4] == (CourseSession.RING_EXIT, CourseSession.RING_EXIT)
    assert result.control_owner == ControlOwner.RING_EXECUTOR


def test_route_loss_clear_only_reaches_active_ring_executor():
    cfg = RaceConfig()
    controller = FixedCourseController(cfg)
    assert controller.clear_route_loss() is False

    cfg.mission.initial_session = "ring_entry"
    controller = FixedCourseController(cfg)
    executor = controller.component.executor
    executor.state = "aligning"
    executor.last_fit = context().visual_fit
    executor.missing_frames = 6

    assert controller.clear_route_loss() is True
    assert executor.missing_frames == 0


def test_ring_rotate_search_is_owned_in_place_rotation_only():
    cfg = RaceConfig()
    stage = _RingEntryStage(cfg, FixedSessionMission(cfg.mission))
    stage.executor.state = "rotate_search"

    command, producer = stage._command(
        context(),
        RingEntryResult(None, "ring_entry_rotating_search"),
        MotionCommand(0.06, 0.15, "cruise", RaceState.TRACK),
    )

    assert producer == CandidateProducer.RING_EXECUTOR
    assert command.v == 0.0
    assert command.w == -cfg.path_memory.roundabout_entry_search_w
    assert command.reason == "ring_entry_rotating_search"


def test_finished_stage_is_stationary_and_terminal():
    cfg = RaceConfig()
    cfg.mission.initial_session = "finished"
    controller = FixedCourseController(cfg)

    result = controller.step(context())

    assert result.stage == CourseSession.FINISHED
    assert result.detector == DetectorKind.NONE
    assert result.candidate_producer == CandidateProducer.NONE
    assert result.control_owner == ControlOwner.NONE
    assert result.candidate.v == result.candidate.w == 0.0
    assert result.stop_cause == StopCause.MISSION_FINISHED
