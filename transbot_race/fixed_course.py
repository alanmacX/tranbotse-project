from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .capture_geometry import (
    CaptureGeometryDebug,
    CaptureGeometryDecision,
    CaptureGeometryObservation,
    CornerGeometryFilter,
    RingEntryGeometryFilter,
    RingExitGeometryFilter,
    analyze_capture_geometry,
)
from .config import RaceConfig, StageGeometryConfig
from .control import ControlOwner, StopCause, TransitionEvent
from .mission import CourseSession, DetectorKind, FixedSessionMission, MissionEvent
from .path_memory import CornerCommandDelay, MotionSample, PathStrategyStatus
from .ring_entry import (
    RingEntryExecutor,
    RingEntryResult,
    RingPhaseEvent,
    RingStageTransfer,
    effective_ring_margin_distance,
    selected_path_fit,
)
from .stage_runtime import (
    AtomicStageRuntime,
    CandidateProducer,
    StageFrameResult,
    TransitionBarrierConfig,
    TransitionBarrierState,
    owner_for_producer,
    zero_command,
)
from .state_machine import MotionCommand, RaceState, RaceStateMachine
from .vision import LineFeatures, TrajectoryFit


@dataclass(frozen=True, slots=True)
class FixedCourseFrameContext:
    frame: np.ndarray
    now: float
    visual_fit: TrajectoryFit
    features: LineFeatures
    motion: MotionSample
    last_command: MotionCommand
    frame_center_x: float
    control_width: float
    capture_geometry_debug: bool = False


@dataclass(frozen=True, slots=True)
class CornerTransfer:
    cruise_fit: TrajectoryFit


@dataclass(frozen=True, slots=True)
class ComponentStep:
    candidate: MotionCommand
    producer: CandidateProducer
    fit: TrajectoryFit
    executor_phase: str
    memory_status: PathStrategyStatus
    detector: DetectorKind
    geometry_observation: CaptureGeometryObservation | None = None
    geometry_decision: CaptureGeometryDecision | None = None
    geometry_debug: CaptureGeometryDebug | None = None
    transition_to: CourseSession | None = None
    transition_event: TransitionEvent = TransitionEvent.NONE
    stop_cause: StopCause | None = None
    ring_result: RingEntryResult | None = None
    route_fit: TrajectoryFit | None = None
    geometry_age_frames: int = 0


@dataclass(frozen=True, slots=True)
class FixedCourseStep:
    stage: CourseSession
    next_stage: CourseSession | None
    detector: DetectorKind
    executor_phase: str
    candidate: MotionCommand
    candidate_producer: CandidateProducer
    control_owner: ControlOwner
    fit: TrajectoryFit
    memory_status: PathStrategyStatus
    transition_event: TransitionEvent
    stop_cause: StopCause | None
    barrier: StageFrameResult
    geometry_observation: CaptureGeometryObservation | None
    geometry_decision: CaptureGeometryDecision | None
    geometry_debug: CaptureGeometryDebug | None
    ring_result: RingEntryResult | None
    route_fit: TrajectoryFit | None
    geometry_age_frames: int = 0

    def __post_init__(self) -> None:
        if owner_for_producer(self.candidate_producer) != self.control_owner:
            raise ValueError("fixed-course producer/owner mismatch")


class _Stage:
    detector = DetectorKind.NONE

    def _geometry_sample(
        self,
        context: FixedCourseFrameContext,
        geometry_cfg: StageGeometryConfig,
        *,
        route_direction: int | None = None,
    ) -> tuple[
        CaptureGeometryObservation,
        CaptureGeometryDebug,
        bool,
        int,
    ]:
        cadence = max(1, int(geometry_cfg.cadence_frames))
        tick = int(getattr(self, "_geometry_tick", 0))
        previous = getattr(self, "_geometry_cached", None)
        fresh = previous is None or tick % cadence == 0
        if fresh:
            observation, debug = analyze_capture_geometry(
                context.frame,
                self.cfg,
                route_direction=route_direction,
                include_debug_images=context.capture_geometry_debug,
                geometry_cfg=geometry_cfg,
            )
            self._geometry_cached = (observation, debug)
            self._geometry_age_frames = 0
        else:
            observation, debug = previous
            self._geometry_age_frames = int(
                getattr(self, "_geometry_age_frames", 0),
            ) + 1
        self._geometry_tick = tick + 1
        return observation, debug, fresh, self._geometry_age_frames

    def initialize(self, transfer: Any) -> None:
        if transfer is not None:
            raise ValueError(f"{type(self).__name__} does not accept stage transfer")

    def dispose(self, next_stage: CourseSession) -> Any:
        return None

    def step(self, context: FixedCourseFrameContext) -> ComponentStep:
        raise NotImplementedError


class _CornerStage(_Stage):
    detector = DetectorKind.CORNER

    def __init__(self, cfg: RaceConfig, mission: FixedSessionMission) -> None:
        self.cfg = cfg
        self.mission = mission
        self.cruise = RaceStateMachine(cfg)
        self.executor = CornerCommandDelay(
            cfg.path_memory,
            handoff_conf_min=cfg.tracker.conf_predict,
            tracker_cfg=cfg.tracker,
        )
        self.geometry_filter = CornerGeometryFilter(
            cfg.path_memory.corner_confirm_frames,
        )
        self.transfer: CornerTransfer | None = None

    def dispose(self, next_stage: CourseSession) -> CornerTransfer:
        if next_stage != CourseSession.RING_ENTRY or self.transfer is None:
            raise RuntimeError("corner stage has no ring-entry transfer")
        self.geometry_filter.reset()
        return self.transfer

    def step(self, context: FixedCourseFrameContext) -> ComponentStep:
        observation = decision = debug = None
        geometry_active = self.executor.state in {"armed", "approach"}
        if self.cfg.path_memory.capture_geometry_enabled and geometry_active:
            observation, debug, fresh_geometry, geometry_age = self._geometry_sample(
                context,
                self.cfg.corner_geometry,
            )
            decision = self.geometry_filter.update(observation) if fresh_geometry else None
        else:
            geometry_age = 0
        accepted = self.mission.gate(
            decision,
            observation,
            context.visual_fit,
        )
        pending = self.executor.will_accept_geometry(accepted)
        owns = self.executor.state not in {"armed", "cooldown"} or pending
        if owns:
            command = MotionCommand(
                0.0,
                0.0,
                "corner_owned",
                self.cruise.state,
                None,
            )
            producer = CandidateProducer.CORNER_EXECUTOR
        else:
            command = self.cruise.step(context.visual_fit, context.now)
            producer = CandidateProducer.CRUISE

        status = PathStrategyStatus(True, "fixed_sessions", self.executor.state)
        state_before = self.executor.state
        delayed = self.executor.step(
            context.visual_fit,
            context.features,
            command.v,
            command.w,
            context.now,
            context.motion.linear,
            context.motion.angular,
            geometry=observation,
            geometry_decision=accepted,
            invert_turn=self.cfg.tracker.invert_turn,
            angular_scale=(
                self.cfg.path_memory.corner_command_yaw_scale
                if context.motion.source == "command_fallback" else 1.0
            ),
        )
        status = replace(delayed.status, mode="fixed_sessions")
        if delayed.v != command.v or delayed.w != command.w:
            command = replace(
                command,
                v=delayed.v,
                w=delayed.w,
                reason=f"corner_{status.reason}",
            )
            producer = CandidateProducer.CORNER_EXECUTOR
        if state_before == "approach" and self.executor.state == "armed":
            self.cruise.reset()
            command = self.cruise.step(context.visual_fit, context.now)
            producer = CandidateProducer.CRUISE

        transition_to = None
        event = status.transition_event
        if event == TransitionEvent.HANDOFF_READY:
            self.transfer = CornerTransfer(context.visual_fit)
            transition_to = CourseSession.RING_ENTRY
            command = zero_command("corner_to_ring_entry")
            producer = CandidateProducer.CORNER_EXECUTOR
        stop_cause = (
            StopCause.EXECUTOR_FAILED
            if self.executor.state in {"failed", "failed_locked"} else None
        )
        return ComponentStep(
            command,
            producer,
            context.visual_fit,
            self.executor.state,
            status,
            self.detector,
            observation,
            decision,
            debug,
            transition_to,
            event,
            stop_cause,
            geometry_age_frames=geometry_age,
        )


class _RingEntryStage(_Stage):
    detector = DetectorKind.RING_ENTRY

    def __init__(self, cfg: RaceConfig, mission: FixedSessionMission) -> None:
        self.cfg = cfg
        self.mission = mission
        self.cruise = RaceStateMachine(cfg)
        self.executor = RingEntryExecutor(
            cfg.mission.ring_entry_direction,
            margin_distance_m=effective_ring_margin_distance(
                cfg.path_memory.roundabout_margin_distance_m,
                cfg.path_memory.roundabout_margin_enabled,
            ),
            entry_left_turn_rad=np.deg2rad(
                cfg.path_memory.roundabout_entry_left_turn_deg
            ),
            align_w=cfg.path_memory.roundabout_align_w,
            align_slow_w=cfg.path_memory.roundabout_align_slow_w,
            align_slowdown_rad=cfg.path_memory.roundabout_align_slowdown_rad,
            arc_v=cfg.path_memory.roundabout_arc_v,
            fixed_radius_m=cfg.path_memory.roundabout_fixed_radius_m,
            chord_distance_scale=(
                cfg.path_memory.roundabout_chord_distance_scale
            ),
            half_arc_yaw_rad=cfg.path_memory.roundabout_half_arc_yaw_rad,
            half_arc_timeout_sec=cfg.path_memory.roundabout_half_arc_timeout_sec,
            exit_reacquire_frames=(
                cfg.path_memory.roundabout_exit_reacquire_frames
            ),
            exit_reacquire_extra_rad=(
                cfg.path_memory.roundabout_exit_reacquire_extra_rad
            ),
            tracker_cfg=cfg.tracker,
        )
        self.geometry_filter = RingEntryGeometryFilter(
            cfg.path_memory.corner_confirm_frames,
        )
        self.seed_fit: TrajectoryFit | None = None
        self.seeded = False

    def initialize(self, transfer: Any) -> None:
        if transfer is None:
            return
        if not isinstance(transfer, CornerTransfer):
            raise ValueError("ring entry requires corner cruise transfer")
        self.seed_fit = transfer.cruise_fit

    def dispose(self, next_stage: CourseSession) -> RingStageTransfer:
        if next_stage != CourseSession.RING_EXIT:
            raise RuntimeError("ring entry may only transition to ring exit")
        transfer = self.executor.export_stage_transfer()
        self.geometry_filter.reset()
        self.executor.reset()
        return transfer

    def step(self, context: FixedCourseFrameContext) -> ComponentStep:
        if self.executor.state == "waiting":
            if not self.seeded and self.seed_fit is not None:
                cruise_command = self.cruise.reacquire_from(self.seed_fit, context.now)
                self.seeded = True
            else:
                cruise_command = self.cruise.step(context.visual_fit, context.now)
        else:
            cruise_command = zero_command("ring_fixed_route_owner")

        observation = None
        debug = None
        fresh_geometry = False
        geometry_age = 0
        decision = None
        accepted = None
        route_fit = None
        # Vision is only a boundary concern: trigger fixed execution before it,
        # then reacquire the straight exit after it. The fixed body runs with no
        # geometry detector, path fitting, or competing cruise controller.
        if self.executor.state in {"waiting", "exit_reacquire"}:
            observation, debug, fresh_geometry, geometry_age = self._geometry_sample(
                context,
                self.cfg.ring_entry_geometry,
                route_direction=self.cfg.mission.ring_entry_direction,
            )
            decision = self.geometry_filter.update(observation) if fresh_geometry else None
            route_fit = selected_path_fit(
                debug,
                observation,
                frame_center_x=context.frame_center_x,
                control_width=context.control_width,
            )
            if self.executor.state == "waiting":
                accepted = self.mission.gate(
                    decision,
                    observation,
                    context.visual_fit,
                    entry_takeover_ready=self.cruise.can_take_ring_entry(context.visual_fit),
                )
        result = self.executor.step(
            observation,
            route_fit,
            now=context.now,
            linear=context.motion.linear,
            angular=context.motion.angular,
            motion_source=context.motion.source,
            accepted_entry=accepted is not None,
            cruise_fit=context.visual_fit,
            incoming_v=context.last_command.v,
            fresh_geometry=fresh_geometry,
        )
        command, producer = self._command(context, result, cruise_command)
        status = PathStrategyStatus(
            True,
            "fixed_sessions",
            result.reason,
            self.executor.margin_remaining_m,
            self.executor.direction,
            self.executor.clear_frames,
        )
        transition_to = (
            CourseSession.RING_EXIT
            if result.phase_event == RingPhaseEvent.ENTRY_ESTABLISHED else None
        )
        event = (
            TransitionEvent.HANDOFF_READY
            if transition_to is not None else
            TransitionEvent.ROUTE_LOST
            if result.phase_event == RingPhaseEvent.ROUTE_LOST else
            TransitionEvent.NONE
        )
        stop_cause = (
            StopCause.ROUTE_LOST
            if producer == CandidateProducer.RING_EXECUTOR
            and result.phase_event == RingPhaseEvent.ROUTE_LOST
            else None
        )
        return ComponentStep(
            command,
            producer,
            result.fit or context.visual_fit,
            self.executor.state,
            status,
            self.detector,
            observation,
            decision,
            debug,
            transition_to,
            event,
            stop_cause,
            result,
            route_fit,
            geometry_age,
        )

    def _command(
        self,
        context: FixedCourseFrameContext,
        result: RingEntryResult,
        cruise_command: MotionCommand,
    ) -> tuple[MotionCommand, CandidateProducer]:
        if self.executor.state == "waiting":
            return cruise_command, CandidateProducer.CRUISE
        if self.executor.state == "margin":
            return MotionCommand(
                self.executor.margin_v,
                0.0,
                result.reason,
                RaceState.TRACK,
                None,
            ), CandidateProducer.RING_EXECUTOR
        if self.executor.state in {"entry_left_align", "entry_left_stop"}:
            v, w = self.executor.entry_left_command(
                invert_turn=self.cfg.tracker.invert_turn,
            )
            return MotionCommand(
                v, w, result.reason, RaceState.TRACK, None,
            ), CandidateProducer.RING_EXECUTOR
        if self.executor.state in {
            "leg1_model", "leg2_model", "leg3_model", "leg4_exit_bridge",
        }:
            v, w = self.executor.model_leg_command(
                invert_turn=self.cfg.tracker.invert_turn,
            )
            return MotionCommand(
                v, w, result.reason, RaceState.TRACK, None,
            ), CandidateProducer.RING_EXECUTOR
        if self.executor.state in {"exit_line_align", "exit_line_stop"}:
            v, w = self.executor.exit_line_command(
                invert_turn=self.cfg.tracker.invert_turn,
            )
            return MotionCommand(
                v, w, result.reason, RaceState.TRACK, None,
            ), CandidateProducer.RING_EXECUTOR
        if self.executor.state == "exit_reacquire":
            return zero_command(result.reason), CandidateProducer.RING_EXECUTOR
        if self.executor.state == "exit_ready":
            return zero_command(result.reason), CandidateProducer.RING_EXECUTOR
        if result.fit is None:
            return zero_command(result.reason), CandidateProducer.RING_EXECUTOR
        v, w = self.executor.control(
            result.fit,
            now=context.now,
            v_max=self.cfg.tracker.v_max,
            k_pursuit=self.cfg.tracker.k_pursuit,
            k_theta=self.cfg.tracker.k_theta,
            max_w=self.cfg.path_memory.roundabout_replay_max_w,
            approach_max_w=self.cfg.path_memory.corner_approach_max_w,
            invert_turn=self.cfg.tracker.invert_turn,
        )
        return MotionCommand(v, w, result.reason, RaceState.TRACK, None), CandidateProducer.RING_EXECUTOR


class _RingExitStage(_Stage):
    detector = DetectorKind.RING_EXIT

    def __init__(self, cfg: RaceConfig, mission: FixedSessionMission) -> None:
        self.cfg = cfg
        self.mission = mission
        self.executor = RingEntryExecutor(
            cfg.mission.ring_exit_direction,
            tracker_cfg=cfg.tracker,
        )
        self.geometry_filter = RingExitGeometryFilter(
            cfg.path_memory.corner_confirm_frames,
        )

    def initialize(self, transfer: Any) -> None:
        if not isinstance(transfer, RingStageTransfer):
            raise ValueError("ring exit requires ring-route transfer")
        self.executor.initialize_exit(transfer)

    def dispose(self, next_stage: CourseSession) -> None:
        if next_stage != CourseSession.FINISHED:
            raise RuntimeError("ring exit may only transition to finished")
        self.geometry_filter.reset()
        self.executor.reset()
        return None

    def step(self, context: FixedCourseFrameContext) -> ComponentStep:
        observation, debug, fresh_geometry, geometry_age = self._geometry_sample(
            context,
            self.cfg.ring_exit_geometry,
            route_direction=self.cfg.mission.ring_exit_direction,
        )
        route_fit = selected_path_fit(
            debug,
            observation,
            frame_center_x=context.frame_center_x,
            control_width=context.control_width,
        )
        armed = (
            self.executor.state == "exiting"
            or self.executor.inside_travelled_m >= self.executor.inside_arm_distance_m
        )
        decision = self.geometry_filter.update(observation) if armed and fresh_geometry else None
        accepted = self.mission.gate(
            decision,
            observation,
            context.visual_fit,
            exit_selection_armed=armed,
        )
        result = self.executor.step(
            observation,
            route_fit,
            now=context.now,
            linear=context.motion.linear,
            accepted_entry=accepted is not None,
            cruise_fit=context.visual_fit,
        )
        if result.fit is None:
            command = zero_command(result.reason)
        else:
            v, w = self.executor.control(
                result.fit,
                now=context.now,
                v_max=self.cfg.tracker.v_max,
                k_pursuit=self.cfg.tracker.k_pursuit,
                k_theta=self.cfg.tracker.k_theta,
                max_w=self.cfg.path_memory.roundabout_replay_max_w,
                approach_max_w=self.cfg.path_memory.corner_approach_max_w,
                invert_turn=self.cfg.tracker.invert_turn,
            )
            command = MotionCommand(v, w, result.reason, RaceState.TRACK, None)
        transition_to = (
            CourseSession.FINISHED
            if result.phase_event == RingPhaseEvent.EXECUTOR_COMPLETED else None
        )
        event = (
            TransitionEvent.EXECUTOR_COMPLETED
            if transition_to is not None else
            TransitionEvent.ROUTE_LOST
            if result.phase_event == RingPhaseEvent.ROUTE_LOST else
            TransitionEvent.NONE
        )
        stop_cause = (
            StopCause.ROUTE_LOST
            if result.fit is None and self.executor.state in {"inside", "exiting"}
            else None
        )
        status = PathStrategyStatus(
            True,
            "fixed_sessions",
            result.reason,
            0.0,
            self.executor.direction,
            self.executor.clear_frames,
        )
        return ComponentStep(
            command,
            CandidateProducer.RING_EXECUTOR,
            result.fit or context.visual_fit,
            self.executor.state,
            status,
            self.detector,
            observation,
            decision,
            debug,
            transition_to,
            event,
            stop_cause,
            result,
            route_fit,
            geometry_age,
        )


class _FinishedStage(_Stage):
    def step(self, context: FixedCourseFrameContext) -> ComponentStep:
        command = zero_command("mission_finished")
        return ComponentStep(
            command,
            CandidateProducer.NONE,
            context.visual_fit,
            "unimplemented",
            PathStrategyStatus(True, "fixed_sessions", command.reason),
            DetectorKind.NONE,
            stop_cause=StopCause.MISSION_FINISHED,
        )


class FixedCourseController:
    def __init__(self, cfg: RaceConfig) -> None:
        self.cfg = cfg
        self.mission = FixedSessionMission(cfg.mission)
        factories = {
            CourseSession.CORNER: lambda: _CornerStage(cfg, self.mission),
            CourseSession.RING_ENTRY: lambda: _RingEntryStage(cfg, self.mission),
            CourseSession.RING_EXIT: lambda: _RingExitStage(cfg, self.mission),
            CourseSession.FINISHED: _FinishedStage,
        }
        initial = self.mission.session
        if initial not in factories:
            raise ValueError(f"unsupported fixed-course initial stage: {initial.value}")
        self.runtime = AtomicStageRuntime(
            initial,
            factories,
            barrier_cfg=TransitionBarrierConfig(
                max_w_slew_rate=cfg.tracker.max_w_slew_rate,
            ),
        )
        self.last_component_step: ComponentStep | None = None

    @property
    def component(self) -> _Stage | None:
        return self.runtime.component

    def clear_route_loss(self) -> bool:
        component = self.runtime.component
        executor = getattr(component, "executor", None)
        return bool(
            self.runtime.barrier.state == TransitionBarrierState.ACTIVE
            and isinstance(executor, RingEntryExecutor)
            and executor.clear_route_loss()
        )

    def step(self, context: FixedCourseFrameContext) -> FixedCourseStep:
        component_step = self._step_component(context)
        if (
            self.runtime.barrier.state == TransitionBarrierState.ACTIVE
            and component_step.transition_to is not None
        ):
            self.runtime.request_transition(
                component_step.transition_to,
                outgoing_owner=owner_for_producer(component_step.producer),
                incoming_owner=self._incoming_owner(component_step.transition_to),
                now=context.now,
            )

        stage_before = self.runtime.current_stage
        barrier_result = self.runtime.resolve(
            candidate=component_step.candidate,
            producer=component_step.producer,
            executor_phase=component_step.executor_phase,
            now=context.now,
            motion=context.motion,
        )
        if self.runtime.current_stage != stage_before:
            committed = self.mission.transition(MissionEvent.PHASE_COMPLETED)
            if committed != self.runtime.current_stage:
                raise RuntimeError(
                    f"mission/runtime stage mismatch: {committed.value} != "
                    f"{self.runtime.current_stage.value}"
                )

        stop_cause = component_step.stop_cause
        if self.runtime.current_stage == CourseSession.FINISHED:
            stop_cause = StopCause.MISSION_FINISHED
        self.last_component_step = component_step
        return FixedCourseStep(
            self.runtime.current_stage,
            self.runtime.next_stage,
            component_step.detector,
            component_step.executor_phase,
            barrier_result.candidate,
            barrier_result.candidate_producer,
            barrier_result.control_owner,
            component_step.fit,
            component_step.memory_status,
            component_step.transition_event,
            stop_cause,
            barrier_result,
            component_step.geometry_observation,
            component_step.geometry_decision,
            component_step.geometry_debug,
            component_step.ring_result,
            component_step.route_fit,
            component_step.geometry_age_frames,
        )

    def _step_component(self, context: FixedCourseFrameContext) -> ComponentStep:
        barrier_state = self.runtime.barrier.state
        component = self.runtime.component
        if component is None or barrier_state not in {
            TransitionBarrierState.ACTIVE,
            TransitionBarrierState.RAMP,
        }:
            phase = barrier_state.value
            return ComponentStep(
                zero_command(f"transition_{phase}"),
                CandidateProducer.NONE,
                context.visual_fit,
                phase,
                PathStrategyStatus(True, "fixed_sessions", f"transition_{phase}"),
                DetectorKind.NONE,
            )
        return component.step(context)

    def _incoming_owner(self, stage: CourseSession) -> ControlOwner:
        if stage == CourseSession.RING_ENTRY:
            return ControlOwner.CRUISE
        if stage == CourseSession.RING_EXIT:
            return ControlOwner.RING_EXECUTOR
        return ControlOwner.NONE
