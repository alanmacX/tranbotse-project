from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

from .capture_geometry import CaptureGeometryDecision, CaptureGeometryObservation
from .config import MissionConfig
from .vision import TrajectoryFit


class CourseSession(str, Enum):
    OBSTACLE = "obstacle"
    CORNER = "corner"
    RING_ENTRY = "ring_entry"
    RING_EXIT = "ring_exit"
    FORK = "fork"
    RETURN = "return"
    FINISHED = "finished"
    STOPPED = "stopped"


class MissionEvent(str, Enum):
    PHASE_COMPLETED = "phase_completed"
    MISSION_COMPLETED = "mission_completed"
    MISSION_RESET = "mission_reset"


class DetectorKind(str, Enum):
    NONE = "none"
    CORNER = "corner"
    RING_ENTRY = "ring_entry"


class ExecutorKind(str, Enum):
    NONE = "none"
    CRUISE = "cruise"
    CORNER = "corner"
    RING_ENTRY = "ring_entry"


@dataclass(frozen=True, slots=True)
class SessionSpec:
    detector: DetectorKind
    executor: ExecutorKind


class FixedSessionMission:
    """Linear course progress and strict per-session detector gating."""

    ORDER = (
        CourseSession.OBSTACLE,
        CourseSession.CORNER,
        CourseSession.RING_ENTRY,
        CourseSession.RING_EXIT,
        CourseSession.FORK,
        CourseSession.RETURN,
        CourseSession.FINISHED,
    )
    SESSION_MAP = {
        CourseSession.OBSTACLE: SessionSpec(DetectorKind.NONE, ExecutorKind.CRUISE),
        CourseSession.CORNER: SessionSpec(DetectorKind.CORNER, ExecutorKind.CORNER),
        CourseSession.RING_ENTRY: SessionSpec(DetectorKind.RING_ENTRY, ExecutorKind.RING_ENTRY),
        CourseSession.RING_EXIT: SessionSpec(DetectorKind.RING_ENTRY, ExecutorKind.RING_ENTRY),
        CourseSession.FORK: SessionSpec(DetectorKind.NONE, ExecutorKind.CRUISE),
        CourseSession.RETURN: SessionSpec(DetectorKind.NONE, ExecutorKind.CRUISE),
        CourseSession.FINISHED: SessionSpec(DetectorKind.NONE, ExecutorKind.NONE),
        CourseSession.STOPPED: SessionSpec(DetectorKind.NONE, ExecutorKind.NONE),
    }

    def __init__(self, cfg: MissionConfig) -> None:
        self.cfg = cfg
        try:
            self.session = CourseSession(cfg.initial_session)
        except ValueError as exc:
            raise ValueError(f"unsupported initial session: {cfg.initial_session}") from exc
        self.last_gate_reason = "not_evaluated"
        self.ring_entry_alignment_frames = 0
        self.last_transition_event: MissionEvent | None = None
        self._initial_session = self.session

    @property
    def detector_enabled(self) -> bool:
        return self.spec.detector != DetectorKind.NONE

    @property
    def spec(self) -> SessionSpec:
        return self.SESSION_MAP[self.session]

    @property
    def detector_kind(self) -> DetectorKind:
        return self.spec.detector

    @property
    def executor_kind(self) -> ExecutorKind:
        return self.spec.executor

    def gate(
        self,
        decision: CaptureGeometryDecision | None,
        observation: CaptureGeometryObservation | None = None,
        fit: TrajectoryFit | None = None,
        incoming_ready: bool = True,
    ) -> CaptureGeometryDecision | None:
        """Accept only the topology signature belonging to the current session.

        The session map selects a dedicated detector before this gate runs.
        This gate validates that detector's physical signature; it never
        reclassifies a generic event into another session.
        """
        if self.session == CourseSession.RING_ENTRY:
            self.ring_entry_alignment_frames = (
                self.ring_entry_alignment_frames + 1
                if incoming_ready else 0
            )
        if decision is None or decision.kind not in {"turn", "corner", "ring_entry"}:
            self.last_gate_reason = "no_turn_decision"
            return None
        if self.session == CourseSession.CORNER:
            if decision.kind != "corner":
                self.last_gate_reason = "corner_reject_wrong_detector"
                return None
            if decision.is_fork:
                self.last_gate_reason = "corner_reject_fork"
                return None
            # The same physical bend changes from ``corner`` to ``curve`` as
            # its rounded outer edge fills more of the image.  Shape labels
            # are telemetry, not a stable mission boundary.
            if observation is not None and observation.kind not in {"corner", "curve"}:
                self.last_gate_reason = f"corner_reject_kind_{observation.kind}"
                return None
            # Mirror the pre-existing CornerCommandDelay ownership check so
            # telemetry cannot claim acceptance for geometry the executor
            # itself will reject.
            if abs(decision.incoming_e) > 0.65 or abs(decision.incoming_theta) > 0.45:
                self.last_gate_reason = "corner_reject_incoming_stem"
                return None
            # Keep the proven corner path intact: the dedicated filter confirms
            # the same rounded bend observations as the former generic filter,
            # and CornerCommandDelay performs the original incoming-stem and
            # motor-prerequisite checks. Session order only prevents a fork or
            # ring detector from reaching this executor.
            self.last_gate_reason = "corner_signature_accepted"
            return decision
        if self.session == CourseSession.RING_ENTRY:
            if decision.kind != "ring_entry":
                self.last_gate_reason = "ring_entry_reject_wrong_detector"
                return None
            if observation is not None and observation.kind not in {"curve", "circle"}:
                self.last_gate_reason = f"ring_entry_reject_kind_{observation.kind}"
                return None
            if decision.vertex_y_frac is None or not 0.35 <= decision.vertex_y_frac <= 0.76:
                self.last_gate_reason = "ring_entry_reject_vertex"
                return None
            if self.ring_entry_alignment_frames < 3:
                self.last_gate_reason = "ring_entry_reject_incoming_alignment"
                return None
            if fit is not None and (not fit.found or fit.n_bands < 3 or fit.conf < 0.30):
                self.last_gate_reason = "ring_entry_reject_track_quality"
                return None
            if decision.is_fork:
                self.last_gate_reason = "ring_entry_fork_accepted"
                return replace(decision, direction=self.cfg.ring_entry_direction)
            # At a tangential entrance the other side of the ring may be out
            # of frame, leaving one unambiguous two-endpoint arc. Fixed course
            # order makes this a valid entry signature; do not invent a fork.
            if observation is None or observation.endpoints != 2:
                self.last_gate_reason = "ring_entry_reject_nonfork_topology"
                return None
            if not math.radians(25.0) <= abs(decision.angle_rad) <= math.radians(80.0):
                self.last_gate_reason = "ring_entry_reject_single_path_angle"
                return None
            self.last_gate_reason = "ring_entry_single_path_accepted"
            return decision
        if self.session == CourseSession.RING_EXIT:
            if decision.kind != "ring_entry":
                self.last_gate_reason = "ring_exit_reject_wrong_detector"
                return None
            if observation is None or not observation.is_fork:
                self.last_gate_reason = "ring_exit_waiting_branch"
                return None
            if decision.vertex_y_frac is None or not 0.30 <= decision.vertex_y_frac <= 0.82:
                self.last_gate_reason = "ring_exit_reject_vertex"
                return None
            self.last_gate_reason = "ring_exit_branch_accepted"
            return replace(decision, direction=self.cfg.ring_exit_direction)
        self.last_gate_reason = f"detector_disabled_{self.session.value}"
        return None

    def transition(self, event: MissionEvent) -> CourseSession:
        if event == MissionEvent.MISSION_RESET:
            self.session = self._initial_session
            self.ring_entry_alignment_frames = 0
            self.last_transition_event = event
            return self.session
        if event == MissionEvent.MISSION_COMPLETED:
            self.session = CourseSession.FINISHED
            self.ring_entry_alignment_frames = 0
            self.last_transition_event = event
            return self.session
        if event != MissionEvent.PHASE_COMPLETED:
            raise ValueError(f"unsupported mission event: {event!r}")
        try:
            index = self.ORDER.index(self.session)
        except ValueError:
            return self.session
        if index + 1 < len(self.ORDER):
            self.session = self.ORDER[index + 1]
        self.ring_entry_alignment_frames = 0
        self.last_transition_event = event
        return self.session

    def advance(self) -> CourseSession:
        """Compatibility wrapper; new control code must emit MissionEvent."""
        return self.transition(MissionEvent.PHASE_COMPLETED)
