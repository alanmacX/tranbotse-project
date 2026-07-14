"""Core race logic for the Transbot SE course project."""

from .config import RaceConfig
from .control import (
    CommandArbiter,
    ControlOwner,
    SafetyState,
    StopCause,
    TransitionEvent,
)
from .state_machine import (
    MotionCommand,
    RaceState,
    RaceStateMachine,
    TrackMode,
    command_summary,
)
from .mission import (
    CourseSession,
    DetectorKind,
    ExecutorKind,
    FixedSessionMission,
    MissionEvent,
    SessionSpec,
)
from .ring_entry import RingPhaseEvent
from .motor import MotorGateway, MotorLeaseError
from .vision import (
    LineFeatures,
    TrajectoryFit,
    fit_line_trajectory,
    preprocess_blackline,
    scan_line_features,
)

__all__ = [
    "LineFeatures",
    "CommandArbiter",
    "ControlOwner",
    "MotionCommand",
    "MotorGateway",
    "MotorLeaseError",
    "RaceConfig",
    "RaceState",
    "RaceStateMachine",
    "RingPhaseEvent",
    "SafetyState",
    "StopCause",
    "TrackMode",
    "TransitionEvent",
    "CourseSession",
    "FixedSessionMission",
    "MissionEvent",
    "DetectorKind",
    "ExecutorKind",
    "SessionSpec",
    "TrajectoryFit",
    "command_summary",
    "fit_line_trajectory",
    "preprocess_blackline",
    "scan_line_features",
]
