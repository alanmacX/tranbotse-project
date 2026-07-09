"""Core race logic for the Transbot SE course project."""

from .config import RaceConfig
from .state_machine import (
    MotionCommand,
    RaceState,
    RaceStateMachine,
    TrackMode,
    command_summary,
)
from .vision import (
    LineFeatures,
    TrajectoryFit,
    fit_line_trajectory,
    preprocess_blackline,
    scan_line_features,
)

__all__ = [
    "LineFeatures",
    "MotionCommand",
    "RaceConfig",
    "RaceState",
    "RaceStateMachine",
    "TrackMode",
    "TrajectoryFit",
    "command_summary",
    "fit_line_trajectory",
    "preprocess_blackline",
    "scan_line_features",
]
