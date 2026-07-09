from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CameraConfig:
    """Camera crop used by the race runner."""

    frame_width: int = 640
    frame_height: int = 480
    crop: tuple[int, int, int, int] = (300, 265, 430, 455)
    expand_left_px: int = 20
    expand_right_px: int = 140


@dataclass(slots=True)
class VisionConfig:
    """Image preprocessing and scan-line feature extraction knobs."""

    percentile: int = 32
    threshold_min: int = 25
    threshold_max: int = 120
    band_count: int = 5
    active_col_ratio: float = 0.18
    min_run_width_px: int = 10
    min_run_area_px: int = 30
    branch_width_ratio: float = 2.2
    branch_min_crop_ratio: float = 0.22
    trigger_y_frac: float = 0.30


@dataclass(slots=True)
class LineControlConfig:
    """Line-following controller parameters."""

    speed: float = 0.06
    kp: float = 0.24
    max_w: float = 0.24
    slow_on_error: float = 0.45
    max_slowdown: float = 0.55
    invert_turn: bool = False


@dataclass(slots=True)
class CornerConfig:
    """Timed right-angle maneuver parameters."""

    mode: str = "right"  # "auto", "left", "right", or "off"
    confirm_frames: int = 2
    forward_sec: float = 5.0
    turn_w: float = 0.38
    turn_sec: float = 2.3
    right_turn_dir: float = -1.0
    left_turn_dir: float = 1.0
    reacquire_confirm_frames: int = 3
    reacquire_err_norm: float = 0.50
    reacquire_timeout_sec: float = 3.0


@dataclass(slots=True)
class GapConfig:
    """Dashed or missing line behavior."""

    enabled: bool = True
    missing_frames: int = 4
    blind_sec: float = 1.2
    blind_speed_factor: float = 0.70
    blind_turn_factor: float = 0.35
    search_w: float = 0.16
    # Minimum |w| while searching, so a line lost near the crop center
    # (last_err_norm ~= 0) still sweeps instead of stalling at w=0.
    search_w_min: float = 0.10
    search_timeout_sec: float = 3.0


@dataclass(slots=True)
class RaceConfig:
    """Single source of truth for the integrated course state machine."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    line: LineControlConfig = field(default_factory=LineControlConfig)
    corner: CornerConfig = field(default_factory=CornerConfig)
    gap: GapConfig = field(default_factory=GapConfig)
