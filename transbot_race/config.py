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
    band_count: int = 7
    active_col_ratio: float = 0.18
    min_run_width_px: int = 10
    min_run_area_px: int = 30
    branch_width_ratio: float = 2.2
    branch_min_crop_ratio: float = 0.22
    trigger_y_frac: float = 0.30


@dataclass(slots=True)
class TrackerConfig:
    """Unified continuous line tracker.

    Corners (any angle), dashed lines and roundabouts are handled by one control
    law over the trajectory fit (e0, theta, kappa) plus a confidence filter,
    rather than by per-case states.
    """

    # Base motion.
    v_max: float = 0.06
    v_min_ratio: float = 0.35        # floor on v as a fraction of v_max
    invert_turn: bool = False

    # Control law: w = k_e*e0 + k_theta*theta + k_ff*kappa.
    k_e: float = 0.24
    k_theta: float = 0.30
    k_ff: float = 0.20
    max_w: float = 0.30
    slow_gain: float = 0.55          # how much |w| cuts speed (0..1)

    # Confidence filter: below conf_predict the controller runs on prediction
    # only, carrying the car through dashed/occluded gaps.
    filter_alpha: float = 0.55       # position blend when a fit is present
    filter_beta: float = 0.25        # rate blend
    conf_predict: float = 0.35       # below this, run on prediction only
    conf_decay: float = 0.15         # conf lost per predicted frame
    conf_lost: float = 0.12          # predicted conf that trips LOST
    predict_speed_factor: float = 0.7

    # Pivot assist: a saturation branch of the same controller, entered when the
    # line is far off / sharply angled (covers corners of any angle).
    e_pivot: float = 0.55
    theta_pivot: float = 0.65        # radians
    pivot_hysteresis: float = 0.12   # fractional widening to exit pivot
    v_pivot_ratio: float = 0.0       # v during pivot, fraction of v_max
    w_pivot: float = 0.34

    # LOST search sweep.
    w_search: float = 0.16
    w_search_min: float = 0.10
    search_timeout_sec: float = 3.0

    # Optional branch bias for roundabout / fork exit selection.
    e_bias: float = 0.0              # + biases toward right side, - toward left


@dataclass(slots=True)
class RaceConfig:
    """Single source of truth for the unified course tracker."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
