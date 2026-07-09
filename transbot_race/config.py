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
class PerspectiveConfig:
    """Lens undistort + inverse-perspective (bird's-eye) mapping.

    All fields default to a no-op: with enabled=False the pipeline runs in raw
    crop pixel space exactly as before. Populate camera_matrix/dist_coeffs from
    an intrinsic calibration and homography from a ground-plane calibration to
    turn e0/theta/kappa into physically meaningful (and left/right symmetric)
    ground-space quantities.
    """

    enabled: bool = False
    undistort: bool = False
    camera_matrix: tuple[float, ...] | None = None   # 9 floats, row-major 3x3
    dist_coeffs: tuple[float, ...] | None = None      # 5 floats k1,k2,p1,p2,k3
    homography: tuple[float, ...] | None = None        # 9 floats, crop->bird 3x3
    output_width: int = 160
    output_height: int = 240
    px_per_cm: float = 4.0


@dataclass(slots=True)
class OcclusionConfig:
    """Static dead zones (arm, gripper, chassis) in the scan coordinate space.

    Rectangles are (x0, y0, x1, y1) in the crop (or bird's-eye, if perspective
    is enabled) frame. Occluded pixels are forced to background before scanning
    and, crucially, occluded bands are excluded from the confidence denominator
    so a known obstruction is not mistaken for line loss.
    """

    enabled: bool = False
    rects: tuple[tuple[int, int, int, int], ...] = ()



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

    # Pure-pursuit lookahead: when > 0, steer toward the fitted curve sampled at
    # this fraction of the ROI height ahead, instead of the reactive e0 law.
    # 0 disables (pure reactive control, unchanged behavior).
    lookahead_frac: float = 0.0
    k_pursuit: float = 0.9


@dataclass(slots=True)
class RaceConfig:
    """Single source of truth for the unified course tracker."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    perspective: PerspectiveConfig = field(default_factory=PerspectiveConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
