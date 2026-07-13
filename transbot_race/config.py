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
class PathMemoryConfig:
    """Short metric path memory referenced to the chassis turning center."""

    enabled: bool = True
    camera_to_axle_physical_m: float = 0.10
    tracking_offset_m: float = 0.0
    lookahead_m: float = 0.07
    heading_lookahead_m: float = 0.04
    max_age_sec: float = 1.20
    max_horizon_m: float = 0.80
    behind_tolerance_m: float = 0.04
    sample_spacing_m: float = 0.01
    max_snapshots: int = 12
    max_motion_dt_sec: float = 0.25

    @property
    def effective_camera_to_axle_m(self) -> float:
        return self.camera_to_axle_physical_m + self.tracking_offset_m


@dataclass(slots=True)
class ObstacleConfig:
    """Parallel monocular obstacle monitor for the straight track corridor."""

    enabled: bool = True
    stable_frames: int = 3
    arm_hold_sec: float = 0.20
    candidate_hold_frames: int = 3
    corridor_half_width_ratio: float = 0.10
    min_confidence: float = 0.52
    approach_bottom_frac: float = 0.55
    stop_bottom_frac: float = 0.78
    approach_width_frac: float = 0.48
    stop_width_frac: float = 0.62
    slow_speed_ratio: float = 0.45


@dataclass(slots=True)
class VisionConfig:
    """Image preprocessing and scan-line feature extraction knobs."""

    percentile: int = 32
    threshold_min: int = 25
    threshold_max: int = 120
    glare_rejection_threshold: int = 140
    band_count: int = 7
    active_col_ratio: float = 0.18
    min_run_width_px: int = 10
    min_run_area_px: int = 30
    branch_width_ratio: float = 2.2
    branch_min_crop_ratio: float = 0.22
    trigger_y_frac: float = 0.30
    fit_mode: str = "classic"        # "classic" contour ROI baseline, or "poly"

    # Contour/sliding-window extraction. These mirror the reliable shape of
    # common OpenCV line followers and lane detectors: clean connected
    # components first, then follow one bottom-anchored window path.
    component_min_area_px: int = 45
    component_min_fill_ratio: float = 0.10
    component_max_width_ratio: float = 0.92
    component_bottom_bar_width_ratio: float = 0.34
    component_bottom_bar_height_ratio: float = 0.20
    sliding_window_margin_px: int = 42
    sliding_window_max_gap_bands: int = 2

    # Classic OpenCV line follower mode: use near-field contour centroids only.
    classic_near_band_count: int = 4
    classic_max_run_width_ratio: float = 0.55
    classic_max_center_error_ratio: float = 0.62
    classic_theta_limit: float = 0.42

    # Preview candidate for blind zones / roundabout arcs. The controller can
    # latch this as an intent without stitching the far segment into the near fit.
    preview_min_score: float = 0.30
    preview_min_dx_ratio: float = 0.16
    preview_corridor_width_ratio: float = 1.8


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
    k_ff: float = 0.0
    max_w: float = 0.24
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
    e_pivot: float = 1.05            # lateral error alone should not stop straight tracking
    theta_pivot: float = 1.30        # radians
    pivot_hysteresis: float = 0.12   # fractional widening to exit pivot
    v_pivot_ratio: float = 0.30      # v during pivot, fraction of v_max
    w_pivot: float = 0.26

    # LOST search sweep.
    w_search: float = 0.16
    w_search_min: float = 0.10
    search_timeout_sec: float = 3.0

    # Optional branch bias for roundabout / fork exit selection.
    e_bias: float = 0.0              # + biases toward right side, - toward left

    # Pure-pursuit lookahead: steer toward a target point ahead on the accepted
    # path instead of only reacting to the bottom-band error.
    lookahead_frac: float = 0.0
    k_pursuit: float = 0.9

    # TRACK sub-mode for short blind-zone execution. A preview is latched while
    # the line is still visible; if the near path breaks, the robot rolls forward
    # briefly to the physical blind point, then turns toward the stored side.
    preview_plan_enabled: bool = False
    preview_conf_min: float = 0.34
    preview_plan_hold_sec: float = 1.10
    preview_forward_sec: float = 0.28
    preview_turn_sec: float = 0.55
    preview_turn_v_ratio: float = 0.45
    preview_turn_w: float = 0.26


@dataclass(slots=True)
class RaceConfig:
    """Single source of truth for the unified course tracker."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    perspective: PerspectiveConfig = field(default_factory=PerspectiveConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    path_memory: PathMemoryConfig = field(default_factory=PathMemoryConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
