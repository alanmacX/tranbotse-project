from __future__ import annotations

from dataclasses import dataclass, field


def coerce_bool(value: object) -> bool:
    """Parse config booleans without Python's truthy-string trap."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"expected boolean value, got {value!r}")


@dataclass(slots=True)
class CameraConfig:
    """Camera crop used by the race runner."""

    frame_width: int = 640
    frame_height: int = 480
    crop: tuple[int, int, int, int] = (300, 265, 430, 442)
    expand_left_px: int = 20
    expand_right_px: int = 140
    # The full crop remains available to stage geometry. Ordinary line
    # following removes the fixed chassis/gripper band at its bottom so the
    # body edge cannot become the first connected path.
    control_bottom_trim_px: int = 22


@dataclass(slots=True)
class OcclusionConfig:
    """Static dead zones (arm, gripper, chassis) in the scan coordinate space.

    Rectangles are (x0, y0, x1, y1) in the crop frame. Occluded pixels are
    forced to background before scanning
    and, crucially, occluded bands are excluded from the confidence denominator
    so a known obstruction is not mistaken for line loss.
    """

    enabled: bool = False
    rects: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(slots=True)
class PathMemoryConfig:
    """Camera-to-axle and fixed-course executor configuration."""

    # Odometric distance from the near-field curvature-onset gate to the axle
    # turn point.  This is a gate-to-axle longitudinal extrinsic compensation,
    # not an image-processing margin and not a delay from first detection.
    camera_to_axle_m: float = 0.10
    capture_geometry_enabled: bool = True
    corner_gate_y_frac: float = 0.52
    corner_gate_confirm_frames: int = 2
    corner_approach_max_w: float = 0.08
    corner_approach_missing_frames: int = 4
    corner_align_missing_frames: int = 4
    corner_exit_confirm_frames: int = 3
    corner_exit_predict_sec: float = 1.0
    corner_exit_predict_v_ratio: float = 0.60
    corner_align_v: float = 0.025
    corner_align_max_w: float = 0.08
    corner_cruise_ready_frames: int = 4
    corner_cruise_ready_distance_m: float = 0.03
    geometry_roi_top_offset_px: int = 80
    geometry_chassis_trim_px: int = 45
    corner_confirm_frames: int = 3
    corner_theta_threshold: float = 0.18
    corner_e_threshold: float = 0.48
    corner_hold_max_w: float = 0.0
    corner_replay_max_w: float = 0.20
    # Ring entry has its own perspective/axle compensation. It must not reuse
    # the corner gate distance: the line/circle intersection is oblique and is
    # detected from a different image-space landmark.
    roundabout_margin_enabled: bool = True
    roundabout_margin_distance_m: float = 0.45
    # Legacy "entry_search" config names now parameterize the bounded runtime
    # radius-acquisition state; there is no rotate-in-place search state.
    roundabout_entry_search_w: float = 0.08
    roundabout_entry_capture_frames: int = 3
    roundabout_entry_search_max_angle_rad: float = 0.75
    roundabout_entry_search_timeout_sec: float = 12.0
    roundabout_arc_v: float = 0.020
    roundabout_radius_window_rad: float = 0.15
    roundabout_radius_stable_e: float = 0.08
    roundabout_radius_w_step: float = 0.01
    roundabout_radius_confirm_windows: int = 2
    roundabout_radius_min_w: float = 0.05
    roundabout_half_arc_yaw_rad: float = 3.141592653589793
    roundabout_half_arc_timeout_sec: float = 50.0
    roundabout_exit_reacquire_extra_rad: float = 0.35
    roundabout_replay_max_w: float = 0.20
    corner_turn_angle_rad: float = 1.57
    corner_max_turn_angle_rad: float = 2.6179938779914944
    corner_command_yaw_scale: float = 1.00
    # Minimum accumulated yaw before the first same-side exit line is eligible.
    corner_reacquire_angle_rad: float = 0.35
    corner_reacquire_max_e: float = 0.18
    corner_reacquire_max_theta: float = 0.20
    corner_image_angle_gain: float = 6.8
    corner_search_extra_rad: float = 0.40
    corner_reacquire_confirm_frames: int = 3
    max_motion_dt_sec: float = 1.0


@dataclass(slots=True)
class StageGeometryConfig:
    """Geometry cost/ROI owned by one fixed-course stage."""

    roi_top_offset_px: int = 80
    chassis_trim_px: int = 45
    cadence_frames: int = 1


@dataclass(slots=True)
class GroundProjectionConfig:
    """Pixel-to-ground calibration retained for capture geometry analysis."""

    homography: tuple[float, ...] | None = None  # crop pixel -> (forward, left) metres
    paper_width_m: float = 0.210
    paper_length_m: float = 0.297
    paper_near_m: float = 0.05
    bird_width_px: int = 200
    bird_height_px: int = 260
    pixels_per_meter: float = 500.0


@dataclass(slots=True)
class MissionConfig:
    """Fixed course order; only the current session may emit an event."""

    initial_session: str = "corner"
    ring_entry_direction: int = 1
    ring_exit_direction: int = 1


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
    fit_mode: str = "poly"           # "classic" contour ROI baseline, or "poly"

    # Contour/sliding-window extraction. These mirror the reliable shape of
    # common OpenCV line followers and lane detectors: clean connected
    # components first, then follow one bottom-anchored window path.
    component_min_area_px: int = 45
    component_min_fill_ratio: float = 0.10
    # 90th-percentile inscribed radius (distance-transform pixels).  Thin tile
    # grout can span every scan band, but unlike tape it has no component core.
    component_min_thickness_px: float = 2.5
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
    """Near-field cruise plus shared session handoff thresholds."""

    # Base motion.
    v_max: float = 0.06
    v_min_ratio: float = 0.35        # floor on v as a fraction of v_max
    invert_turn: bool = False

    # Ordinary cruise restores the original near-field proportional follower:
    # w = k_e * e0. Session executors own corners and roundabout geometry.
    k_e: float = 0.24
    k_theta: float = 0.30  # session path controllers only
    k_ff: float = 0.20     # retained for session-specific path control
    max_w: float = 0.24
    max_w_slew_rate: float = 0.45  # rad/s^2; bounds single-frame reversals
    slow_gain: float = 0.55          # how much |w| cuts speed (0..1)

    # Confidence filter: below conf_predict the controller runs on prediction
    # only, carrying the car through dashed/occluded gaps.
    filter_alpha: float = 0.55       # position blend when a fit is present
    filter_beta: float = 0.25        # rate blend
    conf_predict: float = 0.35       # below this, run on prediction only
    conf_decay: float = 0.15         # conf lost per predicted frame
    conf_lost: float = 0.12          # predicted conf that trips LOST
    predict_speed_factor: float = 0.7

    # Stable session-to-cruise handoff corridor. Generic cruise has no pivot.
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
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    path_memory: PathMemoryConfig = field(default_factory=PathMemoryConfig)
    corner_geometry: StageGeometryConfig = field(
        default_factory=lambda: StageGeometryConfig(roi_top_offset_px=60),
    )
    ring_entry_geometry: StageGeometryConfig = field(default_factory=StageGeometryConfig)
    ring_exit_geometry: StageGeometryConfig = field(default_factory=StageGeometryConfig)
    ground_projection: GroundProjectionConfig = field(default_factory=GroundProjectionConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
