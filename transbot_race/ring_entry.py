from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

import cv2 as cv
import numpy as np

from .capture_geometry import CaptureGeometryDebug, CaptureGeometryObservation
from .config import TrackerConfig
from .state_machine import moving_follow_handoff_ready
from .vision import TrajectoryFit


class RingPhaseEvent(str, Enum):
    NONE = "none"
    ENTRY_ESTABLISHED = "ring_entry_established"
    EXIT_SELECTED = "ring_exit_selected"
    EXECUTOR_COMPLETED = "ring_executor_completed"
    ROUTE_LOST = "ring_route_lost"


@dataclass(frozen=True, slots=True)
class RingEntryResult:
    fit: TrajectoryFit | None
    reason: str
    started: bool = False
    completed: bool = False
    phase_event: RingPhaseEvent = RingPhaseEvent.NONE


@dataclass(frozen=True, slots=True)
class RingStageTransfer:
    direction: int
    raw_fit: TrajectoryFit | None
    pending_fit: TrajectoryFit | None
    pending_fit_frames: int
    last_fit: TrajectoryFit | None
    missing_frames: int
    command_w: float


@dataclass(frozen=True, slots=True)
class RingCircleModel:
    """Entry-time image-space ellipse used to register four straight chords."""

    center: tuple[float, float]
    axes: tuple[float, float]
    angle_deg: float
    support_points: int
    branch_count: int = 0
    median_residual_px: float = 0.0
    p95_residual_px: float = 0.0
    coverage_deg: float = 0.0
    quality_score: float = 0.0
    entry_phase_deg: float = 0.0
    image_arc_sign: int = 1


def circle_model_chord_points(model: RingCircleModel) -> tuple[tuple[float, float], ...]:
    """Return the five fitted ellipse points bounding the four half-ring chords."""
    cx, cy = model.center
    short_radius, long_radius = (0.5 * model.axes[0], 0.5 * model.axes[1])
    rotation = math.radians(model.angle_deg)
    cosine, sine = math.cos(rotation), math.sin(rotation)
    entry_phase = math.radians(model.entry_phase_deg)
    points: list[tuple[float, float]] = []
    for index in range(5):
        phase = entry_phase + model.image_arc_sign * index * math.pi / 4.0
        local_x = short_radius * math.cos(phase)
        local_y = long_radius * math.sin(phase)
        points.append((
            cx + local_x * cosine - local_y * sine,
            cy + local_x * sine + local_y * cosine,
        ))
    return tuple(points)


def fit_entry_circle_model(
    debug: CaptureGeometryDebug | None,
    frame_shape: tuple[int, ...] | None = None,
    route_direction: int = 1,
) -> RingCircleModel | None:
    if debug is None:
        return None
    branch_pairs = [
        (direction, path)
        for direction, path in zip(
            debug.candidate_directions, debug.candidate_paths,
        )
        if len(path) >= 80
    ]
    branches = [path for _direction, path in branch_pairs]
    if len(branches) < 2:
        return None
    # All candidates start on the incoming stem. Remove their common prefix so
    # the line into the roundabout cannot pull the fitted centre/radius.
    common = 0
    common_limit = min(len(path) for path in branches)
    while (
        common < common_limit
        and all(path[common] == branches[0][common] for path in branches[1:])
    ):
        common += 1
    points = np.unique(np.asarray(
        [point for path in branches for point in path[common:]],
        dtype=np.float32,
    ), axis=0)
    if len(points) < 250:
        return None
    try:
        (cx, cy), (axis_a, axis_b), angle = cv.fitEllipseDirect(
            points.reshape(-1, 1, 2),
        )
    except cv.error:
        return None
    axis_a, axis_b = float(axis_a), float(axis_b)
    if axis_a <= axis_b:
        short_axis, long_axis = axis_a, axis_b
    else:
        short_axis, long_axis = axis_b, axis_a
        angle = (float(angle) + 90.0) % 180.0
    if short_axis <= 1e-6:
        return None
    axis_ratio = long_axis / short_axis
    theta = math.radians(float(angle))
    cosine, sine = math.cos(theta), math.sin(theta)
    centered = points - np.asarray([cx, cy], dtype=np.float32)
    local_x = centered[:, 0] * cosine + centered[:, 1] * sine
    local_y = -centered[:, 0] * sine + centered[:, 1] * cosine
    normalized_radius = np.sqrt(
        (local_x / (0.5 * short_axis)) ** 2
        + (local_y / (0.5 * long_axis)) ** 2
    )
    residuals = np.abs(normalized_radius - 1.0) * (0.5 * short_axis)
    median_residual = float(np.median(residuals))
    p95_residual = float(np.percentile(residuals, 95))
    angles = np.sort(np.mod(np.arctan2(
        local_y / (0.5 * long_axis),
        local_x / (0.5 * short_axis),
    ), 2.0 * math.pi))
    gaps = np.diff(np.concatenate((angles, [angles[0] + 2.0 * math.pi])))
    coverage_deg = math.degrees(2.0 * math.pi - float(np.max(gaps)))
    if debug.roi is not None:
        height, width = debug.roi.shape[:2]
    elif frame_shape is not None and len(frame_shape) >= 2:
        width = int(frame_shape[1])
        height = max(1, int(frame_shape[0]) - 10 - int(debug.roi_y0))
    else:
        width = max(1, int(np.max(points[:, 0])) + 1)
        height = max(1, int(np.max(points[:, 1])) + 1)
    if (
        short_axis < 85.0
        or long_axis < 270.0
        or not 1.7 <= axis_ratio <= 4.2
        or median_residual > 3.0
        or p95_residual > 8.0
        or coverage_deg < 180.0
        or not -0.15 * width <= cx <= 1.15 * width
        or not -0.25 * height <= cy <= 0.85 * height
    ):
        return None
    quality_score = (
        short_axis * long_axis
        * min(1.5, coverage_deg / 180.0)
        / max(1.0, 1.0 + median_residual)
    )
    entry_point = np.asarray(
        branches[0][max(0, common - 1)], dtype=np.float32,
    )

    def ellipse_phase(point: np.ndarray) -> float:
        delta = point - np.asarray([cx, cy], dtype=np.float32)
        phase_x = (
            delta[0] * cosine + delta[1] * sine
        ) / (0.5 * short_axis)
        phase_y = (
            -delta[0] * sine + delta[1] * cosine
        ) / (0.5 * long_axis)
        return math.atan2(float(phase_y), float(phase_x))

    entry_phase = ellipse_phase(entry_point)
    desired = 1 if int(route_direction) >= 0 else -1
    selected_branch = next(
        (path for direction, path in branch_pairs if direction == desired),
        max(branches, key=len),
    )
    probe_index = min(len(selected_branch) - 1, common + 24)
    probe_phase = ellipse_phase(np.asarray(
        selected_branch[probe_index], dtype=np.float32,
    ))
    phase_delta = math.atan2(
        math.sin(probe_phase - entry_phase),
        math.cos(probe_phase - entry_phase),
    )
    image_arc_sign = 1 if phase_delta >= 0.0 else -1
    return RingCircleModel(
        (float(cx), float(cy)), (short_axis, long_axis),
        float(angle), len(points), len(branches), median_residual,
        p95_residual, coverage_deg, quality_score,
        math.degrees(entry_phase) % 360.0, image_arc_sign,
    )


def effective_ring_margin_distance(margin_distance_m: float, enabled: bool) -> float:
    return max(0.0, float(margin_distance_m)) if enabled else 0.0


def selected_path_fit(
    debug: CaptureGeometryDebug | None,
    observation: CaptureGeometryObservation | None,
    *,
    frame_center_x: float,
    control_width: float,
    lookahead_px: float = 145.0,
) -> TrajectoryFit | None:
    """Convert the route-selected skeleton path into a pure-pursuit fit.

    The path is already ordered from the chassis-side anchor to the selected
    exit.  A target chosen by arc length remains meaningful on the horizontal
    part of a ring, where fitting x as a single-valued polynomial of image y
    does not.
    """
    if debug is None or observation is None or len(debug.path) < 24:
        return None
    points = np.asarray(debug.path, dtype=np.float64)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    if cumulative[-1] < 45.0:
        return None
    target_distance = min(cumulative[-1], max(45.0, float(lookahead_px)))
    target_index = int(np.searchsorted(cumulative, target_distance, side="left"))
    target_index = min(len(points) - 1, max(1, target_index))
    anchor = points[0]
    target = points[target_index]
    forward = float(anchor[1] - target[1])
    lateral = float(target[0] - anchor[0])
    if forward <= 4.0 and abs(lateral) <= 12.0:
        return None
    half_width = max(1.0, 0.5 * float(control_width))
    e0 = float(np.clip((anchor[0] - frame_center_x) / half_width, -1.0, 1.0))
    e_look = float(np.clip((target[0] - frame_center_x) / half_width, -1.0, 1.0))
    theta = float(np.clip(math.atan2(lateral, max(4.0, forward)), -1.35, 1.35))
    confidence = max(0.35, min(1.0, observation.confidence))
    return TrajectoryFit(
        found=True,
        e0=e0,
        e_look=e_look,
        theta=theta,
        kappa=0.0,
        conf=confidence,
        # This is route memory derived from a selected skeleton, not a fresh
        # ordinary scan-line observation. Keep scan-specific support explicit.
        n_bands=0,
        quadratic=False,
        disconnected=True,
        preview_dir=observation.direction,
        preview_e=e_look,
        preview_theta=theta,
        preview_conf=confidence,
        path_memory=True,
    )


class RingEntryExecutor:
    """Fixed four-leg ring traversal with one linear owner."""

    _REFERENCE_ELLIPSE_AXES = (203.36, 394.51)
    _MODEL_CONFIRM_FRAMES = 3
    # The real IMU keeps roughly 0.015 rad/s of residual yaw after the chassis
    # has stopped.  Keep the stop gate above that measured floor, while still
    # rejecting visible rotation before advancing to the next linear state.
    _STOP_LINEAR_RATE_MPS = 0.002
    _STOP_ANGULAR_RATE_RAD_SEC = 0.025

    def __init__(
        self,
        direction: int,
        clear_frames: int = 4,
        inside_arm_distance_m: float = 0.18,
        exit_distance_m: float = 0.08,
        margin_distance_m: float = 0.0,
        entry_left_turn_rad: float = math.radians(45.0),
        align_w: float = 0.20,
        align_slow_w: float = 0.12,
        align_slowdown_rad: float = math.radians(5.0),
        arc_v: float = 0.020,
        fixed_radius_m: float | None = None,
        chord_distance_scale: float = 2.0,
        half_arc_yaw_rad: float = math.pi,
        half_arc_timeout_sec: float = 50.0,
        exit_reacquire_frames: int = 3,
        exit_reacquire_extra_rad: float = 0.35,
        tracker_cfg: TrackerConfig | None = None,
    ) -> None:
        self.direction = 1 if direction >= 0 else -1
        self.clear_frames_required = max(2, int(clear_frames))
        self.inside_arm_distance_m = max(0.0, float(inside_arm_distance_m))
        self.exit_distance_m = max(0.0, float(exit_distance_m))
        self.margin_distance_m = max(0.0, float(margin_distance_m))
        self.entry_left_turn_rad = float(entry_left_turn_rad)
        self.align_w = abs(float(align_w))
        self.align_slow_w = min(self.align_w, abs(float(align_slow_w)))
        self.align_slowdown_rad = max(0.0, float(align_slowdown_rad))
        self.arc_v = max(0.0, float(arc_v))
        self.fixed_radius_m = max(
            0.01,
            float(fixed_radius_m)
            if fixed_radius_m is not None
            else self.arc_v / max(self.align_w, 1e-4),
        )
        self.chord_distance_scale = max(0.05, float(chord_distance_scale))
        self.half_arc_yaw_rad = max(0.1, float(half_arc_yaw_rad))
        self.half_arc_timeout_sec = max(0.1, float(half_arc_timeout_sec))
        self.exit_reacquire_frames_required = max(1, int(exit_reacquire_frames))
        self.exit_reacquire_extra_rad = max(0.0, float(exit_reacquire_extra_rad))
        self.tracker_cfg = tracker_cfg or TrackerConfig()
        self.margin_remaining_m = 0.0
        self.margin_v = 0.0
        self.margin_w = 0.0
        self.state = "waiting"
        self.clear_frames = 0
        self.travelled_m = 0.0
        self.inside_travelled_m = 0.0
        self.last_now: float | None = None
        self.control_now: float | None = None
        self.command_w = 0.0
        self.raw_fit: TrajectoryFit | None = None
        self.pending_fit: TrajectoryFit | None = None
        self.pending_fit_frames = 0
        self.last_fit: TrajectoryFit | None = None
        self.missing_frames = 0
        self.entry_candidate_fit: TrajectoryFit | None = None
        self.entry_candidate_frames = 0
        self.circle_model: RingCircleModel | None = None
        self.circle_model_candidate: RingCircleModel | None = None
        self.circle_model_candidate_frames = 0
        self.leg_index = 0
        self.leg_phase = "idle"
        self.leg_yaw_rad = 0.0
        self.leg_distance_m = 0.0
        self.leg_chord_length_m = 0.0
        self.leg_turn_deltas_rad: tuple[float, ...] = ()
        self.leg_turn_delta_rad = 0.0
        self.settled_frames = 0
        self.model_turn_sign = 0
        self.command_turn_sign = 0
        self.wrong_way_elapsed_sec = 0.0
        self.entry_left_yaw_rad = 0.0
        self.entry_left_command_sign = 0
        self.model_elapsed_sec = 0.0
        self.model_aligned_rad = 0.0
        self.radius_estimate_m: float | None = None
        self.radius_source = "configured_fixed"
        self.failure_reason = "ring_entry_search_path_lost"

    def reset(self) -> None:
        self.state = "waiting"
        self.clear_frames = 0
        self.travelled_m = 0.0
        self.inside_travelled_m = 0.0
        self.margin_remaining_m = 0.0
        self.margin_v = 0.0
        self.margin_w = 0.0
        self.entry_candidate_fit = None
        self.entry_candidate_frames = 0
        self.circle_model = None
        self.circle_model_candidate = None
        self.circle_model_candidate_frames = 0
        self.leg_index = 0
        self.leg_phase = "idle"
        self.leg_yaw_rad = 0.0
        self.leg_distance_m = 0.0
        self.leg_chord_length_m = 0.0
        self.leg_turn_deltas_rad = ()
        self.leg_turn_delta_rad = 0.0
        self.settled_frames = 0
        self.model_turn_sign = 0
        self.command_turn_sign = 0
        self.wrong_way_elapsed_sec = 0.0
        self.entry_left_yaw_rad = 0.0
        self.entry_left_command_sign = 0
        self.model_elapsed_sec = 0.0
        self.model_aligned_rad = 0.0
        self.radius_estimate_m = None
        self.radius_source = "configured_fixed"
        self.failure_reason = "ring_entry_search_path_lost"
        self.last_now = None
        self._reset_route_control()

    def _reset_route_control(self) -> None:
        self.control_now = None
        self.command_w = 0.0
        self.raw_fit = None
        self.pending_fit = None
        self.pending_fit_frames = 0
        self.last_fit = None
        self.missing_frames = 0

    @classmethod
    def _is_chassis_settled(cls, linear: float, angular: float) -> bool:
        return bool(
            abs(float(linear)) < cls._STOP_LINEAR_RATE_MPS
            and abs(float(angular)) < cls._STOP_ANGULAR_RATE_RAD_SEC
        )

    def set_direction(self, direction: int) -> None:
        self.direction = 1 if direction >= 0 else -1

    def clear_route_loss(self) -> bool:
        if self.state not in {"inside", "exiting"}:
            return False
        if self.last_fit is None or self.missing_frames <= 5:
            return False
        # An operator clear only permits a fresh selected path to recover the
        # controller. It must never restart movement from the route that caused
        # the safety latch.
        self.raw_fit = None
        self.last_fit = None
        self.missing_frames = 0
        self.pending_fit = None
        self.pending_fit_frames = 0
        self.control_now = None
        self.command_w = 0.0
        return True

    def pause(self, now: float) -> None:
        """Advance clocks without accumulating safety-hold motion or steering."""
        self.last_now = float(now)
        self.control_now = float(now)

    def export_stage_transfer(self) -> RingStageTransfer:
        return RingStageTransfer(
            self.direction,
            self.raw_fit,
            self.pending_fit,
            self.pending_fit_frames,
            self.last_fit,
            self.missing_frames,
            self.command_w,
        )

    def initialize_exit(self, transfer: RingStageTransfer) -> None:
        self.reset()
        self.direction = transfer.direction
        self.state = "inside"
        self.raw_fit = transfer.raw_fit
        self.pending_fit = transfer.pending_fit
        self.pending_fit_frames = transfer.pending_fit_frames
        self.last_fit = transfer.last_fit
        self.missing_frames = transfer.missing_frames
        self.command_w = transfer.command_w

    def _continuous_fit(self, route_fit: TrajectoryFit | None) -> TrajectoryFit | None:
        """Reject branch swaps and low-pass the carrot in robot image space."""
        if route_fit is None:
            return None
        jumped = self.raw_fit is not None and (
            abs(route_fit.e_look - self.raw_fit.e_look) > 0.70
            or abs(route_fit.theta - self.raw_fit.theta) > 0.65
        )
        if jumped:
            same_pending = bool(
                self.pending_fit is not None
                and abs(route_fit.e_look - self.pending_fit.e_look) <= 0.25
                and abs(route_fit.theta - self.pending_fit.theta) <= 0.25
            )
            self.pending_fit_frames = self.pending_fit_frames + 1 if same_pending else 1
            self.pending_fit = route_fit
            if self.pending_fit_frames < 2:
                return None
        else:
            self.pending_fit = None
            self.pending_fit_frames = 0
        self.raw_fit = route_fit
        self.pending_fit = None
        self.pending_fit_frames = 0
        if self.last_fit is None:
            return route_fit
        # The selected skeleton is rebuilt from pixels every frame. Bound its
        # motion before filtering so a gradually changing arc remains usable
        # without allowing the carrot to sweep across the full image in one
        # control interval.
        bounded_e = self.last_fit.e_look + float(np.clip(
            route_fit.e_look - self.last_fit.e_look, -0.20, 0.20,
        ))
        bounded_theta = self.last_fit.theta + float(np.clip(
            route_fit.theta - self.last_fit.theta, -0.14, 0.14,
        ))
        alpha = 0.38
        return replace(
            route_fit,
            e0=(1.0 - alpha) * self.last_fit.e0 + alpha * route_fit.e0,
            e_look=(1.0 - alpha) * self.last_fit.e_look + alpha * bounded_e,
            theta=(1.0 - alpha) * self.last_fit.theta + alpha * bounded_theta,
        )

    def _cruise_ready(self, fit: TrajectoryFit | None) -> bool:
        return bool(
            fit is not None
            and moving_follow_handoff_ready(fit, self.tracker_cfg)
        )

    def _observe_entry_candidate(
        self,
        observation: CaptureGeometryObservation | None,
        route_fit: TrajectoryFit | None,
        *,
        fresh_geometry: bool,
    ) -> None:
        """Collect evidence for the next state without granting motor control."""
        if not fresh_geometry:
            return
        valid = bool(
            observation is not None
            and route_fit is not None
            and getattr(observation, "confidence", 1.0) >= 0.55
            and route_fit.conf >= 0.55
            and route_fit.control_valid
            and abs(route_fit.e0) <= 0.85
            and abs(route_fit.theta) <= 1.20
        )
        if not valid:
            self.entry_candidate_fit = None
            self.entry_candidate_frames = 0
            return
        continuous = bool(
            self.entry_candidate_fit is not None
            and abs(route_fit.e_look - self.entry_candidate_fit.e_look) <= 0.25
            and abs(route_fit.theta - self.entry_candidate_fit.theta) <= 0.30
        )
        self.entry_candidate_frames = self.entry_candidate_frames + 1 if continuous else 1
        self.entry_candidate_fit = route_fit

    def model_leg_command(self, *, invert_turn: bool = False) -> tuple[float, float]:
        """Execute exactly one owner action using gyro-closed-loop phases."""
        if self.leg_phase == "align":
            sign = -self.model_turn_sign if invert_turn else self.model_turn_sign
            self.command_turn_sign = sign
            remaining = max(0.0, self.leg_target_yaw_rad - self.leg_yaw_rad)
            speed = (
                self.align_slow_w
                if remaining <= self.align_slowdown_rad else self.align_w
            )
            return 0.0, sign * speed
        if self.leg_phase == "drive":
            return self.arc_v, 0.0
        return 0.0, 0.0

    def exit_line_command(self, *, invert_turn: bool = False) -> tuple[float, float]:
        """Final fixed tangent alignment after the fourth equal chord."""
        if self.state != "exit_line_align":
            return 0.0, 0.0
        sign = -self.model_turn_sign if invert_turn else self.model_turn_sign
        self.command_turn_sign = sign
        remaining = max(0.0, abs(self.leg_turn_delta_rad) - self.leg_yaw_rad)
        speed = self.align_slow_w if remaining <= self.align_slowdown_rad else self.align_w
        return 0.0, sign * speed

    def entry_left_command(self, *, invert_turn: bool = False) -> tuple[float, float]:
        """Signed margin handoff: positive is left and negative is right."""
        if self.state != "entry_left_align":
            return 0.0, 0.0
        physical_sign = 1 if self.entry_left_turn_rad >= 0.0 else -1
        self.entry_left_command_sign = -physical_sign if invert_turn else physical_sign
        remaining = max(
            0.0,
            abs(self.entry_left_turn_rad) - abs(self.entry_left_yaw_rad),
        )
        if abs(self.entry_left_turn_rad) <= 1e-6:
            return 0.0, 0.0
        speed = self.align_slow_w if remaining <= self.align_slowdown_rad else self.align_w
        return 0.0, self.entry_left_command_sign * speed

    @property
    def leg_target_yaw_rad(self) -> float:
        return abs(self.leg_turn_delta_rad)

    def _begin_model_leg(self, index: int) -> None:
        states = (
            "leg1_model", "leg2_model", "leg3_model", "leg4_exit_bridge",
        )
        self.leg_index = index
        self.state = states[index - 1]
        self.leg_phase = "align"
        self.leg_yaw_rad = 0.0
        self.leg_distance_m = 0.0
        self.leg_turn_delta_rad = self.leg_turn_deltas_rad[index - 1]
        self.model_turn_sign = 1 if self.leg_turn_delta_rad >= 0.0 else -1
        self.command_turn_sign = 0
        self.wrong_way_elapsed_sec = 0.0
        self.settled_frames = 0

    def _configure_fixed_chords(self) -> None:
        """Generate four immutable equal legs without visual geometry."""
        arc_segment_angle = math.pi / 4.0
        chord_half_angle = arc_segment_angle / 2.0
        endpoint_alignment_angle = 3.0 * math.pi / 8.0
        # Route direction +1 means right, while positive physical yaw is left.
        # The configurable post-margin correction establishes the local radial
        # heading and is deliberately outside this model.  The fixed polygon is
        # self-contained: turn 67.5 degrees onto the first chord, advance around
        # the selected half with three 45-degree turns, then turn 67.5 degrees
        # back to the same local radial heading at the exit.  The left route is
        # the exact mirror of the right route.
        first_chord_sign = -self.direction
        circle_progress_sign = self.direction
        self.leg_turn_deltas_rad = (
            first_chord_sign * endpoint_alignment_angle,
            circle_progress_sign * arc_segment_angle,
            circle_progress_sign * arc_segment_angle,
            circle_progress_sign * arc_segment_angle,
        )
        self.radius_estimate_m = self.fixed_radius_m
        self.radius_source = "configured_fixed_four_leg"
        base_chord = 2.0 * self.radius_estimate_m * math.sin(chord_half_angle)
        self.leg_chord_length_m = base_chord * self.chord_distance_scale

    def _begin_exit_line_align(self) -> None:
        self.state = "exit_line_align"
        self.leg_phase = "idle"
        self.leg_yaw_rad = 0.0
        # The fourth chord ends 67.5 degrees away from the vertical exit line.
        self.leg_turn_delta_rad = -self.direction * (3.0 * math.pi / 8.0)
        self.model_turn_sign = 1 if self.leg_turn_delta_rad >= 0.0 else -1
        self.command_turn_sign = 0
        self.wrong_way_elapsed_sec = 0.0
        self.settled_frames = 0

    def _apply_circle_model_radius(self) -> None:
        if self.circle_model is None:
            return
        reference_area = math.prod(self._REFERENCE_ELLIPSE_AXES)
        observed_area = math.prod(self.circle_model.axes)
        scale = math.sqrt(max(1e-6, observed_area / reference_area))
        scale = float(np.clip(scale, 0.60, 1.60))
        self.radius_estimate_m = self.fixed_radius_m * scale
        self.radius_source = "entry_ellipse_four_leg"

    def step(
        self,
        observation: CaptureGeometryObservation | None,
        route_fit: TrajectoryFit | None,
        *,
        now: float,
        linear: float,
        accepted_entry: bool,
        angular: float = 0.0,
        motion_source: str = "command_fallback",
        cruise_fit: TrajectoryFit | None = None,
        incoming_v: float = 0.0,
        fresh_geometry: bool = True,
        circle_model: RingCircleModel | None = None,
    ) -> RingEntryResult:
        dt = 0.0 if self.last_now is None else max(0.0, min(1.0, now - self.last_now))
        self.last_now = now
        forward_travelled = max(0.0, float(linear)) * dt
        started = False
        margin_started = False
        if self.state == "completed":
            return RingEntryResult(
                cruise_fit or route_fit or self.last_fit,
                "ring_exit_cruise_established",
                False,
                True,
                RingPhaseEvent.EXECUTOR_COMPLETED,
            )
        if self.state == "failed":
            return RingEntryResult(
                None, self.failure_reason, started, False,
                RingPhaseEvent.ROUTE_LOST,
            )
        if self.state == "waiting":
            if route_fit is None or not accepted_entry:
                return RingEntryResult(None, "ring_entry_waiting_topology")
            self.entry_candidate_fit = None
            self.entry_candidate_frames = 0
            self.margin_v = max(0.0, float(incoming_v))
            self.margin_w = 0.0
            self.state = "margin"
            self.margin_remaining_m = self.margin_distance_m
            margin_started = True
            started = True

        if self.state == "margin":
            if not margin_started:
                self.margin_remaining_m = max(
                    0.0, self.margin_remaining_m - forward_travelled,
                )
            if self.margin_remaining_m > 1e-6:
                return RingEntryResult(
                    None, "ring_entry_waiting_margin", started, False,
                )
            self.state = "entry_left_align"
            self.travelled_m = 0.0
            self.clear_frames = 0
            self.entry_candidate_fit = None
            self.entry_candidate_frames = 0
            self.model_elapsed_sec = 0.0
            self.model_aligned_rad = 0.0
            self.entry_left_yaw_rad = 0.0
            self.entry_left_command_sign = 0
            self.settled_frames = 0
            self.wrong_way_elapsed_sec = 0.0
            # Generate all four legs here. No detected ellipse, path, or image
            # direction is allowed to modify this fixed runtime route.
            self._configure_fixed_chords()
            self._reset_route_control()
            return RingEntryResult(
                None, "ring_entry_left_align", started, False,
            )

        if self.state in {"entry_left_align", "entry_left_stop"}:
            if motion_source != "measured":
                self.state = "failed"
                self.failure_reason = "ring_gyro_feedback_lost"
                return RingEntryResult(
                    None, self.failure_reason, started, False,
                    RingPhaseEvent.ROUTE_LOST,
                )
            self.model_elapsed_sec += dt
            if self.model_elapsed_sec >= self.half_arc_timeout_sec:
                self.state = "failed"
                self.failure_reason = "ring_entry_left_turn_timeout"
                return RingEntryResult(
                    None, self.failure_reason, started, False,
                    RingPhaseEvent.ROUTE_LOST,
                )
            if self.state == "entry_left_align":
                physical_sign = 1 if self.entry_left_turn_rad >= 0.0 else -1
                directed_rate = float(angular) * physical_sign
                if directed_rate < -0.02:
                    self.wrong_way_elapsed_sec += dt
                    if self.wrong_way_elapsed_sec >= 0.35:
                        self.state = "failed"
                        self.failure_reason = "ring_gyro_wrong_direction"
                        return RingEntryResult(
                            None, self.failure_reason, started, False,
                            RingPhaseEvent.ROUTE_LOST,
                        )
                else:
                    self.wrong_way_elapsed_sec = 0.0
                rotated = max(0.0, directed_rate) * dt
                self.entry_left_yaw_rad += physical_sign * rotated
                if abs(self.entry_left_yaw_rad) >= abs(self.entry_left_turn_rad):
                    self.state = "entry_left_stop"
                    self.settled_frames = 0
                    return RingEntryResult(None, "ring_entry_left_stop", started)
                return RingEntryResult(None, "ring_entry_left_align", started)

            stationary = self._is_chassis_settled(linear, angular)
            self.settled_frames = self.settled_frames + 1 if stationary else 0
            if self.settled_frames >= 2:
                # The mandatory entry turn is a separate state.  Its time and
                # yaw must not alter the fixed four-leg route that follows.
                self.model_elapsed_sec = 0.0
                self.model_aligned_rad = 0.0
                self._begin_model_leg(1)
                return RingEntryResult(None, "ring_leg1_align", started)
            return RingEntryResult(None, "ring_entry_left_stop", started)

        if self.state in {
            "leg1_model", "leg2_model", "leg3_model", "leg4_exit_bridge",
        }:
            if motion_source != "measured":
                self.state = "failed"
                self.failure_reason = "ring_gyro_feedback_lost"
                return RingEntryResult(
                    None, self.failure_reason, started, False,
                    RingPhaseEvent.ROUTE_LOST,
                )
            self.model_elapsed_sec += dt
            if self.model_elapsed_sec >= self.half_arc_timeout_sec:
                self.state = "failed"
                self.failure_reason = "ring_half_arc_timeout"
                return RingEntryResult(
                    None, self.failure_reason, started, False,
                    RingPhaseEvent.ROUTE_LOST,
                )
            reason_root = {
                "leg1_model": "ring_leg1",
                "leg2_model": "ring_leg2",
                "leg3_model": "ring_leg3",
                "leg4_exit_bridge": "ring_leg4_exit_bridge",
            }[self.state]
            if self.leg_phase == "align":
                directed_rate = float(angular) * self.model_turn_sign
                if self.command_turn_sign and directed_rate < -0.02:
                    self.wrong_way_elapsed_sec += dt
                    if self.wrong_way_elapsed_sec >= 0.35:
                        self.state = "failed"
                        self.failure_reason = "ring_gyro_wrong_direction"
                        return RingEntryResult(
                            None, self.failure_reason, started, False,
                            RingPhaseEvent.ROUTE_LOST,
                        )
                else:
                    self.wrong_way_elapsed_sec = 0.0
                rotated = max(0.0, directed_rate) * dt
                self.leg_yaw_rad += rotated
                self.model_aligned_rad += rotated
                if self.leg_yaw_rad >= self.leg_target_yaw_rad:
                    self.leg_phase = "align_stop"
                    self.settled_frames = 0
                    self.leg_distance_m = 0.0
                    return RingEntryResult(None, f"{reason_root}_align_stop", started)
                return RingEntryResult(None, f"{reason_root}_align", started)

            if self.leg_phase == "align_stop":
                stationary = self._is_chassis_settled(linear, angular)
                self.settled_frames = self.settled_frames + 1 if stationary else 0
                if self.settled_frames >= 2:
                    self.leg_phase = "drive"
                    self.settled_frames = 0
                    return RingEntryResult(None, f"{reason_root}_straight", started)
                return RingEntryResult(None, f"{reason_root}_align_stop", started)

            if self.leg_phase == "drive":
                self.leg_distance_m += forward_travelled
                if self.leg_distance_m < self.leg_chord_length_m:
                    return RingEntryResult(None, f"{reason_root}_straight", started)
                self.leg_phase = "drive_stop"
                self.settled_frames = 0
                return RingEntryResult(None, f"{reason_root}_straight_stop", started)

            stationary = self._is_chassis_settled(linear, angular)
            self.settled_frames = self.settled_frames + 1 if stationary else 0
            if self.settled_frames >= 2:
                if self.leg_index < 4:
                    self._begin_model_leg(self.leg_index + 1)
                    next_root = (
                        f"ring_leg{self.leg_index}"
                        if self.leg_index < 4 else "ring_leg4_exit_bridge"
                    )
                    return RingEntryResult(None, f"{next_root}_align", started)
                self._begin_exit_line_align()
                return RingEntryResult(None, "ring_exit_line_align", started)
            return RingEntryResult(None, f"{reason_root}_straight_stop", started)

        if self.state in {"exit_line_align", "exit_line_stop"}:
            if motion_source != "measured":
                self.state = "failed"
                self.failure_reason = "ring_gyro_feedback_lost"
                return RingEntryResult(
                    None, self.failure_reason, started, False,
                    RingPhaseEvent.ROUTE_LOST,
                )
            self.model_elapsed_sec += dt
            if self.model_elapsed_sec >= self.half_arc_timeout_sec:
                self.state = "failed"
                self.failure_reason = "ring_exit_line_align_timeout"
                return RingEntryResult(
                    None, self.failure_reason, started, False,
                    RingPhaseEvent.ROUTE_LOST,
                )
            if self.state == "exit_line_align":
                directed_rate = float(angular) * self.model_turn_sign
                if self.command_turn_sign and directed_rate < -0.02:
                    self.wrong_way_elapsed_sec += dt
                    if self.wrong_way_elapsed_sec >= 0.35:
                        self.state = "failed"
                        self.failure_reason = "ring_gyro_wrong_direction"
                        return RingEntryResult(
                            None, self.failure_reason, started, False,
                            RingPhaseEvent.ROUTE_LOST,
                        )
                else:
                    self.wrong_way_elapsed_sec = 0.0
                rotated = max(0.0, directed_rate) * dt
                self.leg_yaw_rad += rotated
                self.model_aligned_rad += rotated
                if self.leg_yaw_rad >= abs(self.leg_turn_delta_rad):
                    self.state = "exit_line_stop"
                    self.settled_frames = 0
                    return RingEntryResult(None, "ring_exit_line_stop", started)
                return RingEntryResult(None, "ring_exit_line_align", started)

            stationary = self._is_chassis_settled(linear, angular)
            self.settled_frames = self.settled_frames + 1 if stationary else 0
            if self.settled_frames >= 2:
                self.state = "exit_reacquire"
                self.entry_candidate_fit = None
                self.entry_candidate_frames = 0
                return RingEntryResult(None, "ring_exit_reacquiring", started)
            return RingEntryResult(None, "ring_exit_line_stop", started)

        if self.state == "exit_reacquire":
            self.model_elapsed_sec += dt
            self._observe_entry_candidate(
                observation, route_fit, fresh_geometry=fresh_geometry,
            )
            if self.entry_candidate_frames >= self.exit_reacquire_frames_required:
                captured_fit = self.entry_candidate_fit
                self._reset_route_control()
                route_fit = self._continuous_fit(captured_fit)
                self.last_fit = route_fit
                self.state = "exit_ready"
                return RingEntryResult(
                    route_fit,
                    "ring_half_arc_exit_path_captured",
                    started,
                    True,
                    RingPhaseEvent.ENTRY_ESTABLISHED,
                )
            if self.model_elapsed_sec >= self.half_arc_timeout_sec:
                self.state = "failed"
                self.failure_reason = "ring_half_arc_exit_path_lost"
                return RingEntryResult(
                    None, self.failure_reason, started, False,
                    RingPhaseEvent.ROUTE_LOST,
                )
            return RingEntryResult(None, "ring_exit_reacquiring", started)

        route_fit = self._continuous_fit(route_fit)
        self.travelled_m += forward_travelled
        if route_fit is not None:
            self.last_fit = route_fit
            self.missing_frames = 0
        else:
            self.missing_frames += 1

        fork_cleared = observation is not None and not observation.is_fork
        if self.state == "inside":
            self.inside_travelled_m += forward_travelled
            if accepted_entry and self.inside_travelled_m >= self.inside_arm_distance_m:
                self.state = "exiting"
                self.travelled_m = 0.0
                self.clear_frames = 0
                if route_fit is not None:
                    self.last_fit = route_fit
                return RingEntryResult(
                    route_fit or self.last_fit,
                    "ring_exit_tracking_selected_path",
                    True,
                    False,
                    RingPhaseEvent.EXIT_SELECTED,
                )
        elif self.state == "exiting":
            exit_ready = self._cruise_ready(cruise_fit) and fork_cleared
            self.clear_frames = self.clear_frames + 1 if exit_ready else 0
            if (
                self.travelled_m >= self.exit_distance_m
                and self.clear_frames >= self.clear_frames_required
            ):
                self.state = "completed"
                return RingEntryResult(
                    cruise_fit,
                    "ring_exit_cruise_established",
                    started,
                    True,
                    RingPhaseEvent.EXECUTOR_COMPLETED,
                )
        usable_fit = route_fit or (self.last_fit if self.missing_frames <= 5 else None)
        if self.state == "inside":
            reason = "ring_inside_tracking" if usable_fit is not None else "ring_inside_path_lost"
        elif self.state == "exiting":
            reason = "ring_exit_tracking_selected_path" if usable_fit is not None else "ring_exit_selected_path_lost"
        else:
            reason = "ring_entry_invalid_state"
        return RingEntryResult(
            usable_fit,
            reason,
            started,
            False,
            RingPhaseEvent.ROUTE_LOST if usable_fit is None else RingPhaseEvent.NONE,
        )

    def control(
        self,
        fit: TrajectoryFit,
        *,
        now: float,
        v_max: float,
        k_pursuit: float,
        k_theta: float,
        max_w: float,
        approach_max_w: float,
        invert_turn: bool = False,
    ) -> tuple[float, float]:
        """Curvature-regulated, slew-limited differential-drive command."""
        limit = max_w
        steering = k_pursuit * fit.e_look + k_theta * fit.theta
        sign = 1.0 if invert_turn else -1.0
        desired_w = max(-limit, min(limit, sign * steering))
        dt = 0.0 if self.control_now is None else max(0.0, min(0.5, now - self.control_now))
        self.control_now = now
        if dt <= 0.0:
            self.command_w = desired_w
        else:
            max_delta = 0.35 * dt
            self.command_w += max(-max_delta, min(max_delta, desired_w - self.command_w))
            self.command_w = max(-limit, min(limit, self.command_w))
        turn_ratio = min(1.0, abs(self.command_w) / max(0.01, limit))
        # Regulated pure pursuit principle: high curvature reduces translation
        # instead of pairing full forward speed with saturated angular speed.
        v = v_max * (0.62 - 0.28 * turn_ratio)
        return max(v_max * 0.25, v), self.command_w
