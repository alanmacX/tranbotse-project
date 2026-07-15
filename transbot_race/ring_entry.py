from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

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
    """Continuous entry, circulation and exit controller for one roundabout."""

    def __init__(
        self,
        direction: int,
        clear_frames: int = 4,
        min_distance_m: float = 0.10,
        inside_arm_distance_m: float = 0.18,
        exit_distance_m: float = 0.08,
        margin_distance_m: float = 0.0,
        align_e_tolerance: float = 0.22,
        align_theta_tolerance: float = 0.28,
        align_confirm_frames: int = 3,
        tracker_cfg: TrackerConfig | None = None,
    ) -> None:
        self.direction = 1 if direction >= 0 else -1
        self.clear_frames_required = max(2, int(clear_frames))
        self.min_distance_m = max(0.0, float(min_distance_m))
        self.inside_arm_distance_m = max(0.0, float(inside_arm_distance_m))
        self.exit_distance_m = max(0.0, float(exit_distance_m))
        self.margin_distance_m = max(0.0, float(margin_distance_m))
        self.align_e_tolerance = max(0.0, float(align_e_tolerance))
        self.align_theta_tolerance = max(0.0, float(align_theta_tolerance))
        self.align_confirm_frames = max(1, int(align_confirm_frames))
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
        self.align_ready_frames = 0

    def reset(self) -> None:
        self.state = "waiting"
        self.clear_frames = 0
        self.travelled_m = 0.0
        self.inside_travelled_m = 0.0
        self.margin_remaining_m = 0.0
        self.margin_v = 0.0
        self.margin_w = 0.0
        self.align_ready_frames = 0
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

    def set_direction(self, direction: int) -> None:
        self.direction = 1 if direction >= 0 else -1

    def clear_route_loss(self) -> bool:
        if self.state not in {"aligning", "tracking", "inside", "exiting"}:
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

    def step(
        self,
        observation: CaptureGeometryObservation | None,
        route_fit: TrajectoryFit | None,
        *,
        now: float,
        linear: float,
        accepted_entry: bool,
        cruise_fit: TrajectoryFit | None = None,
        incoming_v: float = 0.0,
    ) -> RingEntryResult:
        dt = 0.0 if self.last_now is None else max(0.0, min(1.0, now - self.last_now))
        self.last_now = now
        travelled = max(0.0, float(linear)) * dt
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
        if self.state == "waiting":
            if route_fit is None or not accepted_entry:
                return RingEntryResult(None, "ring_entry_waiting_topology")
            self.margin_v = max(0.0, float(incoming_v))
            self.margin_w = 0.0
            self.state = "margin"
            self.margin_remaining_m = self.margin_distance_m
            margin_started = True
            started = True

        if self.state == "margin":
            if not margin_started:
                self.margin_remaining_m = max(0.0, self.margin_remaining_m - travelled)
            if self.margin_remaining_m > 1e-6:
                return RingEntryResult(
                    None, "ring_entry_waiting_margin", started, False,
                )
            self.state = "aligning"
            self.travelled_m = 0.0
            self.clear_frames = 0
            self.align_ready_frames = 0
            self._reset_route_control()

        route_fit = self._continuous_fit(route_fit)
        self.travelled_m += travelled
        if route_fit is not None:
            self.last_fit = route_fit
            self.missing_frames = 0
        else:
            self.missing_frames += 1

        coherent_arc = route_fit is not None and observation is not None
        fork_cleared = observation is not None and not observation.is_fork
        if self.state == "aligning":
            alignment_fit = self.raw_fit if route_fit is not None else None
            aligned = bool(
                coherent_arc
                and alignment_fit is not None
                and abs(alignment_fit.e_look) <= self.align_e_tolerance
                and abs(alignment_fit.theta) <= self.align_theta_tolerance
            )
            self.align_ready_frames = self.align_ready_frames + 1 if aligned else 0
            if self.align_ready_frames >= self.align_confirm_frames:
                self.state = "tracking"
                self.travelled_m = 0.0
                self.clear_frames = 0
                return RingEntryResult(
                    route_fit,
                    "ring_entry_alignment_established",
                    started,
                )
        if self.state == "tracking":
            # Entry progress is path-relative, not a shape-classification
            # event. A real ring remains fork/circle/curve from different
            # viewpoints; requiring the fork label to disappear trapped the
            # executor at the entrance indefinitely.
            self.clear_frames = self.clear_frames + 1 if coherent_arc else 0
            if (
                self.travelled_m >= self.min_distance_m
                and self.clear_frames >= self.clear_frames_required
            ):
                # Entry topology is cleared, but the roundabout controller
                # retains motor ownership. This is a phase boundary, not a
                # handoff to generic cruise while still on the circle.
                self.state = "inside"
                self.inside_travelled_m = 0.0
                self.clear_frames = 0
                return RingEntryResult(
                    route_fit or self.last_fit,
                    "ring_entry_path_established",
                    started,
                    True,
                    RingPhaseEvent.ENTRY_ESTABLISHED,
                )
        elif self.state == "inside":
            self.inside_travelled_m += travelled
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
        elif self.state == "aligning":
            reason = "ring_entry_aligning_selected_path" if usable_fit is not None else "ring_entry_alignment_path_lost"
        else:
            reason = "ring_entry_tracking_selected_path" if usable_fit is not None else "ring_entry_selected_path_lost"
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
        limit = approach_max_w if self.state in {"aligning", "tracking"} else max_w
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
