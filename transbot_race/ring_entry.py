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
    """Linear fixed-arc entry followed by visual circulation and exit."""

    def __init__(
        self,
        direction: int,
        clear_frames: int = 4,
        inside_arm_distance_m: float = 0.18,
        exit_distance_m: float = 0.08,
        margin_distance_m: float = 0.0,
        entry_search_w: float = 0.20,
        entry_capture_frames: int = 1,
        entry_search_max_angle_rad: float = 1.75,
        entry_search_timeout_sec: float = 9.0,
        arc_v: float = 0.020,
        arc_motion_sign: int = -1,
        radius_window_rad: float = 0.15,
        radius_stable_e: float = 0.08,
        radius_w_step: float = 0.01,
        radius_confirm_windows: int = 2,
        radius_min_w: float = 0.05,
        half_arc_yaw_rad: float = math.pi,
        half_arc_timeout_sec: float = 50.0,
        exit_reacquire_extra_rad: float = 0.35,
        tracker_cfg: TrackerConfig | None = None,
    ) -> None:
        self.direction = 1 if direction >= 0 else -1
        self.clear_frames_required = max(2, int(clear_frames))
        self.inside_arm_distance_m = max(0.0, float(inside_arm_distance_m))
        self.exit_distance_m = max(0.0, float(exit_distance_m))
        self.margin_distance_m = max(0.0, float(margin_distance_m))
        self.entry_search_w = abs(float(entry_search_w))
        self.entry_capture_frames_required = max(1, int(entry_capture_frames))
        self.entry_search_max_angle_rad = max(
            0.0, float(entry_search_max_angle_rad),
        )
        self.entry_search_timeout_sec = max(
            0.0, float(entry_search_timeout_sec),
        )
        self.arc_v = max(0.0, float(arc_v))
        self.arc_motion_sign = 1 if int(arc_motion_sign) >= 0 else -1
        self.radius_window_rad = max(0.01, float(radius_window_rad))
        self.radius_stable_e = max(0.0, float(radius_stable_e))
        self.radius_w_step = max(0.0, float(radius_w_step))
        self.radius_confirm_windows_required = max(1, int(radius_confirm_windows))
        self.radius_min_w = max(0.01, abs(float(radius_min_w)))
        self.half_arc_yaw_rad = max(0.1, float(half_arc_yaw_rad))
        self.half_arc_timeout_sec = max(0.1, float(half_arc_timeout_sec))
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
        self.entry_search_yaw_rad = 0.0
        self.entry_search_elapsed_sec = 0.0
        self.arc_w = self.entry_search_w
        self.arc_turned_rad = 0.0
        self.radius_window_yaw_rad = 0.0
        self.radius_window_distance_m = 0.0
        self.radius_window_e_sum = 0.0
        self.radius_window_e_frames = 0
        self.radius_previous_e: float | None = None
        self.radius_stable_windows = 0
        self.radius_estimate_m: float | None = None
        self.radius_source = "command_proxy"
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
        self.entry_search_yaw_rad = 0.0
        self.entry_search_elapsed_sec = 0.0
        self.arc_w = self.entry_search_w
        self.arc_turned_rad = 0.0
        self.radius_window_yaw_rad = 0.0
        self.radius_window_distance_m = 0.0
        self.radius_window_e_sum = 0.0
        self.radius_window_e_frames = 0
        self.radius_previous_e = None
        self.radius_stable_windows = 0
        self.radius_estimate_m = None
        self.radius_source = "command_proxy"
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

    def fixed_arc_command(self, *, invert_turn: bool = False) -> tuple[float, float]:
        """The sole command source during radius acquisition and the half arc."""
        image_to_motor_sign = 1.0 if invert_turn else -1.0
        return (
            self.arc_motion_sign * self.arc_v,
            image_to_motor_sign * self.direction * self.arc_w,
        )

    def _reset_radius_window(self) -> None:
        self.radius_window_yaw_rad = 0.0
        self.radius_window_distance_m = 0.0
        self.radius_window_e_sum = 0.0
        self.radius_window_e_frames = 0

    def _observe_radius_anchor(
        self, route_fit: TrajectoryFit | None, *, fresh_geometry: bool,
    ) -> None:
        if not fresh_geometry or route_fit is None:
            return
        if not route_fit.control_valid or route_fit.conf < 0.55 or abs(route_fit.e0) > 0.90:
            return
        self.radius_window_e_sum += route_fit.e0
        self.radius_window_e_frames += 1

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
    ) -> RingEntryResult:
        dt = 0.0 if self.last_now is None else max(0.0, min(1.0, now - self.last_now))
        self.last_now = now
        forward_travelled = max(0.0, float(linear)) * dt
        arc_travelled = abs(float(linear)) * dt
        rotated = abs(float(angular)) * dt
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
            self.state = "radius_acquire"
            self.travelled_m = 0.0
            self.clear_frames = 0
            self.entry_candidate_fit = None
            self.entry_candidate_frames = 0
            self.entry_search_yaw_rad = 0.0
            self.entry_search_elapsed_sec = 0.0
            self.arc_w = self.entry_search_w
            self.arc_turned_rad = 0.0
            self.radius_previous_e = None
            self.radius_stable_windows = 0
            self.radius_estimate_m = None
            self._reset_radius_window()
            self._reset_route_control()
            return RingEntryResult(
                None, "ring_entry_radius_acquiring", started, False,
            )

        if self.state == "radius_acquire":
            self.entry_search_yaw_rad += rotated
            self.entry_search_elapsed_sec += dt
            self.arc_turned_rad += rotated
            self.radius_window_yaw_rad += rotated
            # The fixed ring arc may deliberately run in reverse. Radius is a
            # distance magnitude, so reverse travel must not be clipped away.
            self.radius_window_distance_m += arc_travelled
            self._observe_radius_anchor(route_fit, fresh_geometry=fresh_geometry)
            timed_out = bool(
                self.entry_search_yaw_rad >= self.entry_search_max_angle_rad
                or self.entry_search_elapsed_sec >= self.entry_search_timeout_sec
            )
            if timed_out:
                self.state = "failed"
                self.failure_reason = "ring_entry_radius_acquire_timeout"
                return RingEntryResult(
                    None, self.failure_reason, started, False,
                    RingPhaseEvent.ROUTE_LOST,
                )
            if self.radius_window_yaw_rad >= self.radius_window_rad:
                enough_path = self.radius_window_e_frames >= 2
                mean_e = (
                    self.radius_window_e_sum / self.radius_window_e_frames
                    if enough_path else None
                )
                stable = bool(
                    mean_e is not None
                    and self.radius_previous_e is not None
                    and abs(mean_e - self.radius_previous_e) <= self.radius_stable_e
                )
                self.radius_stable_windows = self.radius_stable_windows + 1 if stable else 0
                if mean_e is not None and self.radius_previous_e is not None and not stable:
                    # Reversing flips the relationship between image drift and
                    # a tighter/looser curvature, while the yaw side stays the
                    # same. Include longitudinal direction exactly once.
                    drift = (
                        self.direction
                        * self.arc_motion_sign
                        * (mean_e - self.radius_previous_e)
                    )
                    if abs(drift) > self.radius_stable_e:
                        self.arc_w += math.copysign(self.radius_w_step, drift)
                        self.arc_w = float(np.clip(
                            self.arc_w, self.radius_min_w, self.entry_search_w * 2.0,
                        ))
                if mean_e is not None:
                    self.radius_previous_e = mean_e
                if self.radius_window_yaw_rad > 1e-4 and self.radius_window_distance_m > 0.0:
                    self.radius_estimate_m = (
                        self.radius_window_distance_m / self.radius_window_yaw_rad
                    )
                else:
                    self.radius_estimate_m = self.arc_v / max(self.arc_w, 1e-4)
                self.radius_source = (
                    str(motion_source)
                    if motion_source != "command_fallback" else "command_proxy"
                )
                self._reset_radius_window()
            if self.radius_stable_windows >= self.radius_confirm_windows_required:
                self.state = "half_arc"
                # Radius acquisition is a separate state. The commanded half
                # circle starts at zero here; its angle and timeout must not
                # inherit motion already spent learning the radius.
                self.arc_turned_rad = 0.0
                self.entry_search_elapsed_sec = 0.0
                return RingEntryResult(
                    None, "ring_entry_radius_locked", started,
                )
            return RingEntryResult(
                None, "ring_entry_radius_acquiring", started, False,
            )

        if self.state == "half_arc":
            self.arc_turned_rad += rotated
            self.entry_search_elapsed_sec += dt
            if self.entry_search_elapsed_sec >= self.half_arc_timeout_sec:
                self.state = "failed"
                self.failure_reason = "ring_half_arc_timeout"
                return RingEntryResult(
                    None, self.failure_reason, started, False,
                    RingPhaseEvent.ROUTE_LOST,
                )
            if self.arc_turned_rad >= self.half_arc_yaw_rad:
                self.state = "exit_reacquire"
                self.entry_candidate_fit = None
                self.entry_candidate_frames = 0
                return RingEntryResult(None, "ring_exit_reacquiring", started)
            return RingEntryResult(None, "ring_half_arc_running", started)

        if self.state == "exit_reacquire":
            self.arc_turned_rad += rotated
            self._observe_entry_candidate(
                observation, route_fit, fresh_geometry=fresh_geometry,
            )
            if self.entry_candidate_frames >= self.entry_capture_frames_required:
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
            if self.arc_turned_rad >= self.half_arc_yaw_rad + self.exit_reacquire_extra_rad:
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
