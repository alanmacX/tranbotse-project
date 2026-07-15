from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

from .config import PathMemoryConfig, TrackerConfig
from .control import TransitionEvent
from .state_machine import moving_follow_command_w, moving_follow_handoff_ready
from .vision import LineFeatures, TrajectoryFit

if TYPE_CHECKING:
    from .capture_geometry import CaptureGeometryDecision, CaptureGeometryObservation


@dataclass(frozen=True, slots=True)
class MotionSample:
    linear: float
    angular: float
    source: str
    fresh: bool = True


@dataclass(frozen=True, slots=True)
class PathStrategyStatus:
    active: bool
    mode: str
    reason: str
    remaining_m: float = 0.0
    intent_dir: int = 0
    point_count: int = 0
    target: tuple[float, float] | None = None
    transition_event: TransitionEvent = TransitionEvent.NONE


@dataclass(frozen=True, slots=True)
class CornerCommandResult:
    v: float
    w: float
    status: PathStrategyStatus


class FirstEntryLineLatch:
    """Latch exactly one exit line as it first enters from the turn side.

    A right-turn exit must enter from image-right and move toward the centre;
    a left turn is the exact mirror.  ``disconnected`` is intentionally not a
    veto here: during a pivot a coherent near-field line naturally appears
    before it connects to every far band.  After latching, observations must
    remain locally continuous with the same line and no replacement is ever
    accepted.
    """

    def __init__(self, confirm_frames: int = 3) -> None:
        self.confirm_frames = max(2, int(confirm_frames))
        self.latched = False
        self.last_e = 0.0
        self.last_theta = 0.0
        self.candidate_frames = 0
        self.candidate_e: float | None = None

    def reset(self) -> None:
        self.__init__(self.confirm_frames)

    def try_latch(self, fit: TrajectoryFit, direction: int) -> bool:
        if self.latched or direction == 0:
            return False
        strong = bool(
            fit.control_valid
            and fit.conf >= 0.30
            and fit.n_bands >= 3
            and abs(fit.theta) <= 0.95
        )
        side = direction * fit.e0
        weak_first_entry = bool(
            fit.control_valid
            and fit.conf >= 0.15
            and fit.n_bands >= 1
            and abs(fit.e0) >= 0.45
        )
        if side <= 0.0 or not (strong or weak_first_entry):
            self.candidate_frames = 0
            self.candidate_e = None
            return False
        continuous = bool(
            self.candidate_e is None
            or abs(fit.e0 - self.candidate_e) <= 0.50
        )
        self.candidate_frames = self.candidate_frames + 1 if continuous else 1
        self.candidate_e = fit.e0
        # Observation quality changes how long confirmation takes, but never
        # bypasses temporal confirmation entirely. This prevents one noisy,
        # high-confidence fragment from becoming an irreversible route ID.
        required = 2 if strong else self.confirm_frames
        if self.candidate_frames < required:
            return False
        # The first coherent line on the commanded turn side is the exit.
        # Waiting for it to move inward discarded the actual first entry in
        # 173358 and allowed a much later line to become eligible instead.
        self.latched = True
        self.last_e = fit.e0
        self.last_theta = fit.theta
        return True

    def observe_latched(self, fit: TrajectoryFit, min_conf: float = 0.35) -> bool:
        if not self.latched:
            return False
        continuous = bool(
            fit.control_valid
            # Confirmation is also the cruise handoff frame, so it must meet
            # the same completeness contract as RaceStateMachine.  A weaker
            # or disconnected fragment may stop the coarse pivot provisionally
            # via ``try_latch``, but it cannot seed moving cruise.
            and fit.conf >= min_conf
            and fit.n_bands >= 3
            and not fit.disconnected
            and abs(fit.e0) <= 0.95
            and abs(fit.theta) <= 1.20
            and abs(fit.e0 - self.last_e) <= 0.32
            and abs(fit.theta - self.last_theta) <= 0.45
        )
        if continuous:
            self.last_e = fit.e0
            self.last_theta = fit.theta
        return continuous


def read_motion_sample(bot: object, fallback_v: float, fallback_w: float) -> MotionSample:
    getter = getattr(bot, "get_motion_data", None)
    if callable(getter):
        try:
            value = getter()
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                linear, angular = float(value[0]), float(value[1])
                valid = math.isfinite(linear) and math.isfinite(angular) and abs(linear) <= 0.5 and abs(angular) <= 3.0
                stale_linear = abs(fallback_v) > 0.01 and abs(linear) < 0.002
                stale_angular = abs(fallback_w) > 0.03 and abs(angular) < 0.005
                if valid and not (stale_linear or stale_angular):
                    # The installed Transbot API returns only (v, w), without a
                    # timestamp or sequence. Such a sample may still be a
                    # frozen old value, so it cannot certify physical stop.
                    fresh = bool(len(value) >= 3 and value[2] is True)
                    return MotionSample(
                        linear,
                        angular,
                        "measured" if fresh else "measured_unverified",
                        fresh,
                    )
        except Exception:
            pass
    return MotionSample(float(fallback_v), float(fallback_w), "command_fallback", False)


def _corner_observation(
    features: LineFeatures, fit: TrajectoryFit, cfg: PathMemoryConfig,
) -> tuple[int, float]:
    left = features.branch_left is not None
    right = features.branch_right is not None
    if left and right:
        return 0, 0.0
    if right:
        return 1, math.pi / 2.0
    if left:
        return -1, math.pi / 2.0
    if fit.preview_dir:
        angle = min(math.pi / 2.0, max(0.60, abs(fit.preview_theta) * cfg.corner_image_angle_gain))
        return (1 if fit.preview_dir > 0 else -1), angle
    if (
        not fit.found
        or fit.conf < 0.80
        or fit.n_bands < 3
        or fit.disconnected
        or abs(fit.theta) < cfg.corner_theta_threshold
    ):
        return 0, 0.0
    direction = 1 if fit.theta > 0 else -1
    # A one-frame fit that points across the current lateral displacement is
    # the failure signature seen in 113957; do not latch it as a corner.
    if abs(fit.e0) >= 0.02 and direction != (1 if fit.e0 > 0 else -1):
        return 0, 0.0
    angle = min(math.pi / 2.0, max(0.60, abs(fit.theta) * cfg.corner_image_angle_gain))
    return direction, angle


def _intent_direction(features: LineFeatures, fit: TrajectoryFit, cfg: PathMemoryConfig) -> int:
    return _corner_observation(features, fit, cfg)[0]


class CornerCommandDelay:
    def __init__(
        self,
        cfg: PathMemoryConfig,
        handoff_conf_min: float = 0.35,
        tracker_cfg: TrackerConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.handoff_conf_min = max(0.0, float(handoff_conf_min))
        self.tracker_cfg = tracker_cfg or TrackerConfig()
        self.state = "armed"
        self.candidate_dir = 0
        self.confirm = 0
        self.candidate_angles: list[float] = []
        self.target_angle_rad = cfg.corner_turn_angle_rad
        self.remaining_m = 0.0
        self.hold_v = 0.0
        self.hold_w = 0.0
        self.stable_v = 0.0
        self.stable_w = 0.0
        self.turned_rad = 0.0
        self.reacquire_frames = 0
        self.align_missing_frames = 0
        self.align_travelled_m = 0.0
        self.clear_frames = 0
        self.gate_frames = 0
        self.approach_missing_frames = 0
        self.approach_travelled_m = 0.0
        self.capture_votes = 0
        self.event_shape = "corner"
        self.exit_latch = FirstEntryLineLatch(cfg.corner_exit_confirm_frames)
        self.recovery_last_e: float | None = None
        self.recovery_last_theta: float | None = None
        self.exit_last_seen_now: float | None = None
        self.exit_track_w = 0.0
        self.last_now: float | None = None

    def will_accept_geometry(
        self, decision: CaptureGeometryDecision | None,
    ) -> bool:
        """Whether an armed controller will atomically take turn ownership."""
        if (
            self.state != "armed"
            # Margin/approach is a forward-distance operation.  Taking motor
            # ownership without a previously verified forward command leaves
            # approach or waiting at v=0 forever because no odometry can
            # consume the remaining distance.  A geometry decision may still
            # be observed while cruise establishes this prerequisite.
            or self.stable_v <= 0.01
            or decision is None
            or decision.kind not in {"turn", "corner"}
            or decision.direction == 0
        ):
            return False
        # Even an ordinary corner must have a plausible incoming stem.  These
        # broad bounds reject clipped/noise geometry without demanding the
        # fork's much tighter centring (historical real corners stayed within
        # |e|=.20 and |theta|=.08; the 193907 shadow was -1/-1.2 or worse).
        if abs(decision.incoming_e) > 0.65 or abs(decision.incoming_theta) > 0.45:
            return False
        if not decision.is_fork:
            return True
        # A route-selected ring is actionable only while its incoming stem is
        # already in the normal closed-loop corridor.  Accepting e=-1 geometry
        # in 193907 handed ownership to a fork the car was not facing.
        return bool(
            abs(decision.incoming_e) <= self.cfg.corner_reacquire_max_e
            and abs(decision.incoming_theta) <= self.cfg.corner_reacquire_max_theta
        )

    def step(
        self,
        fit: TrajectoryFit,
        features: LineFeatures,
        command_v: float,
        command_w: float,
        now: float,
        linear: float,
        angular: float = 0.0,
        geometry: CaptureGeometryObservation | None = None,
        geometry_decision: CaptureGeometryDecision | None = None,
        invert_turn: bool = False,
        angular_scale: float = 1.0,
    ) -> CornerCommandResult:
        travelled, yaw_delta = self._motion_delta(now, linear, angular, angular_scale)
        visual_takeover = False
        direction, observed_angle = (0, 0.0)
        if not self.cfg.capture_geometry_enabled:
            direction, observed_angle = _corner_observation(features, fit, self.cfg)
        if self.state == "armed":
            # Freeze this frame's arbitration result before updating the
            # cached approach command below.  Runner makes the same precheck
            # before deciding whether cruise may advance; changing false to
            # true midway through this method would create two motor owners in
            # one frame (a TOCTOU takeover).
            accept_geometry_now = self.will_accept_geometry(geometry_decision)
            if (
                (direction == 0 or not self.cfg.capture_geometry_enabled)
                and fit.found
                and fit.conf >= 0.65
                and command_v > 0.01
                and not accept_geometry_now
            ):
                # Remember the last trustworthy incoming-line correction.
                # It is frozen once a turn is latched; the bend itself must not
                # pull the robot into an early turn during axle compensation.
                # In particular, never let a zero-speed PIVOT frame erase a
                # valid forward approach command.  That was the pending-
                # takeover deadlock exposed by the 193907 sequence.
                limit = self.cfg.corner_approach_max_w
                self.stable_v = max(0.0, command_v)
                self.stable_w = max(-limit, min(limit, command_w))
            if self.cfg.capture_geometry_enabled:
                if accept_geometry_now:
                    assert geometry_decision is not None
                    self.state = "approach"
                    self.candidate_dir = geometry_decision.direction
                    self.event_shape = (
                        "fork" if geometry_decision.is_fork else "corner"
                    )
                    self.capture_votes = geometry_decision.votes
                    self.hold_v = self.stable_v if self.stable_v > 0.01 else max(0.0, command_v)
                    # The first geometry detection can happen anywhere in the
                    # far field.  Distance delay is anchored only after the
                    # vertex crosses the stable image gate below.
                    self.remaining_m = 0.0
                    self.hold_w = self.stable_w
                    # A partially visible rounded bend does not reveal its
                    # total maneuver angle.  Use the configured angle only as
                    # a blind-search safety envelope; first exit-line capture
                    # may finish earlier. Scaling the visible fragment caused the
                    # old controller to stop searching around 50 degrees on a
                    # real 90-degree bend.
                    self.target_angle_rad = self.cfg.corner_turn_angle_rad
                    self.gate_frames = 0
                    self.approach_missing_frames = 0
                    self.approach_travelled_m = 0.0
                    self.turned_rad = 0.0
                    self.reacquire_frames = 0
                    self.exit_latch.reset()
            else:
                if direction and direction == self.candidate_dir:
                    self.confirm += 1
                    self.candidate_angles.append(observed_angle)
                else:
                    self.candidate_dir, self.confirm = direction, int(direction != 0)
                    self.candidate_angles = [observed_angle] if direction else []
                if (
                    self.confirm >= max(3, self.cfg.corner_confirm_frames)
                    and self.stable_v > 0.01
                ):
                    self.event_shape = "corner"
                    self._start_margin(self.stable_v)
                    angles = [angle for angle in self.candidate_angles if angle > 0.0]
                    fitted_angle = float(np.median(angles)) if angles else self.cfg.corner_turn_angle_rad
                    self.target_angle_rad = max(self.cfg.corner_reacquire_angle_rad, fitted_angle)
        elif self.state == "approach":
            self.approach_travelled_m += travelled
            expected_fork = self.event_shape == "fork"
            shape_valid = bool(
                geometry is not None
                and geometry.vertex_y_frac is not None
                and geometry.direction == self.candidate_dir
                and geometry.confidence >= 0.55
                and (
                    (
                        expected_fork
                        and geometry.is_fork
                        and geometry.kind in {"curve", "circle"}
                    )
                    or (
                        not expected_fork
                        and not geometry.is_fork
                        and geometry.kind in {"corner", "curve"}
                    )
                )
            )
            incoming_aligned = bool(
                not expected_fork
                or (
                    geometry is not None
                    and abs(geometry.incoming_e) <= self.cfg.corner_reacquire_max_e
                    and abs(geometry.incoming_theta) <= self.cfg.corner_reacquire_max_theta
                )
            )
            vertex_valid = shape_valid and incoming_aligned
            if vertex_valid:
                self.approach_missing_frames = 0
                if (
                    geometry.vertex_y_frac >= self.cfg.corner_gate_y_frac
                ):
                    if self.gate_frames == 0:
                        # Anchor distance at the first observed crossing.  The
                        # following frame confirms the gate without adding a
                        # systematic one-cycle delay to the physical turn point.
                        self.remaining_m = self.cfg.camera_to_axle_m
                    else:
                        self.remaining_m = max(0.0, self.remaining_m - travelled)
                    self.gate_frames += 1
                else:
                    self.gate_frames = 0
                    self.remaining_m = 0.0
                # Keep the incoming-line command frozen.  Updating it from the
                # visible outgoing arc is precisely the early-turn feedback
                # that the longitudinal offset is intended to prevent.
                self.hold_w = self.stable_w
            else:
                self.approach_missing_frames += 1
                self.gate_frames = 0
                self.remaining_m = 0.0
            if self.approach_travelled_m > max(0.25, self.cfg.camera_to_axle_m):
                # A real floor vertex must approach the gate as the chassis
                # advances.  A persistent wall/stage seam can otherwise keep
                # shape_valid true forever and own forward motion without any
                # reachable transition.
                self._reset_armed()
            elif self.gate_frames >= self.cfg.corner_gate_confirm_frames:
                if self.event_shape == "fork" and not self.cfg.roundabout_margin_enabled:
                    # The line/circle intersection is oblique and already
                    # exposes both routes. A corner's axial distance delay has
                    # no valid physical meaning here.
                    self.state = "turning"
                    self.remaining_m = 0.0
                    self.turned_rad = 0.0
                    self.reacquire_frames = 0
                    self.align_missing_frames = 0
                else:
                    self._start_margin(self.hold_v, reset_distance=False)
            elif self.approach_missing_frames > self.cfg.corner_approach_missing_frames:
                self._reset_armed()
        elif self.state == "waiting":
            self.remaining_m = max(0.0, self.remaining_m - travelled)
            if self.remaining_m <= 1e-6:
                self.state = "turning"
                self.turned_rad = 0.0
        elif self.state == "turning":
            self._advance_turn(yaw_delta, invert_turn)
            # The incoming fork stem can remain slightly on the requested side
            # at commit.  Applying the same small minimum yaw to every turn
            # prevents a zero-angle capture without making angle the success
            # criterion; first coherent exit entry still ends the search.
            ready = self.turned_rad >= self.cfg.corner_reacquire_angle_rad
            if self._turn_limit_reached():
                self._enter_failed()
            elif self.event_shape != "fork":
                if ready and not self.exit_latch.latched:
                    self.exit_latch.try_latch(fit, self.candidate_dir)
                if self.exit_latch.latched and self._latched_exit_trackable(fit):
                    # The first exit is now both identified and close enough
                    # to control. From this point onward its continuous visual
                    # track owns the maneuver. Never keep blind-pivoting until
                    # another complete line (often the incoming road) happens
                    # to satisfy a generic cruise predicate.
                    self.state = "exit_tracking"
                    self.reacquire_frames = 0
                    self.align_missing_frames = 0
                    self.exit_last_seen_now = now
                    self.exit_track_w = self._tracked_exit_w(fit, invert_turn)
            elif ready:
                exit_latched = self.exit_latch.try_latch(fit, self.candidate_dir)
                if exit_latched:
                    # First confirmed sight of the exit is a hard edge: cancel
                    # coarse pivot immediately.
                    self.state = "captured"
                    self.align_missing_frames = 0
                    self.reacquire_frames = 0
                elif self.exit_latch.candidate_frames > 0:
                    # Hold still while a weak far-field entry accumulates its
                    # temporal votes; do not cross into blind seeking.
                    pass
                elif self.turned_rad >= self.target_angle_rad:
                    self.state = "seeking"
                    self.reacquire_frames = 0
        elif self.state == "captured":
            # Account for physical coasting after the stop command.  If this
            # provisional candidate disappears and search resumes, the angular
            # safety budget must include that motion.
            self._advance_turn(yaw_delta, invert_turn)
            if self._turn_limit_reached():
                self._enter_failed()
            elif self.exit_latch.observe_latched(fit, self.handoff_conf_min):
                # Exit identity is established, but an edge-of-frame line is
                # not a safe cruise handoff. Keep ownership for bounded moving
                # alignment until normal cruise would remain in FOLLOW.
                self.state = "aligning"
                self.align_missing_frames = 0
                self.reacquire_frames = 0
                self.align_travelled_m = 0.0
            else:
                self.align_missing_frames += 1
                if self.align_missing_frames > self.cfg.corner_align_missing_frames:
                    # ``captured`` is only a provisional, one-frame edge.  In
                    # 192114 a tiny component clipped at e0=1.0 satisfied the
                    # first-entry gate, disappeared on the following frames,
                    # and the old transition made that visual false positive
                    # terminal for the whole maneuver.  Resume the *same*
                    # bounded turn search from the accumulated yaw instead.
                    # A true failure is declared only by ``seeking`` after the
                    # configured angular safety envelope has been exhausted.
                    self.exit_latch.reset()
                    self.align_missing_frames = 0
                    self.reacquire_frames = 0
                    # Coasting while the provisional line is being checked is
                    # part of the angular safety budget.  If it already spent
                    # the envelope, fail stopped now; emitting one more blind
                    # turn cycle would reproduce the observed over-rotation at
                    # slow/night frame rates.
                    self.state = "seeking"
        elif self.state == "aligning":
            self.align_travelled_m += travelled
            complete_line = bool(
                fit.found
                and fit.conf >= self.handoff_conf_min
                and fit.n_bands >= 3
                and not fit.disconnected
                and features.branch_left is None
                and features.branch_right is None
            )
            self.align_missing_frames = 0 if complete_line else self.align_missing_frames + 1
            cruise_ready = bool(
                complete_line
                and moving_follow_handoff_ready(fit, self.tracker_cfg)
            )
            self.reacquire_frames = self.reacquire_frames + 1 if cruise_ready else 0
            if (
                self.reacquire_frames >= self.cfg.corner_cruise_ready_frames
                and self.align_travelled_m >= self.cfg.corner_cruise_ready_distance_m
            ):
                self.state = "cooldown"
                self.clear_frames = 0
                visual_takeover = True
        elif self.state == "exit_tracking":
            # Residual yaw still spends the same 150-degree safety budget.
            self._advance_turn(yaw_delta, invert_turn)
            if self._turn_limit_reached():
                self._enter_failed()
            else:
                trackable_exit = self._latched_exit_trackable(fit)
                if trackable_exit:
                    self.exit_last_seen_now = now
                    self.exit_track_w = self._tracked_exit_w(fit, invert_turn)
                    cruise_ready = self._corner_cruise_ready(fit, features)
                    self.reacquire_frames = (
                        self.reacquire_frames + 1 if cruise_ready else 0
                    )
                if self.reacquire_frames >= self.cfg.corner_cruise_ready_frames:
                    self.state = "cooldown"
                    self.clear_frames = 0
                    visual_takeover = True
                elif (
                    self.exit_last_seen_now is None
                    or now - self.exit_last_seen_now > self.cfg.corner_exit_predict_sec
                ):
                    # The committed path is retained across dashed gaps, as a
                    # local planner retains and prunes its path. Only expiry of
                    # the bounded prediction horizon is a real path loss.
                    self._enter_failed(lock_identity=True)
        elif self.state == "seeking":
            self._advance_turn(yaw_delta, invert_turn)
            # The angular envelope is a hard safety boundary.  Check it before
            # considering this frame's line: once crossed, a same-side line
            # may already be the route we came from.
            if self._turn_limit_reached():
                self._enter_failed()
            elif self.exit_latch.try_latch(fit, self.candidate_dir):
                self.state = "captured"
                self.align_missing_frames = 0
                self.reacquire_frames = 0
        elif self.state == "failed":
            # Failure is terminal with respect to the turn: never resume
            # rotating and never bind another candidate as its exit.  It must
            # not, however, be terminal for the whole race.  A complete,
            # centred straight line is sufficient independent evidence that
            # the robot is back in an ordinary cruise pose.
            safe_cruise = self._safe_cruise_recovery(fit, features)
            continuous = bool(
                safe_cruise
                and (
                    self.recovery_last_e is None
                    or (
                        abs(fit.e0 - self.recovery_last_e) <= 0.32
                        and self.recovery_last_theta is not None
                        and abs(fit.theta - self.recovery_last_theta) <= 0.45
                    )
                )
            )
            self.reacquire_frames = self.reacquire_frames + 1 if continuous else int(safe_cruise)
            if safe_cruise:
                self.recovery_last_e = fit.e0
                self.recovery_last_theta = fit.theta
            else:
                self.recovery_last_e = None
                self.recovery_last_theta = None
            if self.reacquire_frames >= self.cfg.corner_reacquire_confirm_frames:
                self.state = "cooldown"
                self.clear_frames = 0
                visual_takeover = True
        elif self.state == "cooldown":
            # A rounded/circular entry remains the same event while the robot
            # is on the ring. Re-arming merely because the arc crosses image
            # centre caused alternating left/right turn events in run 173927.
            # Sharp corners can clear as soon as their exit crosses centre.
            cleared = (
                self._stable_straight(fit, features)
                if self.event_shape in {"curve", "fork"}
                else self._exit_cleared(fit)
            )
            self.clear_frames = self.clear_frames + 1 if cleared else 0
            required = 3 if self.event_shape in {"curve", "fork"} else 2
            if self.clear_frames >= required:
                self._reset_armed()

        if self.state == "approach":
            status = PathStrategyStatus(
                True, "corner_event", "approaching_corner", self.remaining_m,
                self.candidate_dir, self.capture_votes,
                (
                    -1.0 if geometry is None or geometry.vertex_y_frac is None else geometry.vertex_y_frac,
                    self.cfg.corner_gate_y_frac,
                ),
            )
            if self.event_shape == "fork":
                return CornerCommandResult(self.hold_v, 0.0, status)
            return CornerCommandResult(self.hold_v, self.stable_w, status)
        if self.state == "waiting":
            status = PathStrategyStatus(
                True, "corner_event", "waiting_margin", self.remaining_m,
                self.candidate_dir, 0,
                (0.0, self.target_angle_rad),
            )
            if self.event_shape == "fork":
                return CornerCommandResult(self.hold_v, 0.0, status)
            return CornerCommandResult(self.hold_v, self.hold_w, status)
        if visual_takeover:
            status = PathStrategyStatus(
                True, "corner_event", "corner_visual_takeover", 0.0,
                self.candidate_dir, self.reacquire_frames,
                (self.turned_rad, self.target_angle_rad),
                TransitionEvent.HANDOFF_READY,
            )
            return CornerCommandResult(command_v, command_w, status)
        if self.state == "turning":
            if self.exit_latch.candidate_frames > 0 and not self.exit_latch.latched:
                status = PathStrategyStatus(
                    True, "corner_event", "corner_exit_confirming", 0.0,
                    self.candidate_dir, self.exit_latch.candidate_frames,
                    (self.turned_rad, self.target_angle_rad),
                )
                return CornerCommandResult(0.0, 0.0, status)
            turn_w = self._turn_command_w(invert_turn)
            status = PathStrategyStatus(
                True, "corner_event",
                (
                    "corner_cruise_confirming"
                    if self.reacquire_frames > 0 else "committed_turn"
                ),
                0.0, self.candidate_dir, self.reacquire_frames,
                (self.turned_rad, self.cfg.corner_max_turn_angle_rad),
            )
            return CornerCommandResult(0.0, turn_w, status)
        if self.state == "exit_tracking":
            if self._latched_exit_trackable(fit):
                track_w = self.exit_track_w
                track_v = self.cfg.corner_align_v
            else:
                # A dash gap is prediction, not STOP. Preserve the last
                # committed curvature and reduce translation until the next
                # observed segment updates the path.
                track_v = (
                    self.cfg.corner_align_v
                    * self.cfg.corner_exit_predict_v_ratio
                )
                track_w = self.exit_track_w
            status = PathStrategyStatus(
                True, "corner_event", "corner_exit_tracking", 0.0,
                self.candidate_dir, self.reacquire_frames,
                (self.turned_rad, self.cfg.corner_max_turn_angle_rad),
            )
            return CornerCommandResult(track_v, track_w, status)
        if self.state == "captured":
            status = PathStrategyStatus(
                True, "corner_event", "corner_exit_captured", 0.0,
                self.candidate_dir, 1,
                (self.turned_rad, self.target_angle_rad),
            )
            return CornerCommandResult(0.0, 0.0, status)
        if self.state == "aligning":
            if fit.found and fit.n_bands >= 2 and not fit.disconnected:
                cruise_w = moving_follow_command_w(
                    fit,
                    self.tracker_cfg,
                    invert_turn=invert_turn,
                )
                align_w = max(
                    -self.cfg.corner_align_max_w,
                    min(self.cfg.corner_align_max_w, cruise_w),
                )
                align_v = self.cfg.corner_align_v
            else:
                align_v = 0.0
                align_w = 0.0
            status = PathStrategyStatus(
                True,
                "corner_event",
                "corner_exit_aligning",
                max(
                    0.0,
                    self.cfg.corner_cruise_ready_distance_m - self.align_travelled_m,
                ),
                self.candidate_dir,
                self.reacquire_frames,
                (fit.e0 if fit.found else 0.0, fit.theta if fit.found else 0.0),
            )
            return CornerCommandResult(align_v, align_w, status)
        if self.state == "seeking":
            # With no visual evidence, only the latched maneuver direction is
            # safe.  A last-frame reverse correction may have been an outlier
            # and must never become a blind search direction.
            confirming_exit = self.exit_latch.candidate_frames > 0
            turn_w = 0.0 if confirming_exit else self._turn_command_w(invert_turn)
            status = PathStrategyStatus(
                True, "corner_event",
                "corner_exit_confirming" if confirming_exit else "corner_reacquire_search",
                0.0, self.candidate_dir, self.exit_latch.candidate_frames,
                (self.turned_rad, self.target_angle_rad),
            )
            return CornerCommandResult(0.0, turn_w, status)
        if self.state in {"failed", "failed_locked"}:
            status = PathStrategyStatus(
                True, "corner_event",
                (
                    "corner_exit_lost"
                    if self.state == "failed_locked"
                    else (
                        "corner_failed_recovery_confirm"
                        if self.reacquire_frames > 0 else "corner_reacquire_failed"
                    )
                ),
                0.0,
                self.candidate_dir, self.reacquire_frames,
                (self.turned_rad, self.target_angle_rad),
            )
            return CornerCommandResult(0.0, 0.0, status)
        if self.state == "armed" and self.cfg.capture_geometry_enabled and geometry is not None:
            if geometry.kind == "circle":
                reason = "geometry_circle_passthrough"
            elif geometry.kind == "curve":
                reason = "geometry_curve_passthrough"
            else:
                reason = "armed"
            status = PathStrategyStatus(True, "corner_event", reason, 0.0, 0, 0)
            return CornerCommandResult(command_v, command_w, status)
        status = PathStrategyStatus(
            True, "corner_event", self.state, 0.0,
            self.candidate_dir, 0,
        )
        return CornerCommandResult(command_v, command_w, status)

    def _start_margin(self, hold_v: float, reset_distance: bool = True) -> None:
        self.state = "waiting"
        if reset_distance:
            self.remaining_m = self.cfg.camera_to_axle_m
        self.hold_v = max(0.0, hold_v)
        hold_limit = self.cfg.corner_hold_max_w
        self.hold_w = max(-hold_limit, min(hold_limit, self.stable_w))
        self.turned_rad = 0.0
        self.reacquire_frames = 0
        self.align_missing_frames = 0
        self.align_travelled_m = 0.0

    def _reset_armed(self) -> None:
        self.state, self.candidate_dir, self.confirm = "armed", 0, 0
        self.candidate_angles = []
        self.remaining_m = 0.0
        # A cached approach command belongs to exactly one event epoch.  The
        # next corner must observe a fresh forward cruise frame; otherwise a
        # post-turn PIVOT/LOST pose can inherit stale forward motion.
        self.hold_v = 0.0
        self.hold_w = 0.0
        self.stable_v = 0.0
        self.stable_w = 0.0
        self.reacquire_frames = 0
        self.align_missing_frames = 0
        self.align_travelled_m = 0.0
        self.gate_frames = 0
        self.approach_missing_frames = 0
        self.approach_travelled_m = 0.0
        self.capture_votes = 0
        self.event_shape = "corner"
        self.exit_latch.reset()
        self.recovery_last_e = None
        self.recovery_last_theta = None
        self.exit_last_seen_now = None
        self.exit_track_w = 0.0

    def _turn_limit_reached(self) -> bool:
        return self.turned_rad >= self._turn_limit_rad()

    def _turn_limit_rad(self) -> float:
        return (
            self.target_angle_rad + self.cfg.corner_search_extra_rad
            if self.event_shape == "fork"
            else self.cfg.corner_max_turn_angle_rad
        )

    def _enter_failed(self, lock_identity: bool = False) -> None:
        self.state = "failed_locked" if lock_identity else "failed"
        self.reacquire_frames = 0
        self.align_missing_frames = 0
        self.exit_latch.reset()
        self.recovery_last_e = None
        self.recovery_last_theta = None

    def _reacquire_valid(self, fit: TrajectoryFit) -> bool:
        return bool(
            fit.control_valid
            and fit.conf >= 0.65
            and fit.n_bands >= 3
            and not fit.disconnected
            and abs(fit.e0) <= self.cfg.corner_reacquire_max_e
            and abs(fit.theta) <= self.cfg.corner_reacquire_max_theta
        )

    def _safe_cruise_recovery(
        self, fit: TrajectoryFit, features: LineFeatures,
    ) -> bool:
        """Recognise a complete coherent line that cruise can safely align.

        ``failed`` owns a stopped chassis, so requiring the line to already be
        centred creates an impossible self-recovery condition.  Identity is
        established by consecutive e/theta continuity in the failed branch;
        RaceStateMachine owns the actual alignment after handoff.
        """
        return bool(
            fit.control_valid
            and fit.conf >= self.handoff_conf_min
            and fit.n_bands >= 3
            and not fit.disconnected
            and abs(fit.e0) <= 0.95
            and abs(fit.theta) <= 1.20
            and features.branch_left is None
            and features.branch_right is None
        )

    def _corner_cruise_ready(
        self, fit: TrajectoryFit, features: LineFeatures,
    ) -> bool:
        """The latched exit satisfies normal cruise's own FOLLOW contract."""
        return bool(
            moving_follow_handoff_ready(fit, self.tracker_cfg)
            and features.branch_left is None
            and features.branch_right is None
        )

    def _latched_exit_trackable(self, fit: TrajectoryFit) -> bool:
        """The already-selected first exit is close enough for slow control."""
        return bool(
            fit.control_valid
            and fit.conf >= self.handoff_conf_min
            and fit.n_bands >= 3
            and not fit.disconnected
            and abs(fit.e0) <= 1.0
            and abs(fit.theta) <= 1.10
        )

    def _tracked_exit_w(self, fit: TrajectoryFit, invert_turn: bool) -> float:
        cruise_w = moving_follow_command_w(
            fit,
            self.tracker_cfg,
            invert_turn=invert_turn,
        )
        limit = self.cfg.corner_replay_max_w
        return max(-limit, min(limit, cruise_w))

    def _exit_cleared(self, fit: TrajectoryFit) -> bool:
        """End event suppression once its exit has reached image centre.

        A following circle or curve might never provide a straight/low-heading
        frame. Requiring one made cooldown permanent in run 170557. Lateral
        passage is the only event-boundary fact needed here.
        """
        return bool(
            self.candidate_dir != 0
            and fit.control_valid
            and fit.conf >= 0.55
            and fit.n_bands >= 4
            and not fit.disconnected
            and abs(fit.e0) <= 0.28
            and self.candidate_dir * fit.e0 <= 0.15
        )

    def _stable_straight(self, fit: TrajectoryFit, features: LineFeatures) -> bool:
        return bool(
            self._reacquire_valid(fit)
            and abs(fit.e0) <= min(0.28, self.cfg.corner_e_threshold)
            and abs(fit.theta) <= min(0.16, self.cfg.corner_theta_threshold)
            and features.branch_left is None
            and features.branch_right is None
        )

    def _motion_delta(
        self, now: float, linear: float, angular: float, angular_scale: float,
    ) -> tuple[float, float]:
        if self.last_now is None:
            self.last_now = now
            return 0.0, 0.0
        dt = max(0.0, min(self.cfg.max_motion_dt_sec, now - self.last_now))
        self.last_now = now
        return max(0.0, linear) * dt, angular * dt * max(0.0, angular_scale)

    def pause(self, now: float) -> None:
        """Advance the clock without integrating a safety-hold interval."""
        self.last_now = float(now)

    def _turn_command_w(self, invert_turn: bool) -> float:
        sign = -self.candidate_dir
        if invert_turn:
            sign *= -1
        magnitude = (
            self.cfg.roundabout_replay_max_w
            if self.event_shape == "fork"
            else self.cfg.corner_replay_max_w
        )
        return sign * abs(magnitude)

    def _advance_turn(self, yaw_delta: float, invert_turn: bool) -> None:
        # candidate_dir +1 is a right turn, whose chassis yaw is negative.
        # Track net progress in that locked direction: a visual correction in
        # the opposite direction must reduce progress, not falsely add to it.
        direction_sign = -1.0 if self.candidate_dir > 0 else 1.0
        if invert_turn:
            direction_sign *= -1.0
        # Keep the controller's safety budget strictly bounded even when the
        # last camera/control interval crosses the threshold between samples.
        self.turned_rad = min(
            self._turn_limit_rad(),
            max(0.0, self.turned_rad + direction_sign * yaw_delta),
        )
