from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .config import RaceConfig, TrackerConfig
from .vision import TrajectoryFit


class RaceState(str, Enum):
    TRACK = "track"
    LOST = "lost"
    STOPPED = "stopped"


class TrackMode(str, Enum):
    """Sub-behavior within TRACK, reported for logging only (not a state)."""

    FOLLOW = "follow"
    PREDICT = "predict"   # running on the confidence filter through a gap
    PIVOT = "pivot"       # saturation branch: near-zero v, strong w (sharp bend)
    PLAN = "plan"         # short blind-zone execution from a latched preview


@dataclass(frozen=True, slots=True)
class MotionCommand:
    v: float
    w: float
    reason: str
    state: RaceState
    mode: TrackMode | None = None


def command_summary(command: MotionCommand, fit: TrajectoryFit) -> dict:
    """Flat JSON-friendly view of a control step, shared by runner and debug app."""
    return {
        "state": command.state.value,
        "mode": None if command.mode is None else command.mode.value,
        "reason": command.reason,
        "v": round(command.v, 4),
        "w": round(command.w, 4),
        "found": fit.found,
        "e0": round(fit.e0, 4),
        "e_look": round(fit.e_look, 4),
        "theta": round(fit.theta, 4),
        "kappa": round(fit.kappa, 4),
        "conf": round(fit.conf, 4),
        "n_bands": fit.n_bands,
        "disconnected": fit.disconnected,
        "preview_dir": fit.preview_dir,
        "preview_e": round(fit.preview_e, 4),
        "preview_theta": round(fit.preview_theta, 4),
        "preview_conf": round(fit.preview_conf, 4),
        "path_memory": fit.path_memory,
        "nearest_band_index": fit.nearest_band_index,
        "near_support": fit.has_near_support,
        "control_valid": fit.control_valid,
        "fit_source": "selected_path" if fit.path_memory else "visual_scan",
        "pose_reference": (
            "selected_path" if fit.path_memory
            else "fixed_bottom" if fit.has_near_support
            else "nearest_observed" if fit.found
            else "none"
        ),
    }


def ring_entry_takeover_ready(
    fit: TrajectoryFit,
    tracker: TrackerConfig,
) -> bool:
    """Whether a coherent incoming track can transfer control to the ring."""
    return bool(
        fit.control_valid
        and fit.conf >= tracker.conf_predict
        and fit.n_bands >= 3
        and not fit.disconnected
    )


def moving_follow_handoff_ready(
    fit: TrajectoryFit,
    tracker: TrackerConfig,
) -> bool:
    """Whether reseeding normal cruise from ``fit`` starts in moving FOLLOW.

    This is the producer/consumer contract at a session boundary. Session
    executors must use it instead of inventing narrower e/theta thresholds.
    """
    reliable = bool(
        fit.control_valid
        and fit.conf >= tracker.conf_predict
        and fit.n_bands >= 3
        and not fit.disconnected
    )
    if not reliable:
        return False
    # Handoff is deliberately stricter than "the generic tracker would not
    # stop". A session ends only after the complete line is actually inside
    # its stable near-field/heading corridor. The old e*theta<0 exception let
    # a 44-degree pose terminate the corner and caused immediate S-turning.
    return bool(
        abs(fit.e0 + tracker.e_bias)
        <= tracker.e_pivot * (1.0 - tracker.pivot_hysteresis)
        and abs(fit.theta) <= tracker.theta_pivot
    )


def moving_follow_command_w(
    fit: TrajectoryFit,
    tracker: TrackerConfig,
    *,
    invert_turn: bool | None = None,
) -> float:
    """Fixed-reference lateral steering for ordinary straight cruise."""
    e0 = fit.e0 + tracker.e_bias
    steering = tracker.k_e * e0
    inverted = tracker.invert_turn if invert_turn is None else invert_turn
    sign = 1.0 if inverted else -1.0
    return max(
        -tracker.max_w,
        min(tracker.max_w, sign * steering),
    )


class RaceStateMachine:
    """Near-field cruise controller: TRACK / LOST / STOPPED.

    Cruise only centres the currently accepted line. Fixed-session executors
    own corners, roundabouts and forks; their geometry never leaks into this
    proportional controller.
    """

    def __init__(self, cfg: RaceConfig | None = None) -> None:
        self.cfg = cfg or RaceConfig()
        self.state = RaceState.TRACK
        self.state_started_at = 0.0
        # Filtered trajectory estimate.
        self.f_e0 = 0.0
        self.f_e_look = 0.0
        self.f_theta = 0.0
        self.f_kappa = 0.0
        self.f_conf = 0.0
        self.observed_e0 = 0.0
        self.observed_theta = 0.0
        self.observation_reliable = False
        self.d_e0 = 0.0
        self.gap_prediction_consumed = False
        self.line_identity_valid = False
        self.line_identity_e0 = 0.0
        self.line_identity_theta = 0.0
        self.follow_w = 0.0
        self.follow_now: float | None = None
        self.use_path_lookahead = False
        self.ever_acquired = False
        self.plan_dir = 0
        self.plan_score = 0.0
        self.plan_expires_at = 0.0
        self.plan_active_since: float | None = None
        self.plan_forward_until = 0.0
        self.plan_turn_until = 0.0
        self.stopped_reacquire_frames = 0
        self.stopped_reacquire_e: float | None = None
        self.stopped_reacquire_theta: float | None = None
        self.last_event = "init"

    def reset(self) -> None:
        self.__init__(self.cfg)

    def can_take_ring_entry(self, fit: TrajectoryFit) -> bool:
        """True when a confirmed ring route can safely take control."""
        return ring_entry_takeover_ready(fit, self.cfg.tracker)

    def can_take_moving_handoff(self, fit: TrajectoryFit) -> bool:
        """True iff ``reacquire_from`` would give this line moving control."""
        return moving_follow_handoff_ready(fit, self.cfg.tracker)

    def reacquire_from(self, fit: TrajectoryFit, now: float) -> MotionCommand:
        """Restart normal tracking from the current line without stale turn history."""
        self.state = RaceState.TRACK
        self.state_started_at = now
        self._seed_filter(fit)
        self.follow_w = moving_follow_command_w(fit, self.cfg.tracker)
        self.follow_now = now
        self.ever_acquired = True
        self.stopped_reacquire_frames = 0
        self.stopped_reacquire_e = None
        self.stopped_reacquire_theta = None
        self.last_event = "corner_visual_takeover"
        return self._track_step(now)

    def _seed_filter(self, fit: TrajectoryFit) -> None:
        """Replace, rather than blend, pose history at an ownership boundary."""
        self.f_e0 = fit.e0
        self.f_e_look = fit.e_look
        self.f_theta = fit.theta
        self.f_kappa = fit.kappa
        self.f_conf = fit.conf
        self.observed_e0 = fit.e0
        self.observed_theta = fit.theta
        self.observation_reliable = self._trackable_observation(fit)
        self.d_e0 = 0.0
        self.gap_prediction_consumed = False
        self.use_path_lookahead = fit.path_memory
        self._seed_line_identity(fit)

    def _seed_line_identity(self, fit: TrajectoryFit) -> None:
        ordinary_near = bool(
            not fit.path_memory
            and fit.has_near_support
            and self._pose_observation_usable(fit)
        )
        self.line_identity_valid = ordinary_near
        if ordinary_near:
            self.line_identity_e0 = fit.e0
            self.line_identity_theta = fit.theta

    def _same_line_identity(self, fit: TrajectoryFit) -> bool:
        if fit.path_memory or not fit.has_near_support:
            return False
        if not self.line_identity_valid:
            return True
        return bool(
            abs(fit.e0 - self.line_identity_e0) <= 0.32
            and abs(fit.theta - self.line_identity_theta) <= 0.45
        )

    def _update_line_identity(self, fit: TrajectoryFit) -> tuple[float, float]:
        if not self.line_identity_valid:
            self._seed_line_identity(fit)
            return fit.e0, fit.theta
        alpha = min(0.25, self.cfg.tracker.filter_alpha)
        self.line_identity_e0 += alpha * (fit.e0 - self.line_identity_e0)
        self.line_identity_theta += alpha * (fit.theta - self.line_identity_theta)
        return self.line_identity_e0, self.line_identity_theta

    def _clear_line_identity(self) -> None:
        self.line_identity_valid = False
        self.line_identity_e0 = 0.0
        self.line_identity_theta = 0.0

    # -- public API -------------------------------------------------------
    def step(self, fit: TrajectoryFit, now: float) -> MotionCommand:
        if self.state == RaceState.STOPPED:
            # Search timeout is an exhausted motion search: keep the chassis
            # still, but allow a complete line that remains visible for three
            # frames to reseed the tracker.
            recoverable = bool(
                self.last_event == "search_timeout"
                and self._trackable_observation(fit)
            )
            continuous = bool(
                recoverable
                and (
                    self.stopped_reacquire_e is None
                    or (
                        abs(fit.e0 - self.stopped_reacquire_e) <= 0.32
                        and self.stopped_reacquire_theta is not None
                        and abs(fit.theta - self.stopped_reacquire_theta) <= 0.45
                    )
                )
            )
            self.stopped_reacquire_frames = (
                self.stopped_reacquire_frames + 1
                if continuous else int(recoverable)
            )
            if recoverable:
                self.stopped_reacquire_e = fit.e0
                self.stopped_reacquire_theta = fit.theta
            else:
                self.stopped_reacquire_e = None
                self.stopped_reacquire_theta = None
            if self.stopped_reacquire_frames >= 3:
                return self.reacquire_from(fit, now)
            return MotionCommand(0.0, 0.0, "stopped", self.state, None)

        # Before the first complete observation, weak two-band fragments are
        # neither motion evidence nor filter history.  Blending them while
        # stopped lets night grout poison the first real command several
        # frames later.  The first trackable line atomically seeds pose.
        if not self.ever_acquired:
            if not self._trackable_observation(fit):
                return MotionCommand(0.0, 0.0, "await_first_line", self.state, None)
            self._seed_filter(fit)
            self.ever_acquired = True
        else:
            self._update_filter(fit)
        # Preview memory starts only after the current line has established a
        # pose.  A weak startup fragment must not leave a latent turn plan that
        # executes after an unrelated first acquisition.
        self._update_plan_latch(fit, now)

        if self.state == RaceState.LOST:
            return self._lost_step(now)

        # TRACK.
        plan_cmd = self._plan_step(fit, now)
        if plan_cmd is not None:
            return plan_cmd

        if self.f_conf <= self.cfg.tracker.conf_lost:
            self._enter(RaceState.LOST, now, "line_lost")
            return self._lost_step(now)

        return self._track_step(now)

    # -- filtering --------------------------------------------------------
    def _trackable_observation(self, fit: TrajectoryFit) -> bool:
        return bool(
            fit.control_valid
            and fit.conf >= self.cfg.tracker.conf_predict
            and fit.n_bands >= 3
            and not fit.disconnected
        )

    def _pose_observation_usable(self, fit: TrajectoryFit) -> bool:
        """A weaker, still coherent dash may update pose but not mode gates."""
        return bool(
            fit.control_valid
            and fit.conf >= 0.60 * self.cfg.tracker.conf_predict
            and fit.n_bands >= 2
            and not fit.disconnected
        )

    def _update_filter(self, fit: TrajectoryFit) -> None:
        t = self.cfg.tracker
        pose_observation_usable = bool(
            self._pose_observation_usable(fit)
            and self._same_line_identity(fit)
        )
        self.observation_reliable = bool(
            self._trackable_observation(fit)
            and self._same_line_identity(fit)
        )
        if self.observation_reliable:
            self.observed_e0 = fit.e0
            self.observed_theta = fit.theta
        if pose_observation_usable:
            route_e0, route_theta = self._update_line_identity(fit)
            self.gap_prediction_consumed = False
            self.use_path_lookahead = fit.path_memory
            prev_e0 = self.f_e0
            a, b = t.filter_alpha, t.filter_beta
            self.f_e0 = (1 - a) * (self.f_e0 + self.d_e0) + a * route_e0
            self.f_e0 = max(-1.0, min(1.0, self.f_e0))
            self.d_e0 = (1 - b) * self.d_e0 + b * (self.f_e0 - prev_e0)
            self.f_theta = (1 - a) * self.f_theta + a * route_theta
            self.f_kappa = (1 - a) * self.f_kappa + a * fit.kappa
            self.f_e_look = (1 - a) * self.f_e_look + a * fit.e_look
            self.f_e_look = max(-1.0, min(1.0, self.f_e_look))
            # Fast-attack, slow-release: snap up to a stronger fit, ease down.
            if self.f_conf < fit.conf:
                self.f_conf = fit.conf
            else:
                self.f_conf = (1 - a) * self.f_conf + a * fit.conf
        else:
            # Apply the last measured trend once per gap. Repeating the same
            # derivative every blank frame drove dashed-line pose to saturation.
            if not self.gap_prediction_consumed:
                self.f_e0 = max(-1.0, min(1.0, self.f_e0 + self.d_e0))
                self.f_e_look = max(-1.0, min(1.0, self.f_e_look + self.d_e0))
                self.gap_prediction_consumed = True
                self.d_e0 = 0.0
            self.f_conf = max(0.0, self.f_conf - t.conf_decay)

    # -- preview plan -------------------------------------------------------
    def _update_plan_latch(self, fit: TrajectoryFit, now: float) -> None:
        t = self.cfg.tracker
        if not t.preview_plan_enabled:
            self._clear_plan()
            return
        if fit.preview_dir != 0 and fit.preview_conf >= t.preview_conf_min:
            self.plan_dir = 1 if fit.preview_dir > 0 else -1
            self.plan_score = fit.preview_conf
            self.plan_expires_at = now + t.preview_plan_hold_sec
            if self.plan_active_since is None:
                self.last_event = f"preview_{'right' if self.plan_dir > 0 else 'left'}"
        elif self.plan_dir and now > self.plan_expires_at and self.plan_active_since is None:
            self._clear_plan()

    def _clear_plan(self) -> None:
        self.plan_dir = 0
        self.plan_score = 0.0
        self.plan_expires_at = 0.0
        self.plan_active_since = None
        self.plan_forward_until = 0.0
        self.plan_turn_until = 0.0

    def _plan_step(self, fit: TrajectoryFit, now: float) -> MotionCommand | None:
        t = self.cfg.tracker
        if not t.preview_plan_enabled or self.plan_dir == 0:
            return None
        if now > self.plan_expires_at and self.plan_active_since is None:
            self._clear_plan()
            return None

        reliable_now = (
            fit.control_valid
            and fit.conf >= t.conf_predict
            and not fit.disconnected
            and fit.n_bands >= 3
        )
        if self.plan_active_since is not None and reliable_now and now - self.plan_active_since > 0.12:
            self._clear_plan()
            return None

        weak_now = (not fit.control_valid) or fit.conf < t.conf_predict
        blind_now = fit.disconnected and fit.n_bands <= 3
        if self.plan_active_since is None:
            if not (weak_now or blind_now):
                return None
            self.plan_active_since = now
            self.plan_forward_until = now + t.preview_forward_sec
            self.plan_turn_until = self.plan_forward_until + t.preview_turn_sec
            self.plan_expires_at = self.plan_turn_until + 0.20

        v = t.v_max * t.preview_turn_v_ratio
        if now < self.plan_forward_until:
            return MotionCommand(v, 0.0, "preview_forward", self.state, TrackMode.PLAN)
        if now < self.plan_turn_until:
            # preview_dir > 0 means target is to image/right side. In the current
            # chassis convention, right steering is negative w when invert is off.
            turn_sign = -1.0 if self.plan_dir > 0 else 1.0
            if t.invert_turn:
                turn_sign *= -1.0
            return MotionCommand(v, turn_sign * t.preview_turn_w, "preview_turn", self.state, TrackMode.PLAN)

        self._clear_plan()
        return None

    # -- TRACK ------------------------------------------------------------
    def _track_step(self, now: float) -> MotionCommand:
        t = self.cfg.tracker
        predicting = self.f_conf < t.conf_predict

        desired_w = moving_follow_command_w(
            TrajectoryFit(
                found=True,
                e0=self.f_e0,
                e_look=self.f_e_look,
                theta=self.f_theta,
                kappa=self.f_kappa,
                conf=self.f_conf,
                n_bands=3,
            ),
            t,
        )
        if self.follow_now is None:
            self.follow_w = desired_w
        else:
            dt = max(0.0, min(0.5, now - self.follow_now))
            max_delta = t.max_w_slew_rate * dt
            delta = max(-max_delta, min(max_delta, desired_w - self.follow_w))
            self.follow_w += delta
        self.follow_now = now
        w = self.follow_w

        slowdown = t.slow_gain * min(1.0, abs(w) / max(t.max_w, 1e-6))
        v = t.v_max * max(t.v_min_ratio, 1.0 - slowdown)
        if predicting:
            v *= t.predict_speed_factor
            mode = TrackMode.PREDICT
        else:
            mode = TrackMode.FOLLOW

        return MotionCommand(v, w, f"track_{mode.value}", self.state, mode)

    # -- LOST -------------------------------------------------------------
    def _lost_step(self, now: float) -> MotionCommand:
        t = self.cfg.tracker
        # Recovered?
        if self.f_conf > t.conf_predict:
            self._enter(RaceState.TRACK, now, "reacquired")
            return self._track_step(now)
        if now - self.state_started_at >= t.search_timeout_sec:
            self._enter(RaceState.STOPPED, now, "search_timeout")
            return MotionCommand(0.0, 0.0, "search_timeout", self.state, None)
        # Sweep toward the last known side; never stall at w=0.
        hint = self.f_e0 if abs(self.f_e0) > 1e-3 else (1.0 if self.d_e0 >= 0 else -1.0)
        sign = -1.0 if hint > 0 else 1.0
        w = sign * max(abs(self.f_e0) * t.w_search, t.w_search_min)
        return MotionCommand(0.0, w, "line_search", self.state, None)

    def _enter(self, state: RaceState, now: float, event: str) -> None:
        self.state = state
        self.state_started_at = now
        self.last_event = event
        self.stopped_reacquire_frames = 0
        self.stopped_reacquire_e = None
        self.stopped_reacquire_theta = None
        if state != RaceState.TRACK:
            self._clear_plan()
            self._clear_line_identity()
