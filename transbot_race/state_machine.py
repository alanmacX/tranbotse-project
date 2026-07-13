from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import RaceConfig
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
    }


class RaceStateMachine:
    """Single continuous controller: TRACK / LOST / STOPPED.

    All curve, corner (any angle), dashed-line and roundabout behavior emerges
    from one control law over the trajectory fit plus a confidence filter. There
    are no per-case states or timed open-loop maneuvers.
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
        self.d_e0 = 0.0
        self.in_pivot = False
        self.use_path_lookahead = False
        self.ever_acquired = False
        self.plan_dir = 0
        self.plan_score = 0.0
        self.plan_expires_at = 0.0
        self.plan_active_since: float | None = None
        self.plan_forward_until = 0.0
        self.plan_turn_until = 0.0
        self.last_event = "init"

    def reset(self) -> None:
        self.__init__(self.cfg)

    # -- public API -------------------------------------------------------
    def step(self, fit: TrajectoryFit, now: float, obstacle: bool = False) -> MotionCommand:
        if obstacle:
            self._enter(RaceState.STOPPED, now, "obstacle")
            return MotionCommand(0.0, 0.0, "obstacle", self.state, None)

        if self.state == RaceState.STOPPED:
            return MotionCommand(0.0, 0.0, "stopped", self.state, None)

        self._update_plan_latch(fit, now)
        self._update_filter(fit)
        if fit.found and fit.conf > 0.0:
            self.ever_acquired = True

        # Camera exposure and the first preprocessing frames may be blank. Do
        # not rotate before the tracker has established which side the line is
        # on; LOST search is only meaningful after a real acquisition.
        if not self.ever_acquired:
            return MotionCommand(0.0, 0.0, "await_first_line", self.state, None)

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
    def _update_filter(self, fit: TrajectoryFit) -> None:
        t = self.cfg.tracker
        if fit.found and fit.conf > 0.0:
            self.use_path_lookahead = fit.path_memory
            prev_e0 = self.f_e0
            a, b = t.filter_alpha, t.filter_beta
            self.f_e0 = (1 - a) * (self.f_e0 + self.d_e0) + a * fit.e0
            self.f_e0 = max(-1.0, min(1.0, self.f_e0))
            self.d_e0 = (1 - b) * self.d_e0 + b * (self.f_e0 - prev_e0)
            self.f_theta = (1 - a) * self.f_theta + a * fit.theta
            self.f_kappa = (1 - a) * self.f_kappa + a * fit.kappa
            self.f_e_look = (1 - a) * self.f_e_look + a * fit.e_look
            self.f_e_look = max(-1.0, min(1.0, self.f_e_look))
            # Fast-attack, slow-release: snap up to a stronger fit, ease down.
            if self.f_conf < fit.conf:
                self.f_conf = fit.conf
            else:
                self.f_conf = (1 - a) * self.f_conf + a * fit.conf
        else:
            # Predict: extrapolate position by its rate, decay confidence.
            self.f_e0 = max(-1.0, min(1.0, self.f_e0 + self.d_e0))
            self.f_e_look = max(-1.0, min(1.0, self.f_e_look + self.d_e0))
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

        reliable_now = fit.found and fit.conf >= t.conf_predict and not fit.disconnected and fit.n_bands >= 3
        if self.plan_active_since is not None and reliable_now and now - self.plan_active_since > 0.12:
            self._clear_plan()
            return None

        weak_now = (not fit.found) or fit.conf < t.conf_predict
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

        e0 = self.f_e0 + t.e_bias
        theta = self.f_theta
        kappa = self.f_kappa

        # Lateral term: pure-pursuit toward the lookahead point when enabled,
        # else reactive error at the bottom of the crop. Same feedback shape.
        if t.lookahead_frac > 0.0 or self.use_path_lookahead:
            lateral = t.k_pursuit * (self.f_e_look + t.e_bias)
        else:
            lateral = t.k_e * e0

        sign = 1.0 if t.invert_turn else -1.0
        w = sign * (lateral + t.k_theta * theta + t.k_ff * kappa)

        # Pivot assist: saturation branch for sharp bends / corners of any angle.
        enter = abs(e0) > t.e_pivot or abs(theta) > t.theta_pivot
        exit_thr_e = t.e_pivot * (1.0 - t.pivot_hysteresis)
        exit_thr_th = t.theta_pivot * (1.0 - t.pivot_hysteresis)
        stay = abs(e0) > exit_thr_e or abs(theta) > exit_thr_th
        self.in_pivot = enter if not self.in_pivot else stay

        if self.in_pivot:
            pivot_sign = -1.0 if w < 0 else 1.0
            w = pivot_sign * t.w_pivot
            v = t.v_max * t.v_pivot_ratio
            mode = TrackMode.PIVOT
        else:
            w = max(-t.max_w, min(t.max_w, w))
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
        if state != RaceState.TRACK:
            self.in_pivot = False
            self._clear_plan()
