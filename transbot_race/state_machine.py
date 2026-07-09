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
        "theta": round(fit.theta, 4),
        "kappa": round(fit.kappa, 4),
        "conf": round(fit.conf, 4),
        "n_bands": fit.n_bands,
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
        self.f_theta = 0.0
        self.f_kappa = 0.0
        self.f_conf = 0.0
        self.d_e0 = 0.0
        self.in_pivot = False
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

        self._update_filter(fit)

        if self.state == RaceState.LOST:
            return self._lost_step(now)

        # TRACK.
        if self.f_conf <= self.cfg.tracker.conf_lost:
            self._enter(RaceState.LOST, now, "line_lost")
            return self._lost_step(now)

        return self._track_step(now)

    # -- filtering --------------------------------------------------------
    def _update_filter(self, fit: TrajectoryFit) -> None:
        t = self.cfg.tracker
        if fit.found and fit.conf > 0.0:
            prev_e0 = self.f_e0
            a, b = t.filter_alpha, t.filter_beta
            self.f_e0 = (1 - a) * (self.f_e0 + self.d_e0) + a * fit.e0
            self.d_e0 = (1 - b) * self.d_e0 + b * (self.f_e0 - prev_e0)
            self.f_theta = (1 - a) * self.f_theta + a * fit.theta
            self.f_kappa = (1 - a) * self.f_kappa + a * fit.kappa
            # Fast-attack, slow-release: snap up to a stronger fit, ease down.
            if self.f_conf < fit.conf:
                self.f_conf = fit.conf
            else:
                self.f_conf = (1 - a) * self.f_conf + a * fit.conf
        else:
            # Predict: extrapolate position by its rate, decay confidence.
            self.f_e0 = max(-1.0, min(1.0, self.f_e0 + self.d_e0))
            self.f_conf = max(0.0, self.f_conf - t.conf_decay)

    # -- TRACK ------------------------------------------------------------
    def _track_step(self, now: float) -> MotionCommand:
        t = self.cfg.tracker
        predicting = self.f_conf < t.conf_predict

        e0 = self.f_e0 + t.e_bias
        theta = self.f_theta
        kappa = self.f_kappa

        sign = 1.0 if t.invert_turn else -1.0
        w = sign * (t.k_e * e0 + t.k_theta * theta + t.k_ff * kappa)

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
