from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import PathMemoryConfig
from .vision import LineFeatures, TrajectoryFit


@dataclass(frozen=True, slots=True)
class MotionSample:
    linear: float
    angular: float
    source: str


@dataclass(frozen=True, slots=True)
class PathStrategyStatus:
    active: bool
    mode: str
    reason: str
    remaining_m: float = 0.0
    intent_dir: int = 0
    point_count: int = 0
    target: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class CornerCommandResult:
    v: float
    w: float
    status: PathStrategyStatus


def read_motion_sample(bot: object, fallback_v: float, fallback_w: float) -> MotionSample:
    getter = getattr(bot, "get_motion_data", None)
    if callable(getter):
        try:
            value = getter()
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                linear, angular = float(value[0]), float(value[1])
                valid = math.isfinite(linear) and math.isfinite(angular) and abs(linear) <= 0.5 and abs(angular) <= 3.0
                stale_zero = abs(fallback_v) > 0.01 and abs(linear) < 0.002
                if valid and not stale_zero:
                    return MotionSample(linear, angular, "measured")
        except Exception:
            pass
    return MotionSample(float(fallback_v), float(fallback_w), "command_fallback")


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
    def __init__(self, cfg: PathMemoryConfig) -> None:
        self.cfg = cfg
        self.state = "armed"
        self.candidate_dir = 0
        self.confirm = 0
        self.candidate_angles: list[float] = []
        self.target_angle_rad = cfg.corner_turn_angle_rad
        self.remaining_m = 0.0
        self.hold_v = 0.0
        self.hold_w = 0.0
        self.stable_w = 0.0
        self.turned_rad = 0.0
        self.reacquire_frames = 0
        self.handoff_index = 0
        self.clear_frames = 0
        self.last_now: float | None = None

    def step(
        self,
        fit: TrajectoryFit,
        features: LineFeatures,
        command_v: float,
        command_w: float,
        now: float,
        linear: float,
        angular: float = 0.0,
    ) -> CornerCommandResult:
        travelled, turned = self._motion_delta(now, linear, angular)
        direction, observed_angle = _corner_observation(features, fit, self.cfg)
        if self.state == "armed":
            if direction == 0 and fit.found and fit.conf >= 0.65:
                limit = self.cfg.corner_hold_max_w
                self.stable_w = max(-limit, min(limit, command_w))
            if direction and direction == self.candidate_dir:
                self.confirm += 1
                self.candidate_angles.append(observed_angle)
            else:
                self.candidate_dir, self.confirm = direction, int(direction != 0)
                self.candidate_angles = [observed_angle] if direction else []
            if self.confirm >= max(3, self.cfg.corner_confirm_frames):
                self.state = "waiting"
                self.remaining_m = self.cfg.camera_to_axle_m
                self.hold_v = max(0.0, command_v)
                self.hold_w = self.stable_w
                self.turned_rad = 0.0
                self.reacquire_frames = 0
                angles = [angle for angle in self.candidate_angles if angle > 0.0]
                fitted_angle = float(np.median(angles)) if angles else self.cfg.corner_turn_angle_rad
                self.target_angle_rad = max(self.cfg.corner_reacquire_angle_rad, fitted_angle)
        elif self.state == "waiting":
            self.remaining_m = max(0.0, self.remaining_m - travelled)
            if self.remaining_m <= 1e-6:
                self.state = "turning"
                self.turned_rad = 0.0
        elif self.state == "turning":
            self.turned_rad += turned
            ready = self.turned_rad >= self.cfg.corner_reacquire_angle_rad
            visible = self._reacquire_valid(fit)
            self.reacquire_frames = self.reacquire_frames + 1 if ready and visible else 0
            if self.reacquire_frames >= self.cfg.corner_reacquire_confirm_frames:
                self.state = "handoff"
                self.handoff_index = 0
            elif self.turned_rad >= self.target_angle_rad:
                self.state = "seeking"
                self.reacquire_frames = 0
        elif self.state == "seeking":
            self.turned_rad += turned
            visible = self._reacquire_valid(fit)
            self.reacquire_frames = self.reacquire_frames + 1 if visible else 0
            if self.reacquire_frames >= self.cfg.corner_reacquire_confirm_frames:
                self.state = "handoff"
                self.handoff_index = 0
            elif self.turned_rad >= self.target_angle_rad + self.cfg.corner_search_extra_rad:
                self.state = "failed"
        elif self.state == "handoff":
            if not self._reacquire_valid(fit):
                self.state = "turning" if self.turned_rad < self.target_angle_rad else "seeking"
                self.reacquire_frames = 0
            else:
                self.handoff_index += 1
                if self.handoff_index >= self.cfg.corner_handoff_blend_frames:
                    self.state = "cooldown"
                    self.clear_frames = 0
        elif self.state == "failed":
            pass
        elif self.state == "cooldown":
            stable = self._stable_straight(fit, features)
            self.clear_frames = self.clear_frames + 1 if stable else 0
            if self.clear_frames >= 3:
                self.state, self.candidate_dir, self.confirm = "armed", 0, 0
                self.candidate_angles = []
                self.reacquire_frames = 0

        if self.state == "waiting":
            status = PathStrategyStatus(
                True, "corner_event", "waiting_margin", self.remaining_m,
                self.candidate_dir, 0,
                (0.0, self.target_angle_rad),
            )
            return CornerCommandResult(self.hold_v, self.hold_w, status)
        if self.state == "turning":
            turn_v = self.hold_v * max(0.0, min(1.0, self.cfg.corner_turn_speed_ratio))
            turn_w = -self.candidate_dir * abs(self.cfg.corner_replay_max_w)
            status = PathStrategyStatus(
                True, "corner_event", "committed_turn", 0.0,
                self.candidate_dir, 0,
                (self.turned_rad, self.target_angle_rad),
            )
            return CornerCommandResult(turn_v, turn_w, status)
        if self.state == "seeking":
            turn_w = -self.candidate_dir * abs(self.cfg.corner_replay_max_w)
            status = PathStrategyStatus(
                True, "corner_event", "corner_reacquire_search", 0.0,
                self.candidate_dir, 0,
                (self.turned_rad, self.target_angle_rad),
            )
            return CornerCommandResult(0.0, turn_w, status)
        if self.state == "handoff":
            frames = max(1, self.cfg.corner_handoff_blend_frames)
            alpha = min(1.0, self.handoff_index / frames)
            turn_v = self.hold_v * max(0.0, min(1.0, self.cfg.corner_turn_speed_ratio))
            turn_w = -self.candidate_dir * abs(self.cfg.corner_replay_max_w)
            status = PathStrategyStatus(
                True, "corner_event", "corner_visual_handoff", 0.0,
                self.candidate_dir, self.handoff_index,
                (self.turned_rad, self.target_angle_rad),
            )
            return CornerCommandResult(
                (1.0 - alpha) * turn_v + alpha * command_v,
                (1.0 - alpha) * turn_w + alpha * command_w,
                status,
            )
        if self.state == "failed":
            status = PathStrategyStatus(
                True, "corner_event", "corner_reacquire_failed", 0.0,
                self.candidate_dir, 0,
                (self.turned_rad, self.target_angle_rad),
            )
            return CornerCommandResult(0.0, 0.0, status)
        status = PathStrategyStatus(
            True, "corner_event", self.state, 0.0,
            self.candidate_dir, 0,
        )
        return CornerCommandResult(command_v, command_w, status)

    @staticmethod
    def _reacquire_valid(fit: TrajectoryFit) -> bool:
        return bool(
            fit.found
            and fit.conf >= 0.65
            and fit.n_bands >= 3
            and not fit.disconnected
        )

    def _stable_straight(self, fit: TrajectoryFit, features: LineFeatures) -> bool:
        return bool(
            self._reacquire_valid(fit)
            and abs(fit.e0) <= min(0.28, self.cfg.corner_e_threshold)
            and abs(fit.theta) <= min(0.16, self.cfg.corner_theta_threshold)
            and features.branch_left is None
            and features.branch_right is None
        )

    def _motion_delta(self, now: float, linear: float, angular: float) -> tuple[float, float]:
        if self.last_now is None:
            self.last_now = now
            return 0.0, 0.0
        dt = max(0.0, min(self.cfg.max_motion_dt_sec, now - self.last_now))
        self.last_now = now
        return max(0.0, linear) * dt, abs(angular) * dt
