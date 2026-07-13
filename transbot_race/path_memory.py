from __future__ import annotations

from dataclasses import dataclass
import math

import cv2 as cv
import numpy as np

from .config import GroundProjectionConfig, PathMemoryConfig
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


class GroundProjector:
    def __init__(self, cfg: GroundProjectionConfig) -> None:
        self.cfg = cfg

    @property
    def active(self) -> bool:
        return self.cfg.homography is not None and len(self.cfg.homography) == 9

    @property
    def matrix(self) -> np.ndarray | None:
        if not self.active:
            return None
        return np.asarray(self.cfg.homography, dtype=np.float64).reshape(3, 3)

    def project(self, points_px: np.ndarray) -> np.ndarray:
        if self.matrix is None or len(points_px) == 0:
            return np.empty((0, 2), dtype=np.float64)
        src = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv.perspectiveTransform(src, self.matrix).reshape(-1, 2)

    def bird_matrix(self) -> np.ndarray | None:
        h = self.matrix
        if h is None:
            return None
        ppm = self.cfg.pixels_per_meter
        cx = self.cfg.bird_width_px / 2.0
        ground_to_bird = np.asarray([[0.0, -ppm, cx], [-ppm, 0.0, self.cfg.bird_height_px - 1.0], [0.0, 0.0, 1.0]])
        return ground_to_bird @ h

    def warp(self, crop: np.ndarray) -> np.ndarray:
        matrix = self.bird_matrix()
        if matrix is None:
            return crop.copy()
        return cv.warpPerspective(crop, matrix, (self.cfg.bird_width_px, self.cfg.bird_height_px))


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
            visible = bool(fit.found and fit.conf >= 0.48 and fit.n_bands >= 2)
            if ready and visible:
                self.state = "cooldown"
                self.clear_frames = 0
            elif self.turned_rad >= self.target_angle_rad:
                self.state = "seeking"
        elif self.state == "seeking":
            self.turned_rad += turned
            visible = bool(fit.found and fit.conf >= 0.48 and fit.n_bands >= 2)
            if visible or self.turned_rad >= self.target_angle_rad + self.cfg.corner_search_extra_rad:
                self.state = "cooldown"
                self.clear_frames = 0
        elif self.state == "cooldown":
            self.clear_frames = self.clear_frames + 1 if direction == 0 else 0
            if self.clear_frames >= 3:
                self.state, self.candidate_dir, self.confirm = "armed", 0, 0

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
        status = PathStrategyStatus(
            True, "corner_event", self.state, 0.0,
            self.candidate_dir, 0,
        )
        return CornerCommandResult(command_v, command_w, status)

    def _motion_delta(self, now: float, linear: float, angular: float) -> tuple[float, float]:
        if self.last_now is None:
            self.last_now = now
            return 0.0, 0.0
        dt = max(0.0, min(self.cfg.max_motion_dt_sec, now - self.last_now))
        self.last_now = now
        return max(0.0, linear) * dt, abs(angular) * dt


class RollingPathPursuit:
    def __init__(self, cfg: PathMemoryConfig, mode: str) -> None:
        self.cfg = cfg
        self.mode = mode
        self.points = np.empty((0, 2), dtype=np.float64)
        self.last_now: float | None = None

    def step(self, observed: np.ndarray, source_fit: TrajectoryFit, now: float, linear: float, angular: float) -> tuple[TrajectoryFit, PathStrategyStatus]:
        self._advance(now, linear, angular)
        if len(observed):
            observed = observed[np.isfinite(observed).all(axis=1)]
            if len(observed):
                near_x = float(np.min(observed[:, 0]))
                blind = self.points[(self.points[:, 0] >= -0.03) & (self.points[:, 0] < near_x)] if len(self.points) else self.points
                self.points = np.vstack((blind, observed))[-self.cfg.path_max_points:]
        self.points = self.points[(self.points[:, 0] >= -0.03) & (self.points[:, 0] <= 0.8)] if len(self.points) else self.points
        candidates = self.points[self.points[:, 0] > 0.0]
        if len(candidates) == 0:
            return source_fit, PathStrategyStatus(False, self.mode, "no_ground_path")
        distances = np.linalg.norm(candidates, axis=1)
        target = candidates[int(np.argmin(np.abs(distances - self.cfg.lookahead_m)))]
        tx, ty = float(target[0]), float(target[1])
        e_look = max(-1.0, min(1.0, -ty / max(self.cfg.lateral_half_width_m, 1e-3)))
        theta = math.atan2(-ty, max(tx, 1e-4))
        curvature = -2.0 * ty / max(tx * tx + ty * ty, 1e-4)
        fit = TrajectoryFit(
            found=True, e0=source_fit.e0, e_look=e_look, theta=theta,
            kappa=max(-1.0, min(1.0, curvature * self.cfg.lookahead_m)),
            conf=max(source_fit.conf, 0.5), n_bands=source_fit.n_bands,
            path_memory=True,
        )
        return fit, PathStrategyStatus(True, self.mode, "tracking", point_count=len(self.points), target=(tx, ty))

    def _advance(self, now: float, linear: float, angular: float) -> None:
        if self.last_now is None:
            self.last_now = now
            return
        dt = now - self.last_now
        self.last_now = now
        if dt <= 0.0 or dt > self.cfg.max_motion_dt_sec:
            if dt > self.cfg.max_motion_dt_sec:
                self.points = np.empty((0, 2), dtype=np.float64)
            return
        if not len(self.points):
            return
        shifted = self.points - np.asarray([max(0.0, linear) * dt, 0.0])
        yaw = angular * dt
        c, s = math.cos(yaw), math.sin(yaw)
        self.points = shifted @ np.asarray([[c, -s], [s, c]], dtype=np.float64)


def path_pixels(features: LineFeatures) -> np.ndarray:
    return np.asarray([(point.x, point.y) for point in features.path], dtype=np.float64)


def bird_path_to_ground(features: LineFeatures, width: int, height: int, ppm: float, axle_offset: float) -> np.ndarray:
    return np.asarray([
        ((height - 1.0 - point.y) / ppm + axle_offset, -(point.x - width / 2.0) / ppm)
        for point in features.path
    ], dtype=np.float64).reshape(-1, 2)
