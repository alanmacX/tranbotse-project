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


def _intent_direction(features: LineFeatures, fit: TrajectoryFit, cfg: PathMemoryConfig) -> int:
    left = features.branch_left is not None
    right = features.branch_right is not None
    if left and right:
        return 0
    if right:
        return 1
    if left:
        return -1
    if fit.preview_dir:
        return 1 if fit.preview_dir > 0 else -1
    uses_theta = abs(fit.theta) >= cfg.corner_theta_threshold
    signal = fit.theta if uses_theta else fit.e_look
    threshold = cfg.corner_theta_threshold if uses_theta else cfg.corner_e_threshold
    if abs(signal) < threshold:
        return 0
    return 1 if signal > 0 else -1


class CornerEventMargin:
    def __init__(self, cfg: PathMemoryConfig) -> None:
        self.cfg = cfg
        self.state = "armed"
        self.candidate_dir = 0
        self.confirm = 0
        self.remaining_m = 0.0
        self.baseline_e0 = 0.0
        self.last_now: float | None = None

    def step(self, fit: TrajectoryFit, features: LineFeatures, now: float, linear: float) -> tuple[TrajectoryFit, PathStrategyStatus]:
        travelled = self._travel(now, linear)
        direction = _intent_direction(features, fit, self.cfg)
        if self.state == "armed":
            if direction and direction == self.candidate_dir:
                self.confirm += 1
            else:
                self.candidate_dir, self.confirm = direction, int(direction != 0)
            if self.confirm >= max(1, self.cfg.corner_confirm_frames):
                self.state = "waiting"
                self.remaining_m = self.cfg.camera_to_axle_m
                self.baseline_e0 = max(-0.35, min(0.35, fit.e0))
        elif self.state == "waiting":
            self.remaining_m = max(0.0, self.remaining_m - travelled)
            if self.remaining_m <= 1e-6:
                self.state = "released"
        elif self.state == "released" and direction == 0:
            self.state, self.candidate_dir, self.confirm = "armed", 0, 0

        if self.state == "waiting":
            held = TrajectoryFit(found=True, e0=self.baseline_e0, e_look=self.baseline_e0, conf=max(fit.conf, 0.55), n_bands=fit.n_bands)
            return held, PathStrategyStatus(True, "corner_event", "waiting_margin", self.remaining_m, self.candidate_dir)
        return fit, PathStrategyStatus(True, "corner_event", self.state, 0.0, self.candidate_dir)

    def _travel(self, now: float, linear: float) -> float:
        if self.last_now is None:
            self.last_now = now
            return 0.0
        dt = max(0.0, min(self.cfg.max_motion_dt_sec, now - self.last_now))
        self.last_now = now
        return max(0.0, linear) * dt


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
