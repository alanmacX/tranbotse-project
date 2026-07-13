from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .config import PathMemoryConfig
from .vision import PathPoint, TrajectoryFit


@dataclass(frozen=True, slots=True)
class MotionSample:
    linear: float
    angular: float
    source: str


@dataclass(frozen=True, slots=True)
class PathMemoryStatus:
    active: bool
    reason: str
    snapshot_age_sec: float = 0.0
    snapshot_count: int = 0
    point_count: int = 0


@dataclass(slots=True)
class _Snapshot:
    points: np.ndarray
    observed_at: float
    fit: TrajectoryFit


def read_motion_sample(bot: object, fallback_v: float, fallback_w: float) -> MotionSample:
    """Read cached encoder-derived velocity, with a bounded command fallback."""

    getter = getattr(bot, "get_motion_data", None)
    if callable(getter):
        try:
            value = getter()
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                linear, angular = float(value[0]), float(value[1])
                if math.isfinite(linear) and math.isfinite(angular) and abs(linear) <= 0.5 and abs(angular) <= 3.0:
                    return MotionSample(linear, angular, "measured")
        except Exception:
            pass
    return MotionSample(float(fallback_v), float(fallback_w), "command_fallback")


def image_path_to_axle(
    path: Iterable[PathPoint],
    *,
    image_width: int,
    image_height: int,
    center_x: float,
    pixels_per_meter: float,
    camera_to_axle_m: float,
) -> np.ndarray:
    """Convert an ordered BEV image path into (forward, left) axle coordinates."""

    if pixels_per_meter <= 0.0:
        return np.empty((0, 2), dtype=np.float64)
    points = [
        (
            (float(image_height - 1) - point.y) / pixels_per_meter + camera_to_axle_m,
            -(point.x - center_x) / pixels_per_meter,
        )
        for point in path
        if 0.0 <= point.x < image_width and 0.0 <= point.y < image_height
    ]
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _resample(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    spacing = max(0.002, float(spacing))
    chunks = [points[0]]
    for p0, p1 in zip(points, points[1:]):
        distance = float(np.linalg.norm(p1 - p0))
        steps = max(1, int(math.ceil(distance / spacing)))
        for index in range(1, steps + 1):
            chunks.append(p0 + (p1 - p0) * (index / steps))
    return np.asarray(chunks, dtype=np.float64)


def _advance_points(points: np.ndarray, ds: float, dyaw: float) -> np.ndarray:
    """Express old axle-frame points in the new axle frame after an SE(2) motion."""

    shifted = points - np.asarray([ds, 0.0], dtype=np.float64)
    c, s = math.cos(dyaw), math.sin(dyaw)
    rotation_inverse = np.asarray([[c, s], [-s, c]], dtype=np.float64)
    return shifted @ rotation_inverse.T


def _point_at_arc(points: np.ndarray, distance: float) -> np.ndarray:
    if len(points) == 1 or distance <= 0.0:
        return points[0]
    remaining = distance
    for p0, p1 in zip(points, points[1:]):
        segment = float(np.linalg.norm(p1 - p0))
        if segment >= remaining and segment > 1e-9:
            return p0 + (p1 - p0) * (remaining / segment)
        remaining -= segment
    return points[-1]


def _signed_curvature(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    a = float(np.linalg.norm(p1 - p0))
    b = float(np.linalg.norm(p2 - p1))
    c = float(np.linalg.norm(p2 - p0))
    denom = a * b * c
    if denom <= 1e-9:
        return 0.0
    v1, v2 = p1 - p0, p2 - p1
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    return 2.0 * cross / denom


class ShortHorizonPathMemory:
    """Keep recent visual paths in the moving axle frame and fit one for control."""

    def __init__(self, cfg: PathMemoryConfig) -> None:
        self.cfg = cfg
        self._snapshots: list[_Snapshot] = []
        self._last_now: float | None = None

    def reset(self) -> None:
        self._snapshots.clear()
        self._last_now = None

    def step(
        self,
        observed_points: np.ndarray,
        source_fit: TrajectoryFit,
        *,
        now: float,
        linear_velocity: float,
        angular_velocity: float,
        lateral_half_width_m: float,
    ) -> tuple[TrajectoryFit | None, PathMemoryStatus]:
        if not self.cfg.enabled:
            self.reset()
            return None, PathMemoryStatus(False, "disabled")

        self._propagate(now, linear_velocity, angular_velocity)
        self._prune(now)

        if len(observed_points) >= 2 and source_fit.found and source_fit.conf > 0.0:
            sampled = _resample(observed_points, self.cfg.sample_spacing_m)
            self._snapshots.append(_Snapshot(sampled, now, source_fit))
            self._snapshots = self._snapshots[-max(1, self.cfg.max_snapshots) :]

        snapshot = self._select_snapshot(now)
        if snapshot is None:
            return None, PathMemoryStatus(False, "no_metric_path", snapshot_count=len(self._snapshots))

        fit, point_count = self._fit_snapshot(snapshot, now, lateral_half_width_m)
        if fit is None:
            return None, PathMemoryStatus(False, "path_exhausted", snapshot_count=len(self._snapshots))
        age = max(0.0, now - snapshot.observed_at)
        return fit, PathMemoryStatus(True, "buffered", age, len(self._snapshots), point_count)

    def _propagate(self, now: float, linear: float, angular: float) -> None:
        if self._last_now is None:
            self._last_now = now
            return
        dt = max(0.0, now - self._last_now)
        self._last_now = now
        if dt <= 0.0:
            return
        if dt > self.cfg.max_motion_dt_sec:
            self._snapshots.clear()
            return
        ds, dyaw = linear * dt, angular * dt
        for snapshot in self._snapshots:
            snapshot.points = _advance_points(snapshot.points, ds, dyaw)

    def _prune(self, now: float) -> None:
        kept = []
        for snapshot in self._snapshots:
            if now - snapshot.observed_at > self.cfg.max_age_sec:
                continue
            points = snapshot.points
            useful = (
                (points[:, 0] >= -self.cfg.behind_tolerance_m)
                & (points[:, 0] <= self.cfg.max_horizon_m)
                & (np.linalg.norm(points, axis=1) <= self.cfg.max_horizon_m * 1.5)
            )
            snapshot.points = points[useful]
            if len(snapshot.points) >= 2:
                kept.append(snapshot)
        self._snapshots = kept

    def _select_snapshot(self, now: float) -> _Snapshot | None:
        if not self._snapshots:
            return None
        # Prefer the snapshot that currently reaches closest to the axle. This is
        # usually an older observation propagated through the camera blind zone.
        # A small age term lets fresh vision win when geometric coverage is equal.
        def score(snapshot: _Snapshot) -> float:
            nearest = float(np.min(np.linalg.norm(snapshot.points, axis=1)))
            age = max(0.0, now - snapshot.observed_at)
            return nearest + 0.02 * age

        return min(self._snapshots, key=score)

    def _fit_snapshot(
        self,
        snapshot: _Snapshot,
        now: float,
        lateral_half_width_m: float,
    ) -> tuple[TrajectoryFit | None, int]:
        points = snapshot.points
        valid_indices = np.flatnonzero(points[:, 0] >= -self.cfg.behind_tolerance_m)
        if valid_indices.size == 0:
            return None, 0
        nearest_local = int(np.argmin(np.linalg.norm(points[valid_indices], axis=1)))
        nearest_index = int(valid_indices[nearest_local])
        path = points[nearest_index:]
        if len(path) < 2:
            return None, len(path)

        # The camera cannot observe the strip between its ground projection and
        # the axle. Account for that spatial gap when measuring lookahead; without
        # this bridge, translating every point by camera_to_axle would have no
        # effect because lookahead would incorrectly start at the first image point.
        if path[0, 0] > 0.002:
            axle_projection = np.asarray([[0.0, path[0, 1]]], dtype=np.float64)
            path = np.vstack((axle_projection, path))

        half_width = max(0.01, lateral_half_width_m)
        near = path[0]
        target = _point_at_arc(path, self.cfg.lookahead_m)
        heading_point = _point_at_arc(path, self.cfg.heading_lookahead_m)
        e0 = float(np.clip(-near[1] / half_width, -1.0, 1.0))
        e_look = float(np.clip(-target[1] / half_width, -1.0, 1.0))
        heading_delta = heading_point - near
        theta = float(math.atan2(-heading_delta[1], max(1e-6, heading_delta[0])))

        middle = _point_at_arc(path, self.cfg.lookahead_m * 0.5)
        curvature = _signed_curvature(near, middle, target)
        kappa = float(np.clip(-curvature * max(self.cfg.lookahead_m, 0.01), -1.0, 1.0))

        age = max(0.0, now - snapshot.observed_at)
        age_conf = max(0.0, 1.0 - age / max(self.cfg.max_age_sec, 1e-6))
        conf = float(np.clip(snapshot.fit.conf * max(0.25, age_conf), 0.0, 1.0))
        return TrajectoryFit(
            found=True,
            e0=e0,
            e_look=e_look,
            theta=theta,
            kappa=kappa,
            conf=conf,
            n_bands=snapshot.fit.n_bands,
            quadratic=snapshot.fit.quadratic,
            disconnected=snapshot.fit.disconnected,
            preview_dir=snapshot.fit.preview_dir,
            preview_e=snapshot.fit.preview_e,
            preview_theta=snapshot.fit.preview_theta,
            preview_conf=snapshot.fit.preview_conf,
            path_memory=True,
        ), len(path)
