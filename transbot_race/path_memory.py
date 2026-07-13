from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from .config import PathMemoryConfig
from .vision import TrajectoryFit


@dataclass(frozen=True, slots=True)
class MotionSample:
    linear: float
    angular: float
    source: str


@dataclass(frozen=True, slots=True)
class PathMemoryStatus:
    active: bool
    reason: str
    queue_count: int = 0
    remaining_m: float = 0.0
    released_age_sec: float = 0.0


@dataclass(slots=True)
class _QueuedFit:
    fit: TrajectoryFit
    captured_at: float
    remaining_m: float


def read_motion_sample(bot: object, fallback_v: float, fallback_w: float) -> MotionSample:
    """Read encoder velocity, rejecting missing/stale zero reports while moving."""

    getter = getattr(bot, "get_motion_data", None)
    if callable(getter):
        try:
            value = getter()
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                linear, angular = float(value[0]), float(value[1])
                valid = (
                    math.isfinite(linear)
                    and math.isfinite(angular)
                    and abs(linear) <= 0.5
                    and abs(angular) <= 3.0
                )
                stale_zero = abs(fallback_v) > 0.01 and abs(linear) < 0.002
                if valid and not stale_zero:
                    return MotionSample(linear, angular, "measured")
        except Exception:
            pass
    return MotionSample(float(fallback_v), float(fallback_w), "command_fallback")


class DistanceDelayPathMemory:
    """Delay preview steering by travelled distance while keeping near e0 live."""

    def __init__(self, cfg: PathMemoryConfig) -> None:
        self.cfg = cfg
        self._queue: deque[_QueuedFit] = deque()
        self._released: _QueuedFit | None = None
        self._released_at: float | None = None
        self._last_now: float | None = None

    def reset(self) -> None:
        self._queue.clear()
        self._released = None
        self._released_at = None
        self._last_now = None

    def step(
        self,
        visual_fit: TrajectoryFit,
        *,
        near_e0: float,
        now: float,
        linear_velocity: float,
    ) -> tuple[TrajectoryFit | None, PathMemoryStatus]:
        if not self.cfg.enabled:
            self.reset()
            return None, PathMemoryStatus(False, "disabled")

        self._advance(now, linear_velocity)
        self._prune(now)
        if visual_fit.found and visual_fit.conf > 0.0:
            self._queue.append(_QueuedFit(
                fit=visual_fit,
                captured_at=now,
                remaining_m=max(0.0, self.cfg.effective_camera_to_axle_m),
            ))
            while len(self._queue) > max(1, self.cfg.max_queue_frames):
                self._queue.popleft()

        while self._queue and self._queue[0].remaining_m <= 1e-6:
            self._released = self._queue.popleft()
            self._released_at = now

        delayed = self._released
        remaining = self._queue[0].remaining_m if self._queue else 0.0
        if delayed is None:
            # Before the first camera-to-axle interval has elapsed, keep only the
            # live near-field correction and suppress preview-induced turning.
            return TrajectoryFit(
                found=visual_fit.found,
                e0=near_e0,
                e_look=near_e0,
                theta=0.0,
                kappa=0.0,
                conf=visual_fit.conf,
                n_bands=visual_fit.n_bands,
                disconnected=visual_fit.disconnected,
                path_memory=False,
            ), PathMemoryStatus(True, "filling", len(self._queue), remaining, 0.0)

        released_at = now if self._released_at is None else self._released_at
        age = max(0.0, now - released_at)
        delayed_conf = delayed.fit.conf * max(0.25, 1.0 - age / max(self.cfg.max_age_sec, 1e-6))
        found = visual_fit.found or delayed_conf > 0.0
        conf = max(visual_fit.conf if visual_fit.found else 0.0, delayed_conf)
        return TrajectoryFit(
            found=found,
            e0=near_e0,
            e_look=delayed.fit.e_look,
            theta=delayed.fit.theta,
            kappa=delayed.fit.kappa,
            conf=conf,
            n_bands=max(visual_fit.n_bands, delayed.fit.n_bands),
            quadratic=delayed.fit.quadratic,
            disconnected=visual_fit.disconnected and delayed.fit.disconnected,
            preview_dir=delayed.fit.preview_dir,
            preview_e=delayed.fit.preview_e,
            preview_theta=delayed.fit.preview_theta,
            preview_conf=delayed.fit.preview_conf,
            path_memory=True,
        ), PathMemoryStatus(True, "distance_delay", len(self._queue), remaining, age)

    def _advance(self, now: float, linear_velocity: float) -> None:
        if self._last_now is None:
            self._last_now = now
            return
        dt = max(0.0, now - self._last_now)
        self._last_now = now
        if dt <= 0.0:
            return
        if dt > self.cfg.max_motion_dt_sec:
            self._queue.clear()
            self._released = None
            self._released_at = None
            return
        travelled = max(0.0, linear_velocity) * dt
        for item in self._queue:
            item.remaining_m -= travelled

    def _prune(self, now: float) -> None:
        while self._queue and now - self._queue[0].captured_at > self.cfg.max_age_sec:
            self._queue.popleft()
        if self._released_at is not None and now - self._released_at > self.cfg.max_age_sec:
            self._released = None
            self._released_at = None
