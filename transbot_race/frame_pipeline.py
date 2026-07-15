from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .config import RaceConfig
from .control import ArbitrationResult, CommandArbiter, StopCause
from .fixed_course import FixedCourseController, FixedCourseFrameContext, FixedCourseStep
from .geometry import apply_occlusion, band_is_occluded
from .path_memory import MotionSample
from .state_machine import MotionCommand
from .vision import (
    LineFeatures,
    TrajectoryFit,
    _band_bounds,
    fit_line_trajectory,
    preprocess_blackline,
    scan_line_features,
)


@dataclass(frozen=True, slots=True)
class FrameTimings:
    ordinary_vision_ms: float
    stage_ms: float
    arbiter_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class FramePipelineResult:
    crop: np.ndarray
    crop_rect: tuple[int, int, int, int]
    track_center: float
    mask: np.ndarray
    features: LineFeatures
    ordinary_fit: TrajectoryFit
    stage: FixedCourseStep
    arbitration: ArbitrationResult
    timings: FrameTimings

    def __post_init__(self) -> None:
        if self.stage.candidate_producer.value != self.stage.control_owner.value:
            raise ValueError("candidate producer/owner mismatch in frame pipeline")
        if self.arbitration.owner != self.stage.control_owner:
            raise ValueError("arbiter owner differs from stage owner")
        if self.arbitration.candidate != self.stage.candidate:
            raise ValueError("arbiter did not receive the stage candidate")


def crop_frame(frame: np.ndarray, cfg: RaceConfig):
    x0, y0, x1, y1 = cfg.camera.crop
    ex0 = max(0, x0 - cfg.camera.expand_left_px)
    ex1 = min(frame.shape[1], x1 + cfg.camera.expand_right_px)
    track_center = ((x0 + x1) / 2.0) - ex0
    return frame[y0:y1, ex0:ex1], (ex0, y0, ex1, y1), track_center


def occluded_band_indices(height: int, width: int, cfg: RaceConfig) -> frozenset[int]:
    if not cfg.occlusion.enabled or not cfg.occlusion.rects:
        return frozenset()
    return frozenset(
        index
        for index in range(cfg.vision.band_count)
        if band_is_occluded(
            *_band_bounds(index, height, cfg.vision.band_count),
            width,
            cfg.occlusion,
        )
    )


class FixedCourseFramePipeline:
    """Pure single-frame course pipeline with no camera or motor dependency."""

    def __init__(
        self,
        cfg: RaceConfig,
        *,
        course: FixedCourseController | None = None,
        arbiter: CommandArbiter | None = None,
    ) -> None:
        self.cfg = cfg
        self.course = course or FixedCourseController(cfg)
        self.arbiter = arbiter or CommandArbiter(
            max_v=cfg.tracker.v_max,
            max_w=max(
                cfg.tracker.max_w,
                cfg.tracker.w_search,
                cfg.tracker.w_pivot,
                cfg.tracker.preview_turn_w,
                cfg.path_memory.corner_replay_max_w,
                cfg.path_memory.corner_align_max_w,
                cfg.path_memory.roundabout_replay_max_w,
            ),
        )
        self._occluded: frozenset[int] | None = None

    def clear_route_loss(self) -> bool:
        if (
            self.arbiter.latched_cause == StopCause.ROUTE_LOST
            and self.course.clear_route_loss()
        ):
            self.arbiter.clear_stop(StopCause.ROUTE_LOST)
            return True
        return False

    def step(
        self,
        frame: np.ndarray,
        *,
        now: float,
        motion: MotionSample,
        last_command: MotionCommand,
        capture_geometry_debug: bool = False,
    ) -> FramePipelineResult:
        started = time.perf_counter()
        crop, rect, track_center = crop_frame(frame, self.cfg)
        trim = int(self.cfg.camera.control_bottom_trim_px)
        if trim:
            crop = crop[:-trim]
            rect = (rect[0], rect[1], rect[2], rect[3] - trim)
        mask = apply_occlusion(
            preprocess_blackline(crop, self.cfg.vision, anchor_x=track_center),
            self.cfg.occlusion,
        )
        if self._occluded is None:
            self._occluded = occluded_band_indices(crop.shape[0], crop.shape[1], self.cfg)
        features = scan_line_features(mask, self.cfg.vision, crop_center=track_center)
        ordinary_fit = fit_line_trajectory(
            features,
            self.cfg.vision,
            crop_center=track_center,
            crop_width=crop.shape[1],
            lookahead_frac=self.cfg.tracker.lookahead_frac,
            occluded_band_indices=self._occluded,
        )
        ordinary_done = time.perf_counter()

        stage = self.course.step(FixedCourseFrameContext(
            frame=frame,
            now=now,
            visual_fit=ordinary_fit,
            features=features,
            motion=motion,
            last_command=last_command,
            frame_center_x=rect[0] + track_center,
            control_width=crop.shape[1],
            capture_geometry_debug=capture_geometry_debug,
        ))
        stage_done = time.perf_counter()

        arbitration = self.arbiter.resolve(
            stage.candidate,
            owner=stage.control_owner,
            stop_cause=stage.stop_cause,
            transition_event=stage.transition_event,
        )
        finished = time.perf_counter()
        return FramePipelineResult(
            crop,
            rect,
            track_center,
            mask,
            features,
            ordinary_fit,
            stage,
            arbitration,
            FrameTimings(
                (ordinary_done - started) * 1000.0,
                (stage_done - ordinary_done) * 1000.0,
                (finished - stage_done) * 1000.0,
                (finished - started) * 1000.0,
            ),
        )
