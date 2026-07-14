from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fallback_config import ParkingMotionConfig, ParkingTriggerConfig
from .state_machine import MotionCommand, RaceState


@dataclass(frozen=True, slots=True)
class ParkingObservation:
    candidate: bool
    confirmed: bool
    confidence: float
    reason: str


class ParkingTriggerDetector:
    """Detect a rectangular parking bay from edge support in a fixed mask ROI."""

    def __init__(self, cfg: ParkingTriggerConfig) -> None:
        self.cfg = cfg
        self._streak = 0

    def reset(self) -> None:
        self._streak = 0

    def analyze(self, mask: np.ndarray) -> ParkingObservation:
        if not self.cfg.enabled:
            self.reset()
            return ParkingObservation(False, False, 0.0, "trigger_disabled")
        if mask.ndim != 2 or mask.size == 0:
            self.reset()
            return ParkingObservation(False, False, 0.0, "invalid_mask")

        height, width = mask.shape
        x0f, y0f, x1f, y1f = self.cfg.roi
        x0, x1 = int(round(x0f * width)), int(round(x1f * width))
        y0, y1 = int(round(y0f * height)), int(round(y1f * height))
        roi = mask[y0:y1, x0:x1]
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            self.reset()
            return ParkingObservation(False, False, 0.0, "trigger_roi_too_small")

        binary = roi > 0
        border = max(2, int(round(min(roi.shape) * self.cfg.border_frac)))
        top = float(binary[:border, :].mean())
        bottom = float(binary[-border:, :].mean())
        left = float(binary[:, :border].mean())
        right = float(binary[:, -border:].mean())
        edges = (top, bottom, left, right)
        edge_hits = sum(value >= self.cfg.edge_occupancy_min for value in edges)
        interior = binary[border:-border, border:-border]
        interior_occupancy = float(interior.mean()) if interior.size else 1.0
        candidate = bool(
            edge_hits >= self.cfg.required_edges
            and interior_occupancy <= self.cfg.interior_occupancy_max
        )
        self._streak = self._streak + 1 if candidate else 0
        confirmed = self._streak >= self.cfg.confirm_frames
        confidence = min(1.0, sum(sorted(edges, reverse=True)[: self.cfg.required_edges]) / self.cfg.required_edges)
        reason = "parking_rectangle_confirmed" if confirmed else (
            "parking_rectangle_candidate" if candidate else "parking_rectangle_absent"
        )
        return ParkingObservation(candidate, confirmed, confidence, reason)


class TimedParkingController:
    """Bounded calibration sequence; it is not a measured-distance controller."""

    def __init__(self, cfg: ParkingMotionConfig) -> None:
        self.cfg = cfg

    def stage_command(self, elapsed_sec: float) -> tuple[MotionCommand, bool]:
        complete = elapsed_sec + 1e-9 >= self.cfg.stage_sec
        if complete:
            return self.stop("parking_stage_complete"), True
        return MotionCommand(
            self.cfg.stage_v_mps,
            self.cfg.stage_w_radps,
            "parking_stage",
            RaceState.TRACK,
        ), False

    def reverse_command(self, elapsed_sec: float) -> tuple[MotionCommand, bool, str | None]:
        if elapsed_sec + 1e-9 >= self.cfg.reverse_hard_max_sec:
            return self.stop("parking_reverse_hard_timeout"), False, "parking_reverse_hard_timeout"
        if elapsed_sec + 1e-9 >= self.cfg.reverse_sec:
            return self.stop("parking_reverse_complete"), True, None
        return MotionCommand(
            -abs(self.cfg.reverse_v_mps),
            self.cfg.reverse_w_radps,
            "parking_reverse_unvalidated",
            RaceState.TRACK,
        ), False, None

    @staticmethod
    def stop(reason: str) -> MotionCommand:
        return MotionCommand(0.0, 0.0, reason, RaceState.STOPPED)
