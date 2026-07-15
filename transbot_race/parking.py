from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .fallback_config import (
    LeftTurnConfig,
    ParkingBayConfig,
    ParkingMotionConfig,
    TerminalLineConfig,
)
from .state_machine import MotionCommand, RaceState
from .vision import LineFeatures


@dataclass(frozen=True, slots=True)
class TerminalLineObservation:
    candidate: bool
    confirmed: bool
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class ParkingObservation:
    candidate: bool
    confirmed: bool
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class LeftTurnResult:
    command: MotionCommand
    complete: bool
    fault: str | None
    turned_rad: float


class TerminalLineDetector:
    """Confirm one wide terminal bar joined to the incoming tracked line."""

    def __init__(self, cfg: TerminalLineConfig) -> None:
        self.cfg = cfg
        self._streak = 0

    def reset(self) -> None:
        self._streak = 0

    def analyze(self, mask: np.ndarray, features: LineFeatures) -> TerminalLineObservation:
        if not self.cfg.enabled:
            return self._absent("terminal_line_disabled")
        if mask.ndim != 2 or mask.size == 0 or not features.found:
            return self._absent("terminal_line_invalid_input")

        height, width = mask.shape
        x0f, y0f, x1f, y1f = self.cfg.roi
        x0, x1 = int(round(x0f * width)), int(round(x1f * width))
        y0, y1 = int(round(y0f * height)), int(round(y1f * height))
        binary = mask > 0
        band_h = max(2, int(round(height * self.cfg.row_band_frac)))
        row_support = np.zeros(max(0, y1 - y0), dtype=float)
        for offset, y in enumerate(range(y0, y1)):
            lo, hi = max(y0, y - band_h // 2), min(y1, y + band_h // 2 + 1)
            row_support[offset] = np.any(binary[lo:hi, x0:x1], axis=0).mean()
        wide = row_support >= self.cfg.horizontal_occupancy_min
        groups = self._groups(wide)
        if len(groups) != self.cfg.max_wide_row_groups:
            return self._absent("terminal_line_rectangle_or_absent")

        start, end = groups[0]
        bar_y = y0 + (start + end) // 2
        bar_rows = binary[max(y0, bar_y - band_h):min(y1, bar_y + band_h + 1), x0:x1]
        occupied_cols = np.flatnonzero(np.any(bar_rows, axis=0))
        if occupied_cols.size == 0:
            return self._absent("terminal_line_absent")
        bar_width = int(occupied_cols[-1] - occupied_cols[0] + 1)
        required_width = max(
            (x1 - x0) * self.cfg.min_width_ratio,
            features.line_width_px * self.cfg.min_width_to_line,
        )
        if bar_width < required_width:
            return self._absent("terminal_line_too_narrow")

        center_x = int(round(features.bottom.cx if features.bottom is not None else width / 2.0))
        stem_half = max(2, int(round(features.line_width_px * self.cfg.stem_half_width_ratio)))
        sx0, sx1 = max(0, center_x - stem_half), min(width, center_x + stem_half + 1)
        below = binary[min(height, bar_y + band_h):height, sx0:sx1]
        above = binary[0:max(0, bar_y - band_h), sx0:sx1]
        stem_support = float(below.mean()) if below.size else 0.0
        above_support = float(above.mean()) if above.size else 0.0
        if stem_support < self.cfg.stem_occupancy_min:
            return self._absent("terminal_line_missing_stem")
        if above_support > self.cfg.above_occupancy_max:
            return self._absent("terminal_line_continues")
        labels_count, labels = cv.connectedComponents(binary.astype(np.uint8), connectivity=8)
        if labels_count <= 1:
            return self._absent("terminal_line_disconnected")
        bar_labels = labels[max(y0, bar_y - band_h):min(y1, bar_y + band_h + 1), x0:x1]
        stem_labels = labels[min(height, bar_y + band_h):height, sx0:sx1]
        shared = set(np.unique(bar_labels[bar_labels > 0])).intersection(np.unique(stem_labels[stem_labels > 0]))
        if not shared:
            return self._absent("terminal_line_disconnected")

        self._streak += 1
        confirmed = self._streak >= self.cfg.confirm_frames
        confidence = min(1.0, 0.5 * bar_width / max(required_width, 1.0) + 0.5 * stem_support)
        return TerminalLineObservation(
            True,
            confirmed,
            confidence,
            "terminal_line_confirmed" if confirmed else "terminal_line_candidate",
        )

    def _absent(self, reason: str) -> TerminalLineObservation:
        self.reset()
        return TerminalLineObservation(False, False, 0.0, reason)

    @staticmethod
    def _groups(values: np.ndarray) -> list[tuple[int, int]]:
        indices = np.flatnonzero(values)
        if indices.size == 0:
            return []
        groups: list[tuple[int, int]] = []
        start = previous = int(indices[0])
        for raw in indices[1:]:
            current = int(raw)
            if current > previous + 1:
                groups.append((start, previous))
                start = current
            previous = current
        groups.append((start, previous))
        return groups


class ParkingBayDetector:
    """Detect a rectangular bay only for reverse-period visibility checks."""

    def __init__(self, cfg: ParkingBayConfig) -> None:
        self.cfg = cfg
        self._streak = 0

    def reset(self) -> None:
        self._streak = 0

    def analyze(self, mask: np.ndarray) -> ParkingObservation:
        if not self.cfg.enabled:
            self.reset()
            return ParkingObservation(False, False, 0.0, "parking_bay_disabled")
        if mask.ndim != 2 or mask.size == 0:
            self.reset()
            return ParkingObservation(False, False, 0.0, "parking_bay_invalid_mask")
        height, width = mask.shape
        x0f, y0f, x1f, y1f = self.cfg.roi
        x0, x1 = int(round(x0f * width)), int(round(x1f * width))
        y0, y1 = int(round(y0f * height)), int(round(y1f * height))
        roi = mask[y0:y1, x0:x1]
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            self.reset()
            return ParkingObservation(False, False, 0.0, "parking_bay_roi_too_small")
        binary = roi > 0
        border = max(2, int(round(min(roi.shape) * self.cfg.border_frac)))
        edges = (
            float(binary[:border, :].mean()),
            float(binary[-border:, :].mean()),
            float(binary[:, :border].mean()),
            float(binary[:, -border:].mean()),
        )
        edge_hits = sum(value >= self.cfg.edge_occupancy_min for value in edges)
        interior = binary[border:-border, border:-border]
        interior_occupancy = float(interior.mean()) if interior.size else 1.0
        candidate = edge_hits >= self.cfg.required_edges and interior_occupancy <= self.cfg.interior_occupancy_max
        self._streak = self._streak + 1 if candidate else 0
        confirmed = self._streak >= self.cfg.confirm_frames
        confidence = min(1.0, sum(sorted(edges, reverse=True)[:self.cfg.required_edges]) / self.cfg.required_edges)
        reason = "parking_bay_confirmed" if confirmed else ("parking_bay_candidate" if candidate else "parking_bay_absent")
        return ParkingObservation(bool(candidate), confirmed, confidence, reason)


class LeftTurnController:
    def __init__(self, cfg: LeftTurnConfig) -> None:
        self.cfg = cfg
        self.started_at = 0.0
        self.turned_rad = 0.0

    def reset(self, now: float) -> None:
        self.started_at = now
        self.turned_rad = 0.0

    def step(self, now: float) -> LeftTurnResult:
        elapsed = max(0.0, now - self.started_at)
        self.turned_rad = min(self.cfg.target_yaw_rad, elapsed * abs(self.cfg.w_radps))
        if elapsed + 1e-9 >= self.cfg.hard_timeout_sec:
            return LeftTurnResult(self.stop("left_turn_hard_timeout"), False, "left_turn_hard_timeout", self.turned_rad)
        calibrated_sec = self.cfg.target_yaw_rad / abs(self.cfg.w_radps)
        if elapsed + 1e-9 >= calibrated_sec:
            return LeftTurnResult(self.stop("left_turn_complete"), True, None, self.turned_rad)
        command = MotionCommand(0.0, abs(self.cfg.w_radps), "left_turn_timed_calibration", RaceState.TRACK)
        return LeftTurnResult(command, False, None, self.turned_rad)

    @staticmethod
    def stop(reason: str) -> MotionCommand:
        return MotionCommand(0.0, 0.0, reason, RaceState.STOPPED)


class TimedParkingController:
    def __init__(self, cfg: ParkingMotionConfig) -> None:
        self.cfg = cfg

    def reverse_command(self, elapsed_sec: float) -> tuple[MotionCommand, bool, str | None]:
        if elapsed_sec + 1e-9 >= self.cfg.reverse_hard_max_sec:
            return self.stop("parking_reverse_hard_timeout"), False, "parking_reverse_hard_timeout"
        if elapsed_sec + 1e-9 >= self.cfg.reverse_sec:
            return self.stop("parking_reverse_complete"), True, None
        return MotionCommand(-abs(self.cfg.reverse_v_mps), self.cfg.reverse_w_radps, "parking_reverse_unvalidated", RaceState.TRACK), False, None

    @staticmethod
    def stop(reason: str) -> MotionCommand:
        return MotionCommand(0.0, 0.0, reason, RaceState.STOPPED)
