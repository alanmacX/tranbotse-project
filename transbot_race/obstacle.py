from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import cv2 as cv
import numpy as np

from .config import ObstacleConfig


class ObstacleStage(str, Enum):
    CLEAR = "clear"
    ENTRY = "entry"
    APPROACH = "approach"
    STOP = "stop"


class ObstacleState(str, Enum):
    DISARMED = "disarmed"
    CLEAR = "clear"
    SUSPECT = "suspect"
    ENTRY = "entry"
    APPROACH = "approach_slow"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class ObstacleCandidate:
    bbox: tuple[int, int, int, int]
    cue: str
    confidence: float
    edge_support: float
    bottom_frac: float
    width_frac: float
    stage: ObstacleStage


@dataclass(frozen=True, slots=True)
class ObstacleEvidence:
    candidate: ObstacleCandidate | None
    line_top_y: int | None
    expected_line_top_y: float | None
    horizontal_lines: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True, slots=True)
class ObstacleDecision:
    state: ObstacleState
    confidence: float
    stop_required: bool
    slow_required: bool
    evidence: ObstacleEvidence


def _stage(bottom_frac: float, width_frac: float, cfg: ObstacleConfig) -> ObstacleStage:
    if bottom_frac >= cfg.stop_bottom_frac or width_frac >= cfg.stop_width_frac:
        return ObstacleStage.STOP
    if bottom_frac >= cfg.approach_bottom_frac or width_frac >= cfg.approach_width_frac:
        return ObstacleStage.APPROACH
    return ObstacleStage.ENTRY


def _horizontal_lines(edges: np.ndarray) -> tuple[tuple[int, int, int, int], ...]:
    width = edges.shape[1]
    raw = cv.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(14, int(width * 0.075)),
        minLineLength=max(24, int(width * 0.155)), maxLineGap=max(6, int(width * 0.035)),
    )
    if raw is None:
        return ()
    lines = []
    for x0, y0, x1, y1 in raw.reshape(-1, 4):
        angle = abs(float(np.degrees(np.arctan2(int(y1) - int(y0), int(x1) - int(x0)))))
        if angle <= 12.0 or angle >= 168.0:
            lines.append((int(x0), int(y0), int(x1), int(y1)))
    return tuple(lines)


def _line_top(gray: np.ndarray, line_x: float) -> int | None:
    height, width = gray.shape
    x0 = max(0, int(round(line_x - width * 0.12)))
    x1 = min(width, int(round(line_x + width * 0.07)))
    threshold = min(60.0, max(38.0, float(np.percentile(gray, 15)) + 8.0))
    row_count = (gray[:, x0:x1] < threshold).sum(axis=1)
    present = (row_count >= max(3, int(width * 0.014))).astype(np.uint8) * 255
    present = cv.morphologyEx(present.reshape(-1, 1), cv.MORPH_CLOSE, np.ones((9, 1), np.uint8)).ravel() > 0
    runs = []
    start = None
    for index, on in enumerate([*present.tolist(), False]):
        if on and start is None:
            start = index
        elif not on and start is not None:
            runs.append((start, index))
            start = None
    return min((run[0] for run in runs if run[1] >= height - 3 and run[1] - run[0] >= 12), default=None)


def _edge_support(lines, bbox, line_x: float, corridor: int) -> float:
    bx, by, bw, bh = bbox
    best = 0.0
    for x0, y0, x1, y1 in lines:
        lo, hi = sorted((x0, x1))
        y_mid = (y0 + y1) / 2.0
        overlap = max(0, min(hi, bx + bw) - max(lo, bx))
        if lo <= line_x + corridor and hi >= line_x - corridor and by - 8 <= y_mid <= by + bh + 8:
            best = max(best, overlap / float(max(1, bw)))
    return float(min(1.0, best))


def _component_candidates(mask, lines, line_x: float, corridor: int, cue: str, cfg: ObstacleConfig):
    height, width = mask.shape
    count, _labels, stats, _centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for component_id in range(1, count):
        x, y, w, h, area = map(int, stats[component_id])
        fill = area / float(max(1, w * h))
        aspect = w / float(max(1, h))
        crosses = x <= line_x + corridor and x + w >= line_x - corridor
        if not (
            area >= width * height * 0.022 and w >= width * 0.24 and h >= height * 0.10
            and fill >= 0.34 and 0.65 <= aspect <= 7.0 and crosses and y < height * 0.88
        ):
            continue
        bbox = (x, y, w, h)
        edge = _edge_support(lines, bbox, line_x, corridor)
        area_score = float(np.clip(area / float(width * height * 0.16), 0.0, 1.0))
        width_score = float(np.clip(w / float(width * 0.55), 0.0, 1.0))
        confidence = min(0.98, 0.48 + 0.18 * area_score + 0.18 * width_score + 0.16 * edge)
        bottom_frac, width_frac = (y + h) / float(height), w / float(width)
        candidates.append(ObstacleCandidate(
            bbox, cue, confidence, edge, bottom_frac, width_frac, _stage(bottom_frac, width_frac, cfg)
        ))
    return candidates


def _neutral_edge_candidates(lab, lines, line_x: float, corridor: int, cfg: ObstacleConfig):
    height, width = lab.shape[:2]
    usable = []
    for x0, y0, x1, y1 in lines:
        lo, hi = sorted((x0, x1))
        if hi - lo >= width * 0.17 and lo <= line_x + corridor and hi >= line_x - corridor:
            usable.append((lo, hi, (y0 + y1) / 2.0))
    candidates = []
    for index, first in enumerate(usable):
        for second in usable[index + 1:]:
            top, bottom = sorted((first, second), key=lambda item: item[2])
            separation = bottom[2] - top[2]
            overlap = max(0.0, min(top[1], bottom[1]) - max(top[0], bottom[0]))
            x0, x1 = min(top[0], bottom[0]), max(top[1], bottom[1])
            w = x1 - x0
            if not (height * 0.10 <= separation <= height * 0.74 and overlap >= width * 0.15 and w >= width * 0.26):
                continue
            iy0, iy1 = int(max(0, top[2] + 3)), int(min(height, bottom[2] - 3))
            if iy1 <= iy0:
                continue
            inside = np.median(lab[iy0:iy1, int(x0):int(x1)], axis=(0, 1))
            rx0, rx1 = int(width * 0.71), int(width * 0.97)
            reference = np.median(lab[iy0:iy1, rx0:rx1], axis=(0, 1))
            contrast = float(np.linalg.norm(inside - reference))
            if contrast < 12.0:
                continue
            bbox = (int(x0), int(round(top[2])), int(w), int(round(separation)))
            width_frac, bottom_frac = w / float(width), bottom[2] / float(height)
            edge = min(1.0, overlap / float(max(1.0, min(top[1] - top[0], bottom[1] - bottom[0]))))
            confidence = min(0.95, 0.58 + 0.20 * min(1.0, contrast / 35.0) + 0.12 * min(1.0, width_frac / 0.55) + 0.10 * edge)
            candidates.append(ObstacleCandidate(
                bbox, "neutral_edges", confidence, edge, bottom_frac, width_frac,
                _stage(bottom_frac, width_frac, cfg),
            ))
    return candidates


def _occlusion_candidate(lines, line_top, expected_top, line_x, shape, corridor, cfg):
    if line_top is None or expected_top is None or line_top - expected_top < shape[0] * 0.10:
        return None
    height, width = shape
    supporting = []
    for x0, y0, x1, y1 in lines:
        lo, hi = sorted((x0, x1))
        y_mid, length = (y0 + y1) / 2.0, hi - lo
        if length >= width * 0.19 and lo <= line_x + corridor and hi >= line_x - corridor and y_mid <= line_top - 8:
            supporting.append((length, lo, hi, y_mid))
    if not supporting:
        return None
    length, x0, x1, top_y = max(supporting)
    bbox = (int(x0), int(round(top_y)), int(x1 - x0), int(line_top - top_y))
    width_frac, bottom_frac = (x1 - x0) / float(width), line_top / float(height)
    edge = float(np.clip(length / (width * 0.45), 0.0, 1.0))
    confidence = min(0.99, 0.58 + 0.24 * min(1.0, (line_top - expected_top) / (height * 0.30)) + 0.18 * edge)
    return ObstacleCandidate(
        bbox, "line_occlusion", confidence, edge, bottom_frac, width_frac,
        _stage(bottom_frac, width_frac, cfg),
    )


def detect_obstacle(crop_bgr: np.ndarray, line_x: float, expected_line_top: float | None, cfg: ObstacleConfig) -> ObstacleEvidence:
    height, width = crop_bgr.shape[:2]
    corridor = max(12, int(round(width * cfg.corridor_half_width_ratio)))
    blurred = cv.GaussianBlur(crop_bgr, (3, 3), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    lab = cv.cvtColor(cv.GaussianBlur(crop_bgr, (5, 5), 0), cv.COLOR_BGR2LAB).astype(np.float32)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    floor_like = saturation[(value > 35) & (value < 190)]
    floor_saturation = float(np.median(floor_like)) if floor_like.size else 24.0
    saturation_threshold = float(min(105.0, max(52.0, floor_saturation + 28.0)))

    chroma = ((saturation > saturation_threshold) & (value > 22) & (value < 205)).astype(np.uint8) * 255
    chroma = cv.morphologyEx(chroma, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    chroma = cv.morphologyEx(chroma, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (17, 9)))
    dark_threshold = min(48.0, max(28.0, float(np.percentile(gray, 18)) - 3.0))
    dark = (gray < dark_threshold).astype(np.uint8) * 255
    dark = cv.morphologyEx(dark, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    dark = cv.morphologyEx(dark, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (15, 9)))
    edges = cv.Canny(cv.GaussianBlur(gray, (5, 5), 0), 25, 75)
    lines = _horizontal_lines(edges)
    line_top = _line_top(gray, line_x)

    candidates = []
    candidates.extend(_component_candidates(chroma, lines, line_x, corridor, "chroma", cfg))
    candidates.extend(_component_candidates(dark, lines, line_x, corridor, "dark", cfg))
    candidates.extend(_neutral_edge_candidates(lab, lines, line_x, corridor, cfg))
    occlusion = _occlusion_candidate(lines, line_top, expected_line_top, line_x, (height, width), corridor, cfg)
    if occlusion is not None:
        candidates.append(occlusion)
    candidate = max(candidates, key=lambda item: item.confidence, default=None)
    if candidate is not None and candidate.confidence < cfg.min_confidence:
        candidate = None
    return ObstacleEvidence(candidate, line_top, expected_line_top, lines)


class ObstacleMonitor:
    def __init__(self, cfg: ObstacleConfig) -> None:
        self.cfg = cfg
        self._history: deque[bool] = deque(maxlen=3)
        self._line_top_history: deque[int] = deque(maxlen=9)
        self._candidate_hold_frames = 0

    def reset(self) -> None:
        self._history.clear()
        self._line_top_history.clear()
        self._candidate_hold_frames = 0

    @property
    def expected_line_top(self) -> float | None:
        if not self._line_top_history:
            return None
        return float(np.median(np.asarray(self._line_top_history)))

    def update(self, crop_bgr: np.ndarray, line_x: float, armed: bool) -> ObstacleDecision:
        if not self.cfg.enabled:
            evidence = ObstacleEvidence(None, None, None, ())
            return ObstacleDecision(ObstacleState.DISARMED, 0.0, False, False, evidence)
        evidence = detect_obstacle(crop_bgr, line_x, self.expected_line_top, self.cfg)
        candidate = evidence.candidate
        effective_armed = armed or self._candidate_hold_frames > 0
        if not effective_armed:
            self._history.clear()
            if candidate is None and evidence.line_top_y is not None:
                self._line_top_history.append(evidence.line_top_y)
            return ObstacleDecision(ObstacleState.DISARMED, 0.0 if candidate is None else candidate.confidence, False, False, evidence)

        if candidate is not None:
            self._candidate_hold_frames = max(1, self.cfg.candidate_hold_frames)
        else:
            self._candidate_hold_frames = max(0, self._candidate_hold_frames - 1)
        if candidate is None and evidence.line_top_y is not None:
            self._line_top_history.append(evidence.line_top_y)
        self._history.append(candidate is not None)
        confirmed = sum(self._history) >= 2
        if confirmed and candidate is not None and candidate.stage is ObstacleStage.STOP:
            state = ObstacleState.STOP
        elif candidate is not None and not confirmed:
            state = ObstacleState.SUSPECT
        elif confirmed and candidate is not None and candidate.stage is ObstacleStage.APPROACH:
            state = ObstacleState.APPROACH
        elif confirmed and candidate is not None:
            state = ObstacleState.ENTRY
        else:
            state = ObstacleState.CLEAR
        return ObstacleDecision(
            state, 0.0 if candidate is None else candidate.confidence,
            state is ObstacleState.STOP, state is ObstacleState.APPROACH, evidence,
        )


def draw_obstacle_overlay(crop_bgr: np.ndarray, decision: ObstacleDecision, line_x: float, cfg: ObstacleConfig) -> np.ndarray:
    out = crop_bgr.copy()
    corridor = int(round(out.shape[1] * cfg.corridor_half_width_ratio))
    cv.rectangle(out, (int(line_x - corridor), 0), (int(line_x + corridor), out.shape[0] - 1), (255, 0, 255), 1)
    candidate = decision.evidence.candidate
    if candidate is not None:
        x, y, w, h = candidate.bbox
        color = (0, 0, 255) if decision.stop_required else (0, 180, 255)
        cv.rectangle(out, (x, y), (x + w - 1, y + h - 1), color, 2)
        cv.putText(out, f"{decision.state.value} {candidate.cue} {candidate.confidence:.2f}",
                   (4, 16), cv.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv.LINE_AA)
    return out
