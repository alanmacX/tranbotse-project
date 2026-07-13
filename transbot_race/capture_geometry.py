from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math

import cv2 as cv
import numpy as np

from .config import RaceConfig
from .vision import preprocess_blackline


@dataclass(frozen=True, slots=True)
class CaptureGeometryObservation:
    kind: str = "straight_or_unknown"
    direction: int = 0
    angle_rad: float = 0.0
    confidence: float = 0.0
    vertex: tuple[float, float] | None = None
    vertex_y_frac: float | None = None
    incoming_e: float = 0.0
    incoming_theta: float = 0.0
    endpoints: int = 0
    component_area: int = 0


@dataclass(frozen=True, slots=True)
class CaptureGeometryDecision:
    kind: str
    direction: int
    angle_rad: float
    vertex_y_frac: float | None
    incoming_e: float
    incoming_theta: float
    votes: int


@dataclass(frozen=True, slots=True)
class CaptureGeometryDebug:
    roi: np.ndarray
    mask: np.ndarray
    component: np.ndarray
    skeleton: np.ndarray
    path: tuple[tuple[int, int], ...]
    endpoints: tuple[tuple[int, int], ...]
    roi_y0: int


def _floor_roi(frame: np.ndarray, cfg: RaceConfig) -> tuple[np.ndarray, int]:
    y0 = max(0, int(cfg.camera.crop[1]) - cfg.path_memory.geometry_roi_top_offset_px)
    y1 = min(frame.shape[0], max(int(cfg.camera.crop[3]) + 20, frame.shape[0] - 10))
    return frame[y0:y1].copy(), y0


def _select_anchor_component(mask: np.ndarray, center_x: float, trim_px: int) -> np.ndarray:
    count, labels, stats, centroids = cv.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask)
    h, w = mask.shape
    anchor_y = h - trim_px - 1
    best_label, best_score = -1, -1e9
    for label in range(1, count):
        x, y, cw, ch, area = stats[label]
        if area < 80:
            continue
        ys, xs = np.where(labels == label)
        anchor_distance = float(np.min(np.hypot(xs - center_x, 0.7 * (ys - anchor_y))))
        bottom_gap = h - (y + ch)
        center_cost = abs(float(centroids[label][0]) - center_x) / max(w, 1)
        score = math.log1p(int(area)) - anchor_distance * 0.08 - bottom_gap * 0.03 - center_cost * 2.0
        if anchor_distance <= 90:
            score += 5.0
        if score > best_score:
            best_label, best_score = label, score
    if best_label < 0:
        return np.zeros_like(mask)
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def _geometry_mask(roi: np.ndarray, cfg: RaceConfig, center_x: float) -> np.ndarray:
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    x0 = max(0, int(cfg.camera.crop[0]) - 50)
    x1 = min(gray.shape[1], int(cfg.camera.crop[2]) + 90)
    corridor = gray[:, x0:x1]
    dark_cap = int(np.clip(np.percentile(corridor, 20), 60, 112))
    fallback = np.where(gray <= dark_cap, 255, 0).astype(np.uint8)
    fallback = cv.morphologyEx(fallback, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    fallback = cv.morphologyEx(fallback, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 7)))

    primary = preprocess_blackline(roi, cfg.vision)
    trim = cfg.path_memory.geometry_chassis_trim_px
    probe = primary.copy()
    probe[-trim:] = 0
    component = _select_anchor_component(probe, center_x, trim)
    ys, xs = np.where(component > 0)
    if len(xs):
        anchor_y = roi.shape[0] - trim - 1
        distance = float(np.min(np.hypot(xs - center_x, 0.7 * (ys - anchor_y))))
        if distance <= 85:
            return primary
    return fallback


def _zhang_suen_skeleton(mask: np.ndarray) -> np.ndarray:
    image = (mask > 0).astype(np.uint8)
    while True:
        changed = False
        for phase in (0, 1):
            padded = np.pad(image, 1)
            p2, p3 = padded[:-2, 1:-1], padded[:-2, 2:]
            p4, p5 = padded[1:-1, 2:], padded[2:, 2:]
            p6, p7 = padded[2:, 1:-1], padded[2:, :-2]
            p8, p9 = padded[1:-1, :-2], padded[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1))
                + ((p4 == 0) & (p5 == 1))
                + ((p5 == 0) & (p6 == 1))
                + ((p6 == 0) & (p7 == 1))
                + ((p7 == 0) & (p8 == 1))
                + ((p8 == 0) & (p9 == 1))
                + ((p9 == 0) & (p2 == 1))
            )
            if phase == 0:
                phase_condition = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                phase_condition = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (image == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & phase_condition
            if np.any(remove):
                image[remove] = 0
                changed = True
        if not changed:
            return image * 255


def _skeleton_points(skeleton: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    binary = (skeleton > 0).astype(np.uint8)
    neighbors = cv.filter2D(binary, cv.CV_16S, np.ones((3, 3), np.int16)) - binary
    endpoints = [(int(x), int(y)) for y, x in np.argwhere((binary > 0) & (neighbors == 1))]
    pixels = [(int(x), int(y)) for y, x in np.argwhere(binary > 0)]
    return pixels, endpoints


def _nearest(points: list[tuple[int, int]], target: tuple[float, float]) -> tuple[int, int] | None:
    if not points:
        return None
    tx, ty = target
    return min(points, key=lambda point: (point[0] - tx) ** 2 + (point[1] - ty) ** 2)


def _endpoint_paths(
    skeleton: np.ndarray, anchor: tuple[int, int], endpoints: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    pixels = set((int(x), int(y)) for y, x in np.argwhere(skeleton > 0))
    if not pixels:
        return []
    if anchor not in pixels:
        anchor = min(pixels, key=lambda point: (point[0] - anchor[0]) ** 2 + (point[1] - anchor[1]) ** 2)
    queue = deque([anchor])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {anchor: None}
    while queue:
        x, y = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxt = (x + dx, y + dy)
                if nxt in pixels and nxt not in parent:
                    parent[nxt] = (x, y)
                    queue.append(nxt)
    paths = []
    for target in endpoints:
        if target not in parent or target == anchor:
            continue
        path, current = [], target
        while current is not None:
            path.append(current)
            current = parent[current]
        paths.append(list(reversed(path)))
    return paths


def _image_plane(points: list[tuple[int, int]]) -> np.ndarray:
    output = np.asarray([(-float(y), float(x)) for x, y in points], dtype=np.float64)
    if len(output):
        output -= output[0]
    return output


def _fit_direction(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 4:
        return None
    vx, vy, _, _ = cv.fitLine(points.astype(np.float32), cv.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    direction = np.asarray([float(vx), float(vy)], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 1e-8:
        return None
    direction /= norm
    if np.dot(direction, points[-1] - points[0]) < 0:
        direction = -direction
    return direction


def _path_angle(points: np.ndarray) -> float | None:
    if len(points) < 16:
        return None
    span = max(6, len(points) // 5)
    incoming, outgoing = _fit_direction(points[:span]), _fit_direction(points[-span:])
    if incoming is None or outgoing is None:
        return None
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    dot = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
    return math.atan2(cross, dot)


def _turn_profile(points: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if len(points) < 24:
        return None, None, None
    indices = np.linspace(0, len(points) - 1, min(31, len(points))).astype(int)
    samples = points[indices]
    directions = samples[2:] - samples[:-2]
    headings = np.unwrap(np.arctan2(directions[:, 1], directions[:, 0]))
    signed = np.diff(headings)
    signed = signed[np.abs(signed) < math.radians(115)]
    if not len(signed):
        return None, None, None
    absolute = np.abs(signed)
    total_abs = float(np.sum(absolute))
    top = min(3, len(absolute))
    concentration = float(np.sort(absolute)[-top:].sum() / max(total_abs, 1e-8))
    return float(np.sum(signed)), total_abs, concentration


def _hole_ratio(component: np.ndarray) -> float:
    contours, hierarchy = cv.findContours(component, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0.0
    areas = [cv.contourArea(contour) for index, contour in enumerate(contours) if hierarchy[0][index][3] >= 0]
    return max(areas, default=0.0) / max(1.0, float(component.shape[0] * component.shape[1]))


def _corner_vertex(path: list[tuple[int, int]]) -> tuple[tuple[float, float] | None, int | None]:
    if len(path) < 32:
        return None, None
    points = np.asarray(path, dtype=np.float64)
    window = max(8, min(30, len(points) // 10))
    best_index, best_angle = None, 0.0
    for index in range(window, len(points) - window):
        before = points[index] - points[index - window]
        after = points[index + window] - points[index]
        norms = np.linalg.norm(before) * np.linalg.norm(after)
        if norms <= 1e-8:
            continue
        angle = math.acos(float(np.clip(np.dot(before, after) / norms, -1.0, 1.0)))
        if angle > best_angle:
            best_index, best_angle = index, angle
    if best_index is None:
        return None, None
    return (float(points[best_index, 0]), float(points[best_index, 1])), best_index


def analyze_capture_geometry(
    frame: np.ndarray, cfg: RaceConfig,
) -> tuple[CaptureGeometryObservation, CaptureGeometryDebug]:
    roi, roi_y0 = _floor_roi(frame, cfg)
    center_x = 0.5 * (float(cfg.camera.crop[0]) + float(cfg.camera.crop[2]))
    trim = cfg.path_memory.geometry_chassis_trim_px
    mask = _geometry_mask(roi, cfg, center_x)
    mask[-trim:] = 0
    component = _select_anchor_component(mask, center_x, trim)
    skeleton = _zhang_suen_skeleton(component)
    pixels, endpoints = _skeleton_points(skeleton)
    anchor = _nearest(pixels, (center_x, roi.shape[0] - trim - 1))
    endpoint_anchor = _nearest(endpoints, anchor) if anchor else None
    paths = _endpoint_paths(skeleton, endpoint_anchor or anchor, endpoints) if anchor else []
    path = max(paths, key=len) if paths else []
    image_path = _image_plane(path)
    angle = _path_angle(image_path)
    total_turn, absolute_curvature, concentration = _turn_profile(image_path)
    branch_turns = []
    for candidate in paths:
        candidate_turn, _, _ = _turn_profile(_image_plane(candidate))
        if candidate_turn is not None:
            branch_turns.append(candidate_turn)

    area = int(cv.countNonZero(component))
    hole_ratio = _hole_ratio(component)
    curved_loop = bool(
        len(endpoints) == 3
        and absolute_curvature is not None
        and absolute_curvature >= math.radians(220)
        and concentration is not None
        and concentration <= 0.50
    )
    split_loop = bool(
        len(endpoints) == 3
        and any(abs(turn) >= math.radians(45) for turn in branch_turns)
        and curved_loop
    )
    cycle = hole_ratio >= 0.012 and len(endpoints) <= 4
    sharp_corner = bool(
        len(endpoints) == 2
        and angle is not None
        and abs(angle) >= math.radians(30)
        and total_turn is not None
        and abs(total_turn) >= math.radians(45)
        and concentration is not None
        and concentration >= 0.48
        and area >= 1000
    )
    visible_curve = angle is not None and abs(angle) >= math.radians(20)
    kind = (
        "circle" if cycle or curved_loop or split_loop
        else "corner" if sharp_corner
        else "curve" if visible_curve
        else "straight_or_unknown"
    )

    vertex, vertex_index = _corner_vertex(path)
    incoming_theta = 0.0
    if vertex_index is not None and vertex_index >= 8:
        incoming = _fit_direction(image_path[: vertex_index + 1])
        if incoming is not None:
            incoming_theta = math.atan2(float(incoming[1]), float(incoming[0]))
    anchor_x = float(anchor[0]) if anchor else center_x
    crop_half_width = max(1.0, 0.5 * (cfg.camera.crop[2] - cfg.camera.crop[0]))
    incoming_e = float(np.clip((anchor_x - center_x) / crop_half_width, -1.0, 1.0))
    vertex_y_frac = None if vertex is None else vertex[1] / max(1.0, float(roi.shape[0]))
    direction = 0 if angle is None or abs(angle) < math.radians(12) else (1 if angle > 0 else -1)
    confidence = min(1.0, 0.25 + min(0.35, area / 5000.0) + (0.2 if angle is not None else 0.0))
    if kind == "circle":
        confidence = min(1.0, confidence + 0.2)

    observation = CaptureGeometryObservation(
        kind=kind,
        direction=direction,
        angle_rad=0.0 if angle is None else float(angle),
        confidence=confidence,
        vertex=vertex,
        vertex_y_frac=vertex_y_frac,
        incoming_e=incoming_e,
        incoming_theta=incoming_theta,
        endpoints=len(endpoints),
        component_area=area,
    )
    debug = CaptureGeometryDebug(
        roi=roi,
        mask=mask,
        component=component,
        skeleton=skeleton,
        path=tuple(path),
        endpoints=tuple(endpoints),
        roi_y0=roi_y0,
    )
    return observation, debug


class CaptureGeometryFilter:
    def __init__(self, confirm_frames: int = 3) -> None:
        self.confirm_frames = max(3, int(confirm_frames))
        self.window: deque[CaptureGeometryObservation] = deque(maxlen=self.confirm_frames)

    def update(self, observation: CaptureGeometryObservation) -> CaptureGeometryDecision | None:
        self.window.append(observation)
        if len(self.window) < self.confirm_frames:
            return None
        if observation.kind == "circle" and all(item.kind == "circle" for item in self.window):
            return self._decision("circle", list(self.window), direction=0)
        if observation.kind != "corner":
            return None
        corners = [item for item in self.window if item.kind == "corner" and item.direction == observation.direction]
        required = max(2, self.confirm_frames - 1)
        if observation.direction == 0 or len(corners) < required:
            return None
        return self._decision("corner", corners, direction=observation.direction)

    @staticmethod
    def _decision(
        kind: str, observations: list[CaptureGeometryObservation], direction: int,
    ) -> CaptureGeometryDecision:
        vertices = [item.vertex_y_frac for item in observations if item.vertex_y_frac is not None]
        return CaptureGeometryDecision(
            kind=kind,
            direction=direction,
            angle_rad=float(np.median([abs(item.angle_rad) for item in observations])),
            vertex_y_frac=None if not vertices else float(np.median(vertices)),
            incoming_e=float(np.median([item.incoming_e for item in observations])),
            incoming_theta=float(np.median([item.incoming_theta for item in observations])),
            votes=len(observations),
        )


def draw_capture_geometry(
    debug: CaptureGeometryDebug, observation: CaptureGeometryObservation, gate_y_frac: float,
) -> np.ndarray:
    overlay = debug.roi.copy()
    overlay[debug.component > 0] = (
        0.45 * overlay[debug.component > 0] + 0.55 * np.asarray([0, 180, 255])
    ).astype(np.uint8)
    overlay[debug.skeleton > 0] = (255, 80, 30)
    for point in debug.endpoints:
        cv.circle(overlay, point, 4, (0, 255, 0), -1)
    for p0, p1 in zip(debug.path, debug.path[1:]):
        cv.line(overlay, p0, p1, (0, 0, 255), 1)
    if observation.vertex is not None:
        cv.circle(overlay, tuple(int(round(value)) for value in observation.vertex), 7, (255, 0, 255), 2)
    gate_y = int(round(overlay.shape[0] * gate_y_frac))
    cv.line(overlay, (0, gate_y), (overlay.shape[1] - 1, gate_y), (0, 255, 255), 2)
    angle_deg = math.degrees(observation.angle_rad)
    label = f"{observation.kind} dir={observation.direction:+d} angle={angle_deg:+.1f} ep={observation.endpoints}"
    cv.rectangle(overlay, (0, 0), (overlay.shape[1], 26), (0, 0, 0), -1)
    cv.putText(overlay, label, (7, 18), cv.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv.LINE_AA)
    return overlay
