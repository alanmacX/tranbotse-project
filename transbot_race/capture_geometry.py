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
    is_fork: bool = False


@dataclass(frozen=True, slots=True)
class CaptureGeometryDecision:
    kind: str
    direction: int
    angle_rad: float
    vertex_y_frac: float | None
    incoming_e: float
    incoming_theta: float
    votes: int
    is_fork: bool = False


@dataclass(frozen=True, slots=True)
class CaptureGeometryDebug:
    roi: np.ndarray
    mask: np.ndarray
    component: np.ndarray
    skeleton: np.ndarray
    path: tuple[tuple[int, int], ...]
    candidate_paths: tuple[tuple[tuple[int, int], ...], ...]
    candidate_directions: tuple[int, ...]
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


def _geometry_mask(roi: np.ndarray, cfg: RaceConfig) -> np.ndarray:
    # Geometry keeps the strict reflection guard.  The anchor-scoped exemption
    # is only for cruise crops, where the incoming track centre is known; a
    # wider geometry ROI also contains walls and stage furniture.
    primary = preprocess_blackline(roi, cfg.vision)
    # The expanded geometry ROI intentionally looks above the cruise crop, but
    # its very top contains the stage fascia / wall tiles rather than drivable
    # floor.  Those long dark seams can connect to the tape in perspective and
    # become a synthetic left/right arm.  Keep a small look-ahead allowance,
    # then make the non-floor strip ineligible in both mask paths.
    floor_top = max(0, int(cfg.path_memory.geometry_roi_top_offset_px) - 20)
    if floor_top:
        primary[:floor_top] = 0
    # Never fall back to a raw global intensity percentile here.  In low light
    # that selects a fixed fraction of the floor by construction; run 193907
    # turned the left underexposed half into a 37k--40k pixel fork with 14--19
    # skeleton endpoints.  No anchor is safer than fabricated geometry.
    return primary


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


def _turn_onset(path: list[tuple[int, int]]) -> tuple[tuple[float, float] | None, int | None]:
    """Return where the path first departs from its incoming tangent.

    The old implementation returned the single largest-curvature pixel.  On a
    rounded 90-degree bend that point jumps between the start, middle and end
    of the arc, so its image row is not a physical trigger.  The onset is the
    quantity needed for camera-to-axle compensation: starting at the chassis
    anchor, find the first sustained heading departure from the incoming line.
    """
    if len(path) < 32:
        return None, None
    points = np.asarray(path, dtype=np.float64)
    span = max(10, min(28, len(points) // 8))
    incoming = _fit_direction(_image_plane(path[: max(2 * span, 16)]))
    if incoming is None:
        return None, None
    threshold = math.radians(16.0)
    candidates: list[int] = []
    for index in range(span, len(points) - span):
        local = points[index + span] - points[index - span]
        norm = float(np.linalg.norm(local))
        if norm <= 1e-8:
            continue
        # Convert image (x,y) to the same forward/side convention as
        # _image_plane before comparing headings.
        local_direction = np.asarray([-local[1], local[0]], dtype=np.float64) / norm
        departure = math.acos(float(np.clip(np.dot(incoming, local_direction), -1.0, 1.0)))
        if departure >= threshold:
            candidates.append(index)
            if len(candidates) >= 3 and candidates[-1] - candidates[-3] <= 4:
                onset_index = candidates[-3]
                return (
                    (float(points[onset_index, 0]), float(points[onset_index, 1])),
                    onset_index,
                )
        else:
            candidates.clear()
    if not candidates:
        return None, None
    onset_index = candidates[0]
    return (float(points[onset_index, 0]), float(points[onset_index, 1])), onset_index


def _path_exit_direction(path: list[tuple[int, int]]) -> int:
    """Return which image side a candidate takes after its turn onset."""
    if not path:
        return 0
    _vertex, vertex_index = _turn_onset(path)
    reference_index = vertex_index if vertex_index is not None else 0
    exit_dx = float(path[-1][0] - path[reference_index][0])
    return 0 if abs(exit_dx) < 12.0 else (1 if exit_dx > 0.0 else -1)


def analyze_capture_geometry(
    frame: np.ndarray, cfg: RaceConfig,
) -> tuple[CaptureGeometryObservation, CaptureGeometryDebug]:
    roi, roi_y0 = _floor_roi(frame, cfg)
    center_x = 0.5 * (float(cfg.camera.crop[0]) + float(cfg.camera.crop[2]))
    trim = cfg.path_memory.geometry_chassis_trim_px
    mask = _geometry_mask(roi, cfg)
    mask[-trim:] = 0
    component = _select_anchor_component(mask, center_x, trim)
    skeleton = _zhang_suen_skeleton(component)
    pixels, endpoints = _skeleton_points(skeleton)
    anchor = _nearest(pixels, (center_x, roi.shape[0] - trim - 1))
    # Give every skeleton path a fixed physical orientation: near the chassis
    # (largest image y) -> far exit. Nearest-to-centre selection changed ends
    # as the robot approached the corner and flipped the same right turn from
    # +1 to -1 in frame 19 of run 161655.
    endpoint_anchor = (
        max(endpoints, key=lambda point: (point[1], -abs(point[0] - center_x)))
        if endpoints else None
    )
    paths = _endpoint_paths(skeleton, endpoint_anchor or anchor, endpoints) if anchor else []
    candidate_directions = [(_path_exit_direction(candidate), candidate) for candidate in paths]
    available_directions = {direction for direction, _candidate in candidate_directions if direction}
    is_fork = bool(
        3 <= len(endpoints) <= 4
        and {-1, 1}.issubset(available_directions)
    )
    desired_direction = 1 if cfg.path_memory.roundabout_direction >= 0 else -1
    desired_paths = [
        candidate for direction, candidate in candidate_directions
        if direction == desired_direction
    ]
    # Only a genuine two-sided fork uses route intent. Ordinary corners retain
    # their measured direction and longest connected path.
    path = (
        max(desired_paths, key=len)
        if is_fork and desired_paths
        else max(paths, key=len) if paths else []
    )
    image_path = _image_plane(path)
    angle = _path_angle(image_path)
    total_turn, absolute_curvature, concentration = _turn_profile(image_path)
    vertex, vertex_index = _turn_onset(path)
    branch_turns = []
    for candidate in paths:
        candidate_turn, _, _ = _turn_profile(_image_plane(candidate))
        if candidate_turn is not None:
            branch_turns.append(candidate_turn)

    area = int(cv.countNonZero(component))
    topology_sane = bool(
        2 <= len(endpoints) <= 4
        and 600 <= area <= int(component.size * 0.08)
    )
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
    # A single nearby dash has only a short, jagged skeleton.  Endpoint tangent
    # noise can easily exceed 20 degrees (32--39 degrees in run 173927), but it
    # has no sustained turn onset and must never become a curve event.  A curve
    # is actionable only when the path contains an actual departure vertex;
    # approach/margin logic needs that same physical point anyway.
    visible_curve = bool(
        topology_sane
        and angle is not None
        and abs(angle) >= math.radians(20)
        and vertex is not None
    )
    kind = (
        "circle" if topology_sane and (cycle or curved_loop or split_loop)
        else "corner" if topology_sane and sharp_corner
        else "curve" if visible_curve
        else "straight_or_unknown"
    )

    incoming_theta = 0.0
    if vertex_index is not None and vertex_index >= 8:
        incoming = _fit_direction(image_path[: vertex_index + 1])
        if incoming is not None:
            incoming_theta = math.atan2(float(incoming[1]), float(incoming[0]))
    anchor_x = float(anchor[0]) if anchor else center_x
    crop_half_width = max(1.0, 0.5 * (cfg.camera.crop[2] - cfg.camera.crop[0]))
    incoming_e = float(np.clip((anchor_x - center_x) / crop_half_width, -1.0, 1.0))
    vertex_y_frac = None if vertex is None else vertex[1] / max(1.0, float(roi.shape[0]))
    # Direction is an image-space fact: after the first sustained departure,
    # does the exit enter from the right or the left?  The sign of a fitted
    # path angle is unstable under perspective and flipped on the same physical
    # right turn in run 161655.  Endpoint displacement is invariant to that
    # fit-line orientation ambiguity.
    direction = 0 if angle is None else _path_exit_direction(path)
    if angle is not None and direction:
        angle = math.copysign(abs(angle), direction)
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
        is_fork=is_fork,
    )
    debug = CaptureGeometryDebug(
        roi=roi,
        mask=mask,
        component=component,
        skeleton=skeleton,
        path=tuple(path),
        candidate_paths=tuple(tuple(candidate) for _direction, candidate in candidate_directions),
        candidate_directions=tuple(direction for direction, _candidate in candidate_directions),
        endpoints=tuple(endpoints),
        roi_y0=roi_y0,
    )
    return observation, debug


class CaptureGeometryFilter:
    def __init__(self, confirm_frames: int = 3) -> None:
        self.confirm_frames = max(3, int(confirm_frames))
        self.window: deque[CaptureGeometryObservation] = deque(maxlen=self.confirm_frames)

    def reset(self) -> None:
        self.window.clear()

    def update(
        self,
        observation: CaptureGeometryObservation,
    ) -> CaptureGeometryDecision | None:
        self.window.append(observation)
        if len(self.window) < self.confirm_frames:
            return None
        if (
            not observation.is_fork
            and observation.kind == "circle"
            and all(item.kind == "circle" and not item.is_fork for item in self.window)
        ):
            return self._decision("circle", list(self.window), direction=0)
        # A rounded bend and a sharp corner are the same navigation event.
        # Shape concentration is useful telemetry, but must not decide whether
        # the turn controller gets ownership (151012 never triggered because
        # the real bend was consistently labelled ``curve``).
        observation_is_turn = observation.kind in {"corner", "curve"} or (
            observation.is_fork and observation.kind == "circle"
        )
        if not observation_is_turn:
            return None
        turns = self._trailing_candidates(
            observation,
            lambda item: (
                item.kind in {"corner", "curve"}
                or (item.is_fork and item.kind == "circle")
            )
        )
        if observation.direction == 0 or len(turns) < self.confirm_frames:
            return None
        return self._decision("turn", turns, direction=observation.direction)

    def _trailing_candidates(
        self,
        observation: CaptureGeometryObservation,
        shape_allowed,
    ) -> list[CaptureGeometryObservation]:
        """Return only the uninterrupted current event epoch.

        A dropout, direction flip, fork flip, or invalid quality terminates the
        epoch. Old votes can no longer leak across an unrelated frame.
        """
        candidates: list[CaptureGeometryObservation] = []
        for item in reversed(self.window):
            valid = bool(
                shape_allowed(item)
                and item.is_fork == observation.is_fork
                and item.direction == observation.direction
                and item.direction != 0
                and item.confidence >= 0.55
                and abs(item.angle_rad) >= math.radians(20.0)
                and item.vertex_y_frac is not None
            )
            if not valid:
                break
            candidates.append(item)
        candidates.reverse()
        return candidates

    def _session_decision(
        self,
        observation: CaptureGeometryObservation,
        *,
        event_kind: str,
        allowed_shapes: set[str],
        require_fork: bool | None,
    ) -> CaptureGeometryDecision | None:
        if observation.kind not in allowed_shapes or observation.direction == 0:
            return None
        if require_fork is not None and observation.is_fork != require_fork:
            return None
        candidates = self._trailing_candidates(
            observation,
            lambda item: (
                item.kind in allowed_shapes
                and (require_fork is None or item.is_fork == require_fork)
            ),
        )
        if len(candidates) < self.confirm_frames:
            return None
        return self._decision(event_kind, candidates, direction=observation.direction)

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
            is_fork=any(item.is_fork for item in observations),
        )


class CornerGeometryFilter(CaptureGeometryFilter):
    """Only emits the event understood by the corner executor."""

    def update(
        self,
        observation: CaptureGeometryObservation,
    ) -> CaptureGeometryDecision | None:
        self.window.append(observation)
        if len(self.window) < self.confirm_frames:
            return None
        return self._session_decision(
            observation,
            event_kind="corner",
            allowed_shapes={"corner", "curve"},
            require_fork=False,
        )


class RingEntryGeometryFilter(CaptureGeometryFilter):
    """Only emits the event understood by the ring-entry executor."""

    def update(
        self,
        observation: CaptureGeometryObservation,
    ) -> CaptureGeometryDecision | None:
        self.window.append(observation)
        if len(self.window) < self.confirm_frames:
            return None
        return self._session_decision(
            observation,
            event_kind="ring_entry",
            allowed_shapes={"curve", "circle"},
            require_fork=None,
        )


def draw_capture_geometry(
    debug: CaptureGeometryDebug,
    observation: CaptureGeometryObservation,
    gate_y_frac: float,
    event_kind: str | None = None,
    session: str | None = None,
) -> np.ndarray:
    overlay = debug.roi.copy()
    overlay[debug.component > 0] = (
        0.45 * overlay[debug.component > 0] + 0.55 * np.asarray([0, 180, 255])
    ).astype(np.uint8)
    overlay[debug.skeleton > 0] = (255, 80, 30)
    for point in debug.endpoints:
        cv.circle(overlay, point, 4, (0, 255, 0), -1)
    for direction, candidate in zip(debug.candidate_directions, debug.candidate_paths):
        color = (255, 220, 0) if direction < 0 else (0, 220, 255)
        for p0, p1 in zip(candidate, candidate[1:]):
            cv.line(overlay, p0, p1, color, 1)
    for p0, p1 in zip(debug.path, debug.path[1:]):
        cv.line(overlay, p0, p1, (0, 0, 255), 2)
    if observation.vertex is not None:
        cv.circle(overlay, tuple(int(round(value)) for value in observation.vertex), 7, (255, 0, 255), 2)
    gate_y = int(round(overlay.shape[0] * gate_y_frac))
    cv.line(overlay, (0, gate_y), (overlay.shape[1] - 1, gate_y), (0, 255, 255), 2)
    angle_deg = math.degrees(observation.angle_rad)
    label = (
        f"session={session or '-'} raw_shape={observation.kind} "
        f"event={event_kind or '-'} fork={int(observation.is_fork)} "
        f"selected={observation.direction:+d} angle={angle_deg:+.1f} ep={observation.endpoints}"
    )
    cv.rectangle(overlay, (0, 0), (overlay.shape[1], 26), (0, 0, 0), -1)
    cv.putText(overlay, label, (7, 18), cv.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv.LINE_AA)
    return overlay
