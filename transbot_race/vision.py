from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2 as cv
import numpy as np

from .config import VisionConfig


@dataclass(frozen=True, slots=True)
class Run:
    x0: int
    x1: int
    cx: float
    width: int
    area: int
    component_id: int = -1
    component_area: int = 0
    component_fill: float = 0.0
    touches_border: bool = False


@dataclass(frozen=True, slots=True)
class ScanBand:
    index: int
    y0: int
    y1: int
    runs: tuple[Run, ...]
    best: Run | None


@dataclass(frozen=True, slots=True)
class BranchFeature:
    direction: str
    band_index: int
    y: float
    run: Run


@dataclass(frozen=True, slots=True)
class PathPoint:
    band_index: int
    x: float
    y: float
    run: Run


@dataclass(frozen=True, slots=True)
class PreviewCandidate:
    """A far segment worth remembering, but not fitting through a blind zone."""

    direction: str
    dir_sign: int          # +1 means target is to image/right side, -1 left.
    anchor_band_index: int
    target_band_index: int
    anchor_x: float
    anchor_y: float
    target_x: float
    target_y: float
    score: float
    gap_bands: int


@dataclass(frozen=True, slots=True)
class LineFeatures:
    found: bool
    err_norm: float = 0.0
    line_width_px: float = 0.0
    bands: tuple[ScanBand, ...] = ()
    path: tuple[PathPoint, ...] = ()
    preview: PreviewCandidate | None = None
    bottom: Run | None = None
    mid: Run | None = None
    branch_left: BranchFeature | None = None
    branch_right: BranchFeature | None = None

    @property
    def has_corner(self) -> bool:
        return self.branch_left is not None or self.branch_right is not None


def _raw_dark_cap(gray: np.ndarray) -> int:
    """Dynamic raw-gray cap that keeps black tape and rejects gray shadows."""

    p50 = float(np.percentile(gray, 50))
    p35 = float(np.percentile(gray, 35))
    p08 = float(np.percentile(gray, 8))
    return int(min(112, max(58, min(p50 - 5, p35 + 12, p08 + 24))))


def _reflection_guards(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build hard and near guards around bright specular reflections."""

    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    bright_threshold = max(150, min(220, int(np.percentile(gray, 98))))
    core = (
        (gray >= bright_threshold)
        | ((hsv[:, :, 2] >= bright_threshold) & (hsv[:, :, 1] <= 75))
    ).astype(np.uint8) * 255
    core = cv.morphologyEx(
        core,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    hard_guard = cv.dilate(
        core,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (27, 23)),
        iterations=1,
    )
    hard_guard = cv.dilate(
        hard_guard,
        cv.getStructuringElement(cv.MORPH_RECT, (11, 3)),
        iterations=1,
    )
    near_guard = cv.dilate(
        hard_guard,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 21)),
        iterations=1,
    )
    return hard_guard, near_guard


def _filter_preprocess_components(mask: np.ndarray, hard_guard: np.ndarray, near_guard: np.ndarray) -> np.ndarray:
    """Remove reflection rims, tile seams, and border slivers from a raw mask."""

    height, width = mask.shape[:2]
    count, labels, stats, _centroids = cv.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    hard_bool = hard_guard > 0
    near_bool = near_guard > 0
    for component_id in range(1, count):
        x = int(stats[component_id, cv.CC_STAT_LEFT])
        y = int(stats[component_id, cv.CC_STAT_TOP])
        w = int(stats[component_id, cv.CC_STAT_WIDTH])
        h = int(stats[component_id, cv.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv.CC_STAT_AREA])
        if w <= 0 or h <= 0 or area < 30:
            continue

        fill = area / float(max(1, w * h))
        aspect = max(w / float(max(1, h)), h / float(max(1, w)))
        touches_right = x + w >= width - 1
        touches_any = x <= 1 or y <= 1 or touches_right or y + h >= height - 1

        if area < 60 and (touches_any or aspect > 3.4):
            continue
        if h > height * 0.55 and w < width * 0.11 and fill < 0.42:
            continue
        if aspect > 7.5 and min(w, h) <= 9 and area < 420:
            continue
        if touches_right and area < 360 and w < width * 0.16:
            continue
        if touches_any and aspect > 8.5 and area < 900:
            continue

        component = labels == component_id
        hard_overlap = float(np.count_nonzero(component & hard_bool)) / float(max(1, area))
        near_overlap = float(np.count_nonzero(component & near_bool)) / float(max(1, area))
        effective_thickness = area / float(max(w, h, 1))
        track_structure = (
            area >= 900
            and h >= height * 0.45
            and effective_thickness >= 10.0
            and (w >= width * 0.18 or fill >= 0.50)
        )
        if hard_overlap > 0.18 and fill < 0.65:
            continue
        if near_overlap > 0.16 and fill < 0.50 and not track_structure:
            continue

        clean[component] = 255
    return clean


def preprocess_blackline(frame_bgr: np.ndarray, cfg: VisionConfig) -> np.ndarray:
    """Return a binary mask where likely black track pixels are 255."""

    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    hard_guard, near_guard = _reflection_guards(frame_bgr)

    work = gray.copy()
    work[hard_guard > 0] = 255
    work[work > getattr(cfg, "glare_rejection_threshold", 140)] = 255

    mask = cv.adaptiveThreshold(
        work,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        81,
        5,
    )
    mask[gray > _raw_dark_cap(gray)] = 0
    mask[hard_guard > 0] = 0

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    mask = cv.morphologyEx(
        mask,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_RECT, (3, 7)),
        iterations=1,
    )
    return _filter_preprocess_components(mask, hard_guard, near_guard)


def _band_bounds(index: int, height: int, band_count: int) -> tuple[int, int]:
    """(y0, y1) of scan band `index`, counting from the bottom of the frame."""
    y1 = height - int(index * height / band_count)
    y0 = height - int((index + 1) * height / band_count)
    return y0, y1


@dataclass(frozen=True, slots=True)
class Component:
    x: int
    y: int
    width: int
    height: int
    area: int
    fill: float
    touches_border: bool


def _component_labels(mask: np.ndarray, cfg: VisionConfig) -> tuple[np.ndarray, dict[int, Component]]:
    """Label plausible black-line components before band scanning.

    This is the contour/moments step used by robust camera line followers. It
    removes sparse tile seams and glare fragments before they can become
    synthetic scan boxes.
    """

    height, width = mask.shape[:2]
    count, labels, stats, _centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    components: dict[int, Component] = {}
    min_area = max(cfg.component_min_area_px, cfg.min_run_area_px)
    for component_id in range(1, count):
        x = int(stats[component_id, cv.CC_STAT_LEFT])
        y = int(stats[component_id, cv.CC_STAT_TOP])
        w = int(stats[component_id, cv.CC_STAT_WIDTH])
        h = int(stats[component_id, cv.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv.CC_STAT_AREA])
        if w <= 0 or h <= 0 or area < min_area:
            continue
        fill = area / float(max(1, w * h))
        if fill < cfg.component_min_fill_ratio and area < min_area * 4:
            continue
        # Very wide, shallow components are usually shadows/reflections, not a
        # tape centerline. A true right-angle branch is wide, but not full-crop.
        if w > width * cfg.component_max_width_ratio and h < height * 0.35:
            continue
        # The camera crop often includes a dark strip from the chassis at the
        # bottom edge. It is wide, shallow, border-touching, and was previously
        # chosen as the bottom anchor, which made straight-line tracking run on
        # prediction after a few frames.
        if (
            y + h >= height - 2
            and w > width * cfg.component_bottom_bar_width_ratio
            and h < height * cfg.component_bottom_bar_height_ratio
        ):
            continue
        touches_border = x <= 1 or x + w >= width - 1
        components[component_id] = Component(x, y, w, h, area, fill, touches_border)
    return labels, components


def _runs_from_binary_band(mask: np.ndarray, y0: int, y1: int, cfg: VisionConfig) -> tuple[Run, ...]:
    band = mask[y0:y1, :]
    if band.size == 0:
        return ()
    height = max(1, y1 - y0)
    col_sum = cv.reduce(band, 0, cv.REDUCE_SUM, dtype=cv.CV_32S).reshape(-1)
    active = col_sum > int(255 * height * cfg.active_col_ratio)
    runs: list[Run] = []
    start: int | None = None
    for idx, on in enumerate([*active.tolist(), False]):
        if on and start is None:
            start = idx
            continue
        if not on and start is not None:
            end = idx - 1
            width = end - start + 1
            if width >= cfg.min_run_width_px:
                area = int(cv.countNonZero(band[:, start : end + 1]))
                if area >= cfg.min_run_area_px:
                    runs.append(Run(start, end, (start + end) / 2.0, width, area))
            start = None
    return tuple(runs)


def _runs_from_band(
    mask: np.ndarray,
    y0: int,
    y1: int,
    cfg: VisionConfig,
    labels: np.ndarray | None = None,
    components: dict[int, Component] | None = None,
) -> tuple[Run, ...]:
    if labels is None or components is None:
        return _runs_from_binary_band(mask, y0, y1, cfg)

    band_labels = labels[y0:y1, :]
    if band_labels.size == 0:
        return ()

    height = max(1, y1 - y0)
    runs: list[Run] = []
    active_threshold = max(1, int(height * cfg.active_col_ratio))
    for component_id in sorted(int(value) for value in np.unique(band_labels) if int(value) in components):
        component = components[component_id]
        component_band = band_labels == component_id
        area_total = int(np.count_nonzero(component_band))
        if area_total < cfg.min_run_area_px:
            continue

        col_sum = component_band.sum(axis=0)
        active = col_sum >= active_threshold
        start: int | None = None
        for idx, on in enumerate([*active.tolist(), False]):
            if on and start is None:
                start = idx
                continue
            if not on and start is not None:
                end = idx - 1
                width = end - start + 1
                if width >= cfg.min_run_width_px:
                    local = component_band[:, start : end + 1]
                    ys, xs = np.nonzero(local)
                    area = int(xs.size)
                    if area >= cfg.min_run_area_px:
                        cx = float(xs.mean() + start)
                        touches = component.touches_border or start <= 1 or end >= mask.shape[1] - 2
                        runs.append(
                            Run(
                                start,
                                end,
                                cx,
                                width,
                                area,
                                component_id=component_id,
                                component_area=component.area,
                                component_fill=component.fill,
                                touches_border=touches,
                            )
                        )
                start = None
    return tuple(runs)


def _run_cost(run: Run, target_x: float, crop_width: float, line_width: float | None = None) -> float:
    half = max(1.0, crop_width / 2.0)
    cost = abs(run.cx - target_x) / half
    cost -= min(run.area, 1200) * 0.00015
    cost -= min(run.component_area, 5000) * 0.000015
    if run.touches_border:
        cost += 0.18
    if line_width and line_width > 0:
        too_wide = max(0.0, run.width / line_width - 3.2)
        cost += too_wide * 0.08
    return cost


def _best_run(runs: Iterable[Run], crop_center: float, crop_width: float | None = None) -> Run | None:
    runs = tuple(runs)
    if not runs:
        return None
    width = crop_width if crop_width is not None else max(crop_center * 2.0, 1.0)
    return min(runs, key=lambda run: _run_cost(run, crop_center, width))


def _predict_next_x(accepted: list[tuple[int, Run]], target_index: int) -> float:
    last_index, last = accepted[-1]
    if len(accepted) < 2:
        return last.cx
    prev_index, prev = accepted[-2]
    di = max(1, last_index - prev_index)
    return last.cx + (last.cx - prev.cx) * ((target_index - last_index) / di)


def _corridor_score(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float, radius: int) -> float:
    """Fraction of black pixels inside a thick line corridor."""

    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1.0))
    if steps <= 0:
        return 0.0
    height, width = mask.shape[:2]
    support = 0
    total = 0
    for t in np.linspace(0.0, 1.0, num=max(8, min(96, steps)), dtype=np.float32):
        x = int(round(x0 + (x1 - x0) * float(t)))
        y = int(round(y0 + (y1 - y0) * float(t)))
        xa = max(0, x - radius)
        xb = min(width, x + radius + 1)
        ya = max(0, y - 1)
        yb = min(height, y + 2)
        if xa >= xb or ya >= yb:
            continue
        roi = mask[ya:yb, xa:xb]
        support += int(cv.countNonZero(roi))
        total += int(roi.size)
    return support / float(max(1, total))


def _preview_candidate(
    mask: np.ndarray,
    bands: list[ScanBand],
    path_bests: list[Run | None],
    accepted: list[tuple[int, Run]],
    crop_center: float,
    width: int,
    line_width: float,
    cfg: VisionConfig,
) -> PreviewCandidate | None:
    if not accepted:
        return None

    anchor_index, anchor = accepted[-1]
    if anchor.width > width * 0.42:
        return None
    anchor_band = bands[anchor_index]
    anchor_y = (anchor_band.y0 + anchor_band.y1) / 2.0
    min_dx = max(width * cfg.preview_min_dx_ratio, line_width * 1.7)
    radius = max(4, int(round(line_width * cfg.preview_corridor_width_ratio / 2.0)))
    best: PreviewCandidate | None = None
    best_score = 0.0

    for band in bands[anchor_index + 1 :]:
        target_y = (band.y0 + band.y1) / 2.0
        dy = max(1.0, anchor_y - target_y)
        for run in band.runs:
            if path_bests[band.index] == run:
                continue
            dx = run.cx - anchor.cx
            branch_like = run.width >= line_width * cfg.branch_width_ratio
            if run.width < max(cfg.min_run_width_px, line_width * 0.50):
                continue
            if run.component_area < cfg.min_run_area_px * 6:
                continue
            if abs(dx) < min_dx and not branch_like:
                continue
            if run.touches_border and run.component_id != anchor.component_id:
                continue
            angle = abs(dx) / dy
            if angle < 0.15 and not branch_like:
                continue

            corridor = _corridor_score(mask, anchor.cx, anchor_y, run.cx, target_y, radius)
            if corridor < 0.12 and run.component_id != anchor.component_id:
                continue
            run_conf = min(1.0, run.area / float(max(1, cfg.min_run_area_px * 8)))
            turn_conf = min(1.0, abs(dx) / float(max(1.0, width * 0.42)))
            branch_bonus = 0.14 if branch_like else 0.0
            component_bonus = 0.08 if run.component_id == anchor.component_id and run.component_id >= 0 else 0.0
            border_penalty = 0.18 if run.touches_border else 0.0
            sparse_penalty = 0.12 if run.component_fill < cfg.component_min_fill_ratio * 1.35 else 0.0
            score = (
                0.48 * corridor
                + 0.22 * run_conf
                + 0.26 * turn_conf
                + branch_bonus
                + component_bonus
                - border_penalty
                - sparse_penalty
            )

            if score > best_score:
                direction = "right" if dx > 0 else "left"
                best_score = score
                best = PreviewCandidate(
                    direction=direction,
                    dir_sign=1 if dx > 0 else -1,
                    anchor_band_index=anchor_index,
                    target_band_index=band.index,
                    anchor_x=anchor.cx,
                    anchor_y=anchor_y,
                    target_x=run.cx,
                    target_y=target_y,
                    score=float(score),
                    gap_bands=max(0, band.index - anchor_index - 1),
                )

    if best is not None and best_score >= cfg.preview_min_score:
        return best
    return None


def _sliding_window_path(
    mask: np.ndarray,
    bands: list[ScanBand],
    crop_center: float,
    width: int,
    cfg: VisionConfig,
) -> tuple[list[Run | None], tuple[PathPoint, ...], PreviewCandidate | None]:
    """Choose one bottom-anchored path and a separate far preview candidate."""

    bests: list[Run | None] = [None] * len(bands)
    anchor_index = next((band.index for band in bands[:2] if band.runs), None)
    if anchor_index is None:
        anchor_index = next((band.index for band in bands if band.runs), None)
    if anchor_index is None:
        return bests, (), None

    anchor_band = bands[anchor_index]
    anchor = _best_run(anchor_band.runs, crop_center, width)
    if anchor is None:
        return bests, (), None

    bests[anchor_index] = anchor
    accepted: list[tuple[int, Run]] = [(anchor_index, anchor)]
    bottom_widths = [
        run.width
        for band in bands[:3]
        for run in band.runs
        if cfg.min_run_width_px <= run.width <= width * 0.25
    ]
    fallback_width = min(float(anchor.width), max(float(cfg.min_run_width_px), width * 0.12))
    line_width = float(np.median(bottom_widths or [fallback_width]))
    line_width = max(float(cfg.min_run_width_px), min(line_width, width * 0.16))
    base_margin = max(float(cfg.sliding_window_margin_px), width * 0.16, line_width * 2.2)
    missed_bands = 0

    for band in bands[anchor_index + 1 :]:
        pred = _predict_next_x(accepted, band.index)
        gap_scale = max(1.0, band.index - accepted[-1][0])
        margin = base_margin * min(1.7, 1.0 + 0.35 * missed_bands + 0.15 * (gap_scale - 1.0))
        candidates = [run for run in band.runs if abs(run.cx - pred) <= margin]
        if candidates:
            chosen = min(candidates, key=lambda run: _run_cost(run, pred, width, line_width))
            bests[band.index] = chosen
            accepted.append((band.index, chosen))
            missed_bands = 0
            continue

        if band.runs:
            break

        missed_bands += 1
        if missed_bands > cfg.sliding_window_max_gap_bands:
            break

    path = tuple(
        PathPoint(
            band_index=index,
            x=run.cx,
            y=(bands[index].y0 + bands[index].y1) / 2.0,
            run=run,
        )
        for index, run in accepted
    )
    preview = None
    # If the sliding window already reaches the far ROI, the near path is enough.
    # Preview is reserved for an early break: a visible far segment after a blind
    # zone, not a second opinion on every stray component in the expanded crop.
    if accepted[-1][0] < cfg.band_count - 2:
        preview = _preview_candidate(mask, bands, bests, accepted, crop_center, width, line_width, cfg)
    return bests, path, preview


def _classic_contour_path(
    bands: list[ScanBand],
    crop_center: float,
    width: int,
    cfg: VisionConfig,
) -> tuple[list[Run | None], tuple[PathPoint, ...], PreviewCandidate | None]:
    """OpenCV contour-centroid baseline used by common line followers.

    It intentionally ignores far-field shape and follows only the near ROI. That
    is less ambitious than the sliding-window planner, but much harder for tile
    seams and reflections to derail on straight-line validation.
    """

    bests: list[Run | None] = [None] * len(bands)
    accepted: list[tuple[int, Run]] = []
    near_count = max(1, min(cfg.classic_near_band_count, len(bands)))
    for band in bands[:near_count]:
        candidates = [
            run
            for run in band.runs
            if run.width <= width * cfg.classic_max_run_width_ratio
            and 0.0 <= run.cx <= float(width)
        ]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda run: _run_cost(run, crop_center, width))
        bests[band.index] = chosen
        accepted.append((band.index, chosen))

    path = tuple(
        PathPoint(
            band_index=index,
            x=run.cx,
            y=(bands[index].y0 + bands[index].y1) / 2.0,
            run=run,
        )
        for index, run in accepted
    )
    return bests, path, None


def scan_line_features(mask: np.ndarray, cfg: VisionConfig, crop_center: float | None = None) -> LineFeatures:
    """Extract line-center and left/right branch features from a black-line mask."""

    height, width = mask.shape[:2]
    if crop_center is None:
        crop_center = width / 2.0

    labels, components = _component_labels(mask, cfg)
    raw_bands: list[ScanBand] = []
    for index in range(cfg.band_count):
        y0, y1 = _band_bounds(index, height, cfg.band_count)
        runs = _runs_from_band(mask, y0, y1, cfg, labels=labels, components=components)
        raw_bands.append(ScanBand(index=index, y0=y0, y1=y1, runs=runs, best=None))

    if cfg.fit_mode == "classic":
        path_bests, path, preview = _classic_contour_path(raw_bands, crop_center, width, cfg)
    else:
        path_bests, path, preview = _sliding_window_path(mask, raw_bands, crop_center, width, cfg)
    bands = [
        ScanBand(index=band.index, y0=band.y0, y1=band.y1, runs=band.runs, best=path_bests[band.index])
        for band in raw_bands
    ]

    valid = [band for band in bands if band.best is not None]
    if not valid:
        return LineFeatures(found=False, bands=tuple(bands))

    bottom = bands[0].best or (bands[1].best if len(bands) > 1 else None)
    mid = bands[1].best or bottom
    near_widths = [band.best.width for band in bands[:2] if band.best is not None]
    all_widths = [band.best.width for band in valid if band.best is not None]
    line_width = float(np.median(near_widths or all_widths or [8.0]))
    if bottom and mid:
        err_cx = 0.65 * bottom.cx + 0.35 * mid.cx
    else:
        err_cx = valid[0].best.cx  # type: ignore[union-attr]
    # Normalize by the actual asymmetric half-width relative to crop_center.
    if err_cx >= crop_center:
        half_width = max(float(width) - crop_center, 1.0)
    else:
        half_width = max(crop_center, 1.0)
        
    err_norm = max(-1.0, min(1.0, (err_cx - crop_center) / half_width))

    branch_left: BranchFeature | None = None
    branch_right: BranchFeature | None = None
    for band_index in [2, 3, 4, 1]:
        if band_index >= len(bands):
            continue
        band = bands[band_index]
        for run in band.runs:
            wide = run.width > max(line_width * cfg.branch_width_ratio, width * cfg.branch_min_crop_ratio)
            crosses_center = run.x0 < crop_center + max(10, line_width * 1.3) and run.x1 > crop_center - max(10, line_width * 1.3)
            if not (wide and crosses_center):
                continue
            y = (band.y0 + band.y1) / 2.0
            if run.x1 > crop_center + max(12, line_width * 1.4):
                candidate = BranchFeature("right", band_index, y, run)
                if branch_right is None or run.area > branch_right.run.area:
                    branch_right = candidate
            if run.x0 < crop_center - max(12, line_width * 1.4):
                candidate = BranchFeature("left", band_index, y, run)
                if branch_left is None or run.area > branch_left.run.area:
                    branch_left = candidate

    return LineFeatures(
        found=True,
        err_norm=err_norm,
        line_width_px=line_width,
        bands=tuple(bands),
        path=path,
        preview=preview,
        bottom=bottom,
        mid=mid,
        branch_left=branch_left,
        branch_right=branch_right,
    )


@dataclass(frozen=True, slots=True)
class TrajectoryFit:
    """Continuous estimate of the line as seen in the crop.

    Angle-agnostic: gentle curves, sharp/right-angle corners, dashed lines and
    roundabouts are all expressed through (e0, theta, kappa, conf) instead of
    discrete states. e0/theta/kappa are normalized so the controller gains are
    resolution-independent.
    """

    found: bool
    e0: float = 0.0          # bottom lateral error, normalized to [-1, 1]
    e_look: float = 0.0      # lateral error at the lookahead point, [-1, 1]
    theta: float = 0.0       # line heading vs car forward, radians (approx)
    kappa: float = 0.0       # normalized curvature (0 straight, >0 bends right)
    conf: float = 0.0        # overall fit confidence [0, 1]
    n_bands: int = 0         # number of bands that contributed
    quadratic: bool = False  # whether the quadratic term was used
    disconnected: bool = False  # visible bands were split by a large jump
    preview_dir: int = 0     # +1 means a right-side far target, -1 left
    preview_e: float = 0.0
    preview_theta: float = 0.0
    preview_conf: float = 0.0
    path_memory: bool = False  # preview steering was released by distance delay


def _band_confidence(run: Run, band_height: float, line_width: float, cfg: VisionConfig) -> float:
    """Per-band confidence from area adequacy and width plausibility."""
    if run is None or line_width <= 0:
        return 0.0
    area_conf = min(1.0, run.area / max(1.0, cfg.min_run_area_px * 3.0))
    width_ratio = run.width / line_width
    # Peak confidence when width ~= expected line width; a branch/junction run
    # is much wider, so it is down-weighted rather than trusted as the center.
    width_conf = max(0.0, 1.0 - abs(width_ratio - 1.0))
    return max(0.0, min(1.0, 0.5 * area_conf + 0.5 * width_conf))


def _preview_terms(features: LineFeatures, crop_center: float, half_width: float) -> tuple[int, float, float, float]:
    preview = features.preview
    if preview is None:
        return 0, 0.0, 0.0, 0.0
    e = max(-1.0, min(1.0, (preview.target_x - crop_center) / half_width))
    dy = max(1.0, preview.anchor_y - preview.target_y)
    theta = float(np.arctan2(preview.target_x - preview.anchor_x, dy))
    conf = max(0.0, min(1.0, preview.score))
    return preview.dir_sign, e, theta, conf


def _fit_classic_contour_trajectory(
    features: LineFeatures,
    cfg: VisionConfig,
    crop_center: float,
    crop_width: float,
    occluded_band_indices: frozenset[int],
) -> TrajectoryFit:
    half_width = max(crop_width / 2.0, 1.0)
    samples: list[tuple[int, float, float, Run]] = []
    for band in features.bands[: max(1, cfg.classic_near_band_count)]:
        if band.index in occluded_band_indices or band.best is None:
            continue
        y_mid = (band.y0 + band.y1) / 2.0
        samples.append((band.index, y_mid, band.best.cx, band.best))

    if not samples:
        return TrajectoryFit(found=False)

    samples.sort(key=lambda item: item[0])
    near = samples[0]
    if len(samples) >= 2:
        e_cx = 0.72 * samples[0][2] + 0.28 * samples[1][2]
    else:
        e_cx = near[2]
    e0 = float(max(-1.0, min(1.0, (e_cx - crop_center) / half_width)))

    theta = 0.0
    if len(samples) >= 2:
        far = samples[min(len(samples) - 1, 2)]
        dy = max(1.0, near[1] - far[1])
        dx = far[2] - near[2]
        theta = float(np.arctan2(dx, dy))
        theta = max(-cfg.classic_theta_limit, min(cfg.classic_theta_limit, theta))

    n = len(samples)
    area_score = min(1.0, sum(item[3].area for item in samples) / float(max(1, cfg.min_run_area_px * 12)))
    count_score = min(1.0, n / float(max(1, cfg.classic_near_band_count)))
    conf = float(max(0.0, min(1.0, 0.34 + 0.40 * count_score + 0.26 * area_score)))

    return TrajectoryFit(
        found=True,
        e0=e0,
        e_look=e0,
        theta=theta,
        kappa=0.0,
        conf=conf,
        n_bands=n,
        quadratic=False,
        disconnected=False,
    )


def fit_line_trajectory(
    features: LineFeatures,
    cfg: VisionConfig,
    crop_center: float,
    crop_width: float,
    lookahead_frac: float = 0.0,
    occluded_band_indices: frozenset[int] = frozenset(),
) -> TrajectoryFit:
    """Weighted polynomial fit of the line centerline across scan bands.

    Uses each band's best run as a sample (y_mid, cx) weighted by a per-band
    confidence. Falls back from quadratic to linear when fewer than 4 bands are
    available, so dashed lines (sparse bands) still yield a stable estimate.

    Bands listed in occluded_band_indices are known dead zones (arm/gripper):
    they never contribute samples and are removed from the confidence
    denominator so a fixed obstruction is not read as line loss.
    """
    if cfg.fit_mode == "classic":
        return _fit_classic_contour_trajectory(
            features,
            cfg,
            crop_center,
            crop_width,
            occluded_band_indices,
        )

    half_width = max(crop_width / 2.0, 1.0)
    line_width = features.line_width_px if features.line_width_px > 0 else 8.0
    preview_dir, preview_e, preview_theta, preview_conf = _preview_terms(features, crop_center, half_width)

    visible_bands = max(1, cfg.band_count - len(occluded_band_indices))

    samples: list[tuple[int, float, float, float]] = []
    y_min = float("inf")
    for band in features.bands:
        if band.index in occluded_band_indices or band.best is None:
            continue
        band_h = float(max(1, band.y1 - band.y0))
        conf = _band_confidence(band.best, band_h, line_width, cfg)
        if conf <= 0.0:
            continue
        y_mid = (band.y0 + band.y1) / 2.0
        y_min = min(y_min, float(band.y0))
        # Weight the bottom of the crop (nearest the car) more heavily.
        near_bias = 1.0 + 0.5 * (band.index == 0)
        samples.append((band.index, y_mid, band.best.cx, conf * near_bias))

    n = len(samples)
    if n == 0:
        return TrajectoryFit(
            found=False,
            preview_dir=preview_dir,
            preview_e=preview_e,
            preview_theta=preview_theta,
            preview_conf=preview_conf,
        )

    samples.sort(key=lambda item: item[0])
    # The crop can lose the connecting arc during tight roundabout turns. In
    # that case far bands may belong to a different visible segment; do not let
    # a polynomial stitch that gap and invent a huge heading for pivot assist.
    max_jump_px = max(40.0, crop_width * 0.16)
    near_samples = [samples[0]]
    disconnected = False
    for sample in samples[1:]:
        prev_index, _prev_y, prev_x, _prev_w = near_samples[-1]
        index, _y, x, _w = sample
        if index == prev_index + 1 and abs(x - prev_x) <= max_jump_px:
            near_samples.append(sample)
        else:
            disconnected = True
            break
    if preview_conf > 0.0 and features.preview is not None and features.preview.gap_bands > 0:
        disconnected = True

    if disconnected and len(near_samples) >= 2:
        fit_samples = near_samples
    else:
        fit_samples = samples

    ys = [sample[1] for sample in fit_samples]
    xs = [sample[2] for sample in fit_samples]
    ws = [sample[3] for sample in fit_samples]
    fit_n = len(fit_samples)

    y_arr = np.asarray(ys, dtype=np.float64)
    x_arr = np.asarray(xs, dtype=np.float64)
    w_arr = np.asarray(ws, dtype=np.float64)

    if fit_n == 1:
        e0 = float(max(-1.0, min(1.0, (float(x_arr[0]) - crop_center) / half_width)))
        e_look = preview_e if preview_conf > 0.0 else e0
        conf = float(min(1.0, w_arr.sum() / (visible_bands * 0.9)))
        if disconnected:
            conf *= 0.45
        return TrajectoryFit(
            found=True,
            e0=e0,
            e_look=e_look,
            theta=0.0,
            kappa=0.0,
            conf=conf,
            n_bands=fit_n,
            quadratic=False,
            disconnected=disconnected,
            preview_dir=preview_dir,
            preview_e=preview_e,
            preview_theta=preview_theta,
            preview_conf=preview_conf,
        )

    # RANSAC-lite: with enough bands, drop the single worst residual once so a
    # roundabout entry/exit stub or a corner branch cannot drag the whole fit.
    degree = 2 if fit_n >= 4 else 1
    use_quadratic = degree == 2
    coeffs = np.polyfit(y_arr, x_arr, degree, w=w_arr)
    if fit_n >= 5:
        resid = np.abs(np.polyval(coeffs, y_arr) - x_arr)
        drop = int(np.argmax(resid))
        keep = np.ones(fit_n, dtype=bool)
        keep[drop] = False
        y_arr, x_arr, w_arr = y_arr[keep], x_arr[keep], w_arr[keep]
        fit_n = int(keep.sum())
        degree = 2 if fit_n >= 4 else 1
        use_quadratic = degree == 2
        coeffs = np.polyfit(y_arr, x_arr, degree, w=w_arr)

    y_bottom = float(y_arr.max())
    cx_bottom = float(np.polyval(coeffs, y_bottom))
    e0 = float(max(-1.0, min(1.0, (cx_bottom - crop_center) / half_width)))

    if use_quadratic:
        a2, a1, _a0 = coeffs
        slope = 2.0 * a2 * y_bottom + a1        # dcx/dy at the bottom
        curvature = 2.0 * a2
    else:
        a1, _a0 = coeffs
        slope = a1
        curvature = 0.0

    # Car forward is "up" the image (decreasing y). Positive theta => line bends
    # toward +x (right) as we look ahead. Sign chosen so theta and e0 agree.
    theta = float(np.arctan2(-slope, 1.0))
    kappa = float(max(-1.0, min(1.0, curvature * half_width)))

    # Lookahead sample: the fitted curve some distance ahead (toward smaller y).
    if lookahead_frac > 0.0 and y_min < float("inf"):
        y_look = y_bottom - lookahead_frac * (y_bottom - y_min)
        cx_look = float(np.polyval(coeffs, y_look))
        e_look = float(max(-1.0, min(1.0, (cx_look - crop_center) / half_width)))
    else:
        e_look = e0
    if preview_conf > 0.0 and (lookahead_frac > 0.0 or disconnected):
        e_look = preview_e

    conf = float(min(1.0, w_arr.sum() / (visible_bands * 0.9)))
    if disconnected:
        # This is still a usable near-field observation, but not a trustworthy
        # whole-trajectory fit. Let the temporal filter/predictor carry history
        # instead of treating a blind-zone bridge as high-confidence curvature.
        if fit_n < 3:
            theta = 0.0
        kappa = 0.0
        conf *= 0.55
    return TrajectoryFit(
        found=True,
        e0=e0,
        e_look=e_look,
        theta=theta,
        kappa=kappa,
        conf=conf,
        n_bands=fit_n,
        quadratic=use_quadratic,
        disconnected=disconnected,
        preview_dir=preview_dir,
        preview_e=preview_e,
        preview_theta=preview_theta,
        preview_conf=preview_conf,
    )


def draw_debug_overlay(frame_bgr: np.ndarray, features: LineFeatures, trigger_y_frac: float, crop_center: float | None = None) -> np.ndarray:
    """Draw scan bands, line center, branches, and the corner trigger line."""

    out = frame_bgr.copy()
    height, width = out.shape[:2]
    if crop_center is None:
        crop_center = width / 2.0
    cv.line(out, (int(crop_center), 0), (int(crop_center), height), (255, 0, 0), 1)
    cv.line(out, (0, int(height * trigger_y_frac)), (width, int(height * trigger_y_frac)), (255, 0, 255), 2)
    for band in features.bands:
        yy = (band.y0 + band.y1) // 2
        cv.line(out, (0, yy), (width, yy), (80, 80, 80), 1)
        for run in band.runs:
            if band.best == run:
                cv.rectangle(out, (run.x0, band.y0), (run.x1, band.y1 - 1), (0, 220, 0), 2)
            else:
                cv.line(out, (int(run.cx), yy - 3), (int(run.cx), yy + 3), (90, 90, 90), 1)
    for branch in [features.branch_left, features.branch_right]:
        if branch is None:
            continue
        band = features.bands[branch.band_index]
        cv.rectangle(out, (branch.run.x0, band.y0), (branch.run.x1, band.y1 - 1), (0, 165, 255), 2)
    if len(features.path) >= 2:
        pts = [(int(round(point.x)), int(round(point.y))) for point in features.path]
        for p0, p1 in zip(pts, pts[1:]):
            cv.line(out, p0, p1, (0, 255, 0), 2)
        for point in pts:
            cv.circle(out, point, 3, (0, 255, 0), -1)
    if features.preview is not None:
        preview = features.preview
        p0 = (int(round(preview.anchor_x)), int(round(preview.anchor_y)))
        p1 = (int(round(preview.target_x)), int(round(preview.target_y)))
        cv.line(out, p0, p1, (0, 140, 255), 2)
        cv.circle(out, p1, 5, (0, 140, 255), -1)
    return out
