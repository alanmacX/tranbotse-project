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
class LineFeatures:
    found: bool
    err_norm: float = 0.0
    line_width_px: float = 0.0
    bands: tuple[ScanBand, ...] = ()
    bottom: Run | None = None
    mid: Run | None = None
    branch_left: BranchFeature | None = None
    branch_right: BranchFeature | None = None

    @property
    def has_corner(self) -> bool:
        return self.branch_left is not None or self.branch_right is not None


def preprocess_blackline(frame_bgr: np.ndarray, cfg: VisionConfig) -> np.ndarray:
    """Return a binary mask where likely black track pixels are 255."""

    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized = clahe.apply(gray)
    threshold = int(np.percentile(equalized, cfg.percentile))
    threshold = max(cfg.threshold_min, min(cfg.threshold_max, threshold))
    mask = (equalized <= threshold).astype(np.uint8) * 255
    mask = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    mask = cv.morphologyEx(
        mask,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_RECT, (5, 17)),
        iterations=1,
    )
    return mask


def _band_bounds(index: int, height: int, band_count: int) -> tuple[int, int]:
    """(y0, y1) of scan band `index`, counting from the bottom of the frame."""
    y1 = height - int(index * height / band_count)
    y0 = height - int((index + 1) * height / band_count)
    return y0, y1


def _runs_from_band(mask: np.ndarray, y0: int, y1: int, cfg: VisionConfig) -> tuple[Run, ...]:
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


def _best_run(runs: Iterable[Run], crop_center: float) -> Run | None:
    runs = tuple(runs)
    if not runs:
        return None
    return min(runs, key=lambda run: abs(run.cx - crop_center) - min(run.area, 800) * 0.001)


def scan_line_features(mask: np.ndarray, cfg: VisionConfig, crop_center: float | None = None) -> LineFeatures:
    """Extract line-center and left/right branch features from a black-line mask."""

    height, width = mask.shape[:2]
    if crop_center is None:
        crop_center = width / 2.0

    bands: list[ScanBand] = []
    for index in range(cfg.band_count):
        y0, y1 = _band_bounds(index, height, cfg.band_count)
        runs = _runs_from_band(mask, y0, y1, cfg)
        bands.append(ScanBand(index=index, y0=y0, y1=y1, runs=runs, best=_best_run(runs, crop_center)))

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
    # Normalize by the crop half-width so left/right deviations are symmetric
    # even when the crop is expanded asymmetrically (see CameraConfig expand_*).
    half_width = max(width / 2.0, 1.0)
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
    half_width = max(crop_width / 2.0, 1.0)
    line_width = features.line_width_px if features.line_width_px > 0 else 8.0

    visible_bands = max(1, cfg.band_count - len(occluded_band_indices))

    ys: list[float] = []
    xs: list[float] = []
    ws: list[float] = []
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
        ys.append(y_mid)
        xs.append(band.best.cx)
        ws.append(conf * near_bias)

    n = len(ys)
    if n == 0:
        return TrajectoryFit(found=False)

    y_arr = np.asarray(ys, dtype=np.float64)
    x_arr = np.asarray(xs, dtype=np.float64)
    w_arr = np.asarray(ws, dtype=np.float64)

    # RANSAC-lite: with enough bands, drop the single worst residual once so a
    # roundabout entry/exit stub or a corner branch cannot drag the whole fit.
    degree = 2 if n >= 4 else 1
    use_quadratic = degree == 2
    coeffs = np.polyfit(y_arr, x_arr, degree, w=w_arr)
    if n >= 5:
        resid = np.abs(np.polyval(coeffs, y_arr) - x_arr)
        drop = int(np.argmax(resid))
        keep = np.ones(n, dtype=bool)
        keep[drop] = False
        y_arr, x_arr, w_arr = y_arr[keep], x_arr[keep], w_arr[keep]
        n = int(keep.sum())
        degree = 2 if n >= 4 else 1
        use_quadratic = degree == 2
        coeffs = np.polyfit(y_arr, x_arr, degree, w=w_arr)

    y_bottom = float(y_arr.max())
    cx_bottom = float(np.polyval(coeffs, y_bottom))
    e0 = max(-1.0, min(1.0, (cx_bottom - crop_center) / half_width))

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
    kappa = max(-1.0, min(1.0, curvature * half_width))

    # Lookahead sample: the fitted curve some distance ahead (toward smaller y).
    if lookahead_frac > 0.0 and y_min < float("inf"):
        y_look = y_bottom - lookahead_frac * (y_bottom - y_min)
        cx_look = float(np.polyval(coeffs, y_look))
        e_look = max(-1.0, min(1.0, (cx_look - crop_center) / half_width))
    else:
        e_look = e0

    conf = float(min(1.0, w_arr.sum() / (visible_bands * 0.9)))
    return TrajectoryFit(
        found=True,
        e0=e0,
        e_look=e_look,
        theta=theta,
        kappa=kappa,
        conf=conf,
        n_bands=n,
        quadratic=use_quadratic,
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
            color = (0, 220, 0) if band.best == run else (180, 180, 0)
            cv.rectangle(out, (run.x0, band.y0), (run.x1, band.y1 - 1), color, 1)
    for branch in [features.branch_left, features.branch_right]:
        if branch is None:
            continue
        band = features.bands[branch.band_index]
        cv.rectangle(out, (branch.run.x0, band.y0), (branch.run.x1, band.y1 - 1), (0, 165, 255), 2)
    return out

