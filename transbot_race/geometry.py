from __future__ import annotations

import numpy as np

try:  # OpenCV is present on the robot and in the test env; keep import lazy-safe.
    import cv2 as cv
except Exception:  # pragma: no cover - import guard only
    cv = None  # type: ignore

from .config import OcclusionConfig, PerspectiveConfig


class PerspectiveTransformer:
    """Optional lens-undistort + inverse-perspective (bird's-eye) warp.

    Constructed once (matrices parsed from config) and reused per frame. When
    the config is disabled it is a pass-through, so the rest of the pipeline is
    identical to raw-crop operation until a calibration is supplied.
    """

    def __init__(self, cfg: PerspectiveConfig) -> None:
        self.cfg = cfg
        self._camera_matrix = _as_3x3(cfg.camera_matrix)
        self._dist = _as_vec(cfg.dist_coeffs)
        self._homography = _as_3x3(cfg.homography)
        self._out_size = (int(cfg.output_width), int(cfg.output_height))

    @property
    def active(self) -> bool:
        return self.cfg.enabled and self._homography is not None

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if not self.cfg.undistort or self._camera_matrix is None or self._dist is None:
            return frame
        return cv.undistort(frame, self._camera_matrix, self._dist)

    def to_birdseye(self, crop: np.ndarray) -> np.ndarray:
        """Warp a crop into the calibrated bird's-eye view (or return as-is)."""
        if not self.active:
            return crop
        return cv.warpPerspective(crop, self._homography, self._out_size)

    def scan_scale(self, crop_width: float) -> float:
        """Ground cm per normalized-half-width unit, for reporting only.

        In bird's-eye space e0 is still normalized to [-1, 1] of the ROI half
        width, but that half width now maps linearly to ground distance.
        """
        if not self.active or self.cfg.px_per_cm <= 0:
            return crop_width
        return (self._out_size[0] / 2.0) / self.cfg.px_per_cm


def apply_occlusion(mask: np.ndarray, cfg: OcclusionConfig) -> np.ndarray:
    """Zero out known static dead zones in a binary mask (in place safe copy)."""
    if not cfg.enabled or not cfg.rects:
        return mask
    out = mask.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in cfg.rects:
        xa, xb = max(0, min(x0, x1)), min(w, max(x0, x1))
        ya, yb = max(0, min(y0, y1)), min(h, max(y0, y1))
        if xb > xa and yb > ya:
            out[ya:yb, xa:xb] = 0
    return out


def band_is_occluded(y0: int, y1: int, width: int, cfg: OcclusionConfig) -> bool:
    """True when a scan band lies (mostly) inside an occlusion rectangle.

    Used so a band that only ever sees the arm/gripper is dropped from the
    confidence denominator rather than counted as a missing line.
    """
    if not cfg.enabled or not cfg.rects:
        return False
    band_area = max(1, (y1 - y0) * width)
    covered = 0
    for x0, ry0, x1, ry1 in cfg.rects:
        ix0, ix1 = max(0, min(x0, x1)), min(width, max(x0, x1))
        iy0, iy1 = max(y0, min(ry0, ry1)), min(y1, max(ry0, ry1))
        if ix1 > ix0 and iy1 > iy0:
            covered += (ix1 - ix0) * (iy1 - iy0)
    return covered >= 0.6 * band_area


def _as_3x3(values: tuple[float, ...] | None) -> np.ndarray | None:
    if not values or len(values) != 9:
        return None
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def _as_vec(values: tuple[float, ...] | None) -> np.ndarray | None:
    if not values:
        return None
    return np.asarray(values, dtype=np.float64).reshape(-1)
