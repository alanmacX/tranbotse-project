from __future__ import annotations

import numpy as np

try:  # OpenCV is present on the robot and in the test env; keep import lazy-safe.
    import cv2 as cv
except Exception:  # pragma: no cover - import guard only
    cv = None  # type: ignore

from .config import OcclusionConfig


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
