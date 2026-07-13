import unittest

import cv2 as cv
import numpy as np

from transbot_race.config import OcclusionConfig, RaceConfig
from transbot_race.geometry import apply_occlusion, band_is_occluded
from transbot_race.vision import _filter_preprocess_components
from transbot_race.vision import fit_line_trajectory, scan_line_features


W, H, CENTER = 180, 200, 90


def straight(x=CENTER, line_w=18):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.rectangle(mask, (x - line_w // 2, 0), (x + line_w // 2, H - 1), 255, -1)
    return mask


class OcclusionTests(unittest.TestCase):
    def test_large_corner_survives_near_reflection_guard(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        cv.rectangle(mask, (50, 55), (68, H - 1), 255, -1)
        cv.rectangle(mask, (50, 55), (145, 73), 255, -1)
        hard = np.zeros_like(mask)
        near = np.zeros_like(mask)
        cv.rectangle(near, (45, 45), (95, 95), 255, -1)
        clean = _filter_preprocess_components(mask, hard, near)
        self.assertGreater(cv.countNonZero(clean), 2000)

    def test_disabled_is_passthrough(self):
        mask = straight()
        out = apply_occlusion(mask, OcclusionConfig(enabled=False, rects=((0, 0, W, H),)))
        self.assertTrue(np.array_equal(mask, out))

    def test_apply_zeros_rect(self):
        mask = np.full((H, W), 255, dtype=np.uint8)
        cfg = OcclusionConfig(enabled=True, rects=((10, 20, 40, 60),))
        out = apply_occlusion(mask, cfg)
        self.assertEqual(int(out[30, 20]), 0)
        self.assertEqual(int(out[0, 0]), 255)
        # Original untouched (copy semantics).
        self.assertEqual(int(mask[30, 20]), 255)

    def test_band_fully_covered_is_occluded(self):
        cfg = OcclusionConfig(enabled=True, rects=((0, 0, W, 30),))
        self.assertTrue(band_is_occluded(0, 30, W, cfg))
        self.assertFalse(band_is_occluded(100, 130, W, cfg))

    def test_occluded_band_excluded_from_fit(self):
        # A spurious bright blob in the bottom band should not corrupt the fit
        # once that band is declared occluded.
        mask = straight()
        mask[H - 25 : H, :] = 255  # bottom band saturated (arm reflection)
        cfg = RaceConfig()
        feats = scan_line_features(mask, cfg.vision, crop_center=CENTER)
        clean = fit_line_trajectory(
            feats, cfg.vision, crop_center=CENTER, crop_width=W,
            occluded_band_indices=frozenset({0}),
        )
        self.assertTrue(clean.found)
        self.assertLess(abs(clean.e0), 0.2)


if __name__ == "__main__":
    unittest.main()
