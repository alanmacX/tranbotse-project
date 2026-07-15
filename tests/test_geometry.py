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
    def test_thin_low_contrast_grout_is_rejected_but_black_tape_is_kept(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        cv.line(mask, (25, H - 1), (125, 10), 255, 3)
        hard = np.zeros_like(mask)
        near = np.zeros_like(mask)

        grout_gray = np.full((H, W), 110, dtype=np.uint8)
        grout_gray[mask > 0] = 63  # 63/110: latest night grout signature.
        grout = _filter_preprocess_components(
            mask, hard, near, min_thickness_px=2.5, gray=grout_gray,
        )
        self.assertEqual(cv.countNonZero(grout), 0)

        tape_gray = np.full((H, W), 112, dtype=np.uint8)
        tape_gray[mask > 0] = 30  # Far tape is thin but decisively black.
        tape = _filter_preprocess_components(
            mask, hard, near, min_thickness_px=2.5, gray=tape_gray,
        )
        self.assertGreater(cv.countNonZero(tape), 0)

    def test_large_sparse_shape_near_reflection_is_rejected(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        cv.rectangle(mask, (50, 55), (68, H - 1), 255, -1)
        cv.rectangle(mask, (50, 55), (145, 73), 255, -1)
        hard = np.zeros_like(mask)
        near = np.zeros_like(mask)
        cv.rectangle(near, (45, 45), (95, 95), 255, -1)
        clean = _filter_preprocess_components(mask, hard, near)
        self.assertEqual(cv.countNonZero(clean), 0)

    def test_near_reflection_exemption_requires_track_anchor_corridor(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        cv.rectangle(mask, (50, 55), (68, H - 1), 255, -1)
        cv.rectangle(mask, (50, 55), (145, 73), 255, -1)
        hard = np.zeros_like(mask)
        near = np.zeros_like(mask)
        cv.rectangle(near, (45, 45), (95, 95), 255, -1)

        anchored = _filter_preprocess_components(
            mask, hard, near, anchor_x=60.0, anchor_margin_px=32.0,
        )
        off_corridor = _filter_preprocess_components(
            mask, hard, near, anchor_x=10.0, anchor_margin_px=32.0,
        )
        self.assertGreater(cv.countNonZero(anchored), 0)
        self.assertEqual(cv.countNonZero(off_corridor), 0)

    def test_wide_geometry_roi_keeps_anchored_l_bend_near_reflection(self):
        height, width, center = 285, 640, 365
        mask = np.zeros((height, width), dtype=np.uint8)
        cv.rectangle(mask, (center - 10, 116), (center + 10, 238), 255, -1)
        cv.rectangle(mask, (center, 116), (center + 87, 136), 255, -1)
        hard = np.zeros_like(mask)
        near = np.zeros_like(mask)
        cv.rectangle(near, (center - 28, 103), (center + 45, 176), 255, -1)

        unanchored = _filter_preprocess_components(mask, hard, near)
        anchored = _filter_preprocess_components(
            mask, hard, near, anchor_x=float(center), anchor_margin_px=103.0,
        )

        self.assertEqual(cv.countNonZero(unanchored), 0)
        self.assertGreater(cv.countNonZero(anchored), 0)

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
