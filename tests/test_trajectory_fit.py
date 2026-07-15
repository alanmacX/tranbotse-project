import unittest

import cv2 as cv
import numpy as np

from transbot_race.config import RaceConfig
from transbot_race.vision import fit_line_trajectory, scan_line_features


W, H, CENTER = 180, 200, 90


def _fit(mask, cfg):
    features = scan_line_features(mask, cfg.vision, crop_center=CENTER)
    return fit_line_trajectory(features, cfg.vision, crop_center=CENTER, crop_width=W)


def poly_config():
    cfg = RaceConfig()
    cfg.vision.fit_mode = "poly"
    return cfg


def straight(x=CENTER, line_w=18):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.rectangle(mask, (x - line_w // 2, 0), (x + line_w // 2, H - 1), 255, -1)
    return mask


def diagonal(line_w=18, dx=40):
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = [(int(CENTER + dx * (y / H)), y) for y in range(H)]
    for p0, p1 in zip(pts, pts[1:]):
        cv.line(mask, p0, p1, 255, line_w)
    return mask


def arc(line_w=18, amp=45):
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = [(int(CENTER + amp * np.sin((y / H) * np.pi / 2.0)), y) for y in range(H)]
    for p0, p1 in zip(pts, pts[1:]):
        cv.line(mask, p0, p1, 255, line_w)
    return mask


def dashed(x=CENTER, line_w=18, dash=22, gap=18, phase=0, dx=0):
    mask = np.zeros((H, W), dtype=np.uint8)
    y = phase - (dash + gap)
    while y < H:
        y0 = max(0, y)
        y1 = min(H - 1, y + dash)
        if y0 <= y1:
            x0 = int(x + dx * (y0 / H))
            x1 = int(x + dx * (y1 / H))
            cv.line(mask, (x0, y0), (x1, y1), 255, line_w)
        y += dash + gap
    return mask


def right_angle(line_w=18, corner_y=90):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.rectangle(mask, (CENTER - line_w // 2, corner_y), (CENTER + line_w // 2, H - 1), 255, -1)
    cv.rectangle(mask, (CENTER - line_w // 2, corner_y - line_w // 2), (W - 1, corner_y + line_w // 2), 255, -1)
    return mask


def straight_with_far_distractors(line_w=18):
    mask = straight(line_w=line_w)
    # Tile seams / glare blobs far from the accepted near-field path should not
    # be stitched into the line estimate.
    cv.rectangle(mask, (145, 20), (170, 120), 255, -1)
    cv.rectangle(mask, (135, 0), (178, 18), 255, -1)
    return mask


def blind_zone_split(line_w=18):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.rectangle(mask, (CENTER - line_w // 2, 150), (CENTER + line_w // 2, H - 1), 255, -1)
    cv.rectangle(mask, (145, 0), (170, 90), 255, -1)
    return mask


def blind_zone_with_border_seam(line_w=18):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.rectangle(mask, (CENTER - line_w // 2, 150), (CENTER + line_w // 2, H - 1), 255, -1)
    cv.line(mask, (W - 2, 0), (W - 10, 105), 255, 8)
    return mask


def straight_with_chassis_bar(line_w=18):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.rectangle(mask, (CENTER - line_w // 2, 0), (CENTER + line_w // 2, H - 34), 255, -1)
    cv.rectangle(mask, (0, H - 18), (150, H - 1), 255, -1)
    return mask


class TrajectoryFitTests(unittest.TestCase):
    def test_far_connected_corner_is_a_branch_before_reaching_center(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        cv.rectangle(mask, (62, 35), (80, H - 1), 255, -1)
        cv.rectangle(mask, (62, 35), (W - 1, 53), 255, -1)
        cfg = RaceConfig()
        features = scan_line_features(mask, cfg.vision, crop_center=50)
        self.assertIsNotNone(features.branch_right)

    def test_straight_is_centered_low_curvature(self):
        fit = _fit(straight(), RaceConfig())
        self.assertTrue(fit.found)
        self.assertLess(abs(fit.e0), 0.1)
        self.assertLess(abs(fit.theta), 0.15)
        self.assertLess(abs(fit.kappa), 0.15)

    def test_offset_line_reports_signed_error(self):
        left = _fit(straight(x=CENTER - 35), RaceConfig())
        right = _fit(straight(x=CENTER + 35), RaceConfig())
        self.assertLess(left.e0, -0.1)
        self.assertGreater(right.e0, 0.1)

    def test_diagonal_reports_nonzero_heading(self):
        fit = _fit(diagonal(), RaceConfig())
        self.assertTrue(fit.found)
        self.assertGreater(abs(fit.theta), 0.1)

    def test_arc_reports_curvature(self):
        fit = _fit(arc(), poly_config())
        self.assertTrue(fit.found)
        self.assertGreater(abs(fit.kappa), 0.05)

    def test_dashed_line_still_found(self):
        fit = _fit(dashed(), RaceConfig())
        self.assertTrue(fit.found)
        self.assertGreaterEqual(fit.n_bands, 2)
        self.assertLess(abs(fit.e0), 0.15)

    def test_dashed_straight_pose_is_stable_across_segment_phases(self):
        fits = [
            _fit(dashed(x=CENTER + 22, phase=phase, dx=18), RaceConfig())
            for phase in (0, 8, 16, 24, 32)
        ]
        self.assertTrue(all(fit.found for fit in fits))
        self.assertTrue(all(fit.e0 > 0.0 for fit in fits))
        self.assertLess(max(fit.e0 for fit in fits) - min(fit.e0 for fit in fits), 0.12)
        self.assertLess(max(fit.theta for fit in fits) - min(fit.theta for fit in fits), 0.12)

    def test_intentional_sliding_window_gap_is_not_disconnected(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        # Two collinear dashes separated by one complete scan band.  The
        # sliding-window policy explicitly permits this gap, so the fit layer
        # must not contradict it and halve confidence afterwards.
        cv.rectangle(mask, (CENTER - 9, 172), (CENTER + 9, H - 1), 255, -1)
        cv.rectangle(mask, (CENTER - 9, 115), (CENTER + 9, 142), 255, -1)
        fit = _fit(mask, RaceConfig())
        self.assertTrue(fit.found)
        self.assertFalse(fit.disconnected)
        self.assertGreaterEqual(fit.n_bands, 2)
        self.assertGreater(fit.conf, 0.30)

    def test_far_only_component_is_visible_but_cannot_control(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        cv.line(mask, (18, 105), (130, 0), 255, 18)
        fit = _fit(mask, RaceConfig())

        self.assertTrue(fit.found)
        self.assertGreaterEqual(fit.nearest_band_index, 3)
        self.assertFalse(fit.has_near_support)
        self.assertFalse(fit.control_valid)
        self.assertFalse(fit.quadratic)
        # The diagnostic error is evaluated at the nearest observed row, not
        # extrapolated to the unseen bottom reference where it would saturate.
        self.assertGreater(fit.e0, -0.95)

    def test_band_two_is_inside_near_control_horizon(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        cv.line(mask, (CENTER + 12, 125), (CENTER + 30, 0), 255, 18)
        fit = _fit(mask, RaceConfig())

        self.assertEqual(fit.nearest_band_index, 2)
        self.assertTrue(fit.has_near_support)
        self.assertTrue(fit.control_valid)

    def test_right_angle_produces_strong_signal(self):
        # A right-angle corner should surface as large heading/curvature so the
        # continuous controller slows and pivots - no dedicated corner state.
        fit = _fit(right_angle(), poly_config())
        self.assertTrue(fit.found)
        self.assertGreater(abs(fit.theta) + abs(fit.kappa), 0.2)

    def test_far_distractors_do_not_drive_fit(self):
        fit = _fit(straight_with_far_distractors(), RaceConfig())
        self.assertTrue(fit.found)
        self.assertLess(abs(fit.e0), 0.15)
        self.assertLess(abs(fit.theta), 0.25)
        self.assertLess(abs(fit.kappa), 0.25)

    def test_blind_zone_split_lowers_confidence(self):
        fit = _fit(blind_zone_split(), poly_config())
        self.assertTrue(fit.found)
        self.assertTrue(fit.disconnected)
        self.assertLess(fit.conf, 0.4)
        self.assertLess(abs(fit.theta), 0.25)
        self.assertLess(abs(fit.kappa), 0.25)

    def test_blind_zone_split_latches_preview_direction(self):
        fit = _fit(blind_zone_split(), poly_config())
        self.assertEqual(fit.preview_dir, 1)
        self.assertGreater(fit.preview_conf, 0.3)
        self.assertGreater(fit.preview_e, 0.2)

    def test_border_seam_does_not_become_preview(self):
        fit = _fit(blind_zone_with_border_seam(), RaceConfig())
        self.assertTrue(fit.found)
        self.assertEqual(fit.preview_dir, 0)
        self.assertEqual(fit.preview_conf, 0.0)

    def test_bottom_chassis_bar_is_not_path_anchor(self):
        cfg = RaceConfig()
        features = scan_line_features(straight_with_chassis_bar(), cfg.vision, crop_center=CENTER)
        fit = fit_line_trajectory(features, cfg.vision, crop_center=CENTER, crop_width=W)
        self.assertTrue(fit.found)
        self.assertGreaterEqual(fit.n_bands, 3)
        self.assertLess(abs(fit.e0), 0.15)
        self.assertTrue(features.bottom is None or features.bottom.width < W * 0.34)


if __name__ == "__main__":
    unittest.main()
