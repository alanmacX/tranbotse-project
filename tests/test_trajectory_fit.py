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


def dashed(x=CENTER, line_w=18, dash=22, gap=18):
    mask = np.zeros((H, W), dtype=np.uint8)
    y = 0
    while y < H:
        cv.rectangle(mask, (x - line_w // 2, y), (x + line_w // 2, min(H - 1, y + dash)), 255, -1)
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
