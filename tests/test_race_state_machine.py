import unittest

import cv2 as cv
import numpy as np

from transbot_race.config import RaceConfig
from transbot_race.state_machine import RaceState, RaceStateMachine
from transbot_race.vision import scan_line_features


def mask_with_vertical_line(width=180, height=120, x=90, line_w=18):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.rectangle(mask, (x - line_w // 2, 0), (x + line_w // 2, height - 1), 255, -1)
    return mask


def add_branch(mask, direction="right", y=45, thickness=18):
    h, w = mask.shape
    center = w // 2
    if direction == "right":
        cv.rectangle(mask, (center, y - thickness // 2), (w - 1, y + thickness // 2), 255, -1)
    else:
        cv.rectangle(mask, (0, y - thickness // 2), (center, y + thickness // 2), 255, -1)
    return mask


def mask_with_curve(width=180, height=120, line_w=18):
    mask = np.zeros((height, width), dtype=np.uint8)
    points = []
    for y in range(height):
        x = int(width // 2 + 28 * np.sin((y / height) * np.pi / 2.0))
        points.append((x, y))
    for p0, p1 in zip(points, points[1:]):
        cv.line(mask, p0, p1, 255, line_w)
    return mask


class RaceStateMachineTests(unittest.TestCase):
    def test_straight_line_follows_without_corner(self):
        cfg = RaceConfig()
        features = scan_line_features(mask_with_vertical_line(x=90), cfg.vision, crop_center=90)
        sm = RaceStateMachine(cfg)
        cmd = sm.step(features, now=0.0)
        self.assertEqual(sm.state, RaceState.LINE_FOLLOW)
        self.assertEqual(cmd.reason, "line_follow")
        self.assertAlmostEqual(cmd.w, 0.0, places=3)

    def test_right_corner_triggers_timed_forward(self):
        cfg = RaceConfig()
        cfg.vision.trigger_y_frac = 0.30
        sm = RaceStateMachine(cfg)
        mask = add_branch(mask_with_vertical_line(), "right", y=50)
        features = scan_line_features(mask, cfg.vision, crop_center=90)
        sm.step(features, now=0.0)
        cmd = sm.step(features, now=0.1)
        self.assertEqual(sm.state, RaceState.TIMED_FORWARD)
        self.assertEqual(cmd.reason, "corner_forward")
        self.assertLess(sm.active_turn_dir, 0)

    def test_left_corner_triggers_opposite_turn(self):
        cfg = RaceConfig()
        cfg.vision.trigger_y_frac = 0.30
        cfg.corner.mode = "left"
        sm = RaceStateMachine(cfg)
        mask = add_branch(mask_with_vertical_line(), "left", y=50)
        features = scan_line_features(mask, cfg.vision, crop_center=90)
        sm.step(features, now=0.0)
        sm.step(features, now=0.1)
        self.assertEqual(sm.state, RaceState.TIMED_FORWARD)
        self.assertGreater(sm.active_turn_dir, 0)

    def test_thin_floor_noise_is_rejected(self):
        cfg = RaceConfig()
        mask = mask_with_vertical_line()
        cv.line(mask, (0, 40), (179, 40), 255, 2)
        features = scan_line_features(mask, cfg.vision, crop_center=90)
        self.assertIsNone(features.branch_left)
        self.assertIsNone(features.branch_right)

    def test_curve_uses_line_follow_not_corner(self):
        cfg = RaceConfig()
        features = scan_line_features(mask_with_curve(), cfg.vision, crop_center=90)
        sm = RaceStateMachine(cfg)
        cmd = sm.step(features, now=0.0)
        self.assertEqual(sm.state, RaceState.LINE_FOLLOW)
        self.assertEqual(cmd.reason, "line_follow")

    def test_branch_above_trigger_does_not_turn_yet(self):
        cfg = RaceConfig()
        cfg.vision.trigger_y_frac = 0.60
        sm = RaceStateMachine(cfg)
        mask = add_branch(mask_with_vertical_line(), "right", y=35)
        features = scan_line_features(mask, cfg.vision, crop_center=90)
        sm.step(features, now=0.0)
        cmd = sm.step(features, now=0.1)
        self.assertEqual(sm.state, RaceState.LINE_FOLLOW)
        self.assertEqual(cmd.reason, "line_follow")

    def test_gap_blind_and_recover(self):
        cfg = RaceConfig()
        cfg.gap.missing_frames = 2
        sm = RaceStateMachine(cfg)
        line = scan_line_features(mask_with_vertical_line(), cfg.vision, crop_center=90)
        missing = scan_line_features(np.zeros((120, 180), dtype=np.uint8), cfg.vision, crop_center=90)
        sm.step(line, now=0.0)
        sm.step(missing, now=0.1)
        cmd = sm.step(missing, now=0.2)
        self.assertEqual(sm.state, RaceState.GAP_BLIND)
        self.assertEqual(cmd.reason, "gap_blind")
        cmd = sm.step(line, now=0.3)
        self.assertEqual(sm.state, RaceState.LINE_FOLLOW)
        self.assertEqual(cmd.reason, "line_follow")

    def test_search_never_stalls_when_centered(self):
        # Line lost with err_norm ~= 0 must still sweep, not freeze at w=0.
        cfg = RaceConfig()
        cfg.gap.missing_frames = 1
        cfg.gap.blind_sec = 0.0
        sm = RaceStateMachine(cfg)
        missing = scan_line_features(np.zeros((120, 180), dtype=np.uint8), cfg.vision, crop_center=90)
        sm.step(missing, now=0.1)
        cmd = sm.step(missing, now=0.2)
        self.assertEqual(cmd.reason, "line_missing")
        self.assertGreaterEqual(abs(cmd.w), cfg.gap.search_w_min)

    def test_search_times_out_to_stopped(self):
        cfg = RaceConfig()
        cfg.gap.missing_frames = 1
        cfg.gap.blind_sec = 0.0
        cfg.gap.search_timeout_sec = 1.0
        sm = RaceStateMachine(cfg)
        missing = scan_line_features(np.zeros((120, 180), dtype=np.uint8), cfg.vision, crop_center=90)
        sm.step(missing, now=0.1)
        sm.step(missing, now=0.2)  # sweep begins here
        cmd = sm.step(missing, now=2.0)
        self.assertEqual(sm.state, RaceState.STOPPED)
        self.assertEqual(cmd.reason, "search_timeout")

    def test_reacquire_times_out_to_stopped(self):
        cfg = RaceConfig()
        cfg.corner.reacquire_timeout_sec = 1.0
        sm = RaceStateMachine(cfg)
        sm.state = RaceState.REACQUIRE
        sm.state_started_at = 0.0
        sm.active_turn_dir = 1.0
        missing = scan_line_features(np.zeros((120, 180), dtype=np.uint8), cfg.vision, crop_center=90)
        cmd = sm.step(missing, now=2.0)
        self.assertEqual(sm.state, RaceState.STOPPED)
        self.assertEqual(cmd.reason, "reacquire_timeout")


if __name__ == "__main__":
    unittest.main()
