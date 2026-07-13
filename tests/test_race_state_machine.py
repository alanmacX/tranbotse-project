import unittest

import cv2 as cv
import numpy as np

from transbot_race.config import RaceConfig
from transbot_race.state_machine import RaceState, RaceStateMachine, TrackMode
from transbot_race.vision import TrajectoryFit, fit_line_trajectory, scan_line_features

W, H, CENTER = 180, 200, 90


def fit_of(mask, cfg):
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


def right_angle(line_w=18, corner_y=150):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.rectangle(mask, (CENTER - line_w // 2, corner_y), (CENTER + line_w // 2, H - 1), 255, -1)
    cv.rectangle(mask, (CENTER - line_w // 2, corner_y - line_w // 2), (W - 1, corner_y + line_w // 2), 255, -1)
    return mask


def empty():
    return np.zeros((H, W), dtype=np.uint8)


def blind_right(line_w=18):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.rectangle(mask, (CENTER - line_w // 2, 150), (CENTER + line_w // 2, H - 1), 255, -1)
    cv.rectangle(mask, (145, 0), (170, 90), 255, -1)
    return mask


def replay(sm, mask, cfg, start=0.0, n=6, dt=0.1):
    fit = fit_of(mask, cfg)
    cmd = None
    for i in range(n):
        cmd = sm.step(fit, now=start + i * dt)
    return cmd


class UnifiedTrackerTests(unittest.TestCase):
    def test_blank_start_waits_without_search_rotation(self):
        sm = RaceStateMachine(RaceConfig())
        cmd = sm.step(TrajectoryFit(found=False), now=0.0)
        self.assertEqual(cmd.reason, "await_first_line")
        self.assertEqual(cmd.v, 0.0)
        self.assertEqual(cmd.w, 0.0)

    def test_straight_tracks_forward(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        cmd = replay(sm, straight(), cfg)
        self.assertEqual(sm.state, RaceState.TRACK)
        self.assertGreater(cmd.v, 0.0)
        self.assertAlmostEqual(cmd.w, 0.0, delta=0.05)

    def test_offset_line_turns_toward_center(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        cmd = replay(sm, straight(x=CENTER + 35), cfg)
        self.assertEqual(sm.state, RaceState.TRACK)
        # Line to the right of center -> steer right (w < 0 with invert_turn off).
        self.assertLess(cmd.w, 0.0)

    def test_right_angle_enters_pivot_not_a_state(self):
        cfg = poly_config()
        cfg.tracker.e_pivot = 0.55
        cfg.tracker.theta_pivot = 0.65
        sm = RaceStateMachine(cfg)
        cmd = replay(sm, right_angle(), cfg, n=8)
        # Still the single TRACK state, but pivot sub-mode with near-zero speed.
        self.assertEqual(sm.state, RaceState.TRACK)
        self.assertEqual(cmd.mode, TrackMode.PIVOT)
        self.assertAlmostEqual(cmd.v, cfg.tracker.v_max * cfg.tracker.v_pivot_ratio, places=4)
        self.assertGreater(abs(cmd.w), 0.0)

    def test_dashed_gap_keeps_moving_via_predict(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        # Build confidence on the line, then a couple of blank frames.
        replay(sm, straight(), cfg, n=4)
        c1 = sm.step(fit_of(empty(), cfg), now=1.0)
        c2 = sm.step(fit_of(empty(), cfg), now=1.1)
        self.assertEqual(sm.state, RaceState.TRACK)
        self.assertIn(c2.mode, (TrackMode.PREDICT, TrackMode.FOLLOW))
        self.assertGreater(c2.v, 0.0)  # keeps rolling through the gap

    def test_long_gap_goes_lost_then_searches_then_stops(self):
        cfg = RaceConfig()
        cfg.tracker.search_timeout_sec = 0.5
        sm = RaceStateMachine(cfg)
        replay(sm, straight(), cfg, n=4)
        blank = fit_of(empty(), cfg)
        # Enough blank frames to decay confidence below conf_lost.
        for i in range(20):
            cmd = sm.step(blank, now=1.0 + i * 0.05)
        # Either searching or already timed out to STOPPED; search never w=0.
        if sm.state == RaceState.LOST:
            self.assertGreaterEqual(abs(cmd.w), cfg.tracker.w_search_min)
        cmd = sm.step(blank, now=10.0)
        self.assertEqual(sm.state, RaceState.STOPPED)

    def test_obstacle_stops_immediately(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        cmd = sm.step(fit_of(straight(), cfg), now=0.0, obstacle=True)
        self.assertEqual(sm.state, RaceState.STOPPED)
        self.assertEqual(cmd.v, 0.0)
        self.assertEqual(cmd.w, 0.0)

    def test_lookahead_steers_toward_ahead_error(self):
        # With pure-pursuit enabled, an approaching bend (line offset ahead but
        # centered at the bottom) still turns toward the lookahead point.
        cfg = poly_config()
        cfg.tracker.lookahead_frac = 0.6
        cfg.tracker.k_theta = 0.0
        cfg.tracker.k_ff = 0.0
        sm = RaceStateMachine(cfg)
        features = scan_line_features(right_angle(corner_y=120), cfg.vision, crop_center=CENTER)
        fit = fit_line_trajectory(
            features, cfg.vision, crop_center=CENTER, crop_width=W, lookahead_frac=0.6
        )
        cmd = None
        for i in range(8):
            cmd = sm.step(fit, now=i * 0.1)
        self.assertEqual(sm.state, RaceState.TRACK)
        self.assertGreater(abs(cmd.w), 0.0)

    def test_blind_preview_executes_plan_inside_track(self):
        cfg = poly_config()
        cfg.tracker.preview_plan_enabled = True
        cfg.tracker.preview_forward_sec = 0.2
        cfg.tracker.preview_turn_sec = 0.4
        sm = RaceStateMachine(cfg)
        fit = fit_of(blind_right(), cfg)
        cmd1 = sm.step(fit, now=0.0)
        cmd2 = sm.step(fit, now=0.25)
        self.assertEqual(sm.state, RaceState.TRACK)
        self.assertEqual(cmd1.mode, TrackMode.PLAN)
        self.assertEqual(cmd1.reason, "preview_forward")
        self.assertEqual(cmd2.mode, TrackMode.PLAN)
        self.assertEqual(cmd2.reason, "preview_turn")
        self.assertLess(cmd2.w, 0.0)



if __name__ == "__main__":
    unittest.main()
