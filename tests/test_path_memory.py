import unittest
from dataclasses import replace
import math

import numpy as np

from transbot_race.config import GroundProjectionConfig, PathMemoryConfig
from transbot_race.path_memory import (
    CornerCommandDelay, GroundProjector, RollingPathPursuit, read_motion_sample,
)
from transbot_race.vision import LineFeatures, TrajectoryFit


def turn_fit(theta=0.4):
    return TrajectoryFit(found=True, e0=0.1, e_look=0.5, theta=theta, conf=0.9, n_bands=5)


class FakeBot:
    def __init__(self, value=None): self.value = value
    def get_motion_data(self):
        if isinstance(self.value, Exception): raise self.value
        return self.value


class PathStrategyTests(unittest.TestCase):
    def test_corner_event_holds_then_commits_to_detected_turn(self):
        cfg = PathMemoryConfig(camera_to_axle_m=0.02, corner_confirm_frames=3)
        gate = CornerCommandDelay(cfg)
        features = LineFeatures(found=True)
        gate.step(turn_fit(), features, 0.05, -0.08, 0.0, 0.0)
        gate.step(turn_fit(), features, 0.05, -0.09, 0.1, 0.1)
        held = gate.step(turn_fit(), features, 0.05, -0.10, 0.2, 0.1)
        self.assertEqual(held.status.reason, "waiting_margin")
        self.assertLessEqual(abs(held.w), cfg.corner_hold_max_w)
        self.assertAlmostEqual(gate.target_angle_rad, math.pi / 2.0, places=2)
        gate.step(turn_fit(), features, 0.05, -0.12, 0.3, 0.1)
        turning = gate.step(turn_fit(), features, 0.05, -0.15, 0.4, 0.1)
        self.assertEqual(turning.status.reason, "committed_turn")
        self.assertAlmostEqual(turning.w, -cfg.corner_replay_max_w)
        self.assertAlmostEqual(turning.v, 0.05 * cfg.corner_turn_speed_ratio)

    def test_corner_event_keeps_trigger_speed_when_live_tracker_loses_line(self):
        cfg = PathMemoryConfig(camera_to_axle_m=0.10, corner_confirm_frames=3)
        gate = CornerCommandDelay(cfg)
        features = LineFeatures(found=True)

        gate.step(turn_fit(), features, 0.06, -0.10, 0.0, 0.0)
        gate.step(turn_fit(), features, 0.06, -0.11, 0.1, 0.0)
        triggered = gate.step(turn_fit(), features, 0.06, -0.12, 0.2, 0.0)
        self.assertEqual(triggered.status.reason, "waiting_margin")
        self.assertAlmostEqual(triggered.v, 0.06)

        lost_fit = replace(turn_fit(), found=False, conf=0.0)
        waiting = gate.step(lost_fit, LineFeatures(found=False), 0.0, 0.0, 0.3, 0.04)
        self.assertEqual(waiting.status.reason, "waiting_margin")
        self.assertAlmostEqual(waiting.v, 0.06)
        self.assertAlmostEqual(waiting.w, 0.0)

    def test_corner_event_uses_measured_yaw_then_reacquires_line(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.01,
            corner_confirm_frames=3,
            corner_turn_angle_rad=1.2,
            corner_reacquire_angle_rad=0.5,
        )
        gate = CornerCommandDelay(cfg)
        features = LineFeatures(found=True)
        gate.step(turn_fit(), features, 0.05, -0.04, 0.0, 0.0)
        gate.step(turn_fit(), features, 0.05, -0.04, 0.1, 0.0, 0.0)
        gate.step(turn_fit(), features, 0.05, -0.04, 0.2, 0.0, 0.0)
        turning = gate.step(turn_fit(), features, 0.05, -0.04, 0.3, 0.1, 0.0)
        self.assertEqual(turning.status.reason, "committed_turn")

        unaligned = turn_fit(theta=0.4)
        turning = gate.step(unaligned, features, 0.0, 0.0, 0.8, 0.0, -1.0)
        self.assertEqual(turning.status.reason, "committed_turn")
        self.assertAlmostEqual(turning.status.target[0], 0.25)

        aligned = TrajectoryFit(found=True, e0=0.1, theta=0.05, conf=0.9, n_bands=5)
        reacquired = gate.step(aligned, features, 0.03, -0.02, 1.3, 0.0, -1.0)
        self.assertEqual(reacquired.status.reason, "cooldown")
        self.assertAlmostEqual(reacquired.v, 0.03)

    def test_corner_event_rejects_heading_lateral_sign_conflict(self):
        gate = CornerCommandDelay(PathMemoryConfig(corner_confirm_frames=3))
        conflict = TrajectoryFit(found=True, e0=-0.08, theta=0.42, conf=0.9, n_bands=3)
        for i in range(6):
            output = gate.step(conflict, LineFeatures(found=True), 0.05, 0.0, i * 0.1, 0.05)
        self.assertEqual(output.status.reason, "armed")

    def test_corner_event_does_not_delay_every_fit(self):
        gate = CornerCommandDelay(PathMemoryConfig(corner_confirm_frames=2))
        straight = TrajectoryFit(found=True, e0=0.08, theta=0.02, conf=0.9)
        output = gate.step(straight, LineFeatures(found=True), 0.05, 0.01, 0.0, 0.0)
        self.assertEqual(output.w, 0.01)
        self.assertEqual(output.status.reason, "armed")

    def test_projector_identity(self):
        cfg = GroundProjectionConfig(homography=(1, 0, 0, 0, 1, 0, 0, 0, 1))
        out = GroundProjector(cfg).project(np.asarray([[2.0, 3.0]]))
        np.testing.assert_allclose(out, [[2.0, 3.0]])

    def test_rolling_pursuit_reexpresses_path_after_turn(self):
        tracker = RollingPathPursuit(PathMemoryConfig(lookahead_m=0.1), "local_pursuit")
        source = TrajectoryFit(found=True, e0=0.0, conf=0.9, n_bands=4)
        points = np.asarray([[0.05, 0.0], [0.10, -0.03], [0.15, -0.05]])
        fit, status = tracker.step(points, source, 0.0, 0.0, 0.0)
        self.assertTrue(fit.path_memory)
        self.assertGreater(fit.e_look, 0.0)
        tracker.step(np.empty((0, 2)), source, 0.2, 0.1, 0.2)
        self.assertEqual(status.mode, "local_pursuit")

    def test_motion_reader_fallback(self):
        measured = read_motion_sample(FakeBot((0.04, -0.2)), 0.01, 0.02)
        self.assertEqual(measured.source, "measured")
        stale = read_motion_sample(FakeBot((0.0, 0.0)), 0.03, -0.1)
        self.assertEqual(stale.source, "command_fallback")


if __name__ == "__main__": unittest.main()
