import unittest

import numpy as np

from transbot_race.config import GroundProjectionConfig, PathMemoryConfig
from transbot_race.path_memory import (
    CornerEventMargin, GroundProjector, RollingPathPursuit, read_motion_sample,
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
    def test_corner_event_holds_then_returns_current_fit(self):
        cfg = PathMemoryConfig(camera_to_axle_m=0.05, corner_confirm_frames=2, max_motion_dt_sec=1.0)
        gate = CornerEventMargin(cfg)
        features = LineFeatures(found=True)
        gate.step(turn_fit(), features, 0.0, 0.0)
        held, status = gate.step(turn_fit(), features, 0.1, 0.1)
        self.assertEqual(status.reason, "waiting_margin")
        self.assertEqual(held.theta, 0.0)
        current = turn_fit(theta=-0.3)
        released, status = gate.step(current, features, 0.6, 0.1)
        self.assertEqual(status.reason, "released")
        self.assertIs(released, current)

    def test_corner_event_does_not_delay_every_fit(self):
        gate = CornerEventMargin(PathMemoryConfig(corner_confirm_frames=2))
        straight = TrajectoryFit(found=True, e0=0.08, theta=0.02, conf=0.9)
        output, status = gate.step(straight, LineFeatures(found=True), 0.0, 0.0)
        self.assertIs(output, straight)
        self.assertEqual(status.reason, "armed")

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
