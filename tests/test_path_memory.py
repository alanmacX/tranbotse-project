import unittest

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
    def test_corner_event_holds_then_replays_recorded_commands(self):
        cfg = PathMemoryConfig(camera_to_axle_m=0.02, corner_confirm_frames=2, corner_record_steps=2)
        gate = CornerCommandDelay(cfg)
        features = LineFeatures(found=True)
        gate.step(turn_fit(), features, 0.05, -0.08, 0.0, 0.0)
        held = gate.step(turn_fit(), features, 0.05, -0.10, 0.1, 0.1)
        self.assertEqual(held.status.reason, "waiting_margin")
        self.assertLessEqual(abs(held.w), cfg.corner_hold_max_w)
        gate.step(turn_fit(), features, 0.05, -0.12, 0.2, 0.1)
        replay = gate.step(turn_fit(), features, 0.05, -0.15, 0.3, 0.1)
        self.assertEqual(replay.status.reason, "replay_step")
        self.assertAlmostEqual(replay.w, -0.10)

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
