import unittest

import numpy as np

from transbot_race.config import PathMemoryConfig
from transbot_race.path_memory import (
    ShortHorizonPathMemory,
    _advance_points,
    image_path_to_axle,
    read_motion_sample,
)
from transbot_race.vision import PathPoint, Run, TrajectoryFit


def source_fit(conf=0.9):
    return TrajectoryFit(found=True, conf=conf, n_bands=5)


class FakeBot:
    def __init__(self, value=None):
        self.value = value

    def get_motion_data(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class PathMemoryTests(unittest.TestCase):
    def test_image_path_uses_tunable_camera_to_axle(self):
        run = Run(0, 0, 0.0, 10, 100)
        path = (PathPoint(0, 50.0, 99.0, run),)
        near = image_path_to_axle(
            path,
            image_width=100,
            image_height=100,
            center_x=50.0,
            pixels_per_meter=100.0,
            camera_to_axle_m=0.05,
        )
        far = image_path_to_axle(
            path,
            image_width=100,
            image_height=100,
            center_x=50.0,
            pixels_per_meter=100.0,
            camera_to_axle_m=0.15,
        )
        self.assertAlmostEqual(float(far[0, 0] - near[0, 0]), 0.10)

    def test_forward_motion_brings_saved_corner_to_axle(self):
        cfg = PathMemoryConfig(
            lookahead_m=0.06,
            heading_lookahead_m=0.04,
            max_age_sec=2.0,
            max_motion_dt_sec=0.25,
        )
        memory = ShortHorizonPathMemory(cfg)
        # Straight ahead, then a right turn (negative left-coordinate).
        observed = np.asarray(
            [[0.15, 0.0], [0.20, 0.0], [0.20, -0.05], [0.20, -0.10]],
            dtype=np.float64,
        )
        fit0, status0 = memory.step(
            observed,
            source_fit(),
            now=0.0,
            linear_velocity=0.0,
            angular_velocity=0.0,
            lateral_half_width_m=0.20,
        )
        self.assertTrue(status0.active)
        self.assertAlmostEqual(fit0.e_look, 0.0, delta=0.03)

        fit = fit0
        for index in range(1, 9):
            fit, _status = memory.step(
                np.empty((0, 2)),
                TrajectoryFit(found=False),
                now=index * 0.2,
                linear_velocity=0.10,
                angular_velocity=0.0,
                lateral_half_width_m=0.20,
            )
        self.assertIsNotNone(fit)
        self.assertTrue(fit.path_memory)
        self.assertGreater(fit.e_look, 0.05)

    def test_larger_camera_offset_delays_turn_signal(self):
        cfg = PathMemoryConfig(lookahead_m=0.08, heading_lookahead_m=0.05)
        close_memory = ShortHorizonPathMemory(cfg)
        far_memory = ShortHorizonPathMemory(cfg)
        shape = np.asarray(
            [[0.00, 0.0], [0.04, 0.0], [0.04, -0.05], [0.04, -0.10]],
            dtype=np.float64,
        )
        close_fit, _ = close_memory.step(
            shape + [0.02, 0.0], source_fit(), now=0.0,
            linear_velocity=0.0, angular_velocity=0.0, lateral_half_width_m=0.20,
        )
        far_fit, _ = far_memory.step(
            shape + [0.12, 0.0], source_fit(), now=0.0,
            linear_velocity=0.0, angular_velocity=0.0, lateral_half_width_m=0.20,
        )
        self.assertGreater(close_fit.e_look, far_fit.e_look + 0.05)

    def test_motion_rotation_is_expressed_in_new_robot_frame(self):
        points = np.asarray([[1.0, 0.0]], dtype=np.float64)
        turned = _advance_points(points, ds=0.0, dyaw=np.pi / 2.0)
        self.assertAlmostEqual(float(turned[0, 0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(turned[0, 1]), -1.0, delta=1e-6)

    def test_motion_reader_falls_back_on_invalid_data(self):
        measured = read_motion_sample(FakeBot((0.04, -0.2)), 0.01, 0.02)
        self.assertEqual(measured.source, "measured")
        self.assertAlmostEqual(measured.linear, 0.04)

        fallback = read_motion_sample(FakeBot(RuntimeError("offline")), 0.03, -0.1)
        self.assertEqual(fallback.source, "command_fallback")
        self.assertAlmostEqual(fallback.linear, 0.03)

    def test_disabled_memory_is_strict_passthrough(self):
        memory = ShortHorizonPathMemory(PathMemoryConfig(enabled=False))
        fit, status = memory.step(
            np.asarray([[0.1, 0.0], [0.2, 0.0]]),
            source_fit(),
            now=0.0,
            linear_velocity=0.0,
            angular_velocity=0.0,
            lateral_half_width_m=0.20,
        )
        self.assertIsNone(fit)
        self.assertFalse(status.active)
        self.assertEqual(status.reason, "disabled")


if __name__ == "__main__":
    unittest.main()
