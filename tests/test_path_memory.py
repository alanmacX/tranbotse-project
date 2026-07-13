import unittest

from transbot_race.config import PathMemoryConfig
from transbot_race.path_memory import DistanceDelayPathMemory, read_motion_sample
from transbot_race.vision import TrajectoryFit


def turn_fit(theta=0.6, e_look=0.7):
    return TrajectoryFit(
        found=True, e0=0.2, e_look=e_look, theta=theta,
        kappa=0.3, conf=0.9, n_bands=5,
    )


class FakeBot:
    def __init__(self, value=None):
        self.value = value

    def get_motion_data(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def advance(memory, distance, *, start=0.0, speed=0.1, step=0.2):
    now = start
    fit = None
    status = None
    steps = round(distance / (speed * step))
    for _ in range(steps):
        now += step
        fit, status = memory.step(
            TrajectoryFit(found=False), near_e0=0.05,
            now=now, linear_velocity=speed,
        )
    return fit, status, now


class PathMemoryTests(unittest.TestCase):
    def test_turn_is_released_after_travelled_camera_offset(self):
        memory = DistanceDelayPathMemory(PathMemoryConfig(camera_to_axle_physical_m=0.10))
        initial, status = memory.step(turn_fit(), near_e0=0.08, now=0.0, linear_velocity=0.0)
        self.assertEqual(status.reason, "filling")
        self.assertFalse(initial.path_memory)
        self.assertEqual(initial.theta, 0.0)
        self.assertAlmostEqual(initial.e0, 0.08)

        before, _, now = advance(memory, 0.08)
        self.assertFalse(before.path_memory)
        released, status, _ = advance(memory, 0.02, start=now)
        self.assertTrue(released.path_memory)
        self.assertEqual(status.reason, "distance_delay")
        self.assertAlmostEqual(released.theta, 0.6)

    def test_larger_offset_releases_later(self):
        close = DistanceDelayPathMemory(PathMemoryConfig(camera_to_axle_physical_m=0.05))
        far = DistanceDelayPathMemory(PathMemoryConfig(camera_to_axle_physical_m=0.15))
        close.step(turn_fit(), near_e0=0.0, now=0.0, linear_velocity=0.0)
        far.step(turn_fit(), near_e0=0.0, now=0.0, linear_velocity=0.0)
        close_fit, _, _ = advance(close, 0.06)
        far_fit, far_status, _ = advance(far, 0.06)
        self.assertTrue(close_fit.path_memory)
        self.assertFalse(far_fit.path_memory)
        self.assertGreater(far_status.remaining_m, 0.08)

    def test_live_near_error_is_not_delayed(self):
        memory = DistanceDelayPathMemory(PathMemoryConfig(camera_to_axle_physical_m=0.2))
        fit, _ = memory.step(turn_fit(), near_e0=-0.17, now=0.0, linear_velocity=0.0)
        self.assertAlmostEqual(fit.e0, -0.17)
        self.assertEqual(fit.theta, 0.0)

    def test_motion_reader_uses_measurement_and_fallback(self):
        measured = read_motion_sample(FakeBot((0.04, -0.2)), 0.01, 0.02)
        self.assertEqual(measured.source, "measured")
        stale = read_motion_sample(FakeBot((0.0, 0.0)), 0.03, -0.1)
        self.assertEqual(stale.source, "command_fallback")
        failed = read_motion_sample(FakeBot(RuntimeError("offline")), 0.03, -0.1)
        self.assertEqual(failed.source, "command_fallback")

    def test_disabled_memory_is_passthrough(self):
        memory = DistanceDelayPathMemory(PathMemoryConfig(enabled=False))
        fit, status = memory.step(turn_fit(), near_e0=0.0, now=0.0, linear_velocity=0.0)
        self.assertIsNone(fit)
        self.assertFalse(status.active)
        self.assertEqual(status.reason, "disabled")


if __name__ == "__main__":
    unittest.main()
