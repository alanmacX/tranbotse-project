import math
import unittest

from transbot_race.capture_geometry import (
    CaptureGeometryFilter,
    CaptureGeometryObservation,
)


def observation(kind, direction=0, angle=0.0, vertex=0.5):
    return CaptureGeometryObservation(
        kind=kind,
        direction=direction,
        angle_rad=angle,
        confidence=0.9,
        vertex_y_frac=vertex,
        incoming_e=0.1,
        incoming_theta=0.02,
    )


class CaptureGeometryFilterTests(unittest.TestCase):
    def test_corner_requires_consistent_direction(self):
        geometry_filter = CaptureGeometryFilter(confirm_frames=3)
        self.assertIsNone(geometry_filter.update(observation("corner", 1, math.pi / 2)))
        self.assertIsNone(geometry_filter.update(observation("corner", -1, -math.pi / 2)))
        decision = geometry_filter.update(observation("corner", 1, math.pi / 2))
        self.assertEqual(decision.kind, "corner")
        self.assertEqual(decision.direction, 1)
        self.assertEqual(decision.votes, 2)

    def test_circle_requires_full_window(self):
        geometry_filter = CaptureGeometryFilter(confirm_frames=3)
        self.assertIsNone(geometry_filter.update(observation("circle")))
        self.assertIsNone(geometry_filter.update(observation("curve", 1, 0.5)))
        self.assertIsNone(geometry_filter.update(observation("circle")))
        self.assertIsNone(geometry_filter.update(observation("circle")))
        decision = geometry_filter.update(observation("circle"))
        self.assertEqual(decision.kind, "circle")
        self.assertEqual(decision.votes, 3)


if __name__ == "__main__":
    unittest.main()
