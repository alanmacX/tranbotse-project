import unittest

import cv2 as cv
import numpy as np

from transbot_race.config import ObstacleConfig
from transbot_race.obstacle import (
    ObstacleMonitor,
    ObstacleState,
    detect_obstacle,
)


W, H, LINE_X = 290, 177, 85.0


def clear_track():
    image = np.full((H, W, 3), 125, dtype=np.uint8)
    cv.line(image, (85, H - 1), (85, 0), (20, 20, 20), 14)
    return image


def brown_block():
    image = clear_track()
    cv.rectangle(image, (10, 28), (175, 142), (45, 90, 145), -1)
    cv.rectangle(image, (10, 28), (175, 142), (25, 45, 70), 2)
    return image


def white_block():
    image = clear_track()
    cv.rectangle(image, (15, 38), (165, 116), (220, 220, 220), -1)
    cv.rectangle(image, (15, 38), (165, 116), (75, 75, 75), 2)
    return image


class ObstacleTests(unittest.TestCase):
    def test_clear_track_has_no_candidate(self):
        evidence = detect_obstacle(clear_track(), LINE_X, 0.0, ObstacleConfig())
        self.assertIsNone(evidence.candidate)

    def test_colored_block_crossing_corridor_is_detected(self):
        evidence = detect_obstacle(brown_block(), LINE_X, 0.0, ObstacleConfig())
        self.assertIsNotNone(evidence.candidate)
        self.assertIn(evidence.candidate.cue, ("chroma", "neutral_edges"))
        self.assertGreater(evidence.candidate.confidence, 0.5)

    def test_white_block_uses_edges_or_line_occlusion(self):
        monitor = ObstacleMonitor(ObstacleConfig())
        for _ in range(3):
            monitor.update(clear_track(), LINE_X, armed=False)
        first = monitor.update(white_block(), LINE_X, armed=True)
        second = monitor.update(white_block(), LINE_X, armed=True)
        self.assertEqual(first.state, ObstacleState.SUSPECT)
        self.assertIn(second.state, (ObstacleState.ENTRY, ObstacleState.APPROACH, ObstacleState.STOP))
        self.assertIsNotNone(second.evidence.candidate)

    def test_disabled_monitor_never_stops(self):
        monitor = ObstacleMonitor(ObstacleConfig(enabled=False))
        decision = monitor.update(brown_block(), LINE_X, armed=True)
        self.assertEqual(decision.state, ObstacleState.DISARMED)
        self.assertFalse(decision.stop_required)


if __name__ == "__main__":
    unittest.main()
