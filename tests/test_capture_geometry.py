import math
import unittest

import cv2 as cv
import numpy as np

from transbot_race.capture_geometry import (
    CaptureGeometryFilter,
    CaptureGeometryObservation,
    CornerGeometryFilter,
    _significant_candidate_paths,
    analyze_capture_geometry,
)
from transbot_race.config import RaceConfig


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
    def test_short_opposite_leaf_is_not_a_route_branch(self):
        main = [(365, 220 - index) for index in range(220)]
        spur = [(365 + index, 220) for index in range(24)]

        significant = _significant_candidate_paths([(1, main), (-1, spur)])

        self.assertEqual(len(significant), 1)
        self.assertEqual(significant[0][0], 1)

    def test_corner_requires_consistent_direction(self):
        geometry_filter = CaptureGeometryFilter(confirm_frames=3)
        self.assertIsNone(geometry_filter.update(observation("corner", 1, math.pi / 2)))
        self.assertIsNone(geometry_filter.update(observation("corner", -1, -math.pi / 2)))
        self.assertIsNone(geometry_filter.update(observation("corner", 1, math.pi / 2)))
        self.assertIsNone(geometry_filter.update(observation("corner", 1, math.pi / 2)))
        decision = geometry_filter.update(observation("corner", 1, math.pi / 2))
        self.assertEqual(decision.kind, "turn")
        self.assertEqual(decision.direction, 1)
        self.assertEqual(decision.votes, 3)

    def test_circle_requires_full_window(self):
        geometry_filter = CaptureGeometryFilter(confirm_frames=3)
        self.assertIsNone(geometry_filter.update(observation("circle")))
        self.assertIsNone(geometry_filter.update(observation("curve", 1, 0.5)))
        self.assertIsNone(geometry_filter.update(observation("circle")))
        self.assertIsNone(geometry_filter.update(observation("circle")))
        decision = geometry_filter.update(observation("circle"))
        self.assertEqual(decision.kind, "circle")
        self.assertEqual(decision.votes, 3)

    def test_reset_prevents_stale_corner_votes_from_relatching(self):
        geometry_filter = CaptureGeometryFilter(confirm_frames=3)
        geometry_filter.update(observation("corner", 1, math.pi / 2))
        geometry_filter.update(observation("corner", 1, math.pi / 2))
        geometry_filter.reset()
        self.assertIsNone(
            geometry_filter.update(observation("corner", 1, math.pi / 2))
        )

    def test_rounded_curve_is_a_turn_event(self):
        geometry_filter = CaptureGeometryFilter(confirm_frames=3)
        self.assertIsNone(geometry_filter.update(observation("straight_or_unknown")))
        self.assertIsNone(
            geometry_filter.update(observation("curve", 1, math.radians(35)))
        )
        self.assertIsNone(geometry_filter.update(
            observation("curve", 1, math.radians(45), vertex=0.56)
        ))
        decision = geometry_filter.update(
            observation("curve", 1, math.radians(45), vertex=0.56)
        )
        self.assertEqual(decision.kind, "turn")
        self.assertEqual(decision.direction, 1)
        self.assertEqual(decision.votes, 3)

    def test_turn_votes_must_be_consecutive_across_dropout(self):
        geometry_filter = CaptureGeometryFilter(confirm_frames=3)
        turn = observation("corner", 1, math.pi / 2)
        self.assertIsNone(geometry_filter.update(turn))
        self.assertIsNone(geometry_filter.update(turn))
        self.assertIsNone(geometry_filter.update(observation("straight_or_unknown")))
        self.assertIsNone(geometry_filter.update(turn))
        self.assertIsNone(geometry_filter.update(turn))
        self.assertIsNotNone(geometry_filter.update(turn))

    def test_corner_session_counts_curve_and_corner_as_one_confidence_epoch(self):
        geometry_filter = CornerGeometryFilter(confirm_frames=3)
        self.assertIsNone(geometry_filter.update(
            observation("curve", 1, math.radians(70))
        ))
        self.assertIsNone(geometry_filter.update(
            observation("curve", 1, math.radians(80))
        ))
        decision = geometry_filter.update(
            observation("corner", 1, math.radians(75))
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.kind, "corner")
        self.assertEqual(decision.direction, 1)
        self.assertEqual(decision.votes, 3)

    def test_pure_curve_confidence_never_commits_corner(self):
        geometry_filter = CornerGeometryFilter(confirm_frames=3)
        for _ in range(5):
            decision = geometry_filter.update(
                observation("curve", 1, math.radians(80))
            )
            self.assertIsNone(decision)

    def test_corner_session_emits_only_from_corner_observations(self):
        geometry_filter = CornerGeometryFilter(confirm_frames=3)
        self.assertIsNone(geometry_filter.update(
            observation("corner", 1, math.radians(70))
        ))
        self.assertIsNone(geometry_filter.update(
            observation("corner", 1, math.radians(80))
        ))
        decision = geometry_filter.update(
            observation("corner", 1, math.radians(90))
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.kind, "corner")

    def test_corner_session_never_emits_ring_entry_event(self):
        geometry_filter = CornerGeometryFilter(confirm_frames=3)
        fork = CaptureGeometryObservation(
            kind="curve", direction=1, angle_rad=math.radians(45),
            confidence=0.9, vertex_y_frac=0.55, is_fork=True,
        )
        for _ in range(4):
            decision = geometry_filter.update(fork)
            self.assertIsNone(decision)

    def test_short_curve_observations_without_vertex_never_vote_as_turn(self):
        geometry_filter = CaptureGeometryFilter(confirm_frames=3)
        for _ in range(5):
            decision = geometry_filter.update(
                observation("curve", 1, math.radians(39), vertex=None)
            )
            self.assertIsNone(decision)

    def test_short_bent_dash_is_not_a_curve(self):
        cfg = RaceConfig()
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)
        # Reproduce the near isolated dash shape from 32.44 s of run 173927.
        # Its endpoint tangents differ by >20 degrees, but it is too short to
        # contain a sustained turn onset.
        points = np.asarray([(365, 425), (365, 408), (373, 397)], np.int32)
        cv.polylines(frame, [points], False, (20, 20, 20), 18)
        captured, _debug = analyze_capture_geometry(frame, cfg)
        self.assertEqual(captured.kind, "straight_or_unknown")

    def test_capture_geometry_is_left_right_symmetric(self):
        cfg = RaceConfig()
        cfg.camera.crop = (300, 265, 430, 442)
        observations = []
        for direction in (-1, 1):
            frame = np.full((480, 640, 3), 230, dtype=np.uint8)
            center_x, vertex_y = 365, 300
            cv.line(frame, (center_x, 430), (center_x, vertex_y), (20, 20, 20), 20)
            cv.line(
                frame,
                (center_x, vertex_y),
                (center_x + direction * 130, vertex_y),
                (20, 20, 20),
                20,
            )
            captured, _debug = analyze_capture_geometry(frame, cfg)
            observations.append(captured)
            self.assertEqual(captured.kind, "corner")
            self.assertEqual(captured.direction, direction)
        self.assertAlmostEqual(
            observations[0].vertex_y_frac,
            observations[1].vertex_y_frac,
            places=3,
        )
        self.assertAlmostEqual(
            observations[0].angle_rad,
            -observations[1].angle_rad,
            places=3,
        )

    def test_stage_tile_above_floor_cannot_flip_right_turn(self):
        cfg = RaceConfig()
        cfg.camera.crop = (300, 265, 430, 442)
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)
        center_x = 365
        # Real right turn on the floor.
        cv.line(frame, (center_x, 430), (350, 305), (20, 20, 20), 20)
        cv.line(frame, (350, 305), (500, 325), (20, 20, 20), 20)
        # Dark stage/tile edge in the expanded top ROI, including a seam that
        # would otherwise join the incoming component and form a false arm.
        cv.line(frame, (120, 220), (520, 220), (25, 25, 25), 14)
        cv.line(frame, (center_x, 220), (center_x, 285), (25, 25, 25), 10)

        captured, debug = analyze_capture_geometry(frame, cfg)

        floor_top = cfg.path_memory.geometry_roi_top_offset_px - 20
        self.assertEqual(cv.countNonZero(debug.mask[:floor_top]), 0)
        self.assertEqual(captured.kind, "corner")
        self.assertEqual(captured.direction, 1)

    def test_slanted_incoming_line_does_not_flip_exit_direction(self):
        cfg = RaceConfig()
        cfg.camera.crop = (300, 265, 430, 442)
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)
        cv.line(frame, (330, 430), (365, 305), (20, 20, 20), 20)
        cv.line(frame, (365, 305), (510, 340), (20, 20, 20), 20)

        captured, _debug = analyze_capture_geometry(frame, cfg)

        self.assertEqual(captured.kind, "corner")
        self.assertEqual(captured.direction, 1)

    def test_fork_uses_explicit_route_direction_not_longest_branch(self):
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)
        center_x = 365
        cv.line(frame, (center_x, 430), (center_x, 310), (20, 20, 20), 20)
        # Deliberately make the left branch longer. Route intent must still be
        # able to select the shorter right branch deterministically.
        cv.line(frame, (center_x, 310), (500, 285), (20, 20, 20), 20)
        cv.line(frame, (center_x, 310), (180, 330), (20, 20, 20), 20)

        for desired in (-1, 1):
            cfg = RaceConfig()
            cfg.camera.crop = (300, 265, 430, 442)
            cfg.mission.ring_entry_direction = desired
            captured, debug = analyze_capture_geometry(frame, cfg)
            self.assertTrue(captured.is_fork)
            self.assertEqual(captured.direction, desired)
            self.assertEqual(1 if debug.path[-1][0] > center_x else -1, desired)

    def test_route_selected_circle_fork_votes_as_turn(self):
        geometry_filter = CaptureGeometryFilter(confirm_frames=3)
        fork = observation("circle", 1, math.radians(45), vertex=0.55)
        fork = CaptureGeometryObservation(
            kind=fork.kind,
            direction=fork.direction,
            angle_rad=fork.angle_rad,
            confidence=fork.confidence,
            vertex_y_frac=fork.vertex_y_frac,
            incoming_e=fork.incoming_e,
            incoming_theta=fork.incoming_theta,
            is_fork=True,
        )
        geometry_filter.update(fork)
        geometry_filter.update(fork)
        decision = geometry_filter.update(fork)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.kind, "turn")
        self.assertEqual(decision.direction, 1)
        self.assertTrue(decision.is_fork)

    def test_many_endpoint_noise_is_neither_fork_nor_turn(self):
        cfg = RaceConfig()
        cfg.camera.crop = (300, 265, 430, 442)
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        center = (365, 420)
        cv.line(frame, center, (365, 300), (25, 25, 25), 18)
        # A shadow/tile blob with many arms reproduced the ep=14/19 geometry
        # topology in the latest night run.
        for endpoint in ((180, 260), (230, 235), (290, 225), (440, 230), (500, 250), (540, 300)):
            cv.line(frame, (365, 300), endpoint, (25, 25, 25), 8)
        captured, _debug = analyze_capture_geometry(frame, cfg)
        self.assertFalse(captured.is_fork)
        self.assertNotIn(captured.kind, {"corner", "curve"})


if __name__ == "__main__":
    unittest.main()
