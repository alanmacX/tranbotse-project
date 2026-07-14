import unittest

import cv2 as cv
import numpy as np

from transbot_race.capture_geometry import analyze_capture_geometry
from transbot_race.config import RaceConfig
from transbot_race.ring_entry import RingEntryExecutor, selected_path_fit
from transbot_race.vision import TrajectoryFit


class RingEntryTests(unittest.TestCase):
    def _fork(self, direction=1):
        cfg = RaceConfig()
        cfg.camera.crop = (300, 265, 430, 442)
        cfg.path_memory.roundabout_direction = direction
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)
        cv.line(frame, (365, 430), (365, 310), (20, 20, 20), 20)
        cv.line(frame, (365, 310), (515, 280), (20, 20, 20), 20)
        cv.line(frame, (365, 310), (195, 325), (20, 20, 20), 20)
        observation, debug = analyze_capture_geometry(frame, cfg)
        return cfg, observation, debug

    def test_debug_retains_both_paths_and_selected_route(self):
        _cfg, observation, debug = self._fork(direction=1)
        self.assertTrue(observation.is_fork)
        self.assertIn(-1, debug.candidate_directions)
        self.assertIn(1, debug.candidate_directions)
        self.assertGreater(debug.path[-1][0], debug.path[0][0])

    def test_selected_skeleton_becomes_right_lookahead_fit(self):
        _cfg, observation, debug = self._fork(direction=1)
        fit = selected_path_fit(
            debug,
            observation,
            frame_center_x=365.0,
            control_width=290.0,
            lookahead_px=180.0,
        )
        self.assertIsNotNone(fit)
        self.assertTrue(fit.path_memory)
        self.assertGreater(fit.e_look, 0.0)
        self.assertGreater(fit.theta, 0.0)

    def test_entry_completes_only_after_motion_and_fork_clear(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1, clear_frames=2, min_distance_m=0.05)
        started = executor.step(
            fork, route_fit, now=0.0, linear=0.0, accepted_entry=True,
        )
        self.assertTrue(started.started)
        self.assertFalse(started.completed)
        # Losing the fork without physical progress is not completion.
        arc = type(fork)(
            # Direction may flip once the chassis is on the sole visible arc;
            # route identity was already locked while the fork was visible.
            kind="curve", direction=-1, angle_rad=-0.7, confidence=0.9,
            vertex_y_frac=0.55, incoming_e=0.0, incoming_theta=0.1,
            endpoints=2, component_area=2000, is_fork=False,
        )
        self.assertFalse(executor.step(
            arc, route_fit, now=0.5, linear=0.04, accepted_entry=False,
        ).completed)
        completed = executor.step(
            arc, route_fit, now=1.5, linear=0.04, accepted_entry=False,
        )
        self.assertTrue(completed.completed)
        self.assertEqual(completed.reason, "ring_entry_path_established")

    def test_committed_route_bridges_five_missing_frames(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1)
        executor.step(fork, route_fit, now=0.0, linear=0.0, accepted_entry=True)
        for index in range(1, 6):
            held = executor.step(
                None, None, now=index * 0.1, linear=0.03, accepted_entry=False,
            )
            self.assertIsNotNone(held.fit)
        lost = executor.step(
            None, None, now=0.6, linear=0.03, accepted_entry=False,
        )
        self.assertIsNone(lost.fit)
        self.assertEqual(lost.reason, "ring_entry_selected_path_lost")

    def test_raw_confirmed_topology_cannot_own_motor_before_gate(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1)
        observing = executor.step(
            fork, route_fit, now=0.0, linear=0.0,
            accepted_entry=False, confirmed_entry=True,
        )
        self.assertEqual(executor.state, "waiting")
        self.assertIsNone(observing.fit)
        self.assertEqual(observing.reason, "ring_entry_waiting_topology")

    def test_entry_uses_frozen_incoming_command_until_axle_margin_is_consumed(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1, margin_distance_m=0.10)
        started = executor.step(
            fork, route_fit, now=0.0, linear=0.0,
            accepted_entry=True, incoming_v=0.05, incoming_w=-0.02,
        )
        self.assertEqual(started.reason, "ring_entry_waiting_margin")
        self.assertEqual(executor.state, "margin")
        self.assertAlmostEqual(executor.margin_remaining_m, 0.10)
        self.assertAlmostEqual(executor.margin_v, 0.05)
        self.assertAlmostEqual(executor.margin_w, -0.02)

        waiting = executor.step(
            fork, route_fit, now=1.0, linear=0.05, accepted_entry=True,
        )
        self.assertEqual(waiting.reason, "ring_entry_waiting_margin")
        self.assertAlmostEqual(executor.margin_remaining_m, 0.05)
        tracking = executor.step(
            fork, route_fit, now=2.0, linear=0.05, accepted_entry=True,
        )
        self.assertEqual(executor.state, "tracking")
        self.assertIsNotNone(tracking.fit)

    def test_tracking_rejects_single_frame_branch_swap(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1)
        executor.step(fork, route_fit, now=0.0, linear=0.0, accepted_entry=True)
        swapped = type(route_fit)(
            **{
                field: getattr(route_fit, field)
                for field in route_fit.__dataclass_fields__
                if field not in {"e_look", "theta"}
            },
            e_look=-route_fit.e_look - 1.0,
            theta=-route_fit.theta,
        )
        held = executor.step(
            fork, swapped, now=0.1, linear=0.03, accepted_entry=True,
        )
        self.assertAlmostEqual(held.fit.e_look, route_fit.e_look)

    def test_persistent_new_local_path_segment_advances_reference(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1)
        executor.step(fork, route_fit, now=0.0, linear=0.0, accepted_entry=True)
        advanced = type(route_fit)(
            **{
                field: getattr(route_fit, field)
                for field in route_fit.__dataclass_fields__
                if field not in {"e_look", "theta"}
            },
            e_look=-route_fit.e_look,
            theta=route_fit.theta - 0.9,
        )
        first = executor.step(
            fork, advanced, now=0.1, linear=0.03, accepted_entry=True,
        )
        second = executor.step(
            fork, advanced, now=0.2, linear=0.03, accepted_entry=True,
        )
        self.assertAlmostEqual(first.fit.e_look, route_fit.e_look)
        self.assertIsNotNone(second.fit)
        self.assertIs(executor.raw_fit, advanced)

    def test_route_carrot_is_slew_limited_and_filtered(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1)
        executor.step(fork, route_fit, now=0.0, linear=0.0, accepted_entry=True)
        moved = type(route_fit)(
            **{
                field: getattr(route_fit, field)
                for field in route_fit.__dataclass_fields__
                if field != "e_look"
            },
            e_look=route_fit.e_look + 0.60,
        )
        result = executor.step(
            fork, moved, now=0.1, linear=0.03, accepted_entry=True,
        )
        self.assertLess(result.fit.e_look - route_fit.e_look, 0.10)

    def test_angular_command_cannot_reverse_at_full_rate_in_one_frame(self):
        executor = RingEntryExecutor(1)
        executor.state = "tracking"
        right = type("Fit", (), {})
        # Use real fits so this test covers the executor's public control API.
        from transbot_race.vision import TrajectoryFit
        first = TrajectoryFit(found=True, e_look=1.0, theta=0.3, conf=0.9, n_bands=6)
        opposite = TrajectoryFit(found=True, e_look=-1.0, theta=-0.3, conf=0.9, n_bands=6)
        _v, first_w = executor.control(
            first, now=0.0, v_max=0.06, k_pursuit=0.9, k_theta=0.3,
            max_w=0.2, approach_max_w=0.08,
        )
        _v, next_w = executor.control(
            opposite, now=0.1, v_max=0.06, k_pursuit=0.9, k_theta=0.3,
            max_w=0.2, approach_max_w=0.08,
        )
        self.assertLess(first_w, 0.0)
        self.assertLess(next_w, 0.0)
        self.assertLessEqual(abs(next_w - first_w), 0.0351)

    def test_roundabout_keeps_control_through_inside_and_confirms_exit_cruise(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(
            1, clear_frames=2, min_distance_m=0.02,
            inside_arm_distance_m=0.02, exit_distance_m=0.02,
        )
        executor.step(fork, route_fit, now=0.0, linear=0.0, accepted_entry=True)
        arc = type(fork)(
            kind="curve", direction=1, angle_rad=0.7, confidence=0.9,
            vertex_y_frac=0.55, incoming_e=0.0, incoming_theta=0.1,
            endpoints=2, component_area=2000, is_fork=False,
        )
        entry_done = executor.step(
            arc, route_fit, now=0.5, linear=0.04, accepted_entry=False,
        )
        self.assertTrue(entry_done.completed)
        self.assertEqual(executor.state, "inside")

        executor.step(arc, route_fit, now=1.0, linear=0.04, accepted_entry=False)
        executor.step(arc, route_fit, now=1.5, linear=0.04, accepted_entry=False)
        exiting = executor.step(
            fork, route_fit, now=2.0, linear=0.04, accepted_entry=True,
        )
        self.assertEqual(executor.state, "exiting")
        self.assertFalse(exiting.completed)

        from transbot_race.vision import TrajectoryFit
        cruise = TrajectoryFit(
            found=True, e0=0.1, e_look=0.1, theta=0.1,
            conf=0.9, n_bands=6, disconnected=False,
        )
        result = None
        for index in range(4):
            result = executor.step(
                arc, route_fit, now=2.5 + index * 0.5,
                linear=0.04, accepted_entry=False, cruise_fit=cruise,
            )
        self.assertTrue(result.completed)
        self.assertEqual(result.reason, "ring_exit_cruise_established")

    def test_ring_exit_uses_normal_cruise_follow_boundary(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(
            1, clear_frames=4, exit_distance_m=0.0,
        )
        executor.state = "exiting"
        cruise = TrajectoryFit(
            found=True, e0=0.463, e_look=0.463, theta=0.338,
            conf=0.90, n_bands=5, disconnected=False,
        )
        arc = type(fork)(
            kind="curve", direction=1, angle_rad=0.7, confidence=0.9,
            vertex_y_frac=0.55, incoming_e=0.0, incoming_theta=0.1,
            endpoints=2, component_area=2000, is_fork=False,
        )

        result = None
        for index in range(4):
            result = executor.step(
                arc, None, now=index * 0.1, linear=0.0,
                accepted_entry=False, cruise_fit=cruise,
            )

        self.assertTrue(result.completed)
        self.assertEqual(result.reason, "ring_exit_cruise_established")


if __name__ == "__main__":
    unittest.main()
