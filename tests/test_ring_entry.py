import unittest

import cv2 as cv
import numpy as np

from transbot_race.capture_geometry import analyze_capture_geometry
from transbot_race.config import RaceConfig
from transbot_race.ring_entry import (
    RingEntryExecutor,
    RingPhaseEvent,
    effective_ring_margin_distance,
    selected_path_fit,
)
from transbot_race.vision import TrajectoryFit


class RingEntryTests(unittest.TestCase):
    def _fork(self, direction=1):
        cfg = RaceConfig()
        cfg.camera.crop = (300, 265, 430, 442)
        cfg.mission.ring_entry_direction = direction
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)
        cv.line(frame, (365, 430), (365, 310), (20, 20, 20), 20)
        cv.line(frame, (365, 310), (515, 280), (20, 20, 20), 20)
        cv.line(frame, (365, 310), (195, 325), (20, 20, 20), 20)
        observation, debug = analyze_capture_geometry(frame, cfg)
        return cfg, observation, debug

    @staticmethod
    def _fit(e0=0.1, e_look=0.2, theta=-0.4):
        return TrajectoryFit(
            found=True,
            e0=e0,
            e_look=e_look,
            theta=theta,
            conf=0.9,
            path_memory=True,
        )

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
        self.assertTrue(fit.control_valid)
        self.assertGreater(fit.e_look, 0.0)
        self.assertGreater(fit.theta, 0.0)

    def test_raw_topology_cannot_own_motor_before_gate(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1)
        result = executor.step(
            fork, route_fit, now=0.0, linear=0.0, accepted_entry=False,
        )
        self.assertEqual(executor.state, "waiting")
        self.assertIsNone(result.fit)
        self.assertEqual(result.reason, "ring_entry_waiting_topology")

    def test_ring_margin_uses_independent_distance(self):
        cfg = RaceConfig()
        cfg.path_memory.camera_to_axle_m = 0.10
        cfg.path_memory.roundabout_margin_distance_m = 0.45
        cfg.path_memory.roundabout_margin_enabled = False
        self.assertEqual(effective_ring_margin_distance(0.45, False), 0.0)
        self.assertEqual(effective_ring_margin_distance(0.45, True), 0.45)

    def test_ring_margin_is_enabled_by_default(self):
        cfg = RaceConfig()
        self.assertTrue(cfg.path_memory.roundabout_margin_enabled)
        self.assertAlmostEqual(cfg.path_memory.roundabout_margin_distance_m, 0.45)

    def test_margin_transitions_only_to_radius_acquire(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1, margin_distance_m=0.10)
        first = executor.step(
            fork, route_fit, now=0.0, linear=0.0,
            accepted_entry=True, incoming_v=0.05,
        )
        self.assertEqual(first.reason, "ring_entry_waiting_margin")
        self.assertEqual(executor.state, "margin")
        self.assertAlmostEqual(executor.margin_v, 0.05)
        executor.step(
            fork, route_fit, now=1.0, linear=0.05, accepted_entry=False,
        )
        transitioned = executor.step(
            fork, route_fit, now=2.0, linear=0.05, accepted_entry=False,
        )
        self.assertEqual(executor.state, "radius_acquire")
        self.assertEqual(transitioned.reason, "ring_entry_radius_acquiring")
        self.assertIsNone(transitioned.fit)

    def _advance_radius_window(self, executor, observation, e0, now):
        fit = self._fit(e0=e0)
        for offset in (0.5, 1.0):
            result = executor.step(
                observation,
                fit,
                now=now + offset,
                linear=-0.02,
                angular=-0.10,
                motion_source="measured",
                accepted_entry=False,
            )
        return result, now + 1.0

    def test_runtime_radius_locks_only_after_stable_windows(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(
            1,
            margin_distance_m=0.0,
            entry_search_w=0.08,
            entry_search_max_angle_rad=0.8,
            radius_window_rad=0.10,
            radius_stable_e=0.05,
            radius_confirm_windows=2,
        )
        executor.step(
            fork, self._fit(), now=0.0, linear=0.0, accepted_entry=True,
        )
        now = 0.0
        for e0 in (0.20, 0.21, 0.22):
            result, now = self._advance_radius_window(executor, fork, e0, now)
        self.assertEqual(executor.state, "half_arc")
        self.assertEqual(result.reason, "ring_entry_radius_locked")
        self.assertEqual(executor.arc_turned_rad, 0.0)
        self.assertEqual(executor.entry_search_elapsed_sec, 0.0)
        self.assertAlmostEqual(executor.radius_estimate_m, 0.20, places=3)
        self.assertEqual(executor.radius_source, "measured")

    def test_radius_adjustment_occurs_only_at_window_boundary(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(
            1,
            margin_distance_m=0.0,
            entry_search_w=0.08,
            entry_search_max_angle_rad=1.0,
            radius_window_rad=0.10,
            radius_stable_e=0.05,
            radius_w_step=0.01,
            radius_confirm_windows=3,
        )
        executor.step(fork, self._fit(), now=0.0, linear=0.0, accepted_entry=True)
        _result, now = self._advance_radius_window(executor, fork, 0.10, 0.0)
        self.assertAlmostEqual(executor.arc_w, 0.08)
        executor.step(
            fork, self._fit(e0=0.30), now=now + 0.5,
            linear=-0.02, angular=-0.10, accepted_entry=False,
        )
        self.assertAlmostEqual(executor.arc_w, 0.08)
        executor.step(
            fork, self._fit(e0=0.30), now=now + 1.0,
            linear=-0.02, angular=-0.10, accepted_entry=False,
        )
        self.assertAlmostEqual(executor.arc_w, 0.07)

        forward = RingEntryExecutor(
            1,
            margin_distance_m=0.0,
            entry_search_w=0.08,
            entry_search_max_angle_rad=1.0,
            arc_motion_sign=1,
            radius_window_rad=0.10,
            radius_stable_e=0.05,
            radius_w_step=0.01,
            radius_confirm_windows=3,
        )
        forward.step(
            fork, self._fit(), now=0.0, linear=0.0, accepted_entry=True,
        )
        self._advance_radius_window(forward, fork, 0.10, 0.0)
        self._advance_radius_window(forward, fork, 0.30, 1.0)
        self.assertAlmostEqual(forward.arc_w, 0.09)

    def test_fixed_arc_command_keeps_motion_and_yaw_axes_independent(self):
        cases = {
            (1, 1): (0.02, -0.08),
            (1, -1): (-0.02, -0.08),
            (-1, 1): (0.02, 0.08),
            (-1, -1): (-0.02, 0.08),
        }
        for (direction, motion_sign), expected in cases.items():
            executor = RingEntryExecutor(
                direction,
                arc_v=0.02,
                arc_motion_sign=motion_sign,
                entry_search_w=0.08,
            )
            self.assertEqual(executor.fixed_arc_command(), expected)

    def test_half_arc_ignores_visual_fit_until_target_yaw(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(1, half_arc_yaw_rad=0.30)
        executor.state = "half_arc"
        executor.arc_w = 0.09
        executor.last_now = 0.0
        wild = self._fit(e0=-0.8, e_look=1.0, theta=1.1)
        first = executor.step(
            fork, wild, now=1.0, linear=0.02, angular=-0.20,
            accepted_entry=False,
        )
        self.assertEqual(executor.state, "half_arc")
        self.assertEqual(first.reason, "ring_half_arc_running")
        self.assertAlmostEqual(executor.arc_w, 0.09)
        executor.step(
            fork, wild, now=1.5, linear=0.02, angular=-0.20,
            accepted_entry=False,
        )
        self.assertEqual(executor.state, "exit_reacquire")
        self.assertAlmostEqual(executor.arc_w, 0.09)

    def test_exit_reacquire_hands_off_only_after_confirmed_path(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(
            1, entry_capture_frames=3,
            half_arc_yaw_rad=0.30,
            exit_reacquire_extra_rad=0.40,
        )
        executor.state = "exit_reacquire"
        executor.arc_turned_rad = 0.30
        executor.last_now = 0.0
        result = None
        for index in range(3):
            result = executor.step(
                fork, self._fit(e0=0.1 + index * 0.01),
                now=0.1 + index * 0.1,
                linear=0.02,
                angular=-0.08,
                accepted_entry=False,
            )
        self.assertEqual(executor.state, "exit_ready")
        self.assertEqual(result.phase_event, RingPhaseEvent.ENTRY_ESTABLISHED)
        self.assertEqual(result.reason, "ring_half_arc_exit_path_captured")

    def test_entry_states_advance_in_one_linear_order(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(
            1,
            margin_distance_m=0.02,
            entry_capture_frames=2,
            entry_search_max_angle_rad=1.0,
            radius_window_rad=0.05,
            radius_confirm_windows=1,
            half_arc_yaw_rad=0.20,
            exit_reacquire_extra_rad=0.20,
        )
        states = []
        executor.step(
            fork, self._fit(), now=0.0, linear=0.0,
            incoming_v=0.02, accepted_entry=True,
        )
        states.append(executor.state)
        executor.step(
            fork, self._fit(), now=1.0, linear=0.02,
            accepted_entry=False,
        )
        states.append(executor.state)
        # One baseline window plus one matching window locks the radius.
        for now in (1.5, 2.0, 2.5, 3.0):
            executor.step(
                fork, self._fit(e0=0.20), now=now, linear=0.02,
                angular=-0.05, accepted_entry=False,
            )
        states.append(executor.state)
        for now in (3.5, 4.0):
            executor.step(
                fork, self._fit(e0=-0.80), now=now, linear=0.02,
                angular=-0.20, accepted_entry=False,
            )
        states.append(executor.state)
        for now in (4.1, 4.2):
            executor.step(
                fork, self._fit(e0=0.10), now=now, linear=0.02,
                angular=-0.08, accepted_entry=False,
            )
        states.append(executor.state)
        self.assertEqual(
            states,
            ["margin", "radius_acquire", "half_arc", "exit_reacquire", "exit_ready"],
        )

    def test_command_fallback_is_reported_as_proxy_radius(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(
            1, margin_distance_m=0.0, radius_window_rad=0.05,
        )
        executor.step(fork, self._fit(), now=0.0, linear=0.0, accepted_entry=True)
        executor.step(
            fork, self._fit(), now=0.5, linear=-0.02, angular=-0.10,
            motion_source="command_fallback", accepted_entry=False,
        )
        self.assertEqual(executor.radius_source, "command_proxy")
        self.assertAlmostEqual(executor.radius_estimate_m, 0.20)

    def test_radius_acquire_stops_at_angular_boundary(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(
            1,
            margin_distance_m=0.0,
            entry_search_max_angle_rad=0.02,
            entry_search_timeout_sec=10.0,
        )
        executor.step(fork, self._fit(), now=0.0, linear=0.0, accepted_entry=True)
        lost = executor.step(
            None, None, now=1.0, linear=0.0,
            angular=-0.03, accepted_entry=False,
        )
        self.assertEqual(executor.state, "failed")
        self.assertEqual(lost.phase_event, RingPhaseEvent.ROUTE_LOST)
        self.assertEqual(lost.reason, "ring_entry_radius_acquire_timeout")
        repeated = executor.step(
            None, None, now=2.0, linear=0.0,
            angular=0.0, accepted_entry=False,
        )
        self.assertEqual(repeated.reason, "ring_entry_radius_acquire_timeout")

    def test_reset_clears_runtime_radius_state(self):
        executor = RingEntryExecutor(1)
        executor.state = "half_arc"
        executor.arc_w = 0.12
        executor.arc_turned_rad = 1.2
        executor.radius_estimate_m = 0.21
        executor.radius_stable_windows = 2
        executor.reset()
        self.assertEqual(executor.state, "waiting")
        self.assertAlmostEqual(executor.arc_w, executor.entry_search_w)
        self.assertEqual(executor.arc_turned_rad, 0.0)
        self.assertIsNone(executor.radius_estimate_m)
        self.assertEqual(executor.radius_stable_windows, 0)

    def test_inside_control_uses_roundabout_angular_limit(self):
        executor = RingEntryExecutor(1)
        executor.state = "inside"
        fit = TrajectoryFit(found=True, e_look=1.0, theta=0.4, conf=0.9, n_bands=6)
        _v, w = executor.control(
            fit, now=0.0, v_max=0.06, k_pursuit=0.9, k_theta=0.3,
            max_w=0.2, approach_max_w=0.08,
        )
        self.assertAlmostEqual(abs(w), 0.2)

    def test_ring_exit_uses_normal_cruise_follow_boundary(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(1, clear_frames=4, exit_distance_m=0.0)
        executor.state = "exiting"
        cruise = TrajectoryFit(
            found=True, e0=0.1, e_look=0.1, theta=0.1,
            conf=0.9, n_bands=6, disconnected=False,
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
        self.assertEqual(result.phase_event, RingPhaseEvent.EXECUTOR_COMPLETED)


if __name__ == "__main__":
    unittest.main()
