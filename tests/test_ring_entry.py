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
        self.assertEqual(fit.n_bands, 0)
        self.assertTrue(fit.disconnected)
        self.assertGreater(fit.e_look, 0.0)
        self.assertGreater(fit.theta, 0.0)

    def _establish_alignment(self, executor, observation, now=0.1):
        aligned = TrajectoryFit(
            found=True, e_look=0.05, theta=0.08, conf=0.9,
            path_memory=True,
        )
        if executor.state == "rotate_search":
            executor.step(
                observation, aligned, now=now,
                linear=0.0, accepted_entry=False,
            )
            now += 0.1
        for index in range(executor.align_confirm_frames):
            executor.step(
                observation, aligned, now=now + index * 0.1,
                linear=0.0, accepted_entry=False,
            )
        self.assertEqual(executor.state, "tracking")
        return aligned

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
        aligned_fit = self._establish_alignment(executor, fork)
        # Losing the fork without physical progress is not completion.
        arc = type(fork)(
            # Direction may flip once the chassis is on the sole visible arc;
            # route identity was already locked while the fork was visible.
            kind="curve", direction=-1, angle_rad=-0.7, confidence=0.9,
            vertex_y_frac=0.55, incoming_e=0.0, incoming_theta=0.1,
            endpoints=2, component_area=2000, is_fork=False,
        )
        self.assertFalse(executor.step(
            arc, aligned_fit, now=0.5, linear=0.05, accepted_entry=False,
        ).completed)
        completed = executor.step(
            arc, aligned_fit, now=1.5, linear=0.05, accepted_entry=False,
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
        executor.step(fork, route_fit, now=0.05, linear=0.0, accepted_entry=False)
        for index in range(1, 6):
            held = executor.step(
                None, None, now=0.05 + index * 0.1, linear=0.03, accepted_entry=False,
            )
            self.assertIsNotNone(held.fit)
        lost = executor.step(
            None, None, now=0.65, linear=0.03, accepted_entry=False,
        )
        self.assertIsNone(lost.fit)
        self.assertEqual(lost.phase_event, RingPhaseEvent.ROUTE_LOST)
        self.assertEqual(lost.reason, "ring_entry_alignment_path_lost")

    def test_route_loss_requires_explicit_clear_before_memory_is_reused(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1)
        executor.step(fork, route_fit, now=0.0, linear=0.0, accepted_entry=True)
        executor.step(fork, route_fit, now=0.05, linear=0.0, accepted_entry=False)
        for index in range(1, 7):
            lost = executor.step(
                None, None, now=0.05 + index * 0.1, linear=0.0,
                accepted_entry=False,
            )
        self.assertIsNone(lost.fit)
        self.assertTrue(executor.clear_route_loss())
        self.assertEqual(executor.missing_frames, 0)
        resumed = executor.step(
            None, None, now=0.75, linear=0.0, accepted_entry=False,
        )
        self.assertIsNone(resumed.fit)
        recovered = executor.step(
            fork, route_fit, now=0.85, linear=0.0, accepted_entry=False,
        )
        self.assertIsNotNone(recovered.fit)

    def test_route_loss_clear_rejects_non_lost_executor(self):
        executor = RingEntryExecutor(1)
        self.assertFalse(executor.clear_route_loss())

    def test_raw_confirmed_topology_cannot_own_motor_before_gate(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1)
        observing = executor.step(
            fork, route_fit, now=0.0, linear=0.0,
            accepted_entry=False,
        )
        self.assertEqual(executor.state, "waiting")
        self.assertIsNone(observing.fit)
        self.assertEqual(observing.reason, "ring_entry_waiting_topology")

    def test_ring_margin_uses_independent_distance(self):
        cfg = RaceConfig()
        cfg.path_memory.camera_to_axle_m = 0.45
        cfg.path_memory.roundabout_margin_distance_m = 0.12
        cfg.path_memory.roundabout_margin_enabled = False
        self.assertEqual(effective_ring_margin_distance(
            cfg.path_memory.roundabout_margin_distance_m,
            cfg.path_memory.roundabout_margin_enabled,
        ), 0.0)
        cfg.path_memory.roundabout_margin_enabled = True
        self.assertEqual(effective_ring_margin_distance(
            cfg.path_memory.roundabout_margin_distance_m,
            cfg.path_memory.roundabout_margin_enabled,
        ), 0.12)

    def test_ring_margin_is_enabled_by_default(self):
        cfg = RaceConfig()
        self.assertTrue(cfg.path_memory.roundabout_margin_enabled)
        self.assertAlmostEqual(cfg.path_memory.roundabout_margin_distance_m, 0.45)

    def test_zero_margin_still_passes_through_rotate_search(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1, margin_distance_m=0.0)
        started = executor.step(
            fork, route_fit, now=0.0, linear=0.0,
            accepted_entry=True, incoming_v=0.05,
        )
        self.assertTrue(started.started)
        self.assertEqual(executor.state, "rotate_search")
        self.assertIsNone(started.fit)
        self.assertEqual(started.reason, "ring_entry_rotating_search")
        captured = executor.step(
            fork, route_fit, now=0.1, linear=0.02, accepted_entry=False,
        )
        self.assertEqual(executor.state, "aligning")
        self.assertIsNotNone(captured.fit)
        self.assertEqual(captured.reason, "ring_entry_search_path_captured")

    def test_rotate_search_requires_fresh_stable_path(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(
            1, margin_distance_m=0.0, entry_capture_frames=3,
        )
        executor.step(
            fork, route_fit, now=0.0, linear=0.0, accepted_entry=True,
        )
        self.assertEqual(executor.state, "rotate_search")

        stale = executor.step(
            fork, route_fit, now=0.1, linear=0.02, accepted_entry=False,
            fresh_geometry=False,
        )
        self.assertEqual(executor.state, "rotate_search")
        self.assertEqual(stale.reason, "ring_entry_rotating_search")

        executor.step(
            fork, route_fit, now=0.2, linear=0.02, accepted_entry=False,
        )
        executor.step(
            fork, route_fit, now=0.3, linear=0.02, accepted_entry=False,
        )
        captured = executor.step(
            fork, route_fit, now=0.4, linear=0.02, accepted_entry=False,
        )
        self.assertEqual(executor.state, "aligning")
        self.assertEqual(captured.reason, "ring_entry_search_path_captured")

    def test_rotate_search_accepts_continuous_path_after_local_direction_flips(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(
            1,
            entry_capture_frames=3,
        )
        executor.step(
            fork, route_fit, now=0.0, linear=0.0, accepted_entry=True,
        )
        opposite = type(fork)(
            kind=fork.kind, direction=-1, angle_rad=fork.angle_rad,
            confidence=fork.confidence, vertex_y_frac=fork.vertex_y_frac,
            incoming_e=fork.incoming_e, incoming_theta=fork.incoming_theta,
            endpoints=fork.endpoints, component_area=fork.component_area,
            is_fork=fork.is_fork,
        )
        for index in range(2):
            searching = executor.step(
                opposite, route_fit, now=0.1 + index * 0.1,
                linear=0.0, angular=-0.2, accepted_entry=False,
            )
            self.assertEqual(executor.state, "rotate_search")
            self.assertIsNone(searching.fit)
        captured = executor.step(
            opposite, route_fit, now=0.3, linear=0.0,
            angular=-0.2, accepted_entry=False,
        )
        self.assertEqual(executor.state, "aligning")
        self.assertIsNotNone(captured.fit)

    def test_rotate_search_times_out_at_angular_safety_boundary(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(
            1, entry_capture_frames=3,
            entry_search_max_angle_rad=0.02,
            entry_search_timeout_sec=10.0,
        )
        executor.step(
            fork, route_fit, now=0.0, linear=0.0, accepted_entry=True,
        )
        lost = executor.step(
            None, None, now=1.0, linear=0.0,
            angular=-0.03, accepted_entry=False,
        )
        self.assertEqual(executor.state, "failed")
        self.assertIsNone(lost.fit)
        self.assertEqual(lost.phase_event, RingPhaseEvent.ROUTE_LOST)

    def test_rotate_search_has_one_fixed_owner_and_zero_translation(self):
        right = RingEntryExecutor(1, entry_search_w=0.2)
        left = RingEntryExecutor(-1, entry_search_w=0.2)
        self.assertEqual(right.entry_search_command(), (0.0, -0.2))
        self.assertEqual(left.entry_search_command(), (0.0, 0.2))
        self.assertEqual(right.entry_search_command(invert_turn=True), (0.0, 0.2))

    def test_alignment_requires_stable_selected_route_tangent(self):
        executor = RingEntryExecutor(
            1,
            margin_distance_m=0.0,
            align_e_tolerance=0.20,
            align_theta_tolerance=0.25,
            align_confirm_frames=2,
        )
        observation = type("Observation", (), {"is_fork": True})()
        off_axis = TrajectoryFit(
            found=True, e_look=0.55, theta=0.60, conf=0.9,
            path_memory=True,
        )
        aligned = TrajectoryFit(
            found=True, e_look=0.08, theta=0.12, conf=0.9,
            path_memory=True,
        )

        first = executor.step(
            observation, off_axis, now=0.0, linear=0.0, accepted_entry=True,
        )
        self.assertEqual(executor.state, "rotate_search")
        self.assertEqual(first.reason, "ring_entry_rotating_search")
        executor.step(
            observation, aligned, now=0.1, linear=0.0, accepted_entry=False,
        )
        self.assertEqual(executor.state, "aligning")
        executor.step(
            observation, aligned, now=0.2, linear=0.0, accepted_entry=False,
        )
        established = executor.step(
            observation, aligned, now=0.3, linear=0.0, accepted_entry=False,
        )
        self.assertEqual(executor.state, "tracking")
        self.assertEqual(established.reason, "ring_entry_alignment_established")

    def test_entry_margin_uses_translation_only_until_axle_margin_is_consumed(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1, margin_distance_m=0.10)
        started = executor.step(
            fork, route_fit, now=0.0, linear=0.0,
            accepted_entry=True, incoming_v=0.05,
        )
        self.assertEqual(started.reason, "ring_entry_waiting_margin")
        self.assertEqual(executor.state, "margin")
        self.assertAlmostEqual(executor.margin_remaining_m, 0.10)
        self.assertAlmostEqual(executor.margin_v, 0.05)
        self.assertAlmostEqual(executor.margin_w, 0.0)

        waiting = executor.step(
            fork, route_fit, now=1.0, linear=0.05, accepted_entry=True,
        )
        self.assertEqual(waiting.reason, "ring_entry_waiting_margin")
        self.assertAlmostEqual(executor.margin_remaining_m, 0.05)
        self.assertAlmostEqual(executor.margin_w, 0.0)
        executor.command_w = -0.18
        executor.control_now = 1.0
        executor.pending_fit = route_fit
        executor.pending_fit_frames = 1
        committing = executor.step(
            fork, route_fit, now=2.0, linear=0.05, accepted_entry=True,
        )
        self.assertEqual(executor.state, "rotate_search")
        self.assertIsNone(committing.fit)
        self.assertAlmostEqual(executor.command_w, 0.0)
        self.assertIsNone(executor.control_now)
        self.assertIsNone(executor.pending_fit)
        self.assertEqual(executor.pending_fit_frames, 0)
        self.assertIsNone(executor.raw_fit)
        captured = executor.step(
            fork, route_fit, now=2.1, linear=0.02, accepted_entry=False,
        )
        self.assertEqual(executor.state, "aligning")
        self.assertIs(captured.fit, route_fit)
        self.assertIs(executor.raw_fit, route_fit)

    def test_reset_clears_margin_and_route_control_state(self):
        executor = RingEntryExecutor(1, margin_distance_m=0.10)
        executor.state = "margin"
        executor.margin_remaining_m = 0.07
        executor.margin_v = 0.05
        executor.margin_w = -0.24
        executor.command_w = -0.18
        executor.control_now = 1.0
        executor.raw_fit = TrajectoryFit(found=True, path_memory=True)
        executor.pending_fit = executor.raw_fit
        executor.pending_fit_frames = 1
        executor.last_fit = executor.raw_fit
        executor.missing_frames = 3

        executor.reset()

        self.assertEqual(executor.state, "waiting")
        self.assertEqual(executor.margin_remaining_m, 0.0)
        self.assertEqual(executor.margin_v, 0.0)
        self.assertEqual(executor.margin_w, 0.0)
        self.assertEqual(executor.command_w, 0.0)
        self.assertIsNone(executor.control_now)
        self.assertIsNone(executor.raw_fit)
        self.assertIsNone(executor.pending_fit)
        self.assertEqual(executor.pending_fit_frames, 0)
        self.assertIsNone(executor.last_fit)
        self.assertEqual(executor.missing_frames, 0)

    def test_tracking_rejects_single_frame_branch_swap(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1)
        executor.step(fork, route_fit, now=0.0, linear=0.0, accepted_entry=True)
        executor.step(fork, route_fit, now=0.05, linear=0.0, accepted_entry=False)
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
        executor.step(fork, route_fit, now=0.05, linear=0.0, accepted_entry=False)
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
        executor.step(
            fork, advanced, now=0.3, linear=0.03, accepted_entry=True,
        )
        self.assertIs(executor.raw_fit, advanced)

    def test_route_carrot_is_slew_limited_and_filtered(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1)
        executor.step(fork, route_fit, now=0.0, linear=0.0, accepted_entry=True)
        executor.step(fork, route_fit, now=0.05, linear=0.0, accepted_entry=False)
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

    def test_tracking_uses_approach_angular_limit(self):
        executor = RingEntryExecutor(1)
        executor.state = "tracking"
        fit = TrajectoryFit(
            found=True, e_look=1.0, theta=0.4, conf=0.9, n_bands=6,
        )
        _v, w = executor.control(
            fit, now=0.0, v_max=0.06, k_pursuit=0.9, k_theta=0.3,
            max_w=0.2, approach_max_w=0.08,
        )
        self.assertAlmostEqual(abs(w), 0.08)

        executor.state = "inside"
        executor.control_now = None
        _v, w = executor.control(
            fit, now=0.1, v_max=0.06, k_pursuit=0.9, k_theta=0.3,
            max_w=0.2, approach_max_w=0.08,
        )
        self.assertAlmostEqual(abs(w), 0.2)

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
        aligned_fit = self._establish_alignment(executor, fork)
        arc = type(fork)(
            kind="curve", direction=1, angle_rad=0.7, confidence=0.9,
            vertex_y_frac=0.55, incoming_e=0.0, incoming_theta=0.1,
            endpoints=2, component_area=2000, is_fork=False,
        )
        first_arc = executor.step(
            arc, aligned_fit, now=0.5, linear=0.05, accepted_entry=False,
        )
        self.assertFalse(first_arc.completed)
        entry_done = executor.step(
            arc, aligned_fit, now=0.9, linear=0.05, accepted_entry=False,
        )
        self.assertTrue(entry_done.completed)
        self.assertEqual(entry_done.phase_event, RingPhaseEvent.ENTRY_ESTABLISHED)
        self.assertEqual(executor.state, "inside")

        executor.step(arc, route_fit, now=1.0, linear=0.04, accepted_entry=False)
        executor.step(arc, route_fit, now=1.5, linear=0.04, accepted_entry=False)
        exiting = executor.step(
            fork, route_fit, now=2.0, linear=0.04, accepted_entry=True,
        )
        self.assertEqual(executor.state, "exiting")
        self.assertFalse(exiting.completed)
        self.assertEqual(exiting.phase_event, RingPhaseEvent.EXIT_SELECTED)

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
        self.assertEqual(result.phase_event, RingPhaseEvent.EXECUTOR_COMPLETED)
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
        self.assertEqual(result.phase_event, RingPhaseEvent.EXECUTOR_COMPLETED)
        self.assertEqual(result.reason, "ring_exit_cruise_established")


if __name__ == "__main__":
    unittest.main()
