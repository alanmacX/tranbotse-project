import unittest

import cv2 as cv
import numpy as np

from transbot_race.capture_geometry import analyze_capture_geometry
from transbot_race.config import RaceConfig
from transbot_race.ring_entry import (
    RingCircleModel,
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

    @staticmethod
    def _model():
        return RingCircleModel((100.0, 80.0), (60.0, 140.0), 15.0, 80)

    def _complete_model_leg(self, executor, fork, now):
        executor.model_leg_command()
        now += 1.0
        executor.step(
            fork, self._fit(), now=now, linear=0.0,
            angular=executor.command_turn_sign * (
                abs(executor.leg_target_yaw_rad) + 0.1
            ),
            motion_source="measured",
            accepted_entry=False,
        )
        self.assertEqual(executor.leg_phase, "align_stop")
        self.assertEqual(executor.model_leg_command(), (0.0, 0.0))
        for _ in range(2):
            now += 0.1
            executor.step(
                fork, self._fit(), now=now, linear=0.0,
                angular=0.0, accepted_entry=False,
                motion_source="measured",
            )
        self.assertEqual(executor.leg_phase, "drive")
        now += 0.1
        executor.step(
            fork, self._fit(), now=now,
            linear=executor.leg_chord_length_m / 0.1 + 0.1,
            angular=0.0, accepted_entry=False,
            motion_source="measured",
        )
        self.assertEqual(executor.leg_phase, "drive_stop")
        self.assertEqual(executor.model_leg_command(), (0.0, 0.0))
        for _ in range(2):
            now += 0.1
            executor.step(
                fork, self._fit(), now=now, linear=0.0,
                angular=0.0, accepted_entry=False,
                motion_source="measured",
            )
        return now

    def _complete_entry_left_turn(self, executor, fork, now):
        original_deltas = executor.leg_turn_deltas_rad
        command = executor.entry_left_command()
        self.assertEqual(command[0], 0.0)
        self.assertGreater(command[1], 0.0)
        now += 1.0
        executor.step(
            fork, self._fit(), now=now, linear=0.0,
            angular=executor.entry_left_turn_rad + 0.1,
            motion_source="measured", accepted_entry=False,
        )
        self.assertEqual(executor.state, "entry_left_stop")
        for _ in range(2):
            now += 0.1
            executor.step(
                fork, self._fit(), now=now, linear=0.0, angular=0.0,
                motion_source="measured", accepted_entry=False,
            )
        self.assertEqual(executor.state, "leg1_model")
        self.assertEqual(executor.leg_turn_deltas_rad, original_deltas)
        self.assertEqual(executor.model_aligned_rad, 0.0)
        return now

    def _complete_exit_line_align(self, executor, fork, now):
        self.assertEqual(executor.state, "exit_line_align")
        command = executor.exit_line_command()
        self.assertEqual(command[0], 0.0)
        self.assertEqual(np.sign(command[1]), np.sign(executor.leg_turn_delta_rad))
        now += 1.0
        executor.step(
            fork, self._fit(), now=now, linear=0.0,
            angular=executor.model_turn_sign * (executor.leg_target_yaw_rad + 0.1),
            motion_source="measured", accepted_entry=False,
        )
        self.assertEqual(executor.state, "exit_line_stop")
        for _ in range(2):
            now += 0.1
            executor.step(
                fork, self._fit(), now=now, linear=0.0, angular=0.0,
                motion_source="measured", accepted_entry=False,
            )
        self.assertEqual(executor.state, "exit_reacquire")
        return now

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
        self.assertAlmostEqual(cfg.path_memory.roundabout_entry_left_turn_deg, 45.0)

    def test_margin_transitions_only_to_first_model_leg(self):
        _cfg, fork, debug = self._fork(direction=1)
        route_fit = selected_path_fit(
            debug, fork, frame_center_x=365.0, control_width=290.0,
        )
        executor = RingEntryExecutor(1, margin_distance_m=0.10)
        first = executor.step(
            fork, route_fit, now=0.0, linear=0.0,
            accepted_entry=True, incoming_v=0.05,
            circle_model=self._model(),
        )
        self.assertEqual(first.reason, "ring_entry_waiting_margin")
        self.assertEqual(executor.state, "margin")
        self.assertAlmostEqual(executor.margin_v, 0.05)
        executor.step(
            fork, route_fit, now=1.0, linear=0.05, accepted_entry=False,
            circle_model=self._model(),
        )
        transitioned = executor.step(
            fork, route_fit, now=2.0, linear=0.05, accepted_entry=False,
            circle_model=self._model(),
        )
        self.assertEqual(executor.state, "entry_left_align")
        self.assertEqual(transitioned.reason, "ring_entry_left_align")
        self.assertIsNone(transitioned.fit)
        self._complete_entry_left_turn(executor, fork, 2.0)

    def test_four_legs_advance_in_order(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(
            1, fixed_radius_m=0.25, half_arc_yaw_rad=0.40,
        )
        executor._configure_fixed_chords()
        executor._begin_model_leg(1)
        now = 0.0
        executor.last_now = now
        states = []
        for _index in range(1, 5):
            now = self._complete_model_leg(executor, fork, now)
            states.append(executor.state)
        self.assertEqual(states, [
            "leg2_model", "leg3_model", "leg4_exit_bridge", "exit_line_align",
        ])
        self._complete_exit_line_align(executor, fork, now)

    def test_margin_does_not_require_a_circle_model(self):
        executor = RingEntryExecutor(1, margin_distance_m=0.01)
        model = self._model()
        executor.step(
            None, self._fit(), now=0.0, linear=0.0,
            accepted_entry=True, incoming_v=0.02, circle_model=model,
        )
        result = executor.step(
            None, self._fit(), now=1.0, linear=0.02,
            accepted_entry=False, circle_model=model,
        )
        self.assertEqual(executor.state, "entry_left_align")
        executor.entry_left_command()
        executor.step(
            None, self._fit(), now=2.0, linear=0.0,
            angular=executor.entry_left_turn_rad + 0.1,
            accepted_entry=False, motion_source="measured",
        )
        executor.step(
            None, self._fit(), now=2.1, linear=0.0, angular=0.0,
            accepted_entry=False, motion_source="measured",
        )
        result = executor.step(
            None, self._fit(), now=2.2, linear=0.0, angular=0.0,
            accepted_entry=False, motion_source="measured",
        )
        self.assertEqual(executor.state, "leg1_model")
        self.assertEqual(result.reason, "ring_leg1_align")
        self.assertEqual(result.phase_event, RingPhaseEvent.NONE)

    def test_entry_ellipse_scales_runtime_model_radius(self):
        executor = RingEntryExecutor(1, arc_v=0.02, fixed_radius_m=0.25)
        executor.circle_model = RingCircleModel(
            (220.0, 290.0), (203.36, 394.51), 81.0, 300,
        )
        executor._apply_circle_model_radius()
        self.assertAlmostEqual(executor.radius_estimate_m, 0.25)
        self.assertEqual(executor.radius_source, "entry_ellipse_four_leg")

    def test_model_leg_ignores_visual_fit_until_its_segment_boundary(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(1, half_arc_yaw_rad=0.80)
        executor._configure_fixed_chords()
        executor._begin_model_leg(2)
        executor.leg_phase = "drive"
        executor.leg_chord_length_m = 0.01
        executor.last_now = 0.0
        wild = self._fit(e0=-0.8, e_look=1.0, theta=1.1)
        first = executor.step(
            fork, wild, now=1.0, linear=0.02, angular=-0.20,
            motion_source="measured", accepted_entry=False,
        )
        self.assertEqual(executor.state, "leg2_model")
        self.assertEqual(executor.leg_phase, "drive_stop")
        self.assertEqual(first.reason, "ring_leg2_straight_stop")
        executor.step(
            fork, wild, now=1.1, linear=0.0, angular=0.0,
            motion_source="measured", accepted_entry=False,
        )
        executor.step(
            fork, wild, now=1.2, linear=0.0, angular=0.0,
            motion_source="measured", accepted_entry=False,
        )
        self.assertEqual(executor.state, "leg3_model")

    def test_exit_reacquire_hands_off_only_after_confirmed_path(self):
        _cfg, fork, _debug = self._fork(direction=1)
        executor = RingEntryExecutor(
            1, exit_reacquire_frames=3,
            half_arc_yaw_rad=0.30,
            exit_reacquire_extra_rad=0.40,
        )
        executor.state = "exit_reacquire"
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
            exit_reacquire_frames=2,
            half_arc_yaw_rad=0.40,
            exit_reacquire_extra_rad=0.20,
        )
        states = []
        executor.step(
            fork, self._fit(), now=0.0, linear=0.0,
            incoming_v=0.02, accepted_entry=True, circle_model=self._model(),
        )
        states.append(executor.state)
        executor.step(
            fork, self._fit(), now=0.5, linear=0.0,
            accepted_entry=False, circle_model=self._model(),
        )
        executor.step(
            fork, self._fit(), now=1.0, linear=0.04,
            accepted_entry=False, circle_model=self._model(),
        )
        states.append(executor.state)
        now = self._complete_entry_left_turn(executor, fork, 1.0)
        states.append(executor.state)
        for _index in range(4):
            now = self._complete_model_leg(executor, fork, now)
            states.append(executor.state)
        now = self._complete_exit_line_align(executor, fork, now)
        states.append(executor.state)
        for _index in range(2):
            now += 0.1
            executor.step(
                fork, self._fit(e0=0.10), now=now, linear=0.02,
                angular=0.08, accepted_entry=False,
            )
        self.assertEqual(
            states,
            [
                "margin", "entry_left_align", "leg1_model", "leg2_model", "leg3_model",
                "leg4_exit_bridge", "exit_line_align", "exit_reacquire",
            ],
        )

    def test_entry_left_turn_is_always_physical_left_and_model_independent(self):
        executor = RingEntryExecutor(
            direction=-1, entry_left_turn_rad=np.deg2rad(45.0),
        )
        executor._configure_fixed_chords()
        deltas = executor.leg_turn_deltas_rad
        executor.state = "entry_left_align"
        executor.last_now = 0.0
        self.assertEqual(executor.entry_left_command(), (0.0, 0.2))
        self._complete_entry_left_turn(executor, None, 0.0)
        self.assertEqual(executor.leg_turn_deltas_rad, deltas)
        self.assertAlmostEqual(executor.entry_left_yaw_rad, np.deg2rad(45.0) + 0.1)

    def test_negative_entry_turn_is_physical_right_and_model_independent(self):
        executor = RingEntryExecutor(
            direction=1, entry_left_turn_rad=np.deg2rad(-45.0),
        )
        executor._configure_fixed_chords()
        deltas = executor.leg_turn_deltas_rad
        executor.state = "entry_left_align"
        executor.last_now = 0.0
        self.assertEqual(executor.entry_left_command(), (0.0, -0.2))
        executor.step(
            None, self._fit(), now=1.0, linear=0.0,
            angular=-(np.deg2rad(45.0) + 0.1),
            motion_source="measured", accepted_entry=False,
        )
        self.assertEqual(executor.state, "entry_left_stop")
        self.assertLess(executor.entry_left_yaw_rad, 0.0)
        for now in (1.1, 1.2):
            executor.step(
                None, self._fit(), now=now, linear=0.0, angular=0.0,
                motion_source="measured", accepted_entry=False,
            )
        self.assertEqual(executor.state, "leg1_model")
        self.assertEqual(executor.leg_turn_deltas_rad, deltas)

    def test_entry_turn_stop_accepts_measured_stationary_gyro_floor(self):
        executor = RingEntryExecutor(
            direction=1, entry_left_turn_rad=np.deg2rad(-45.0),
        )
        executor._configure_fixed_chords()
        executor.state = "entry_left_stop"
        executor.last_now = 0.0
        executor.step(
            None, self._fit(), now=0.1, linear=0.0, angular=0.016,
            motion_source="measured", accepted_entry=False,
        )
        self.assertEqual(executor.state, "entry_left_stop")
        executor.step(
            None, self._fit(), now=0.2, linear=0.0, angular=0.015,
            motion_source="measured", accepted_entry=False,
        )
        self.assertEqual(executor.state, "leg1_model")

    def test_entry_turn_stop_rejects_actual_residual_rotation(self):
        executor = RingEntryExecutor(direction=1)
        executor._configure_fixed_chords()
        executor.state = "entry_left_stop"
        executor.last_now = 0.0
        for now in (0.1, 0.2, 0.3):
            executor.step(
                None, self._fit(), now=now, linear=0.0, angular=0.03,
                motion_source="measured", accepted_entry=False,
            )
        self.assertEqual(executor.state, "entry_left_stop")
        self.assertEqual(executor.settled_frames, 0)

    def test_fixed_turn_direction_follows_configured_ring_side(self):
        leg1_commands = []
        leg2_commands = []
        for configured_direction in (-1, 1):
            executor = RingEntryExecutor(configured_direction, fixed_radius_m=0.25)
            executor._configure_fixed_chords()
            executor._begin_model_leg(1)
            leg1_commands.append(executor.model_leg_command())
            executor._begin_model_leg(2)
            leg2_commands.append(executor.model_leg_command())
        self.assertEqual(leg1_commands, [(0.0, 0.2), (0.0, -0.2)])
        self.assertEqual(leg2_commands, [(0.0, -0.2), (0.0, 0.2)])

    def test_left_and_right_fixed_routes_are_exact_mirrors(self):
        left = RingEntryExecutor(-1, fixed_radius_m=0.25)
        right = RingEntryExecutor(1, fixed_radius_m=0.25)
        left._configure_fixed_chords()
        right._configure_fixed_chords()
        self.assertEqual(
            left.leg_turn_deltas_rad,
            tuple(-delta for delta in right.leg_turn_deltas_rad),
        )
        left._begin_exit_line_align()
        right._begin_exit_line_align()
        self.assertAlmostEqual(left.leg_turn_delta_rad, -right.leg_turn_delta_rad)

    def test_fixed_circle_closes_independently_of_post_margin_correction(self):
        for direction in (-1, 1):
            for correction_deg in (-20.0, 30.0, 45.0):
                with self.subTest(direction=direction, correction_deg=correction_deg):
                    executor = RingEntryExecutor(
                        direction,
                        entry_left_turn_rad=np.deg2rad(correction_deg),
                    )
                    executor._configure_fixed_chords()
                    executor._begin_exit_line_align()
                    self.assertAlmostEqual(
                        sum(executor.leg_turn_deltas_rad)
                        + executor.leg_turn_delta_rad,
                        0.0,
                    )

    def test_each_model_leg_aligns_then_drives_a_straight_chord(self):
        executor = RingEntryExecutor(1, fixed_radius_m=0.25)
        executor._configure_fixed_chords()
        executor._begin_model_leg(1)
        align = executor.model_leg_command()
        self.assertEqual(align[0], 0.0)
        self.assertNotEqual(align[1], 0.0)
        executor.leg_phase = "drive"
        self.assertEqual(executor.model_leg_command(), (0.02, 0.0))
        self.assertGreater(executor.leg_chord_length_m, 0.0)

    def test_fixed_four_chords_use_hardcoded_half_circle_geometry(self):
        executor = RingEntryExecutor(1, fixed_radius_m=0.25)
        executor._configure_fixed_chords()
        self.assertEqual(executor.leg_turn_deltas_rad, (
            -3.0 * np.pi / 8.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0,
        ))
        self.assertLess(executor.leg_turn_deltas_rad[0], 0.0)
        self.assertTrue(all(delta > 0.0 for delta in executor.leg_turn_deltas_rad[1:]))
        executor._begin_exit_line_align()
        self.assertAlmostEqual(executor.leg_turn_delta_rad, -3.0 * np.pi / 8.0)
        self.assertAlmostEqual(
            sum(executor.leg_turn_deltas_rad) + executor.leg_turn_delta_rad,
            0.0,
        )
        self.assertAlmostEqual(
            executor.leg_chord_length_m,
            2.0 * 0.25 * np.sin(np.pi / 8.0) * executor.chord_distance_scale,
        )

    def test_command_fallback_is_rejected_for_model_rotation(self):
        executor = RingEntryExecutor(1)
        executor._configure_fixed_chords()
        executor._begin_model_leg(1)
        executor.last_now = 0.0
        result = executor.step(
            None, None, now=1.0, linear=0.0,
            angular=0.1,
            motion_source="command_fallback", accepted_entry=False,
        )
        self.assertEqual(executor.state, "failed")
        self.assertEqual(result.reason, "ring_gyro_feedback_lost")

    def test_model_alignment_slows_near_gyro_target(self):
        executor = RingEntryExecutor(
            1, align_w=0.20, align_slow_w=0.12,
            align_slowdown_rad=0.10,
        )
        executor.leg_phase = "align"
        executor.leg_turn_delta_rad = 0.50
        executor.model_turn_sign = 1
        self.assertEqual(executor.model_leg_command(), (0.0, 0.20))
        executor.leg_yaw_rad = 0.41
        self.assertEqual(executor.model_leg_command(), (0.0, 0.12))

    def test_sustained_wrong_way_gyro_fails_closed(self):
        executor = RingEntryExecutor(1)
        executor._configure_fixed_chords()
        executor._begin_model_leg(1)
        executor.model_leg_command()
        executor.last_now = 0.0
        result = None
        for now in (0.2, 0.4):
            result = executor.step(
                None, None, now=now, linear=0.0,
                angular=-executor.command_turn_sign * 0.10,
                motion_source="measured", accepted_entry=False,
            )
        self.assertEqual(executor.state, "failed")
        self.assertEqual(result.reason, "ring_gyro_wrong_direction")

    def test_reset_restores_configured_fixed_radius(self):
        executor = RingEntryExecutor(1, arc_v=0.02, fixed_radius_m=0.20)
        executor.state = "leg3_model"
        executor.model_aligned_rad = 1.2
        executor.reset()
        self.assertEqual(executor.state, "waiting")
        self.assertEqual(executor.model_aligned_rad, 0.0)
        self.assertIsNone(executor.radius_estimate_m)

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
