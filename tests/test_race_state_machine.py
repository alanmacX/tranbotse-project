import unittest

import cv2 as cv
import numpy as np

from transbot_race.config import RaceConfig
from transbot_race.state_machine import RaceState, RaceStateMachine, TrackMode
from transbot_race.vision import TrajectoryFit, fit_line_trajectory, scan_line_features

W, H, CENTER = 180, 200, 90


def fit_of(mask, cfg):
    features = scan_line_features(mask, cfg.vision, crop_center=CENTER)
    return fit_line_trajectory(features, cfg.vision, crop_center=CENTER, crop_width=W)


def poly_config():
    cfg = RaceConfig()
    cfg.vision.fit_mode = "poly"
    return cfg


def straight(x=CENTER, line_w=18):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.rectangle(mask, (x - line_w // 2, 0), (x + line_w // 2, H - 1), 255, -1)
    return mask


def right_angle(line_w=18, corner_y=150):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.rectangle(mask, (CENTER - line_w // 2, corner_y), (CENTER + line_w // 2, H - 1), 255, -1)
    cv.rectangle(mask, (CENTER - line_w // 2, corner_y - line_w // 2), (W - 1, corner_y + line_w // 2), 255, -1)
    return mask


def empty():
    return np.zeros((H, W), dtype=np.uint8)


def blind_right(line_w=18):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv.rectangle(mask, (CENTER - line_w // 2, 150), (CENTER + line_w // 2, H - 1), 255, -1)
    cv.rectangle(mask, (145, 0), (170, 90), 255, -1)
    return mask


def replay(sm, mask, cfg, start=0.0, n=6, dt=0.1):
    fit = fit_of(mask, cfg)
    cmd = None
    for i in range(n):
        cmd = sm.step(fit, now=start + i * dt)
    return cmd


class UnifiedTrackerTests(unittest.TestCase):
    def test_corner_handoff_uses_the_same_moving_follow_boundary(self):
        sm = RaceStateMachine(RaceConfig())
        fit = TrajectoryFit(
            found=True, e0=0.463, e_look=0.463, theta=0.338,
            conf=0.90, n_bands=5, disconnected=False,
        )

        self.assertTrue(sm.can_take_moving_handoff(fit))
        command = sm.reacquire_from(fit, now=0.0)
        self.assertEqual(command.mode, TrackMode.FOLLOW)
        self.assertGreater(command.v, 0.0)

    def test_corner_handoff_rejects_crossed_but_large_heading_pose(self):
        sm = RaceStateMachine(RaceConfig())
        fit = TrajectoryFit(
            found=True, e0=-0.1823, e_look=-0.1823, theta=0.7693,
            conf=0.88, n_bands=6, disconnected=False,
        )
        self.assertLess(fit.e0 * fit.theta, 0.0)
        self.assertFalse(sm.can_take_moving_handoff(fit))

    def test_follow_slew_limit_prevents_single_frame_direction_reversal(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        first = TrajectoryFit(
            found=True, e0=-1.0, e_look=-1.0, theta=0.0,
            conf=0.90, n_bands=6,
        )
        opposite = TrajectoryFit(
            found=True, e0=1.0, e_look=1.0, theta=0.0,
            conf=0.90, n_bands=6,
        )
        initial = sm.reacquire_from(first, now=0.0)
        next_command = sm.step(opposite, now=0.1)

        self.assertGreater(initial.w, 0.0)
        self.assertGreater(next_command.w, 0.0)
        self.assertLessEqual(
            abs(next_command.w - initial.w),
            cfg.tracker.max_w_slew_rate * 0.1 + 1e-9,
        )

    def test_corner_handoff_rejects_pose_outside_stable_corridor(self):
        sm = RaceStateMachine(RaceConfig())
        fit = TrajectoryFit(
            found=True, e0=0.70, e_look=0.70, theta=0.80,
            conf=0.90, n_bands=5, disconnected=False,
        )

        self.assertFalse(sm.can_take_moving_handoff(fit))

    def test_blank_start_waits_without_search_rotation(self):
        sm = RaceStateMachine(RaceConfig())
        cmd = sm.step(TrajectoryFit(found=False), now=0.0)
        self.assertEqual(cmd.reason, "await_first_line")
        self.assertEqual(cmd.v, 0.0)
        self.assertEqual(cmd.w, 0.0)

    def test_single_band_start_does_not_arm_lost_search(self):
        sm = RaceStateMachine(RaceConfig())
        weak = TrajectoryFit(
            found=True, e0=0.8, e_look=0.8, conf=0.3, n_bands=1,
        )
        command = sm.step(weak, now=0.0)
        self.assertEqual(command.reason, "await_first_line")
        self.assertFalse(sm.ever_acquired)

    def test_weak_two_band_start_only_seeds_filter_without_moving(self):
        sm = RaceStateMachine(RaceConfig())
        weak = TrajectoryFit(
            found=True, e0=0.9, e_look=0.9, conf=0.22, n_bands=2,
        )
        command = sm.step(weak, now=0.0)
        self.assertEqual(command.reason, "await_first_line")
        self.assertEqual(command.v, 0.0)
        self.assertEqual(command.w, 0.0)
        self.assertFalse(sm.ever_acquired)

    def test_weak_startup_fragments_cannot_bias_first_complete_line(self):
        sm = RaceStateMachine(RaceConfig())
        fragment = TrajectoryFit(
            found=True, e0=0.95, e_look=0.95, theta=0.70,
            conf=0.22, n_bands=2,
        )
        for index in range(12):
            command = sm.step(fragment, now=index * 0.1)
            self.assertEqual(command.reason, "await_first_line")

        straight_fit = TrajectoryFit(
            found=True, e0=0.0, e_look=0.0, theta=0.0,
            conf=0.90, n_bands=6,
        )
        command = sm.step(straight_fit, now=1.2)
        self.assertTrue(sm.ever_acquired)
        self.assertAlmostEqual(sm.f_e0, 0.0)
        self.assertAlmostEqual(sm.f_theta, 0.0)
        self.assertAlmostEqual(command.w, 0.0)

    def test_straight_tracks_forward(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        cmd = replay(sm, straight(), cfg)
        self.assertEqual(sm.state, RaceState.TRACK)
        self.assertGreater(cmd.v, 0.0)
        self.assertAlmostEqual(cmd.w, 0.0, delta=0.05)

    def test_offset_line_turns_toward_center(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        cmd = replay(sm, straight(x=CENTER + 35), cfg)
        self.assertEqual(sm.state, RaceState.TRACK)
        # Line to the right of center -> steer right (w < 0 with invert_turn off).
        self.assertLess(cmd.w, 0.0)

    def test_straight_cruise_ignores_noisy_heading_for_same_lateral_fit(self):
        cfg = RaceConfig()
        right_heading = TrajectoryFit(
            found=True, e0=0.2, e_look=0.2, theta=0.3,
            conf=0.9, n_bands=5,
        )
        left_heading = TrajectoryFit(
            found=True, e0=0.2, e_look=0.2, theta=-0.3,
            conf=0.9, n_bands=5,
        )
        right = RaceStateMachine(cfg).step(right_heading, now=0.0)
        left = RaceStateMachine(cfg).step(left_heading, now=0.0)
        self.assertLess(right.w, 0.0)
        self.assertAlmostEqual(right.w, left.w)

    def test_image_curvature_does_not_steer_ordinary_straight_cruise(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        perspective_curve = TrajectoryFit(
            found=True, e0=0.0, e_look=0.0, theta=0.0, kappa=0.28,
            conf=0.95, n_bands=6, path_memory=False,
        )
        command = sm.step(perspective_curve, now=0.0)
        self.assertAlmostEqual(command.w, 0.0)

    def test_generic_cruise_never_uses_session_curvature_feedforward(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        route_curve = TrajectoryFit(
            found=True, e0=0.0, e_look=0.0, theta=0.0, kappa=0.28,
            conf=0.95, n_bands=6, path_memory=True,
        )
        command = sm.step(route_curve, now=0.0)
        self.assertAlmostEqual(command.w, 0.0)

    def test_corner_reacquire_reseeds_filter_from_current_line(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        sm.f_e0 = 1.0
        sm.f_theta = 1.0
        sm.d_e0 = 0.5
        sm.state = RaceState.LOST
        fit = TrajectoryFit(
            found=True, e0=0.10, e_look=0.10, theta=0.20,
            conf=0.9, n_bands=4,
        )
        command = sm.reacquire_from(fit, now=1.0)
        self.assertEqual(sm.state, RaceState.TRACK)
        self.assertAlmostEqual(sm.f_e0, fit.e0)
        self.assertAlmostEqual(sm.f_theta, fit.theta)
        self.assertAlmostEqual(sm.d_e0, 0.0)
        expected_w = -(cfg.tracker.k_e * fit.e0)
        self.assertAlmostEqual(command.w, expected_w)

    def test_crossed_line_uses_moving_lateral_recovery_not_pivot(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        fit = TrajectoryFit(
            found=True, e0=-0.56, e_look=-0.56, theta=0.42,
            conf=0.9, n_bands=4,
        )
        command = sm.reacquire_from(fit, now=1.0)
        self.assertEqual(command.mode, TrackMode.FOLLOW)
        self.assertGreater(command.v, 0.0)
        self.assertGreater(command.w, 0.0)

    def test_offcentre_line_remains_moving_proportional_follow(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        command = sm.reacquire_from(
            TrajectoryFit(
                found=True, e0=0.875, e_look=0.875, theta=0.495,
                conf=0.9, n_bands=6,
            ),
            now=1.0,
        )
        self.assertEqual(command.mode, TrackMode.FOLLOW)
        self.assertGreater(command.v, 0.0)

        command = sm.step(
            TrajectoryFit(
                found=True, e0=0.376, e_look=0.376, theta=0.510,
                conf=0.9, n_bands=6,
            ),
            now=1.1,
        )
        self.assertEqual(command.mode, TrackMode.FOLLOW)
        self.assertGreater(command.v, 0.0)
        self.assertLess(command.w, 0.0)

    def test_large_offset_does_not_invoke_generic_pivot(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        sm.reacquire_from(
            TrajectoryFit(
                found=True, e0=0.875, e_look=0.875, theta=0.495,
                conf=0.9, n_bands=6,
            ),
            now=1.0,
        )
        command = sm.step(
            TrajectoryFit(
                found=True, e0=0.55, e_look=0.55, theta=0.50,
                conf=0.9, n_bands=6,
            ),
            now=1.1,
        )
        self.assertEqual(command.mode, TrackMode.FOLLOW)
        self.assertGreater(command.v, 0.0)

    def test_large_offset_stays_in_one_follow_mode(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        sm.reacquire_from(
            TrajectoryFit(
                found=True, e0=0.70, e_look=0.70, theta=0.45,
                conf=0.9, n_bands=6,
            ),
            now=0.0,
        )
        for now in (0.1, 0.2):
            command = sm.step(
                TrajectoryFit(
                    found=True, e0=0.60, e_look=0.60, theta=0.10,
                    conf=0.9, n_bands=6,
                ),
                now=now,
            )
            self.assertEqual(command.mode, TrackMode.FOLLOW)
            self.assertGreater(command.v, 0.0)

    def test_single_band_false_line_cannot_flip_filtered_side(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        sm.reacquire_from(
            TrajectoryFit(
                found=True, e0=-0.45, e_look=-0.45, theta=0.0,
                conf=0.9, n_bands=5,
            ),
            now=0.0,
        )
        command = sm.step(
            TrajectoryFit(
                found=True, e0=0.90, e_look=0.90, theta=0.0,
                conf=0.9, n_bands=1,
            ),
            now=0.1,
        )
        self.assertLess(sm.f_e0, 0.0)
        self.assertGreater(command.w, 0.0)

    def test_far_only_fit_cannot_acquire_or_reacquire_tracker(self):
        cfg = RaceConfig()
        far = TrajectoryFit(
            found=True, e0=-0.8, theta=0.9, conf=0.9, n_bands=5,
            nearest_band_index=3,
        )
        sm = RaceStateMachine(cfg)
        command = sm.step(far, now=0.0)
        self.assertEqual(command.reason, "await_first_line")
        self.assertFalse(sm.ever_acquired)

        sm.state = RaceState.STOPPED
        sm.last_event = "search_timeout"
        for index in range(4):
            command = sm.step(far, now=1.0 + index * 0.1)
        self.assertEqual(command.state, RaceState.STOPPED)
        self.assertEqual(sm.stopped_reacquire_frames, 0)

    def test_far_only_fit_predicts_without_polluting_valid_history(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        sm.reacquire_from(
            TrajectoryFit(
                found=True, e0=0.25, e_look=0.25, theta=0.1,
                conf=0.9, n_bands=5,
            ),
            now=0.0,
        )
        observed_e0 = sm.observed_e0
        observed_theta = sm.observed_theta
        command = sm.step(
            TrajectoryFit(
                found=True, e0=-0.9, e_look=-0.9, theta=1.0,
                conf=0.9, n_bands=5, nearest_band_index=3,
            ),
            now=0.1,
        )
        self.assertEqual(sm.observed_e0, observed_e0)
        self.assertEqual(sm.observed_theta, observed_theta)
        self.assertGreater(sm.f_e0, 0.0)
        self.assertLess(sm.f_conf, 0.9)
        self.assertGreater(command.w, -0.2)

    def test_large_lateral_error_cannot_be_cancelled_by_heading_and_curvature(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        command = sm.reacquire_from(
            TrajectoryFit(
                found=True, e0=-0.392, e_look=-0.392, theta=0.494,
                kappa=-0.243, conf=0.9, n_bands=6,
            ),
            now=1.0,
        )
        self.assertEqual(command.mode, TrackMode.FOLLOW)
        # The line is substantially left of the chassis, so steering must stay
        # left (positive chassis w) instead of cancelling to zero/wrong-way.
        self.assertGreater(command.w, 0.05)

    def test_pivot_releases_when_line_crosses_centre_before_full_alignment(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        sm.reacquire_from(
            TrajectoryFit(
                found=True, e0=0.875, e_look=0.875, theta=0.495,
                conf=0.9, n_bands=6,
            ),
            now=1.0,
        )
        command = sm.step(
            TrajectoryFit(
                found=True, e0=-0.20, e_look=-0.20, theta=0.40,
                conf=0.9, n_bands=6,
            ),
            now=1.1,
        )
        self.assertEqual(command.mode, TrackMode.FOLLOW)
        self.assertGreater(command.v, 0.0)

        # A still-large error on the next coherent frame must not re-enter the
        # stationary branch.  Once the exposed line has crossed centre,
        # translation—not another pivot pulse—owns convergence.
        command = sm.step(
            TrajectoryFit(
                found=True, e0=-0.70, e_look=-0.70, theta=0.40,
                conf=0.9, n_bands=6,
            ),
            now=1.2,
        )
        self.assertEqual(command.mode, TrackMode.FOLLOW)
        self.assertGreater(command.v, 0.0)

    def test_right_angle_cannot_invoke_generic_pivot(self):
        cfg = poly_config()
        cfg.tracker.e_pivot = 0.55
        cfg.tracker.theta_pivot = 0.65
        sm = RaceStateMachine(cfg)
        replay(sm, straight(), cfg, start=-0.4, n=4)
        cmd = replay(sm, right_angle(), cfg, n=8)
        # Fixed-session geometry owns corners; ordinary cruise must never
        # manufacture a generic pivot. Without stage takeover this sample may
        # safely enter route-loss handling because it is not a valid ordinary
        # near-field line.
        self.assertIn(sm.state, (RaceState.TRACK, RaceState.LOST))
        self.assertNotEqual(cmd.mode, TrackMode.PIVOT)
        if sm.state == RaceState.TRACK:
            self.assertGreater(cmd.v, 0.0)

    def test_dashed_gap_keeps_moving_via_predict(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        # Build confidence on the line, then a couple of blank frames.
        replay(sm, straight(), cfg, n=4)
        c1 = sm.step(fit_of(empty(), cfg), now=1.0)
        c2 = sm.step(fit_of(empty(), cfg), now=1.1)
        self.assertEqual(sm.state, RaceState.TRACK)
        self.assertIn(c2.mode, (TrackMode.PREDICT, TrackMode.FOLLOW))
        self.assertGreater(c2.v, 0.0)  # keeps rolling through the gap

    def test_dashed_gap_applies_lateral_prediction_only_once(self):
        cfg = RaceConfig()
        cfg.tracker.conf_decay = 0.05
        sm = RaceStateMachine(cfg)
        sm.reacquire_from(
            TrajectoryFit(
                found=True, e0=0.10, e_look=0.10, theta=0.0,
                conf=0.95, n_bands=6,
            ),
            now=0.0,
        )
        sm.step(
            TrajectoryFit(
                found=True, e0=0.30, e_look=0.30, theta=0.0,
                conf=0.95, n_bands=6,
            ),
            now=0.1,
        )
        derivative = sm.d_e0
        self.assertGreater(derivative, 0.0)

        blank = TrajectoryFit(found=False)
        sm.step(blank, now=0.2)
        predicted_once = sm.f_e0
        confidence_once = sm.f_conf
        self.assertTrue(sm.gap_prediction_consumed)
        self.assertEqual(sm.d_e0, 0.0)

        sm.step(blank, now=0.3)
        sm.step(blank, now=0.4)
        self.assertAlmostEqual(sm.f_e0, predicted_once)
        self.assertLess(sm.f_conf, confidence_once)

    def test_new_dash_reopens_single_gap_prediction(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        first = TrajectoryFit(
            found=True, e0=0.0, e_look=0.0, theta=0.0,
            conf=0.95, n_bands=6,
        )
        moved = TrajectoryFit(
            found=True, e0=0.20, e_look=0.20, theta=0.0,
            conf=0.95, n_bands=6,
        )
        sm.reacquire_from(first, now=0.0)
        sm.step(moved, now=0.1)
        sm.step(TrajectoryFit(found=False), now=0.2)
        self.assertTrue(sm.gap_prediction_consumed)

        sm.step(moved, now=0.3)
        self.assertFalse(sm.gap_prediction_consumed)
        sm.step(TrajectoryFit(found=False), now=0.4)
        self.assertTrue(sm.gap_prediction_consumed)

    def test_alternating_dashes_update_one_route_without_steering_zigzag(self):
        cfg = RaceConfig()
        sm = RaceStateMachine(cfg)
        first = TrajectoryFit(
            found=True, e0=0.18, e_look=0.18, theta=0.04,
            conf=0.95, n_bands=5,
        )
        sm.reacquire_from(first, now=0.0)
        commands = []
        for index, e0 in enumerate((0.30, 0.08, 0.28, 0.06), start=1):
            sm.step(TrajectoryFit(found=False), now=index * 0.2 - 0.1)
            commands.append(sm.step(
                TrajectoryFit(
                    found=True, e0=e0, e_look=e0, theta=0.04,
                    conf=0.95, n_bands=5,
                ),
                now=index * 0.2,
            ))

        self.assertTrue(sm.line_identity_valid)
        self.assertGreater(sm.line_identity_e0, 0.0)
        self.assertTrue(all(command.w < 0.0 for command in commands))
        self.assertLess(
            max(abs(command.w) for command in commands)
            - min(abs(command.w) for command in commands),
            0.08,
        )

    def test_discontinuous_dash_cannot_replace_straight_route_identity(self):
        sm = RaceStateMachine(RaceConfig())
        first = TrajectoryFit(
            found=True, e0=0.05, e_look=0.05, theta=0.02,
            conf=0.95, n_bands=6,
        )
        sm.reacquire_from(first, now=0.0)
        sm.step(TrajectoryFit(found=False), now=0.1)
        before_e0 = sm.f_e0
        before_theta = sm.f_theta
        self.assertTrue(sm.gap_prediction_consumed)

        swapped = TrajectoryFit(
            found=True, e0=0.75, e_look=0.75, theta=0.70,
            conf=0.95, n_bands=6,
        )
        sm.step(swapped, now=0.2)

        self.assertAlmostEqual(sm.f_e0, before_e0)
        self.assertAlmostEqual(sm.f_theta, before_theta)
        self.assertTrue(sm.gap_prediction_consumed)
        self.assertAlmostEqual(sm.line_identity_e0, first.e0)
        self.assertAlmostEqual(sm.line_identity_theta, first.theta)
        self.assertFalse(sm.observation_reliable)

    def test_path_memory_and_far_only_cannot_refresh_ordinary_line_identity(self):
        sm = RaceStateMachine(RaceConfig())
        first = TrajectoryFit(
            found=True, e0=0.0, e_look=0.0, theta=0.0,
            conf=0.95, n_bands=6,
        )
        sm.reacquire_from(first, now=0.0)
        for fit in (
            TrajectoryFit(
                found=True, e0=0.2, e_look=0.2, theta=0.1,
                conf=0.95, n_bands=6, path_memory=True,
            ),
            TrajectoryFit(
                found=True, e0=0.2, e_look=0.2, theta=0.1,
                conf=0.95, n_bands=5, nearest_band_index=4,
            ),
        ):
            sm.step(fit, now=0.1)
            self.assertAlmostEqual(sm.line_identity_e0, first.e0)
            self.assertAlmostEqual(sm.line_identity_theta, first.theta)

    def test_lost_and_stopped_clear_straight_route_identity(self):
        sm = RaceStateMachine(RaceConfig())
        fit = TrajectoryFit(
            found=True, e0=0.1, e_look=0.1, theta=0.0,
            conf=0.95, n_bands=6,
        )
        sm.reacquire_from(fit, now=0.0)
        self.assertTrue(sm.line_identity_valid)
        sm._enter(RaceState.LOST, now=0.1, event="test")
        self.assertFalse(sm.line_identity_valid)

        sm.reacquire_from(fit, now=0.2)
        sm._enter(RaceState.STOPPED, now=0.3, event="operator_stop")
        self.assertFalse(sm.line_identity_valid)

    def test_long_gap_goes_lost_then_searches_then_stops(self):
        cfg = RaceConfig()
        cfg.tracker.search_timeout_sec = 0.5
        sm = RaceStateMachine(cfg)
        replay(sm, straight(), cfg, n=4)
        blank = fit_of(empty(), cfg)
        # Enough blank frames to decay confidence below conf_lost.
        for i in range(20):
            cmd = sm.step(blank, now=1.0 + i * 0.05)
        # Either searching or already timed out to STOPPED; search never w=0.
        if sm.state == RaceState.LOST:
            self.assertGreaterEqual(abs(cmd.w), cfg.tracker.w_search_min)
        cmd = sm.step(blank, now=10.0)
        self.assertEqual(sm.state, RaceState.STOPPED)

    def test_search_timeout_recovers_from_three_complete_line_frames(self):
        cfg = RaceConfig()
        cfg.tracker.search_timeout_sec = 0.1
        sm = RaceStateMachine(cfg)
        replay(sm, straight(), cfg, n=4)
        blank = fit_of(empty(), cfg)
        for index in range(20):
            sm.step(blank, now=1.0 + index * 0.05)
        self.assertEqual(sm.state, RaceState.STOPPED)

        recovered_fit = fit_of(straight(x=CENTER + 10), cfg)
        first = sm.step(recovered_fit, now=3.0)
        second = sm.step(recovered_fit, now=3.1)
        recovered = sm.step(recovered_fit, now=3.2)
        self.assertEqual(first.state, RaceState.STOPPED)
        self.assertEqual(second.state, RaceState.STOPPED)
        self.assertEqual(recovered.state, RaceState.TRACK)
        self.assertGreater(recovered.v, 0.0)

    def test_search_timeout_recovery_requires_same_complete_line(self):
        sm = RaceStateMachine(RaceConfig())
        sm.state = RaceState.STOPPED
        sm.last_event = "search_timeout"
        right = TrajectoryFit(
            found=True, e0=0.90, e_look=0.90, theta=0.80,
            conf=0.90, n_bands=5,
        )
        left = TrajectoryFit(
            found=True, e0=-0.90, e_look=-0.90, theta=-0.80,
            conf=0.90, n_bands=5,
        )
        for index, fit in enumerate((right, left, right)):
            command = sm.step(fit, now=index * 0.1)
            self.assertEqual(command.state, RaceState.STOPPED)
        self.assertEqual(sm.stopped_reacquire_frames, 1)

        sm.step(right, now=0.3)
        recovered = sm.step(right, now=0.4)
        self.assertEqual(recovered.state, RaceState.TRACK)

    def test_startup_preview_cannot_leak_into_first_real_line(self):
        cfg = RaceConfig()
        cfg.tracker.preview_plan_enabled = True
        sm = RaceStateMachine(cfg)
        weak_preview = TrajectoryFit(
            found=True, conf=0.10, n_bands=1,
            preview_dir=1, preview_conf=0.90,
        )
        command = sm.step(weak_preview, now=0.0)
        self.assertEqual(command.reason, "await_first_line")
        self.assertEqual(sm.plan_dir, 0)

        sm.step(
            TrajectoryFit(found=True, conf=0.90, n_bands=6),
            now=0.1,
        )
        command = sm.step(TrajectoryFit(found=False), now=0.2)
        self.assertNotEqual(command.mode, TrackMode.PLAN)

    def test_lookahead_steers_toward_ahead_error(self):
        # With pure-pursuit enabled, an approaching bend (line offset ahead but
        # centered at the bottom) still turns toward the lookahead point.
        cfg = poly_config()
        cfg.tracker.lookahead_frac = 0.6
        cfg.tracker.k_theta = 0.0
        cfg.tracker.k_ff = 0.0
        sm = RaceStateMachine(cfg)
        features = scan_line_features(right_angle(corner_y=120), cfg.vision, crop_center=CENTER)
        fit = fit_line_trajectory(
            features, cfg.vision, crop_center=CENTER, crop_width=W, lookahead_frac=0.6
        )
        cmd = None
        for i in range(8):
            cmd = sm.step(fit, now=i * 0.1)
        self.assertEqual(sm.state, RaceState.TRACK)
        self.assertGreater(abs(cmd.w), 0.0)

    def test_blind_preview_executes_plan_inside_track(self):
        cfg = poly_config()
        cfg.tracker.preview_plan_enabled = True
        cfg.tracker.preview_forward_sec = 0.2
        cfg.tracker.preview_turn_sec = 0.4
        sm = RaceStateMachine(cfg)
        replay(sm, straight(), cfg, start=-0.4, n=4)
        fit = fit_of(blind_right(), cfg)
        cmd1 = sm.step(fit, now=0.0)
        cmd2 = sm.step(fit, now=0.25)
        self.assertEqual(sm.state, RaceState.TRACK)
        self.assertEqual(cmd1.mode, TrackMode.PLAN)
        self.assertEqual(cmd1.reason, "preview_forward")
        self.assertEqual(cmd2.mode, TrackMode.PLAN)
        self.assertEqual(cmd2.reason, "preview_turn")
        self.assertLess(cmd2.w, 0.0)



if __name__ == "__main__":
    unittest.main()
