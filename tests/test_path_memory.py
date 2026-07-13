import unittest
from dataclasses import replace
import math

from transbot_race.config import PathMemoryConfig
from transbot_race.capture_geometry import (
    CaptureGeometryDecision,
    CaptureGeometryObservation,
)
from transbot_race.path_memory import (
    CornerCommandDelay, FirstEntryLineLatch, read_motion_sample,
)
from transbot_race.vision import LineFeatures, TrajectoryFit


def turn_fit(theta=0.4):
    return TrajectoryFit(found=True, e0=0.1, e_look=0.5, theta=theta, conf=0.9, n_bands=5)


class FakeBot:
    def __init__(self, value=None): self.value = value
    def get_motion_data(self):
        if isinstance(self.value, Exception): raise self.value
        return self.value


class PathStrategyTests(unittest.TestCase):
    def test_corner_event_holds_then_commits_to_detected_turn(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.02,
            corner_confirm_frames=3,
            capture_geometry_enabled=False,
        )
        gate = CornerCommandDelay(cfg)
        features = LineFeatures(found=True)
        gate.step(turn_fit(), features, 0.05, -0.08, 0.0, 0.0)
        gate.step(turn_fit(), features, 0.05, -0.09, 0.1, 0.1)
        held = gate.step(turn_fit(), features, 0.05, -0.10, 0.2, 0.1)
        self.assertEqual(held.status.reason, "waiting_margin")
        self.assertLessEqual(abs(held.w), cfg.corner_hold_max_w)
        self.assertAlmostEqual(gate.target_angle_rad, math.pi / 2.0, places=2)
        gate.step(turn_fit(), features, 0.05, -0.12, 0.3, 0.1)
        turning = gate.step(turn_fit(), features, 0.05, -0.15, 0.4, 0.1)
        self.assertEqual(turning.status.reason, "committed_turn")
        self.assertAlmostEqual(turning.w, -cfg.corner_replay_max_w)
        self.assertAlmostEqual(turning.v, 0.0)

    def test_corner_event_keeps_trigger_speed_when_live_tracker_loses_line(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.10,
            corner_confirm_frames=3,
            capture_geometry_enabled=False,
        )
        gate = CornerCommandDelay(cfg)
        features = LineFeatures(found=True)

        gate.step(turn_fit(), features, 0.06, -0.10, 0.0, 0.0)
        gate.step(turn_fit(), features, 0.06, -0.11, 0.1, 0.0)
        triggered = gate.step(turn_fit(), features, 0.06, -0.12, 0.2, 0.0)
        self.assertEqual(triggered.status.reason, "waiting_margin")
        self.assertAlmostEqual(triggered.v, 0.06)

        lost_fit = replace(turn_fit(), found=False, conf=0.0)
        waiting = gate.step(lost_fit, LineFeatures(found=False), 0.0, 0.0, 0.3, 0.04)
        self.assertEqual(waiting.status.reason, "waiting_margin")
        self.assertAlmostEqual(waiting.v, 0.06)
        self.assertAlmostEqual(waiting.w, 0.0)

    def test_corner_event_uses_measured_yaw_then_reacquires_line(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.01,
            corner_confirm_frames=3,
            corner_turn_angle_rad=1.2,
            corner_reacquire_angle_rad=0.5,
            corner_reacquire_confirm_frames=3,
            capture_geometry_enabled=False,
        )
        gate = CornerCommandDelay(cfg)
        features = LineFeatures(found=True)
        gate.step(turn_fit(), features, 0.05, -0.04, 0.0, 0.0)
        gate.step(turn_fit(), features, 0.05, -0.04, 0.1, 0.0, 0.0)
        gate.step(turn_fit(), features, 0.05, -0.04, 0.2, 0.0, 0.0)
        turning = gate.step(turn_fit(), features, 0.05, -0.04, 0.3, 0.1, 0.0)
        self.assertEqual(turning.status.reason, "committed_turn")

        unaligned = replace(turn_fit(theta=0.4), e0=0.4)
        turning = gate.step(
            unaligned, features, 0.0, 0.0, 0.8, 0.0, -1.0,
        )
        self.assertEqual(turning.status.reason, "corner_exit_captured")
        self.assertAlmostEqual(turning.w, 0.0)
        self.assertAlmostEqual(turning.status.target[0], 0.5)

        reacquired = gate.step(unaligned, features, 0.03, -0.02, 1.3, 0.0, -1.0)
        self.assertEqual(reacquired.status.reason, "corner_visual_takeover")
        self.assertAlmostEqual(reacquired.v, 0.03)

    def test_cooldown_clears_when_exit_crosses_center_on_a_curve(self):
        cfg = PathMemoryConfig(corner_confirm_frames=3)
        gate = CornerCommandDelay(cfg)
        gate.state = "cooldown"
        gate.candidate_dir = 1
        curve = TrajectoryFit(found=True, e0=0.42, theta=0.55, conf=0.9, n_bands=5)
        features = LineFeatures(found=True)

        output = gate.step(curve, features, 0.04, -0.08, 0.0, 0.04, -0.08)
        self.assertEqual(output.status.reason, "cooldown")

        centred_curve = replace(curve, e0=0.10)
        gate.step(centred_curve, features, 0.04, -0.08, 0.1, 0.04, -0.08)
        output = gate.step(centred_curve, features, 0.04, -0.08, 0.2, 0.04, -0.08)
        self.assertEqual(output.status.reason, "armed")

    def test_ring_curve_does_not_rearm_on_center_crossing(self):
        gate = CornerCommandDelay(PathMemoryConfig())
        gate.state = "cooldown"
        gate.candidate_dir = -1
        gate.event_shape = "curve"
        features = LineFeatures(found=True)
        centred_arc = TrajectoryFit(
            found=True, e0=0.04, theta=-0.62, conf=0.94,
            n_bands=6, disconnected=False,
        )

        for index in range(8):
            output = gate.step(
                centred_arc, features, 0.05, 0.02,
                index * 0.1, 0.05, 0.02,
            )
        self.assertEqual(output.status.reason, "cooldown")

        straight = replace(centred_arc, e0=0.02, theta=0.03)
        for index in range(3):
            output = gate.step(
                straight, features, 0.05, 0.0,
                1.0 + index * 0.1, 0.05, 0.0,
            )
        self.assertEqual(output.status.reason, "armed")

    def test_corner_event_rejects_heading_lateral_sign_conflict(self):
        gate = CornerCommandDelay(PathMemoryConfig(
            corner_confirm_frames=3,
            capture_geometry_enabled=False,
        ))
        conflict = TrajectoryFit(found=True, e0=-0.08, theta=0.42, conf=0.9, n_bands=3)
        for i in range(6):
            output = gate.step(conflict, LineFeatures(found=True), 0.05, 0.0, i * 0.1, 0.05)
        self.assertEqual(output.status.reason, "armed")

    def test_corner_event_does_not_delay_every_fit(self):
        gate = CornerCommandDelay(PathMemoryConfig(
            corner_confirm_frames=2,
            capture_geometry_enabled=False,
        ))
        straight = TrajectoryFit(found=True, e0=0.08, theta=0.02, conf=0.9)
        output = gate.step(straight, LineFeatures(found=True), 0.05, 0.01, 0.0, 0.0)
        self.assertEqual(output.w, 0.01)
        self.assertEqual(output.status.reason, "armed")

    def test_motion_reader_fallback(self):
        measured = read_motion_sample(FakeBot((0.04, -0.2)), 0.01, 0.02)
        self.assertEqual(measured.source, "measured")
        stale = read_motion_sample(FakeBot((0.0, 0.0)), 0.03, -0.1)
        self.assertEqual(stale.source, "command_fallback")
        stale_pivot = read_motion_sample(FakeBot((0.0, 0.0)), 0.0, -0.2)
        self.assertEqual(stale_pivot.source, "command_fallback")
        self.assertAlmostEqual(stale_pivot.angular, -0.2)

    def test_capture_geometry_waits_for_distance_and_vertex_gate(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.02,
            corner_gate_y_frac=0.50,
            corner_gate_confirm_frames=2,
        )
        gate = CornerCommandDelay(cfg)
        gate.stable_v, gate.stable_w = 0.10, 0.0
        features = LineFeatures(found=True)
        fit = turn_fit()
        decision = CaptureGeometryDecision(
            kind="corner",
            direction=1,
            angle_rad=math.pi / 2.0,
            vertex_y_frac=0.35,
            incoming_e=0.0,
            incoming_theta=0.0,
            votes=3,
        )
        below = CaptureGeometryObservation(
            kind="corner", direction=1, vertex_y_frac=0.40, confidence=0.9,
        )
        above = CaptureGeometryObservation(
            kind="curve", direction=1, vertex_y_frac=0.60, confidence=0.9,
        )

        captured = gate.step(
            fit, features, 0.10, -0.12, 0.0, 0.0,
            geometry=below, geometry_decision=decision,
        )
        self.assertEqual(captured.status.reason, "approaching_corner")
        self.assertAlmostEqual(captured.w, 0.0)

        gate.step(fit, features, 0.10, -0.12, 0.1, 0.10, geometry=below)
        first_gate = gate.step(fit, features, 0.10, -0.12, 0.2, 0.10, geometry=above)
        self.assertEqual(first_gate.status.reason, "approaching_corner")
        self.assertAlmostEqual(first_gate.status.remaining_m, 0.02)

        waiting = gate.step(fit, features, 0.10, -0.12, 0.3, 0.10, geometry=above)
        self.assertEqual(waiting.status.reason, "waiting_margin")
        self.assertAlmostEqual(waiting.status.remaining_m, 0.01)
        turning = gate.step(fit, features, 0.10, -0.12, 0.4, 0.10, geometry=above)
        self.assertEqual(turning.status.reason, "committed_turn")

    def test_capture_geometry_gate_does_not_skip_remaining_distance(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.10,
            corner_gate_y_frac=0.50,
            corner_gate_confirm_frames=1,
        )
        gate = CornerCommandDelay(cfg)
        gate.stable_v, gate.stable_w = 0.05, 0.08
        observation = CaptureGeometryObservation(
            kind="corner", direction=-1, vertex_y_frac=0.60, confidence=0.9,
        )
        decision = CaptureGeometryDecision(
            kind="corner", direction=-1, angle_rad=math.pi / 2.0,
            vertex_y_frac=0.60, incoming_e=0.0, incoming_theta=0.0, votes=3,
        )
        gate.step(
            turn_fit(-0.4), LineFeatures(found=True), 0.05, 0.1, 0.0, 0.0,
            geometry=observation, geometry_decision=decision,
        )
        waiting = gate.step(
            turn_fit(-0.4), LineFeatures(found=True), 0.05, 0.1, 0.1, 0.05,
            geometry=observation,
        )
        self.assertEqual(waiting.status.reason, "waiting_margin")
        self.assertAlmostEqual(waiting.status.remaining_m, 0.10)
        advanced = gate.step(
            turn_fit(-0.4), LineFeatures(found=True), 0.05, 0.1, 0.2, 0.05,
            geometry=observation,
        )
        self.assertAlmostEqual(advanced.status.remaining_m, 0.095)

    def test_roundabout_fork_skips_margin_and_uses_lower_turn_rate(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.365,
            corner_gate_y_frac=0.50,
            corner_gate_confirm_frames=1,
            corner_replay_max_w=0.35,
            roundabout_margin_enabled=False,
            roundabout_replay_max_w=0.20,
        )
        gate = CornerCommandDelay(cfg)
        gate.stable_v, gate.stable_w = 0.05, 0.0
        observation = CaptureGeometryObservation(
            kind="curve", direction=1, vertex_y_frac=0.60,
            confidence=0.9, is_fork=True,
        )
        decision = CaptureGeometryDecision(
            kind="turn", direction=1, angle_rad=math.pi / 2.0,
            vertex_y_frac=0.60, incoming_e=0.0, incoming_theta=0.0,
            votes=3, is_fork=True,
        )
        gate.step(
            turn_fit(), LineFeatures(found=True), 0.05, 0.0, 0.0, 0.0,
            geometry=observation, geometry_decision=decision,
        )
        output = gate.step(
            turn_fit(), LineFeatures(found=True), 0.05, 0.0, 0.1, 0.05,
            geometry=observation,
        )
        self.assertEqual(output.status.reason, "committed_turn")
        self.assertAlmostEqual(output.status.remaining_m, 0.0)
        self.assertAlmostEqual(output.w, -0.20)

        selected_right_path = TrajectoryFit(
            found=True, e0=0.40, e_look=0.40, theta=0.20,
            conf=0.90, n_bands=5, disconnected=False,
        )
        still_turning = gate.step(
            selected_right_path, LineFeatures(found=True),
            0.0, 0.0, 0.2, 0.0, angular=-0.05,
        )
        self.assertEqual(still_turning.status.reason, "committed_turn")
        captured = gate.step(
            selected_right_path, LineFeatures(found=True),
            0.0, 0.0, 1.2, 0.0, angular=-0.35,
        )
        self.assertEqual(captured.status.reason, "corner_exit_captured")
        self.assertAlmostEqual(captured.w, 0.0)
        self.assertGreaterEqual(gate.turned_rad, cfg.corner_reacquire_angle_rad)

    def test_roundabout_margin_switch_preserves_legacy_delay(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.10,
            corner_gate_y_frac=0.50,
            corner_gate_confirm_frames=1,
            roundabout_margin_enabled=True,
        )
        gate = CornerCommandDelay(cfg)
        gate.stable_v, gate.stable_w = 0.05, 0.0
        observation = CaptureGeometryObservation(
            kind="curve", direction=1, vertex_y_frac=0.60,
            confidence=0.9, is_fork=True,
        )
        decision = CaptureGeometryDecision(
            kind="turn", direction=1, angle_rad=math.pi / 2.0,
            vertex_y_frac=0.60, incoming_e=0.0, incoming_theta=0.0,
            votes=3, is_fork=True,
        )
        gate.step(
            turn_fit(), LineFeatures(found=True), 0.05, 0.0, 0.0, 0.0,
            geometry=observation, geometry_decision=decision,
        )
        output = gate.step(
            turn_fit(), LineFeatures(found=True), 0.05, 0.0, 0.1, 0.05,
            geometry=observation,
        )
        self.assertEqual(output.status.reason, "waiting_margin")
        self.assertAlmostEqual(output.status.remaining_m, 0.10)

    def test_roundabout_margin_gate_waits_until_incoming_stem_is_aligned(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.10,
            corner_gate_y_frac=0.50,
            corner_gate_confirm_frames=1,
            roundabout_margin_enabled=True,
        )
        gate = CornerCommandDelay(cfg)
        gate.stable_v, gate.stable_w = 0.05, 0.03
        misaligned = CaptureGeometryObservation(
            kind="curve", direction=1, vertex_y_frac=0.65,
            incoming_e=-0.60, incoming_theta=0.05,
            confidence=0.9, is_fork=True,
        )
        decision = CaptureGeometryDecision(
            kind="turn", direction=1, angle_rad=math.pi / 2.0,
            vertex_y_frac=0.65, incoming_e=-0.60, incoming_theta=0.05,
            votes=3, is_fork=True,
        )
        gate.step(
            turn_fit(), LineFeatures(found=True), 0.05, 0.03, 0.0, 0.0,
            geometry=misaligned, geometry_decision=decision,
        )
        output = gate.step(
            turn_fit(), LineFeatures(found=True), 0.05, 0.03, 0.1, 0.05,
            geometry=misaligned,
        )
        self.assertEqual(output.status.reason, "geometry_curve_passthrough")
        self.assertAlmostEqual(output.status.remaining_m, 0.0)
        # A misaligned fork is observation only and cannot take motor ownership.
        self.assertAlmostEqual(output.w, 0.03)

    def test_aligned_fork_approach_owns_a_straight_motor_command(self):
        cfg = PathMemoryConfig(corner_gate_y_frac=0.70)
        gate = CornerCommandDelay(cfg)
        gate.stable_v, gate.stable_w = 0.05, -0.08
        observation = CaptureGeometryObservation(
            kind="curve", direction=1, vertex_y_frac=0.40,
            incoming_e=0.05, incoming_theta=0.03,
            confidence=0.9, is_fork=True,
        )
        decision = CaptureGeometryDecision(
            kind="turn", direction=1, angle_rad=math.pi / 2,
            vertex_y_frac=0.40, incoming_e=0.05, incoming_theta=0.03,
            votes=3, is_fork=True,
        )
        output = gate.step(
            turn_fit(), LineFeatures(found=True), 0.05, -0.22,
            0.0, 0.0, geometry=observation, geometry_decision=decision,
        )
        self.assertEqual(output.status.reason, "approaching_corner")
        self.assertAlmostEqual(output.v, 0.05)
        self.assertAlmostEqual(output.w, 0.0)

    def test_zero_speed_pivot_cannot_erase_forward_approach_command(self):
        gate = CornerCommandDelay(PathMemoryConfig())
        features = LineFeatures(found=True)
        fit = turn_fit()

        gate.step(fit, features, 0.05, -0.03, 0.0, 0.0)
        self.assertAlmostEqual(gate.stable_v, 0.05)
        gate.step(fit, features, 0.0, -0.34, 0.1, 0.0)

        self.assertAlmostEqual(gate.stable_v, 0.05)
        self.assertAlmostEqual(gate.stable_w, -0.03)

    def test_geometry_takeover_waits_for_a_forward_command(self):
        gate = CornerCommandDelay(PathMemoryConfig(corner_gate_y_frac=0.70))
        features = LineFeatures(found=True)
        fit = turn_fit()
        observation = CaptureGeometryObservation(
            kind="corner", direction=1, vertex_y_frac=0.40,
            incoming_e=0.0, incoming_theta=0.0, confidence=0.9,
        )
        decision = CaptureGeometryDecision(
            kind="turn", direction=1, angle_rad=math.pi / 2,
            vertex_y_frac=0.40, incoming_e=0.0, incoming_theta=0.0,
            votes=3,
        )

        self.assertFalse(gate.will_accept_geometry(decision))
        stopped = gate.step(
            fit, features, 0.0, 0.0, 0.0, 0.0,
            geometry=observation, geometry_decision=decision,
        )
        self.assertEqual(gate.state, "armed")
        self.assertNotEqual(stopped.status.reason, "approaching_corner")

        moving = gate.step(
            fit, features, 0.05, -0.02, 0.1, 0.05,
            geometry=observation, geometry_decision=decision,
        )
        self.assertEqual(gate.state, "armed")
        self.assertNotEqual(moving.status.reason, "approaching_corner")

        takeover = gate.step(
            fit, features, 0.0, 0.0, 0.2, 0.0,
            geometry=observation, geometry_decision=decision,
        )
        self.assertEqual(gate.state, "approach")
        self.assertEqual(takeover.status.reason, "approaching_corner")
        self.assertAlmostEqual(takeover.v, 0.05)

    def test_direction_flip_cannot_keep_ordinary_approach_alive(self):
        cfg = PathMemoryConfig(corner_approach_missing_frames=1)
        gate = CornerCommandDelay(cfg)
        gate.stable_v = 0.05
        initial = CaptureGeometryObservation(
            kind="corner", direction=1, vertex_y_frac=0.30, confidence=0.9,
        )
        decision = CaptureGeometryDecision(
            kind="turn", direction=1, angle_rad=math.pi / 2,
            vertex_y_frac=0.30, incoming_e=0.0, incoming_theta=0.0,
            votes=3,
        )
        gate.step(
            turn_fit(), LineFeatures(found=True), 0.05, 0.0,
            0.0, 0.0, geometry=initial, geometry_decision=decision,
        )
        flipped = CaptureGeometryObservation(
            kind="corner", direction=-1, vertex_y_frac=0.35, confidence=0.9,
        )
        gate.step(
            turn_fit(), LineFeatures(found=True), 0.05, 0.0,
            0.1, 0.05, geometry=flipped,
        )
        output = gate.step(
            turn_fit(), LineFeatures(found=True), 0.05, 0.0,
            0.2, 0.05, geometry=flipped,
        )
        self.assertEqual(gate.state, "armed")
        self.assertEqual(output.status.reason, "armed")

    def test_clipped_incoming_geometry_cannot_take_ordinary_corner(self):
        gate = CornerCommandDelay(PathMemoryConfig())
        gate.stable_v = 0.05
        clipped = CaptureGeometryDecision(
            kind="turn", direction=1, angle_rad=math.pi / 2,
            vertex_y_frac=0.60, incoming_e=-1.0, incoming_theta=-1.2,
            votes=3,
        )
        self.assertFalse(gate.will_accept_geometry(clipped))

    def test_capture_geometry_circle_never_starts_corner_margin(self):
        gate = CornerCommandDelay(PathMemoryConfig())
        observation = CaptureGeometryObservation(kind="circle", confidence=0.9)
        decision = CaptureGeometryDecision(
            kind="circle", direction=0, angle_rad=0.0,
            vertex_y_frac=0.5, incoming_e=0.0, incoming_theta=0.0, votes=3,
        )
        output = gate.step(
            turn_fit(), LineFeatures(found=True), 0.05, -0.1, 0.0, 0.0,
            geometry=observation, geometry_decision=decision,
        )
        self.assertEqual(output.status.reason, "geometry_circle_passthrough")
        self.assertAlmostEqual(output.w, -0.1)

    def test_partial_capture_angle_does_not_shorten_blind_search_envelope(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.0,
            corner_reacquire_angle_rad=0.20,
            corner_reacquire_confirm_frames=2,
        )
        gate = CornerCommandDelay(cfg)
        gate.stable_v = 0.05
        observation = CaptureGeometryObservation(
            kind="corner", direction=1, vertex_y_frac=0.60, confidence=0.9,
        )
        decision = CaptureGeometryDecision(
            kind="corner", direction=1, angle_rad=math.radians(65.0),
            vertex_y_frac=0.60, incoming_e=0.0, incoming_theta=0.0, votes=3,
        )
        missing = replace(turn_fit(), found=False, conf=0.0, n_bands=0)
        gate.step(
            missing, LineFeatures(found=False), 0.05, 0.0, 0.0, 0.0,
            geometry=observation, geometry_decision=decision,
        )
        gate.step(
            missing, LineFeatures(found=False), 0.05, 0.0, 0.1, 0.05,
            geometry=observation,
        )
        gate.step(
            missing, LineFeatures(found=False), 0.05, 0.0, 0.2, 0.05,
            geometry=observation,
        )
        turning = gate.step(missing, LineFeatures(found=False), 0.0, 0.0, 0.3, 0.0)
        self.assertEqual(turning.status.reason, "committed_turn")
        self.assertAlmostEqual(gate.target_angle_rad, cfg.corner_turn_angle_rad)
        still_turning = gate.step(
            missing, LineFeatures(found=False), 0.0, 0.0, 1.1, 0.0,
            angular=-1.0,
        )
        self.assertEqual(still_turning.status.reason, "committed_turn")
        searching = gate.step(
            missing, LineFeatures(found=False), 0.0, 0.0, 1.9, 0.0,
            angular=-1.0,
        )
        self.assertEqual(searching.status.reason, "corner_reacquire_search")
        stopped = gate.step(
            missing, LineFeatures(found=False), 0.0, 0.0, 2.4, 0.0,
            angular=-1.0,
        )
        self.assertEqual(stopped.status.reason, "corner_reacquire_failed")
        self.assertAlmostEqual(stopped.w, 0.0)

    def test_first_exit_line_stops_turn_then_hands_directly_to_cruise(self):
        cfg = PathMemoryConfig(
            corner_reacquire_angle_rad=0.20,
            corner_reacquire_confirm_frames=2,
        )
        gate = CornerCommandDelay(cfg)
        gate.state = "turning"
        gate.candidate_dir = 1
        gate.hold_v = 0.05
        gate.target_angle_rad = math.radians(45.0)
        gate.turned_rad = 0.70
        trackable = TrajectoryFit(
            found=True, e0=0.40, theta=0.10, conf=0.9, n_bands=4,
        )
        output = gate.step(
            trackable, LineFeatures(found=True), 0.04, 0.03,
            0.0, 0.0, 0.0,
        )
        self.assertEqual(output.status.reason, "corner_exit_captured")
        self.assertAlmostEqual(output.v, 0.0)
        self.assertAlmostEqual(output.w, 0.0)
        takeover = gate.step(
            trackable, LineFeatures(found=True), 0.04, 0.03,
            0.1, 0.0, 0.0,
        )
        self.assertEqual(takeover.status.reason, "corner_visual_takeover")
        self.assertAlmostEqual(takeover.w, 0.03)
        self.assertEqual(gate.state, "cooldown")

    def test_161732_exit_hands_off_despite_large_heading(self):
        gate = CornerCommandDelay(PathMemoryConfig(corner_reacquire_angle_rad=0.35))
        gate.state = "turning"
        gate.candidate_dir = 1
        gate.turned_rad = 0.80
        gate.target_angle_rad = 1.57
        features = LineFeatures(found=True)
        first_entry = TrajectoryFit(
            found=True, e0=0.7930, theta=0.8267, conf=0.5916,
            n_bands=4, disconnected=False,
        )
        same_exit = replace(first_entry, e0=0.7457, theta=0.8272, conf=0.5905)

        captured = gate.step(first_entry, features, 0.0, 0.0, 22.28, 0.0, -0.2)
        takeover = gate.step(same_exit, features, 0.0, 0.0, 22.37, 0.0, 0.0)

        self.assertEqual(captured.status.reason, "corner_exit_captured")
        self.assertEqual(takeover.status.reason, "corner_visual_takeover")
        self.assertEqual(gate.state, "cooldown")

    def test_run_regression_first_entry_prevents_visual_alignment_drift(self):
        cfg = PathMemoryConfig(
            corner_reacquire_angle_rad=0.35,
            corner_reacquire_confirm_frames=2,
            corner_replay_max_w=0.20,
        )
        gate = CornerCommandDelay(cfg)
        gate.state = "turning"
        gate.candidate_dir = 1
        gate.hold_v = 0.0572
        gate.target_angle_rad = 1.0188
        gate.turned_rad = 0.80
        features = LineFeatures(found=True)

        # Representative reliable frames from 20260713-143944_final.  The old
        # controller ignored all of them and kept v=0.0257,w=-0.2.
        entering = TrajectoryFit(
            found=True, e0=0.7256, theta=0.9217, kappa=-1.0,
            conf=0.6596, n_bands=5,
        )
        first = gate.step(
            entering, features, 0.02, -0.2, 0.0, 0.0, 0.0,
        )
        self.assertEqual(first.status.reason, "corner_exit_captured")
        self.assertEqual(first.v, 0.0)
        self.assertEqual(first.w, 0.0)

        takeover = gate.step(
            entering, features, 0.02, -0.2, 0.05, 0.0, 0.0,
        )
        self.assertEqual(takeover.status.reason, "corner_visual_takeover")
        self.assertEqual(gate.state, "cooldown")

    def test_failed_turn_hands_complete_offcentre_line_to_cruise(self):
        gate = CornerCommandDelay(
            PathMemoryConfig(corner_reacquire_confirm_frames=3),
        )
        gate.state = "failed"
        gate.candidate_dir = 1
        gate.turned_rad = 2.0
        recovered_fit = TrajectoryFit(
            found=True, e0=0.8, theta=-0.5, conf=0.95, n_bands=6,
        )
        features = LineFeatures(found=True)
        first = gate.step(recovered_fit, features, 0.0, 0.0, 0.0, 0.0, 0.0)
        second = gate.step(recovered_fit, features, 0.0, 0.0, 0.1, 0.0, 0.0)
        output = gate.step(recovered_fit, features, 0.0, 0.0, 0.2, 0.0, 0.0)
        self.assertEqual(first.status.reason, "corner_failed_recovery_confirm")
        self.assertEqual(second.status.reason, "corner_failed_recovery_confirm")
        self.assertEqual(output.status.reason, "corner_visual_takeover")
        self.assertEqual(gate.state, "cooldown")

    def test_failed_turn_recovers_from_confirmed_complete_cruise_line(self):
        gate = CornerCommandDelay(PathMemoryConfig(corner_reacquire_confirm_frames=3))
        gate.state = "failed"
        gate.candidate_dir = 1
        gate.turned_rad = 2.0
        features = LineFeatures(found=True)
        cruise_fit = TrajectoryFit(
            found=True, e0=0.12, theta=0.04, conf=0.92, n_bands=6,
            disconnected=False,
        )

        first = gate.step(cruise_fit, features, 0.0, 0.0, 0.0, 0.0, 0.0)
        second = gate.step(cruise_fit, features, 0.0, 0.0, 0.1, 0.0, 0.0)
        recovered = gate.step(cruise_fit, features, 0.0, 0.0, 0.2, 0.0, 0.0)

        self.assertEqual(first.status.reason, "corner_failed_recovery_confirm")
        self.assertEqual(second.status.reason, "corner_failed_recovery_confirm")
        self.assertEqual(first.v, 0.0)
        self.assertEqual(recovered.status.reason, "corner_visual_takeover")
        self.assertEqual(gate.state, "cooldown")

    def test_unreliable_tracker_search_cannot_override_corner_approach(self):
        gate = CornerCommandDelay(PathMemoryConfig())
        gate.state = "approach"
        gate.candidate_dir = 1
        gate.hold_v = 0.05
        gate.stable_w = 0.025
        unreliable = TrajectoryFit(
            found=True, e0=0.9, theta=-1.0, conf=0.30, n_bands=2,
        )
        geometry = CaptureGeometryObservation(
            kind="corner", direction=1, vertex_y_frac=0.40,
        )
        output = gate.step(
            unreliable, LineFeatures(found=True), 0.0, -0.16,
            0.0, 0.0, 0.0, geometry=geometry,
        )
        self.assertEqual(output.status.reason, "approaching_corner")
        self.assertAlmostEqual(output.w, 0.025)

    def test_provisional_capture_loss_resumes_bounded_same_direction_search(self):
        cfg = PathMemoryConfig(corner_align_missing_frames=1)
        gate = CornerCommandDelay(cfg)
        gate.state = "captured"
        gate.candidate_dir = 1
        gate.turned_rad = 0.95
        gate.target_angle_rad = 1.57
        gate.exit_latch.latched = True
        missing = TrajectoryFit(found=False)
        features = LineFeatures(found=False)

        held = gate.step(missing, features, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(held.status.reason, "corner_exit_captured")
        searched = gate.step(missing, features, 0.0, 0.0, 0.1, 0.0, 0.0)
        self.assertEqual(searched.status.reason, "corner_reacquire_search")
        self.assertLess(searched.w, 0.0)
        self.assertAlmostEqual(gate.turned_rad, 0.95)
        self.assertFalse(gate.exit_latch.latched)

        # The real exit can now enter and be confirmed; the discarded sliver
        # did not reverse the turn or reset its yaw safety budget.
        exit_fit = TrajectoryFit(
            found=True, e0=0.70, theta=0.30, conf=0.75, n_bands=5,
        )
        captured = gate.step(
            exit_fit, LineFeatures(found=True), 0.0, 0.0,
            0.2, 0.0, -0.10,
        )
        takeover = gate.step(
            exit_fit, LineFeatures(found=True), 0.04, 0.02,
            0.3, 0.0, 0.0,
        )
        self.assertEqual(captured.status.reason, "corner_exit_captured")
        self.assertEqual(takeover.status.reason, "corner_visual_takeover")

    def test_disconnected_exit_cannot_seed_cruise_handoff(self):
        gate = CornerCommandDelay(PathMemoryConfig())
        gate.state = "turning"
        gate.candidate_dir = 1
        gate.turned_rad = 0.80
        gate.target_angle_rad = 1.57
        fragment = TrajectoryFit(
            found=True, e0=0.70, theta=0.30, conf=0.90, n_bands=5,
            disconnected=True,
        )
        features = LineFeatures(found=True)

        captured = gate.step(fragment, features, 0.0, 0.0, 0.0, 0.0, 0.0)
        held = gate.step(fragment, features, 0.0, 0.0, 0.1, 0.0, 0.0)
        self.assertEqual(captured.status.reason, "corner_exit_captured")
        self.assertEqual(held.status.reason, "corner_exit_captured")
        self.assertEqual(gate.state, "captured")

        complete = replace(fragment, disconnected=False)
        takeover = gate.step(complete, features, 0.04, -0.02, 0.2, 0.0, 0.0)
        self.assertEqual(takeover.status.reason, "corner_visual_takeover")
        self.assertEqual(gate.state, "cooldown")

    def test_capture_coast_past_safety_envelope_fails_without_extra_turn(self):
        cfg = PathMemoryConfig(corner_align_missing_frames=0)
        gate = CornerCommandDelay(cfg)
        gate.state = "captured"
        gate.candidate_dir = 1
        gate.target_angle_rad = 1.57
        gate.turned_rad = 1.96
        gate.exit_latch.latched = True
        gate.last_now = 0.0

        output = gate.step(
            TrajectoryFit(found=False), LineFeatures(found=False),
            0.0, 0.0, 0.1, 0.0, angular=-0.30,
        )
        self.assertEqual(output.status.reason, "corner_reacquire_failed")
        self.assertEqual(gate.state, "failed")
        self.assertAlmostEqual(output.w, 0.0)

    def test_seeking_checks_hard_limit_before_latching_same_side_line(self):
        gate = CornerCommandDelay(PathMemoryConfig())
        gate.state = "seeking"
        gate.candidate_dir = 1
        gate.target_angle_rad = 1.57
        gate.turned_rad = 1.90
        gate.last_now = 0.0
        late_line = TrajectoryFit(
            found=True, e0=0.50, theta=0.30, conf=0.90, n_bands=5,
        )

        output = gate.step(
            late_line, LineFeatures(found=True), 0.0, 0.0,
            1.0, 0.0, angular=-0.20,
        )
        self.assertEqual(output.status.reason, "corner_reacquire_failed")
        self.assertEqual(gate.state, "failed")
        self.assertFalse(gate.exit_latch.latched)

    def test_completed_event_clears_cached_forward_command_epoch(self):
        gate = CornerCommandDelay(PathMemoryConfig())
        gate.state = "cooldown"
        gate.event_shape = "corner"
        gate.candidate_dir = 1
        gate.stable_v, gate.stable_w = 0.06, -0.05
        cleared = TrajectoryFit(
            found=True, e0=0.10, theta=0.05, conf=0.90,
            n_bands=5, disconnected=False,
        )
        features = LineFeatures(found=True)
        gate.step(cleared, features, 0.05, 0.0, 0.0, 0.05, 0.0)
        gate.step(cleared, features, 0.05, 0.0, 0.1, 0.05, 0.0)

        self.assertEqual(gate.state, "armed")
        self.assertAlmostEqual(gate.stable_v, 0.0)
        decision = CaptureGeometryDecision(
            kind="turn", direction=1, angle_rad=math.pi / 2,
            vertex_y_frac=0.4, incoming_e=0.0, incoming_theta=0.0,
            votes=3,
        )
        self.assertFalse(gate.will_accept_geometry(decision))

    def test_persistent_below_gate_geometry_has_bounded_approach(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.10,
            corner_gate_y_frac=0.80,
        )
        gate = CornerCommandDelay(cfg)
        gate.stable_v = 0.05
        observation = CaptureGeometryObservation(
            kind="corner", direction=1, vertex_y_frac=0.40,
            incoming_e=0.0, incoming_theta=0.0, confidence=0.90,
        )
        decision = CaptureGeometryDecision(
            kind="turn", direction=1, angle_rad=math.pi / 2,
            vertex_y_frac=0.40, incoming_e=0.0, incoming_theta=0.0,
            votes=3,
        )
        features = LineFeatures(found=True)
        gate.step(
            turn_fit(), features, 0.05, 0.0, 0.0, 0.0,
            geometry=observation, geometry_decision=decision,
        )
        for now in (1.0, 2.0, 3.0):
            output = gate.step(
                turn_fit(), features, 0.05, 0.0, now, 0.10,
                geometry=observation,
            )
        self.assertEqual(gate.state, "armed")
        self.assertEqual(output.status.reason, "armed")

    def test_visual_fallback_never_starts_zero_speed_margin(self):
        cfg = PathMemoryConfig(
            capture_geometry_enabled=False,
            camera_to_axle_m=0.10,
            corner_confirm_frames=3,
        )
        gate = CornerCommandDelay(cfg)
        features = LineFeatures(found=True)
        for index in range(10):
            output = gate.step(
                turn_fit(), features, 0.0, -0.34,
                index * 0.1, 0.0, -0.34,
            )
        self.assertEqual(gate.state, "armed")
        self.assertNotEqual(output.status.reason, "waiting_margin")

    def test_first_entry_latch_captures_first_coherent_side_line(self):
        latch = FirstEntryLineLatch()
        samples = (
            TrajectoryFit(found=True, e0=1.0, theta=0.451, conf=0.364, n_bands=4, disconnected=True),
            TrajectoryFit(found=True, e0=0.897, theta=0.452, conf=0.349, n_bands=4, disconnected=True),
            TrajectoryFit(found=True, e0=0.801, theta=0.458, conf=0.351, n_bands=4, disconnected=True),
        )
        self.assertTrue(latch.try_latch(samples[0], 1))
        self.assertFalse(latch.try_latch(samples[1], 1))
        self.assertFalse(latch.try_latch(samples[2], 1))
        self.assertTrue(latch.latched)

        # A later border/return line cannot replace the first identity.
        return_line = TrajectoryFit(
            found=True, e0=1.0, theta=-0.266, conf=0.95, n_bands=6,
        )
        self.assertFalse(latch.try_latch(return_line, 1))
        self.assertFalse(latch.observe_latched(return_line))

    def test_first_entry_identity_rejects_opposite_heading_replacement(self):
        latch = FirstEntryLineLatch()
        first = TrajectoryFit(
            found=True, e0=0.55, theta=0.85, conf=0.90, n_bands=5,
        )
        replacement = replace(first, e0=0.50, theta=-0.85)
        self.assertTrue(latch.try_latch(first, 1))
        self.assertFalse(latch.observe_latched(replacement))

    def test_first_entry_latch_is_left_right_symmetric(self):
        right = FirstEntryLineLatch()
        left = FirstEntryLineLatch()
        right_fit = TrajectoryFit(
            found=True, e0=0.6, theta=0.4, conf=0.4, n_bands=5,
            disconnected=True,
        )
        left_fit = replace(right_fit, e0=-0.6, theta=-0.4)
        self.assertTrue(right.try_latch(right_fit, 1))
        self.assertTrue(left.try_latch(left_fit, -1))

    def test_coarse_turn_is_left_right_symmetric_and_honors_inversion(self):
        missing = TrajectoryFit(found=False)
        features = LineFeatures(found=False)
        cases = (
            (1, False, -0.2),
            (-1, False, 0.2),
            (1, True, 0.2),
            (-1, True, -0.2),
        )
        for direction, inverted, expected_w in cases:
            with self.subTest(direction=direction, inverted=inverted):
                gate = CornerCommandDelay(PathMemoryConfig())
                gate.state = "turning"
                gate.candidate_dir = direction
                gate.target_angle_rad = 2.0
                first = gate.step(
                    missing, features, 0.0, 0.0, 0.0, 0.0, 0.0,
                    invert_turn=inverted,
                )
                self.assertAlmostEqual(first.w, expected_w)
                gate.step(
                    missing, features, 0.0, 0.0, 1.0, 0.0, expected_w,
                    invert_turn=inverted,
                )
                self.assertAlmostEqual(gate.turned_rad, 0.2)


if __name__ == "__main__": unittest.main()
