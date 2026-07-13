import unittest
from dataclasses import replace
import math

from transbot_race.config import PathMemoryConfig
from transbot_race.capture_geometry import (
    CaptureGeometryDecision,
    CaptureGeometryObservation,
)
from transbot_race.path_memory import (
    CornerCommandDelay, read_motion_sample,
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
        self.assertAlmostEqual(turning.v, 0.05 * cfg.corner_turn_speed_ratio)

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
            corner_handoff_blend_frames=3,
            capture_geometry_enabled=False,
        )
        gate = CornerCommandDelay(cfg)
        features = LineFeatures(found=True)
        gate.step(turn_fit(), features, 0.05, -0.04, 0.0, 0.0)
        gate.step(turn_fit(), features, 0.05, -0.04, 0.1, 0.0, 0.0)
        gate.step(turn_fit(), features, 0.05, -0.04, 0.2, 0.0, 0.0)
        turning = gate.step(turn_fit(), features, 0.05, -0.04, 0.3, 0.1, 0.0)
        self.assertEqual(turning.status.reason, "committed_turn")

        unaligned = turn_fit(theta=0.4)
        turning = gate.step(unaligned, features, 0.0, 0.0, 0.8, 0.0, -1.0)
        self.assertEqual(turning.status.reason, "corner_visual_align")
        self.assertAlmostEqual(turning.status.target[0], 0.5)

        aligned = TrajectoryFit(found=True, e0=0.1, theta=0.05, conf=0.9, n_bands=5)
        gate.step(aligned, features, 0.03, -0.02, 1.3, 0.0, -1.0)
        gate.step(aligned, features, 0.03, -0.02, 1.8, 0.0, -1.0)
        handoff = gate.step(aligned, features, 0.03, -0.02, 2.3, 0.0, -1.0)
        self.assertEqual(handoff.status.reason, "corner_visual_handoff")
        self.assertGreater(handoff.w, -cfg.corner_replay_max_w)

        blended = gate.step(aligned, features, 0.03, -0.02, 2.4, 0.0, 0.0)
        self.assertEqual(blended.status.reason, "corner_visual_handoff")
        self.assertGreater(blended.w, -cfg.corner_replay_max_w)
        gate.step(aligned, features, 0.03, -0.02, 2.5, 0.0, 0.0)
        reacquired = gate.step(aligned, features, 0.03, -0.02, 2.6, 0.0, 0.0)
        self.assertEqual(reacquired.status.reason, "cooldown")
        self.assertAlmostEqual(reacquired.v, 0.03)

    def test_corner_event_stays_latched_while_following_a_curve(self):
        cfg = PathMemoryConfig(corner_confirm_frames=3)
        gate = CornerCommandDelay(cfg)
        gate.state = "cooldown"
        gate.candidate_dir = 1
        curve = TrajectoryFit(found=True, e0=0.12, theta=0.3, conf=0.9, n_bands=4)
        features = LineFeatures(found=True)

        for i in range(8):
            output = gate.step(curve, features, 0.04, -0.08, i * 0.1, 0.04, -0.08)
        self.assertEqual(output.status.reason, "cooldown")

        straight = TrajectoryFit(found=True, e0=0.02, theta=0.03, conf=0.9, n_bands=4)
        for i in range(3):
            output = gate.step(straight, features, 0.05, 0.0, 1.0 + i * 0.1, 0.05, 0.0)
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

    def test_capture_geometry_waits_for_distance_and_vertex_gate(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.02,
            corner_gate_y_frac=0.50,
            corner_gate_confirm_frames=2,
        )
        gate = CornerCommandDelay(cfg)
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
        below = CaptureGeometryObservation(kind="corner", direction=1, vertex_y_frac=0.40)
        above = CaptureGeometryObservation(kind="curve", direction=1, vertex_y_frac=0.60)

        captured = gate.step(
            fit, features, 0.10, -0.12, 0.0, 0.0,
            geometry=below, geometry_decision=decision, approach_w=0.01,
        )
        self.assertEqual(captured.status.reason, "approaching_corner")
        self.assertAlmostEqual(captured.w, 0.01)

        gate.step(fit, features, 0.10, -0.12, 0.1, 0.10, geometry=below)
        first_gate = gate.step(fit, features, 0.10, -0.12, 0.2, 0.10, geometry=above)
        self.assertEqual(first_gate.status.reason, "approaching_corner")
        self.assertAlmostEqual(first_gate.status.remaining_m, 0.0)

        waiting = gate.step(fit, features, 0.10, -0.12, 0.3, 0.10, geometry=above)
        self.assertEqual(waiting.status.reason, "waiting_margin")
        turning = gate.step(fit, features, 0.10, -0.12, 0.4, 0.10, geometry=above)
        self.assertEqual(turning.status.reason, "committed_turn")

    def test_capture_geometry_gate_does_not_skip_remaining_distance(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.10,
            corner_gate_y_frac=0.50,
            corner_gate_confirm_frames=1,
        )
        gate = CornerCommandDelay(cfg)
        observation = CaptureGeometryObservation(kind="corner", direction=-1, vertex_y_frac=0.60)
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
        self.assertAlmostEqual(waiting.status.remaining_m, 0.095)

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

    def test_capture_angle_is_calibrated_and_used_as_hard_maximum(self):
        cfg = PathMemoryConfig(
            camera_to_axle_m=0.0,
            corner_capture_angle_scale=0.70,
            corner_reacquire_angle_rad=0.20,
            corner_reacquire_confirm_frames=2,
        )
        gate = CornerCommandDelay(cfg)
        observation = CaptureGeometryObservation(kind="corner", direction=1, vertex_y_frac=0.60)
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
        self.assertAlmostEqual(gate.target_angle_rad, math.radians(45.5), places=3)
        stopped = gate.step(
            missing, LineFeatures(found=False), 0.0, 0.0, 1.1, 0.0,
            angular=-1.0,
        )
        self.assertEqual(stopped.status.reason, "corner_reacquire_failed")
        self.assertAlmostEqual(stopped.w, 0.0)

    def test_visual_alignment_reduces_fixed_turn_before_handoff(self):
        cfg = PathMemoryConfig(
            corner_reacquire_angle_rad=0.20,
            corner_reacquire_confirm_frames=2,
            corner_visual_align_blend=0.70,
        )
        gate = CornerCommandDelay(cfg)
        gate.state = "turning"
        gate.candidate_dir = 1
        gate.hold_v = 0.05
        gate.target_angle_rad = math.radians(45.0)
        gate.turned_rad = 0.20
        trackable = TrajectoryFit(
            found=True, e0=0.20, theta=0.10, conf=0.9, n_bands=4,
        )
        output = gate.step(
            trackable, LineFeatures(found=True), 0.04, 0.03,
            0.0, 0.0, 0.0,
        )
        self.assertEqual(output.status.reason, "corner_visual_align")
        self.assertGreater(output.w, -cfg.corner_replay_max_w)


if __name__ == "__main__": unittest.main()
