import unittest

from transbot_race.capture_geometry import (
    CaptureGeometryDecision,
    CaptureGeometryObservation,
    CornerGeometryFilter,
    RingEntryGeometryFilter,
    RingExitGeometryFilter,
)
from transbot_race.config import MissionConfig
from transbot_race.mission import (
    CourseSession,
    DetectorKind,
    ExecutorKind,
    FixedSessionMission,
    MissionEvent,
)
from transbot_race.vision import TrajectoryFit


def decision(
    direction=-1, is_fork=False, angle=1.5, vertex=0.4,
    incoming_theta=0.0, kind="corner",
):
    return CaptureGeometryDecision(
        kind=kind, direction=direction, angle_rad=angle,
        vertex_y_frac=vertex, incoming_e=0.0, incoming_theta=incoming_theta,
        votes=3, is_fork=is_fork,
    )


def observation(kind="corner", direction=-1, is_fork=False, endpoints=2):
    return CaptureGeometryObservation(
        kind=kind, direction=direction, angle_rad=1.5, confidence=0.9,
        vertex_y_frac=0.4, incoming_e=0.0, incoming_theta=0.0,
        is_fork=is_fork, endpoints=endpoints,
    )


GOOD_FIT = TrajectoryFit(found=True, conf=0.95, n_bands=6)


class FixedSessionMissionTests(unittest.TestCase):
    def test_phase_progress_requires_typed_mission_event(self):
        mission = FixedSessionMission(MissionConfig(initial_session="corner"))
        self.assertEqual(
            mission.transition(MissionEvent.PHASE_COMPLETED),
            CourseSession.RING_ENTRY,
        )
        self.assertEqual(mission.last_transition_event, MissionEvent.PHASE_COMPLETED)

    def test_mission_completed_and_reset_are_explicit(self):
        mission = FixedSessionMission(MissionConfig(initial_session="corner"))
        self.assertEqual(
            mission.transition(MissionEvent.MISSION_COMPLETED),
            CourseSession.FINISHED,
        )
        self.assertEqual(
            mission.transition(MissionEvent.MISSION_RESET),
            CourseSession.CORNER,
        )

    def test_only_corner_is_allowed_first(self):
        mission = FixedSessionMission(MissionConfig())
        accepted = mission.gate(decision(-1, False), observation(), GOOD_FIT)
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted.direction, -1)
        self.assertIsNone(mission.gate(
            decision(1, True), observation("curve", 1, True), GOOD_FIT,
        ))

    def test_corner_direction_is_selected_from_visible_geometry(self):
        mission = FixedSessionMission(MissionConfig())
        accepted = mission.gate(decision(1), observation(direction=1), GOOD_FIT)
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted.direction, 1)
        accepted = mission.gate(decision(-1), observation(direction=-1), GOOD_FIT)
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted.direction, -1)

    def test_real_curve_cannot_be_upgraded_to_corner(self):
        mission = FixedSessionMission(MissionConfig())
        accepted = mission.gate(
            decision(-1, angle=1.947, vertex=0.5965, incoming_theta=0.1477),
            observation("curve", -1, False),
            TrajectoryFit(found=True, conf=0.6548, n_bands=4, disconnected=False),
        )
        self.assertIsNone(accepted)
        self.assertEqual(mission.last_gate_reason, "corner_reject_kind_curve")

    def test_partial_curve_cannot_latch_corner_stage(self):
        mission = FixedSessionMission(MissionConfig())
        accepted = mission.gate(
            decision(1, angle=0.488, vertex=0.4421, incoming_theta=0.1384),
            observation("curve", 1, False),
            TrajectoryFit(found=True, conf=0.8415, n_bands=5, disconnected=False),
        )
        self.assertIsNone(accepted)
        self.assertEqual(mission.last_gate_reason, "corner_reject_kind_curve")

    def test_ring_interior_curve_cannot_impersonate_corner(self):
        mission = FixedSessionMission(MissionConfig())
        noisy_fit = TrajectoryFit(
            found=True, conf=0.35, n_bands=5, disconnected=True,
        )
        rejected = mission.gate(
            decision(1, angle=2.46, vertex=0.698, incoming_theta=-1.295),
            observation("curve", 1, False),
            noisy_fit,
        )
        self.assertIsNone(rejected)
        self.assertEqual(mission.last_gate_reason, "corner_reject_kind_curve")

    def test_ring_entry_requires_fork_and_uses_route_direction(self):
        mission = FixedSessionMission(MissionConfig(ring_entry_direction=-1))
        self.assertEqual(mission.advance(), CourseSession.RING_ENTRY)
        self.assertIsNone(mission.gate(
            decision(1, False, kind="ring_entry"), observation("curve", 1, False), GOOD_FIT,
        ))
        accepted = mission.gate(
            decision(1, True, kind="ring_entry"), observation("curve", 1, True), GOOD_FIT,
        )
        accepted = mission.gate(
            decision(1, True, kind="ring_entry"), observation("curve", 1, True), GOOD_FIT,
        )
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted.direction, -1)

    def test_ring_entry_accepts_unambiguous_single_tangent_path(self):
        mission = FixedSessionMission(MissionConfig(ring_entry_direction=1))
        mission.advance()
        accepted = None
        for _ in range(3):
            accepted = mission.gate(
                decision(
                    -1, False, angle=1.07, vertex=0.586,
                    incoming_theta=-0.49, kind="ring_entry",
                ),
                observation("curve", -1, False, endpoints=2),
                GOOD_FIT,
            )
        self.assertIsNotNone(accepted)
        # With only one visible route, follow that route rather than inventing
        # the configured fork direction.
        self.assertEqual(accepted.direction, -1)
        self.assertEqual(mission.last_gate_reason, "ring_entry_single_path_accepted")

    def test_ring_entry_requires_three_takeover_ready_frames(self):
        mission = FixedSessionMission(MissionConfig(ring_entry_direction=1))
        mission.advance()
        candidate = decision(1, True, kind="ring_entry")
        seen = observation("curve", 1, True)

        for _ in range(4):
            self.assertIsNone(mission.gate(
                candidate, seen, GOOD_FIT, entry_takeover_ready=False,
            ))
        self.assertEqual(
            mission.last_gate_reason,
            "ring_entry_reject_takeover_readiness",
        )
        self.assertIsNone(mission.gate(
            candidate, seen, GOOD_FIT, entry_takeover_ready=True,
        ))
        self.assertIsNone(mission.gate(
            candidate, seen, GOOD_FIT, entry_takeover_ready=True,
        ))
        self.assertIsNotNone(mission.gate(
            candidate, seen, GOOD_FIT, entry_takeover_ready=True,
        ))

    def test_ring_takeover_readiness_is_weaker_than_cruise_handoff(self):
        from transbot_race.config import TrackerConfig
        from transbot_race.state_machine import (
            moving_follow_handoff_ready,
            ring_entry_takeover_ready,
        )

        tracker = TrackerConfig()
        off_center = TrajectoryFit(
            found=True, e0=0.75, theta=0.9,
            conf=0.9, n_bands=5, disconnected=False,
        )
        self.assertTrue(ring_entry_takeover_ready(off_center, tracker))
        self.assertFalse(moving_follow_handoff_ready(off_center, tracker))

        mission = FixedSessionMission(MissionConfig(ring_entry_direction=1))
        mission.advance()
        accepted = None
        for _ in range(3):
            accepted = mission.gate(
                decision(1, True, kind="ring_entry"),
                observation("curve", 1, True),
                off_center,
                entry_takeover_ready=ring_entry_takeover_ready(off_center, tracker),
            )
        self.assertIsNotNone(accepted)

    def test_ring_takeover_readiness_must_be_consecutive(self):
        mission = FixedSessionMission(MissionConfig(ring_entry_direction=1))
        mission.advance()
        candidate = decision(1, True, kind="ring_entry")
        seen = observation("curve", 1, True)
        self.assertIsNone(mission.gate(candidate, seen, GOOD_FIT, entry_takeover_ready=True))
        self.assertIsNone(mission.gate(candidate, seen, GOOD_FIT, entry_takeover_ready=True))
        self.assertIsNone(mission.gate(candidate, seen, GOOD_FIT, entry_takeover_ready=False))
        self.assertIsNone(mission.gate(candidate, seen, GOOD_FIT, entry_takeover_ready=True))
        self.assertIsNone(mission.gate(candidate, seen, GOOD_FIT, entry_takeover_ready=True))
        self.assertIsNotNone(mission.gate(candidate, seen, GOOD_FIT, entry_takeover_ready=True))

    def test_ring_exit_is_next_and_uses_dedicated_executor(self):
        mission = FixedSessionMission(MissionConfig())
        mission.advance()
        mission.advance()
        self.assertEqual(mission.session, CourseSession.RING_EXIT)
        self.assertTrue(mission.detector_enabled)
        self.assertEqual(mission.detector_kind, DetectorKind.RING_EXIT)
        self.assertEqual(mission.executor_kind, ExecutorKind.RING_EXIT)
        candidate = decision(1, True, kind="ring_exit")
        seen = observation("curve", 1, True)
        self.assertIsNone(mission.gate(
            candidate, seen, GOOD_FIT, exit_selection_armed=False,
        ))
        self.assertEqual(mission.last_gate_reason, "ring_exit_waiting_arm_distance")
        accepted = mission.gate(
            candidate, seen, GOOD_FIT, exit_selection_armed=True,
        )
        self.assertIsNotNone(accepted)

    def test_real_course_order_is_linear_from_corner(self):
        self.assertEqual(FixedSessionMission.ORDER, (
            CourseSession.CORNER,
            CourseSession.RING_ENTRY,
            CourseSession.RING_EXIT,
            CourseSession.FINISHED,
        ))

    def test_session_map_selects_filter_and_executor_before_detection(self):
        mission = FixedSessionMission(MissionConfig())
        filters = {
            DetectorKind.CORNER: CornerGeometryFilter(3),
            DetectorKind.RING_ENTRY: RingEntryGeometryFilter(3),
            DetectorKind.RING_EXIT: RingExitGeometryFilter(3),
        }
        self.assertEqual(mission.detector_kind, DetectorKind.CORNER)
        self.assertEqual(mission.executor_kind, ExecutorKind.CORNER)

        ring_shape = observation("curve", 1, True, endpoints=3)
        for _ in range(3):
            decision_now = filters[mission.detector_kind].update(ring_shape)
        self.assertIsNone(decision_now)
        self.assertEqual(len(filters[DetectorKind.RING_ENTRY].window), 0)

        filters[DetectorKind.CORNER].reset()
        curve_shape = observation("curve", 1, False, endpoints=2)
        for _ in range(3):
            corner_event = filters[mission.detector_kind].update(curve_shape)
        self.assertIsNone(corner_event)

        corner_shape = observation("corner", 1, False, endpoints=2)
        for _ in range(3):
            corner_event = filters[mission.detector_kind].update(corner_shape)
        self.assertEqual(corner_event.kind, "corner")
        self.assertIsNotNone(mission.gate(corner_event, corner_shape, GOOD_FIT))

        mission.advance()
        self.assertEqual(mission.detector_kind, DetectorKind.RING_ENTRY)
        self.assertEqual(mission.executor_kind, ExecutorKind.RING_ENTRY)
        for _ in range(3):
            ring_event = filters[mission.detector_kind].update(ring_shape)
        self.assertEqual(ring_event.kind, "ring_entry")
        accepted = None
        for _ in range(3):
            accepted = mission.gate(ring_event, ring_shape, GOOD_FIT)
        self.assertIsNotNone(accepted)

        mission.advance()
        self.assertEqual(mission.detector_kind, DetectorKind.RING_EXIT)
        entry_window_size = len(filters[DetectorKind.RING_ENTRY].window)
        exit_event = None
        for _ in range(3):
            exit_event = filters[mission.detector_kind].update(ring_shape)
        self.assertEqual(exit_event.kind, "ring_exit")
        self.assertEqual(len(filters[DetectorKind.RING_ENTRY].window), entry_window_size)
        self.assertIsNotNone(mission.gate(
            exit_event, ring_shape, GOOD_FIT, exit_selection_armed=True,
        ))

    def test_finished_stage_has_no_detector_or_executor(self):
        mission = FixedSessionMission(MissionConfig(initial_session="finished"))
        self.assertEqual(mission.detector_kind, DetectorKind.NONE)
        self.assertEqual(mission.executor_kind, ExecutorKind.NONE)
        mission.advance()
        self.assertEqual(mission.session, CourseSession.FINISHED)
        self.assertEqual(mission.detector_kind, DetectorKind.NONE)
        self.assertEqual(mission.executor_kind, ExecutorKind.NONE)


if __name__ == "__main__":
    unittest.main()
