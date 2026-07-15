from dataclasses import replace

from transbot_race.capture_geometry import CaptureGeometryObservation
from transbot_race.ring_entry import RingEntryExecutor, RingPhaseEvent
from transbot_race.vision import TrajectoryFit


ARC = CaptureGeometryObservation(
    kind="curve",
    direction=1,
    angle_rad=0.7,
    confidence=0.9,
    vertex=(360.0, 100.0),
    vertex_y_frac=0.55,
    incoming_e=0.0,
    incoming_theta=0.0,
    endpoints=2,
    component_area=2400,
    is_fork=False,
)


def route(e=0.55, theta=0.30):
    return TrajectoryFit(
        found=True,
        e0=0.1,
        e_look=e,
        theta=theta,
        conf=0.9,
        n_bands=0,
        disconnected=True,
        path_memory=True,
    )


def inside_executor():
    executor = RingEntryExecutor(1)
    executor.state = "inside"
    executor.raw_fit = route()
    executor.last_fit = executor.raw_fit
    executor.last_now = 0.0
    return executor


def test_single_frame_branch_swap_cannot_replace_locked_route():
    executor = inside_executor()
    locked = executor.last_fit
    swapped = route(-0.75, -0.45)

    result = executor.step(
        ARC,
        swapped,
        now=0.1,
        linear=0.0,
        accepted_entry=False,
    )

    assert result.fit == locked
    assert executor.raw_fit == locked
    assert executor.pending_fit == swapped
    assert executor.pending_fit_frames == 1


def test_branch_swap_requires_repeated_matching_candidate_and_is_bounded():
    executor = inside_executor()
    locked = executor.last_fit
    swapped = route(-0.75, -0.45)
    executor.step(ARC, swapped, now=0.1, linear=0.0, accepted_entry=False)
    confirmed = executor.step(
        ARC,
        replace(swapped, e_look=-0.72, theta=-0.43),
        now=0.2,
        linear=0.0,
        accepted_entry=False,
    )

    assert confirmed.fit is not None
    assert confirmed.fit.e_look < locked.e_look
    assert confirmed.fit.e_look > locked.e_look - 0.10
    assert executor.pending_fit is None


def test_route_memory_bridges_only_bounded_gap_then_emits_typed_loss():
    executor = inside_executor()
    for index in range(5):
        bridged = executor.step(
            None,
            None,
            now=0.1 * (index + 1),
            linear=0.0,
            accepted_entry=False,
        )
        assert bridged.fit is not None
        assert bridged.phase_event == RingPhaseEvent.NONE

    lost = executor.step(
        None,
        None,
        now=0.6,
        linear=0.0,
        accepted_entry=False,
    )
    assert lost.fit is None
    assert lost.phase_event == RingPhaseEvent.ROUTE_LOST


def test_operator_clear_discards_route_that_caused_latch():
    executor = inside_executor()
    executor.missing_frames = 6

    assert executor.clear_route_loss()
    assert executor.last_fit is None
    assert executor.raw_fit is None
    assert executor.pending_fit is None
    assert executor.missing_frames == 0
