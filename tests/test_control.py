import math

from transbot_race.control import (
    CommandArbiter,
    ControlOwner,
    SafetyState,
    StopCause,
)
from transbot_race.state_machine import MotionCommand, RaceState


def command(v=0.06, w=0.1):
    return MotionCommand(v, w, "candidate", RaceState.TRACK)


def test_safety_stop_preserves_owner_and_candidate():
    arbiter = CommandArbiter(max_v=0.1, max_w=0.3)
    result = arbiter.resolve(
        command(), owner=ControlOwner.RING_EXECUTOR, stop_cause=StopCause.OBSTACLE,
    )
    assert result.owner == ControlOwner.RING_EXECUTOR
    assert result.candidate.v == 0.06
    assert result.final.v == result.final.w == 0.0
    assert result.safety_state == SafetyState.STOP_LATCHED
    assert result.stop_cause == StopCause.OBSTACLE


def test_slow_override_does_not_destroy_candidate_reason():
    arbiter = CommandArbiter(max_v=0.1, max_w=0.3)
    result = arbiter.resolve(
        command(), owner=ControlOwner.CRUISE, slow_v_limit=0.02,
    )
    assert result.candidate.reason == "candidate"
    assert result.final.reason == "candidate"
    assert result.final.v == 0.02
    assert result.safety_state == SafetyState.SLOW


def test_owner_epoch_changes_only_when_owner_changes():
    arbiter = CommandArbiter(max_v=0.1, max_w=0.3)
    first = arbiter.resolve(command(), owner=ControlOwner.CRUISE)
    second = arbiter.resolve(command(), owner=ControlOwner.CRUISE)
    third = arbiter.resolve(command(), owner=ControlOwner.CORNER_EXECUTOR)
    assert first.owner_epoch == second.owner_epoch
    assert third.owner_epoch == second.owner_epoch + 1


def test_invalid_command_is_stopped():
    arbiter = CommandArbiter(max_v=0.1, max_w=0.3)
    result = arbiter.resolve(
        command(v=math.nan), owner=ControlOwner.CRUISE,
    )
    assert result.stop_cause == StopCause.INVALID_COMMAND
    assert result.final.v == result.final.w == 0.0


def test_out_of_bounds_command_is_stopped():
    arbiter = CommandArbiter(max_v=0.1, max_w=0.3)
    result = arbiter.resolve(command(w=0.31), owner=ControlOwner.CRUISE)
    assert result.stop_cause == StopCause.INVALID_COMMAND


def test_hard_stop_stays_latched_until_explicit_clear():
    arbiter = CommandArbiter(max_v=0.1, max_w=0.3)
    arbiter.resolve(
        command(), owner=ControlOwner.RING_EXECUTOR,
        stop_cause=StopCause.ROUTE_LOST,
    )
    still_stopped = arbiter.resolve(command(), owner=ControlOwner.RING_EXECUTOR)
    assert still_stopped.stop_cause == StopCause.ROUTE_LOST
    assert still_stopped.final.v == 0.0
    arbiter.clear_stop(StopCause.ROUTE_LOST)
    clear = arbiter.resolve(command(), owner=ControlOwner.RING_EXECUTOR)
    assert clear.safety_state == SafetyState.CLEAR


def test_obstacle_stop_auto_clears_when_debounced_request_clears():
    arbiter = CommandArbiter(max_v=0.1, max_w=0.3)
    arbiter.resolve(
        command(), owner=ControlOwner.CRUISE, stop_cause=StopCause.OBSTACLE,
    )
    clear = arbiter.resolve(command(), owner=ControlOwner.CRUISE)
    assert clear.safety_state == SafetyState.CLEAR
