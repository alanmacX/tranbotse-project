import cv2 as cv
import numpy as np
import pytest

from transbot_race.control import StopCause
from transbot_race.fallback_config import FallbackConfig, validate_fallback_config
from transbot_race.fallback_mission import FallbackMission, FallbackState
from transbot_race.fan import DryRunFan, UnavailableFan
from transbot_race.parking import (
    LeftTurnController,
    ParkingBayDetector,
    ParkingObservation,
    TerminalLineDetector,
    TerminalLineObservation,
    TimedParkingController,
)
from transbot_race.vision import TrajectoryFit, scan_line_features


def good_fit() -> TrajectoryFit:
    return TrajectoryFit(found=True, conf=0.9, n_bands=6)


def no_terminal() -> TerminalLineObservation:
    return TerminalLineObservation(False, False, 0.0, "absent")


def confirmed_terminal() -> TerminalLineObservation:
    return TerminalLineObservation(True, True, 0.9, "confirmed")


def no_parking() -> ParkingObservation:
    return ParkingObservation(False, False, 0.0, "absent")


def confirmed_parking() -> ParkingObservation:
    return ParkingObservation(True, True, 0.9, "confirmed")


def candidate_parking() -> ParkingObservation:
    return ParkingObservation(True, False, 0.6, "candidate")


def terminal_mask() -> np.ndarray:
    mask = np.zeros((180, 240), dtype=np.uint8)
    cv.line(mask, (120, 179), (120, 92), 255, 12)
    cv.line(mask, (45, 92), (195, 92), 255, 12)
    return mask


def u_bend_mask() -> np.ndarray:
    mask = np.zeros((180, 240), dtype=np.uint8)
    points = np.array([[120, 179], [120, 145], [105, 115], [78, 98], [52, 112]], np.int32)
    cv.polylines(mask, [points], False, 255, 12)
    return mask


def rectangle_mask() -> np.ndarray:
    mask = np.zeros((180, 240), dtype=np.uint8)
    cv.rectangle(mask, (45, 45), (195, 125), 255, 12)
    cv.line(mask, (120, 179), (120, 125), 255, 12)
    return mask


def disconnected_terminal_mask() -> np.ndarray:
    mask = np.zeros((180, 240), dtype=np.uint8)
    cv.line(mask, (120, 179), (120, 112), 255, 12)
    cv.line(mask, (45, 92), (195, 92), 255, 12)
    return mask


def features(mask: np.ndarray, cfg: FallbackConfig):
    return scan_line_features(mask, cfg.vision, crop_center=mask.shape[1] / 2)


def parking_mask(cfg: FallbackConfig) -> np.ndarray:
    mask = np.zeros((160, 200), dtype=np.uint8)
    x0f, y0f, x1f, y1f = cfg.parking_bay.roi
    x0, x1 = int(round(x0f * 200)), int(round(x1f * 200))
    y0, y1 = int(round(y0f * 160)), int(round(y1f * 160))
    thickness = max(4, int(min(x1 - x0, y1 - y0) * cfg.parking_bay.border_frac))
    cv.rectangle(mask, (x0, y0), (x1 - 1, y1 - 1), 255, thickness)
    return mask


def advance_to_follow(mission: FallbackMission, now: float = 0.0) -> float:
    for _ in range(mission.cfg.runtime.startup_line_frames):
        mission.step(fit=good_fit(), terminal=no_terminal(), parking=no_parking(), now=now)
        now += 0.1
    assert mission.state == FallbackState.FOLLOW_LINE
    return now


def test_terminal_line_requires_consecutive_frames_and_rejects_course_shapes():
    cfg = FallbackConfig()
    detector = TerminalLineDetector(cfg.terminal_line)
    mask = terminal_mask()
    line_features = features(mask, cfg)
    for _ in range(cfg.terminal_line.confirm_frames - 1):
        observation = detector.analyze(mask, line_features)
        assert observation.candidate
        assert not observation.confirmed
    assert detector.analyze(mask, line_features).confirmed

    for negative in (u_bend_mask(), rectangle_mask(), disconnected_terminal_mask()):
        detector.reset()
        observation = detector.analyze(negative, features(negative, cfg))
        assert not observation.candidate
        assert not observation.confirmed


def test_parking_bay_requires_consecutive_frames():
    cfg = FallbackConfig()
    detector = ParkingBayDetector(cfg.parking_bay)
    mask = parking_mask(cfg)
    for _ in range(cfg.parking_bay.confirm_frames - 1):
        assert not detector.analyze(mask).confirmed
    assert detector.analyze(mask).confirmed


def test_bay_confirmation_requires_confirmed_observation_and_times_out():
    cfg = FallbackConfig()
    mission = FallbackMission(cfg, allow_unvalidated_parking=True)
    mission.state = FallbackState.BAY_CONFIRM
    mission.state_started_at = 0.0

    waiting = mission.step(
        fit=good_fit(),
        terminal=no_terminal(),
        parking=candidate_parking(),
        now=0.1,
    )
    assert waiting.state == FallbackState.BAY_CONFIRM
    assert waiting.command.v == waiting.command.w == 0.0

    failed = mission.step(
        fit=good_fit(),
        terminal=no_terminal(),
        parking=candidate_parking(),
        now=cfg.parking_bay.confirm_timeout_sec,
    )
    assert failed.state == FallbackState.FAULT
    assert failed.event == "parking_bay_not_confirmed"
    assert failed.command.v == failed.command.w == 0.0


def test_bay_rectangle_does_not_end_line_following_and_locked_terminal_stops():
    mission = FallbackMission(FallbackConfig(), allow_unvalidated_parking=False)
    now = advance_to_follow(mission)
    following = mission.step(fit=good_fit(), terminal=no_terminal(), parking=confirmed_parking(), now=now)
    assert following.state == FallbackState.FOLLOW_LINE

    stopped = mission.step(fit=good_fit(), terminal=confirmed_terminal(), parking=no_parking(), now=now + 0.1)
    assert stopped.state == FallbackState.TERMINAL_STOP
    assert stopped.command.v == stopped.command.w == 0.0
    locked = mission.step(fit=good_fit(), terminal=no_terminal(), parking=no_parking(), now=now + 0.2)
    assert locked.state == FallbackState.TERMINAL_STOP
    assert locked.command.v == locked.command.w == 0.0


def test_left_turn_is_positive_bounded_and_stops_on_completion():
    cfg = FallbackConfig()
    cfg.left_turn.target_yaw_rad = 0.02
    turn = LeftTurnController(cfg.left_turn)
    turn.reset(0.0)
    active = turn.step(0.1)
    assert active.command.v == 0.0
    assert active.command.w > 0.0
    assert not active.complete
    complete = turn.step(0.125)
    assert complete.complete
    assert complete.command.v == complete.command.w == 0.0


def test_left_turn_timeout_faults_stopped():
    cfg = FallbackConfig()
    cfg.left_turn.hard_timeout_sec = 0.1
    turn = LeftTurnController(cfg.left_turn)
    turn.reset(0.0)
    result = turn.step(0.1)
    assert result.fault == "left_turn_hard_timeout"
    assert result.command.v == result.command.w == 0.0


def test_unlocked_sequence_stops_turns_left_stops_reverses_and_runs_fan():
    cfg = FallbackConfig()
    cfg.left_turn.stop_hold_sec = 0.1
    cfg.left_turn.target_yaw_rad = 0.02
    cfg.parking.reverse_sec = 0.2
    cfg.parking.reverse_hard_max_sec = 0.4
    cfg.parking.verify_hold_sec = 0.1
    cfg.fan.run_sec = 0.2
    cfg.fan.hard_max_sec = 0.4
    mission = FallbackMission(cfg, allow_unvalidated_parking=True)
    now = advance_to_follow(mission)

    trigger = mission.step(fit=good_fit(), terminal=confirmed_terminal(), parking=no_parking(), now=now)
    assert trigger.state == FallbackState.TERMINAL_STOP
    assert trigger.command.v == trigger.command.w == 0.0

    now += 0.1
    turn_boundary = mission.step(fit=good_fit(), terminal=no_terminal(), parking=no_parking(), now=now)
    assert turn_boundary.state == FallbackState.LEFT_TURN
    assert turn_boundary.command.v == turn_boundary.command.w == 0.0

    now += 0.1
    turning = mission.step(fit=good_fit(), terminal=no_terminal(), parking=no_parking(), now=now)
    assert turning.command.v == 0.0 and turning.command.w > 0.0

    now += 0.1
    bay_boundary = mission.step(fit=good_fit(), terminal=no_terminal(), parking=confirmed_parking(), now=now)
    assert bay_boundary.state == FallbackState.BAY_CONFIRM
    assert bay_boundary.command.v == bay_boundary.command.w == 0.0

    now += 0.1
    reverse_boundary = mission.step(fit=good_fit(), terminal=no_terminal(), parking=confirmed_parking(), now=now)
    assert reverse_boundary.state == FallbackState.PARK_REVERSE
    assert reverse_boundary.command.v == reverse_boundary.command.w == 0.0

    now += 0.1
    reverse = mission.step(fit=good_fit(), terminal=no_terminal(), parking=confirmed_parking(), now=now)
    assert reverse.command.v < 0.0
    assert reverse.command.w == 0.0

    now += 0.2
    mission.step(fit=good_fit(), terminal=no_terminal(), parking=confirmed_parking(), now=now)
    assert mission.state == FallbackState.PARK_VERIFY
    now += 0.1
    fan_step = mission.step(fit=good_fit(), terminal=no_terminal(), parking=confirmed_parking(), now=now)
    assert fan_step.state == FallbackState.FAN_RUN
    assert fan_step.fan_requested
    assert fan_step.command.v == fan_step.command.w == 0.0
    now += 0.2
    finished = mission.step(fit=good_fit(), terminal=no_terminal(), parking=confirmed_parking(), now=now)
    assert finished.state == FallbackState.FINISHED
    assert finished.stop_cause == StopCause.MISSION_FINISHED


def test_reverse_geometry_loss_latches_fault():
    cfg = FallbackConfig()
    cfg.left_turn.stop_hold_sec = 0.0
    cfg.left_turn.target_yaw_rad = 0.016
    cfg.parking.geometry_missing_frames = 1
    mission = FallbackMission(cfg, allow_unvalidated_parking=True)
    now = advance_to_follow(mission)
    mission.step(fit=good_fit(), terminal=confirmed_terminal(), parking=no_parking(), now=now)
    mission.step(fit=good_fit(), terminal=no_terminal(), parking=no_parking(), now=now + 0.1)
    mission.step(fit=good_fit(), terminal=no_terminal(), parking=no_parking(), now=now + 0.2)
    mission.step(fit=good_fit(), terminal=no_terminal(), parking=confirmed_parking(), now=now + 0.3)
    assert mission.state == FallbackState.PARK_REVERSE
    mission.step(fit=good_fit(), terminal=no_terminal(), parking=no_parking(), now=now + 0.4)
    lost = mission.step(fit=good_fit(), terminal=no_terminal(), parking=no_parking(), now=now + 0.5)
    assert lost.state == FallbackState.FAULT
    assert lost.command.v == lost.command.w == 0.0


def test_timed_parking_reverse_is_bounded_and_negative():
    cfg = FallbackConfig()
    parking = TimedParkingController(cfg.parking)
    command, complete, fault = parking.reverse_command(0.1)
    assert command.v < 0.0 and not complete and fault is None
    command, complete, fault = parking.reverse_command(cfg.parking.reverse_sec)
    assert command.v == command.w == 0.0 and complete and fault is None
    command, complete, fault = parking.reverse_command(cfg.parking.reverse_hard_max_sec)
    assert command.v == command.w == 0.0 and not complete
    assert fault == "parking_reverse_hard_timeout"


def test_camera_failure_preserves_first_fault():
    mission = FallbackMission(FallbackConfig(), allow_unvalidated_parking=True)
    failed = mission.step(
        fit=TrajectoryFit(found=False),
        terminal=no_terminal(),
        parking=no_parking(),
        now=0.0,
        camera_ok=False,
    )
    assert failed.stop_cause == StopCause.CAMERA_FAILURE
    repeated = mission.step(fit=good_fit(), terminal=no_terminal(), parking=no_parking(), now=0.1)
    assert repeated.stop_cause == StopCause.CAMERA_FAILURE


def test_fan_fakes_fail_closed():
    fan = DryRunFan()
    fan.on()
    fan.close()
    assert fan.events == ["on", "off"]
    unavailable = UnavailableFan()
    with pytest.raises(RuntimeError, match="adapter unavailable"):
        unavailable.on()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cfg: setattr(cfg.camera, "crop", (0, 0, 900, 100)), "within configured frame"),
        (lambda cfg: setattr(cfg.terminal_line, "min_width_ratio", 1.1), "within \\(0, 1\\]"),
        (lambda cfg: setattr(cfg.parking_bay, "edge_occupancy_min", 1.1), "within \\[0, 1\\]"),
        (lambda cfg: setattr(cfg.left_turn, "target_yaw_rad", cfg.left_turn.w_radps * cfg.left_turn.hard_timeout_sec), "below its hard timeout"),
        (lambda cfg: setattr(cfg.runtime, "loop_period_sec", float("nan")), "must be finite"),
    ],
)
def test_config_rejects_invalid_boundaries(mutate, message):
    cfg = FallbackConfig()
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        validate_fallback_config(cfg)
