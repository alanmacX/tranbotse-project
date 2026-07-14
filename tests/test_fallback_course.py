import cv2 as cv
import numpy as np
import pytest

from transbot_race.control import StopCause
from transbot_race.fallback_config import FallbackConfig, validate_fallback_config
from transbot_race.fallback_mission import FallbackMission, FallbackState
from transbot_race.fan import DryRunFan, UnavailableFan
from transbot_race.parking import ParkingObservation, ParkingTriggerDetector, TimedParkingController
from transbot_race.vision import TrajectoryFit


def good_fit() -> TrajectoryFit:
    return TrajectoryFit(found=True, conf=0.9, n_bands=6)


def no_parking() -> ParkingObservation:
    return ParkingObservation(False, False, 0.0, "absent")


def confirmed_parking() -> ParkingObservation:
    return ParkingObservation(True, True, 0.9, "confirmed")


def parking_mask(cfg: FallbackConfig) -> np.ndarray:
    mask = np.zeros((160, 200), dtype=np.uint8)
    x0f, y0f, x1f, y1f = cfg.parking_trigger.roi
    x0, x1 = int(round(x0f * 200)), int(round(x1f * 200))
    y0, y1 = int(round(y0f * 160)), int(round(y1f * 160))
    thickness = max(4, int(min(x1 - x0, y1 - y0) * cfg.parking_trigger.border_frac))
    cv.rectangle(mask, (x0, y0), (x1 - 1, y1 - 1), 255, thickness)
    return mask


def advance_to_follow(mission: FallbackMission, now: float = 0.0) -> float:
    for _ in range(mission.cfg.runtime.startup_line_frames):
        mission.step(fit=good_fit(), parking=no_parking(), now=now)
        now += 0.1
    assert mission.state == FallbackState.FOLLOW_LINE
    return now


def test_parking_trigger_requires_consecutive_frames():
    cfg = FallbackConfig()
    detector = ParkingTriggerDetector(cfg.parking_trigger)
    mask = parking_mask(cfg)

    for _ in range(cfg.parking_trigger.confirm_frames - 1):
        observation = detector.analyze(mask)
        assert observation.candidate
        assert not observation.confirmed

    detector.analyze(np.zeros_like(mask))
    for _ in range(cfg.parking_trigger.confirm_frames - 1):
        assert not detector.analyze(mask).confirmed
    assert detector.analyze(mask).confirmed


def test_unlocked_parking_stops_and_never_reverses():
    cfg = FallbackConfig()
    mission = FallbackMission(cfg, allow_unvalidated_parking=False)
    now = advance_to_follow(mission)

    trigger = mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    assert trigger.state == FallbackState.PARK_TRIGGER
    assert trigger.command.v == trigger.command.w == 0.0

    locked = mission.step(fit=good_fit(), parking=confirmed_parking(), now=now + 0.1)
    assert locked.state == FallbackState.PARK_TRIGGER
    assert locked.command.v == locked.command.w == 0.0
    assert locked.event == "parking_motion_requires_explicit_unlock"


def test_timed_parking_reverse_is_bounded_and_nonpositive():
    cfg = FallbackConfig()
    parking = TimedParkingController(cfg.parking)

    command, complete, fault = parking.reverse_command(0.1)
    assert command.v < 0.0
    assert abs(command.v) <= cfg.runtime.max_v_mps
    assert not complete
    assert fault is None

    command, complete, fault = parking.reverse_command(cfg.parking.reverse_sec)
    assert command.v == command.w == 0.0
    assert complete
    assert fault is None

    command, complete, fault = parking.reverse_command(cfg.parking.reverse_hard_max_sec)
    assert command.v == command.w == 0.0
    assert not complete
    assert fault == "parking_reverse_hard_timeout"


def test_unlocked_sequence_keeps_chassis_still_during_fan():
    cfg = FallbackConfig()
    cfg.parking.stage_sec = 0.1
    cfg.parking.reverse_sec = 0.2
    cfg.parking.reverse_hard_max_sec = 0.4
    cfg.parking.verify_hold_sec = 0.1
    cfg.fan.run_sec = 0.2
    cfg.fan.hard_max_sec = 0.4
    mission = FallbackMission(cfg, allow_unvalidated_parking=True)
    now = advance_to_follow(mission)

    mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    now += 0.1
    mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    assert mission.state == FallbackState.PARK_STAGE

    now += 0.1
    mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    assert mission.state == FallbackState.PARK_REVERSE

    now += 0.1
    reverse = mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    assert reverse.command.v < 0.0
    assert not reverse.fan_requested

    now += 0.2
    mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    assert mission.state == FallbackState.PARK_VERIFY

    now += 0.1
    fan_step = mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    assert mission.state == FallbackState.FAN_RUN
    assert fan_step.fan_requested
    assert fan_step.command.v == fan_step.command.w == 0.0

    now += 0.1
    fan_step = mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    assert fan_step.fan_requested
    assert fan_step.command.v == fan_step.command.w == 0.0

    now += 0.2
    finished = mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    assert finished.state == FallbackState.FINISHED
    assert not finished.fan_requested
    assert finished.stop_cause == StopCause.MISSION_FINISHED


def test_reverse_geometry_loss_latches_fault():
    cfg = FallbackConfig()
    cfg.parking.stage_sec = 0.0
    cfg.parking.geometry_missing_frames = 1
    mission = FallbackMission(cfg, allow_unvalidated_parking=True)
    now = advance_to_follow(mission)

    mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    now += 0.1
    mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    now += 0.1
    mission.step(fit=good_fit(), parking=confirmed_parking(), now=now)
    assert mission.state == FallbackState.PARK_REVERSE

    now += 0.1
    first_missing = mission.step(fit=good_fit(), parking=no_parking(), now=now)
    assert mission.state == FallbackState.PARK_REVERSE
    assert first_missing.command.v < 0.0

    now += 0.1
    lost = mission.step(fit=good_fit(), parking=no_parking(), now=now)
    assert mission.state == FallbackState.FAULT
    assert lost.command.v == lost.command.w == 0.0
    assert lost.stop_cause == StopCause.EXECUTOR_FAILED
    assert not lost.fan_requested


def test_camera_failure_latches_fault_and_stops():
    mission = FallbackMission(FallbackConfig(), allow_unvalidated_parking=True)
    step = mission.step(
        fit=TrajectoryFit(found=False),
        parking=no_parking(),
        now=0.0,
        camera_ok=False,
    )
    assert mission.state == FallbackState.FAULT
    assert step.command.v == step.command.w == 0.0
    assert step.stop_cause == StopCause.CAMERA_FAILURE
    assert not step.fan_requested

    repeated = mission.step(fit=good_fit(), parking=no_parking(), now=0.1)
    assert repeated.stop_cause == StopCause.CAMERA_FAILURE
    assert repeated.command.v == repeated.command.w == 0.0


def test_fan_fakes_fail_closed():
    fan = DryRunFan()
    fan.on()
    assert fan.is_on
    fan.close()
    assert not fan.is_on
    assert fan.events == ["on", "off"]

    unavailable = UnavailableFan()
    with pytest.raises(RuntimeError, match="adapter unavailable"):
        unavailable.on()
    assert not unavailable.is_on


def test_config_rejects_unbounded_reverse_duration():
    cfg = FallbackConfig()
    cfg.parking.reverse_sec = cfg.parking.reverse_hard_max_sec + 0.1
    with pytest.raises(ValueError, match="hard maximum"):
        validate_fallback_config(cfg)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cfg: setattr(cfg.camera, "crop", (0, 0, 900, 100)), "within configured frame"),
        (lambda cfg: setattr(cfg.camera, "expand_left_px", -1), "expansion cannot be negative"),
        (lambda cfg: setattr(cfg.parking_trigger, "edge_occupancy_min", 1.1), "within \\[0, 1\\]"),
        (lambda cfg: setattr(cfg.runtime, "loop_period_sec", float("nan")), "must be finite"),
    ],
)
def test_config_rejects_invalid_boundaries(mutate, message):
    cfg = FallbackConfig()
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        validate_fallback_config(cfg)
