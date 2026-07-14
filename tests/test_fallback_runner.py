from argparse import Namespace
from pathlib import Path
import threading

from apps.fallback_runner import CameraReader, apply_step, run_dry
from transbot_race.control import CommandArbiter, ControlOwner
from transbot_race.fallback_config import FallbackConfig, load_fallback_config
from transbot_race.fallback_mission import FallbackState, FallbackStep
from transbot_race.fan import DryRunFan
from transbot_race.motor import MotorGateway
from transbot_race.state_machine import MotionCommand, RaceState


class FakeBot:
    def __init__(self) -> None:
        self.commands = []

    def set_car_motion(self, v, w) -> None:
        self.commands.append((v, w))


class FailingBot(FakeBot):
    def set_car_motion(self, v, w) -> None:
        if v == 0.0 and w == 0.0:
            raise RuntimeError("motor write failed")
        super().set_car_motion(v, w)


class BlockingCapture:
    def __init__(self) -> None:
        self.released = threading.Event()

    def read(self):
        self.released.wait()
        return False, None

    def release(self) -> None:
        self.released.set()


def dry_args(tmp_path: Path, *, unlocked: bool) -> Namespace:
    return Namespace(
        allow_unvalidated_parking=unlocked,
        max_sec=8.0,
        lease_path=tmp_path / "fallback.lock",
    )


def test_fallback_config_loads_without_full_mission_sections():
    root = Path(__file__).resolve().parents[1]
    cfg = load_fallback_config(root / "configs" / "fallback_course.json")
    assert isinstance(cfg, FallbackConfig)
    assert cfg.tracker.v_max == 0.04
    assert not hasattr(cfg, "mission")
    assert not hasattr(cfg, "path_memory")


def test_default_dry_run_stops_before_unvalidated_parking(tmp_path):
    summary = run_dry(dry_args(tmp_path, unlocked=False), FallbackConfig())
    assert summary["state"] == FallbackState.PARK_TRIGGER.value
    assert summary["fan_events"] == []
    assert not summary["unvalidated_parking_unlocked"]


def test_unlocked_dry_run_finishes_and_turns_fan_off(tmp_path):
    cfg = FallbackConfig()
    cfg.parking.stage_sec = 0.05
    cfg.parking.reverse_sec = 0.10
    cfg.parking.reverse_hard_max_sec = 0.20
    cfg.parking.verify_hold_sec = 0.05
    cfg.fan.run_sec = 0.10
    cfg.fan.hard_max_sec = 0.20
    summary = run_dry(dry_args(tmp_path, unlocked=True), cfg)
    assert summary["state"] == FallbackState.FINISHED.value
    assert summary["fan_events"] == ["on", "off"]


def test_dry_run_timeout_latches_fault(tmp_path):
    args = dry_args(tmp_path, unlocked=True)
    args.max_sec = 0.01
    summary = run_dry(args, FallbackConfig())
    assert summary["state"] == FallbackState.FAULT.value
    assert summary["event"] == "runtime_timeout"
    assert summary["fan_events"] == []


def test_camera_reader_times_out_and_releases_blocked_capture():
    capture = BlockingCapture()
    reader = CameraReader(capture)
    reader.start()
    try:
        assert reader.read(0.01) == (False, None)
    finally:
        reader.close()
    assert not reader.thread.is_alive()


def test_apply_step_rejects_motion_while_fan_requested(tmp_path):
    bot = FakeBot()
    motor = MotorGateway(
        bot,
        role="test",
        max_v=0.06,
        max_w=0.24,
        lease_path=tmp_path / "motor.lock",
    )
    fan = DryRunFan()
    arbiter = CommandArbiter(max_v=0.06, max_w=0.24)
    unsafe = FallbackStep(
        FallbackState.FAN_RUN,
        MotionCommand(0.01, 0.0, "unsafe", RaceState.TRACK),
        ControlOwner.TERMINAL_EXECUTOR,
        True,
    )
    try:
        try:
            apply_step(unsafe, arbiter, motor, fan)
        except RuntimeError as exc:
            assert "chassis command is non-zero" in str(exc)
        else:
            raise AssertionError("unsafe fan/motion step was accepted")
        assert not fan.is_on
    finally:
        fan.close()
        motor.close(stop_delay=0.0)


def test_apply_step_keeps_fan_off_when_zero_motor_write_fails(tmp_path):
    motor = MotorGateway(
        FailingBot(),
        role="test",
        max_v=0.06,
        max_w=0.24,
        lease_path=tmp_path / "failing-motor.lock",
    )
    fan = DryRunFan()
    arbiter = CommandArbiter(max_v=0.06, max_w=0.24)
    fan_step = FallbackStep(
        FallbackState.FAN_RUN,
        MotionCommand(0.0, 0.0, "fan", RaceState.STOPPED),
        ControlOwner.TERMINAL_EXECUTOR,
        True,
    )
    try:
        try:
            apply_step(fan_step, arbiter, motor, fan)
        except RuntimeError as exc:
            assert "motor write failed" in str(exc)
        else:
            raise AssertionError("fan step ignored motor write failure")
        assert not fan.is_on
        assert fan.events == []
    finally:
        try:
            motor.close(stop_count=1, stop_delay=0.0)
        except RuntimeError:
            pass
