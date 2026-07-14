import tempfile
from pathlib import Path

import pytest

from transbot_race.motor import MotorGateway, MotorLeaseError


class FakeBot:
    def __init__(self):
        self.commands = []

    def set_car_motion(self, v, w):
        self.commands.append((v, w))


def test_process_lease_rejects_second_writer():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "motor.lock"
        first = MotorGateway(FakeBot(), role="auto", max_v=0.1, max_w=0.4, lease_path=path)
        try:
            with pytest.raises(MotorLeaseError):
                MotorGateway(FakeBot(), role="manual", max_v=0.1, max_w=0.4, lease_path=path)
        finally:
            first.close(stop_delay=0)


def test_release_allows_next_writer():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "motor.lock"
        first = MotorGateway(FakeBot(), role="auto", max_v=0.1, max_w=0.4, lease_path=path)
        first.close(stop_delay=0)
        second = MotorGateway(FakeBot(), role="manual", max_v=0.1, max_w=0.4, lease_path=path)
        second.close(stop_delay=0)


def test_gateway_rejects_out_of_bounds_motion():
    with tempfile.TemporaryDirectory() as tmp:
        gateway = MotorGateway(
            FakeBot(), role="auto", max_v=0.1, max_w=0.4,
            lease_path=Path(tmp) / "motor.lock",
        )
        try:
            with pytest.raises(ValueError):
                gateway.set_car_motion(0.11, 0.0)
        finally:
            gateway.close(stop_delay=0)
