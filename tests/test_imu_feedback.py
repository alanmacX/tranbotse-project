import math
import struct

from transbot_race.imu_feedback import RawImuFrameParser, TransbotImuFeedback


def frame(ax, ay, az, gx, gy, gz, battery=108):
    return struct.pack("<hhhhhhB", ax, ay, az, gx, gy, gz, battery)


def test_headerless_parser_recovers_alignment_and_frames():
    parser = RawImuFrameParser()
    payload = b"bad-prefix" + b"".join(
        frame(500, -15300, 4700, -4, -79, gz) for gz in (-22, -23, -1050, -21)
    )
    parsed = parser.feed(payload)
    assert [sample[5] for sample in parsed] == [-22, -23, -1050, -21]


def test_vendor_gyro_scale_is_radians_per_second():
    value = 65.5 * (180.0 / math.pi)
    assert math.isclose(
        value * TransbotImuFeedback.GYRO_RAD_PER_SEC_PER_LSB,
        1.0,
    )
