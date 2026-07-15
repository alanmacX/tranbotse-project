from __future__ import annotations

from collections import deque
import math
import statistics
import struct
import threading
import time


class RawImuFrameParser:
    """Parse the headerless 13-byte stream emitted by Transbot-SE V3.2.

    The installed 3.2.5 SDK expects a framed 16-byte auto-report payload, but
    this controller emits six little-endian int16 IMU channels followed by one
    battery byte.  Alignment is recovered from gravity magnitude and voltage.
    """

    FRAME_SIZE = 13

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.synced = False

    @staticmethod
    def _decode(frame: bytes) -> tuple[int, int, int, int, int, int, int]:
        return struct.unpack("<hhhhhhB", frame)

    @staticmethod
    def _plausible(values: tuple[int, ...]) -> bool:
        gravity = math.sqrt(sum(value * value for value in values[:3]))
        return 8_000.0 < gravity < 24_000.0 and 60 <= values[6] <= 180

    def feed(self, data: bytes) -> list[tuple[int, int, int, int, int, int, int]]:
        self.buffer.extend(data)
        frames: list[tuple[int, int, int, int, int, int, int]] = []
        while True:
            if not self.synced:
                if len(self.buffer) < self.FRAME_SIZE * 4:
                    break
                best_offset = -1
                best_score = 0
                for offset in range(self.FRAME_SIZE):
                    score = 0
                    for index in range(offset, len(self.buffer) - 12, self.FRAME_SIZE):
                        values = self._decode(bytes(self.buffer[index:index + 13]))
                        if self._plausible(values):
                            score += 1
                        else:
                            break
                    if score > best_score:
                        best_offset, best_score = offset, score
                if best_score < 3:
                    del self.buffer[:self.FRAME_SIZE]
                    continue
                del self.buffer[:best_offset]
                self.synced = True
            if len(self.buffer) < self.FRAME_SIZE:
                break
            values = self._decode(bytes(self.buffer[:self.FRAME_SIZE]))
            if not self._plausible(values):
                del self.buffer[0]
                self.synced = False
                continue
            del self.buffer[:self.FRAME_SIZE]
            frames.append(values)
        return frames


class TransbotImuFeedback:
    """Add fresh physical yaw-rate feedback to the vendor Transbot object."""

    GYRO_RAD_PER_SEC_PER_LSB = 1.0 / 65.5 / (180.0 / math.pi)

    def __init__(self, bot: object, *, calibration_sec: float = 0.8) -> None:
        self._bot = bot
        self._parser = RawImuFrameParser()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._samples: deque[int] = deque(maxlen=200)
        self._gyro_z_raw = 0
        self._gyro_bias_raw = 0.0
        self._sample_at = 0.0
        self._last_v = 0.0
        self._last_w = 0.0
        self._bot.ser.reset_input_buffer()
        self._bot.set_auto_report_state(True, forever=False)
        self._thread = threading.Thread(
            target=self._read_loop, name="transbot_raw_imu", daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + max(1.5, calibration_sec + 1.0)
        while time.monotonic() < deadline:
            with self._lock:
                enough = len(self._samples) >= 12
            if enough:
                break
            time.sleep(0.02)
        time.sleep(max(0.0, float(calibration_sec)))
        with self._lock:
            if len(self._samples) < 12:
                self.close_feedback()
                raise RuntimeError("Transbot IMU auto-report stream is unavailable")
            self._gyro_bias_raw = float(statistics.median(self._samples))

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                waiting = int(self._bot.ser.in_waiting)
                if waiting <= 0:
                    time.sleep(0.002)
                    continue
                data = self._bot.ser.read(waiting)
                for frame in self._parser.feed(data):
                    now = time.monotonic()
                    with self._lock:
                        self._gyro_z_raw = frame[5]
                        self._samples.append(frame[5])
                        self._sample_at = now
            except Exception:
                time.sleep(0.01)

    def set_car_motion(self, v: float, w: float) -> None:
        self._last_v, self._last_w = float(v), float(w)
        self._bot.set_car_motion(v, w)

    def get_motion_data(self) -> tuple[float, float, bool]:
        with self._lock:
            age = time.monotonic() - self._sample_at
            gyro_raw = self._gyro_z_raw - self._gyro_bias_raw
        if age > 0.25:
            raise RuntimeError(f"Transbot IMU sample stale: {age:.3f}s")
        gyro_z = gyro_raw * self.GYRO_RAD_PER_SEC_PER_LSB
        return self._last_v, gyro_z, True

    def close_feedback(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        try:
            self._bot.set_auto_report_state(False, forever=False)
        except Exception:
            pass

    def __getattr__(self, name: str):
        return getattr(self._bot, name)
