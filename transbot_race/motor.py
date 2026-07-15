from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import time


class MotorLeaseError(RuntimeError):
    pass


class MotorGateway:
    """Exclusive process-level access to the chassis motor writer."""

    def __init__(
        self,
        bot,
        *,
        role: str,
        max_v: float,
        max_w: float,
        lease_path: str | Path = "/tmp/transbotse-motor.lock",
    ) -> None:
        self._bot = bot
        self.role = str(role)
        self.max_v = abs(float(max_v))
        self.max_w = abs(float(max_w))
        self.lease_path = Path(lease_path)
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        self._lease = self.lease_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lease.seek(0)
            holder = self._lease.read().strip() or "unknown holder"
            self._lease.close()
            raise MotorLeaseError(f"motor lease already held: {holder}") from exc
        self._held = True
        self._lease.seek(0)
        self._lease.truncate()
        self._lease.write(json.dumps({
            "pid": os.getpid(), "role": self.role, "acquired_at": time.time(),
        }))
        self._lease.flush()
        os.fsync(self._lease.fileno())

    @property
    def lease_held(self) -> bool:
        return self._held

    def set_car_motion(self, v: float, w: float) -> None:
        if not self._held:
            raise MotorLeaseError("motor write rejected: lease is not held")
        v, w = float(v), float(w)
        if not math.isfinite(v) or not math.isfinite(w):
            raise ValueError("motor command must be finite")
        if abs(v) > self.max_v + 1e-9 or abs(w) > self.max_w + 1e-9:
            raise ValueError(
                f"motor command out of bounds: v={v}, w={w}, "
                f"limits=({self.max_v}, {self.max_w})"
            )
        self._bot.set_car_motion(v, w)

    def close(self, *, stop_count: int = 5, stop_delay: float = 0.02) -> None:
        if not self._held:
            return
        try:
            for _ in range(max(1, int(stop_count))):
                self._bot.set_car_motion(0.0, 0.0)
                if stop_delay > 0.0:
                    time.sleep(stop_delay)
        finally:
            close_feedback = getattr(self._bot, "close_feedback", None)
            if callable(close_feedback):
                close_feedback()
            self._held = False
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_UN)
            self._lease.close()

    def __getattr__(self, name: str):
        return getattr(self._bot, name)
