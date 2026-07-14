from __future__ import annotations

from typing import Protocol


class Fan(Protocol):
    @property
    def is_on(self) -> bool: ...

    def on(self) -> None: ...

    def off(self) -> None: ...

    def close(self) -> None: ...


class DryRunFan:
    """Software-only fan used by dry-run and tests; never accesses GPIO."""

    def __init__(self) -> None:
        self._is_on = False
        self.events: list[str] = []

    @property
    def is_on(self) -> bool:
        return self._is_on

    def on(self) -> None:
        if not self._is_on:
            self._is_on = True
            self.events.append("on")

    def off(self) -> None:
        if self._is_on:
            self._is_on = False
            self.events.append("off")

    def close(self) -> None:
        self.off()


class UnavailableFan:
    """Fail-closed placeholder until a verified fan adapter is supplied."""

    @property
    def is_on(self) -> bool:
        return False

    def on(self) -> None:
        raise RuntimeError("fan adapter unavailable")

    def off(self) -> None:
        pass

    def close(self) -> None:
        self.off()
