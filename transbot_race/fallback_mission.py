from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .control import ControlOwner, StopCause, TransitionEvent
from .fallback_config import FallbackConfig
from .parking import ParkingObservation, TimedParkingController
from .state_machine import MotionCommand, RaceState, RaceStateMachine
from .vision import TrajectoryFit


class FallbackState(str, Enum):
    STARTUP = "startup"
    FOLLOW_LINE = "follow_line"
    PARK_TRIGGER = "park_trigger"
    PARK_STAGE = "park_stage"
    PARK_REVERSE = "park_reverse"
    PARK_VERIFY = "park_verify"
    FAN_RUN = "fan_run"
    FINISHED = "finished"
    FAULT = "fault"


@dataclass(frozen=True, slots=True)
class FallbackStep:
    state: FallbackState
    command: MotionCommand
    owner: ControlOwner
    fan_requested: bool
    stop_cause: StopCause | None = None
    transition_event: TransitionEvent = TransitionEvent.NONE
    event: str = ""


class FallbackMission:
    """Independent line-follow, timed parking calibration, and fan sequence."""

    def __init__(self, cfg: FallbackConfig, *, allow_unvalidated_parking: bool) -> None:
        self.cfg = cfg
        self.allow_unvalidated_parking = bool(allow_unvalidated_parking)
        self.cruise = RaceStateMachine(self._race_config_view())
        self.parking = TimedParkingController(cfg.parking)
        self.state = FallbackState.STARTUP
        self.state_started_at = 0.0
        self.startup_line_streak = 0
        self.parking_geometry_missing_frames = 0
        self.fault_cause: StopCause | None = None
        self.last_event = "init"

    def _race_config_view(self):
        from .config import RaceConfig

        race = RaceConfig()
        race.camera = self.cfg.camera
        race.vision = self.cfg.vision
        race.tracker = self.cfg.tracker
        return race

    def step(
        self,
        *,
        fit: TrajectoryFit,
        parking: ParkingObservation,
        now: float,
        camera_ok: bool = True,
    ) -> FallbackStep:
        if not camera_ok:
            return self.fail(now, "camera_failure", StopCause.CAMERA_FAILURE)

        if self.state == FallbackState.STARTUP:
            ready = self._line_ready(fit)
            self.startup_line_streak = self.startup_line_streak + 1 if ready else 0
            if self.startup_line_streak >= self.cfg.runtime.startup_line_frames:
                self._enter(FallbackState.FOLLOW_LINE, now, "startup_line_ready")
                command = self.cruise.reacquire_from(fit, now)
                return self._step(command, ControlOwner.CRUISE, event="startup_line_ready")
            return self._step(self._stop("startup_waiting_for_line"), ControlOwner.NONE)

        if self.state == FallbackState.FOLLOW_LINE:
            if parking.confirmed:
                self._enter(FallbackState.PARK_TRIGGER, now, "parking_trigger_confirmed")
                return self._step(
                    self._stop("parking_trigger_confirmed"),
                    ControlOwner.TERMINAL_EXECUTOR,
                    transition=TransitionEvent.TAKEOVER_ACCEPTED,
                    event="parking_trigger_confirmed",
                )
            command = self.cruise.step(fit, now)
            if command.reason == "search_timeout":
                return self.fail(now, "line_search_timeout", StopCause.SEARCH_TIMEOUT)
            return self._step(command, ControlOwner.CRUISE)

        if self.state == FallbackState.PARK_TRIGGER:
            if not self.allow_unvalidated_parking:
                return self._step(
                    self._stop("parking_motion_locked"),
                    ControlOwner.TERMINAL_EXECUTOR,
                    event="parking_motion_requires_explicit_unlock",
                )
            if not parking.candidate:
                if now - self.state_started_at + 1e-9 >= self.cfg.parking.trigger_timeout_sec:
                    return self.fail(now, "parking_trigger_lost", StopCause.EXECUTOR_FAILED)
                return self._step(self._stop("parking_trigger_recheck"), ControlOwner.TERMINAL_EXECUTOR)
            self._enter(FallbackState.PARK_STAGE, now, "parking_stage_started")
            return self._step(
                self._stop("parking_stage_started"),
                ControlOwner.TERMINAL_EXECUTOR,
                event="parking_stage_started",
            )

        if self.state == FallbackState.PARK_STAGE:
            command, complete = self.parking.stage_command(now - self.state_started_at)
            if complete:
                self._enter(FallbackState.PARK_REVERSE, now, "parking_reverse_started")
                return self._step(command, ControlOwner.TERMINAL_EXECUTOR, event="parking_reverse_started")
            return self._step(command, ControlOwner.TERMINAL_EXECUTOR)

        if self.state == FallbackState.PARK_REVERSE:
            self.parking_geometry_missing_frames = (
                0 if parking.candidate else self.parking_geometry_missing_frames + 1
            )
            if self.parking_geometry_missing_frames > self.cfg.parking.geometry_missing_frames:
                return self.fail(now, "parking_geometry_lost", StopCause.EXECUTOR_FAILED)
            command, complete, fault = self.parking.reverse_command(now - self.state_started_at)
            if fault is not None:
                return self.fail(now, fault, StopCause.EXECUTOR_FAILED)
            if complete:
                self._enter(FallbackState.PARK_VERIFY, now, "parking_reverse_complete")
                return self._step(command, ControlOwner.TERMINAL_EXECUTOR, event="parking_reverse_complete")
            return self._step(command, ControlOwner.TERMINAL_EXECUTOR)

        if self.state == FallbackState.PARK_VERIFY:
            if now - self.state_started_at + 1e-9 >= self.cfg.parking.verify_hold_sec:
                self._enter(FallbackState.FAN_RUN, now, "fan_started")
                return self._step(
                    self._stop("fan_started"),
                    ControlOwner.TERMINAL_EXECUTOR,
                    fan=True,
                    event="fan_started",
                )
            return self._step(self._stop("parking_verify_hold"), ControlOwner.TERMINAL_EXECUTOR)

        if self.state == FallbackState.FAN_RUN:
            elapsed = now - self.state_started_at
            if elapsed + 1e-9 >= self.cfg.fan.hard_max_sec:
                return self.fail(now, "fan_hard_timeout", StopCause.EXECUTOR_FAILED)
            if elapsed + 1e-9 >= self.cfg.fan.run_sec:
                self._enter(FallbackState.FINISHED, now, "mission_finished")
                return self._step(
                    self._stop("mission_finished"),
                    ControlOwner.TERMINAL_EXECUTOR,
                    stop=StopCause.MISSION_FINISHED,
                    transition=TransitionEvent.EXECUTOR_COMPLETED,
                    event="mission_finished",
                )
            return self._step(self._stop("fan_running"), ControlOwner.TERMINAL_EXECUTOR, fan=True)

        if self.state == FallbackState.FINISHED:
            return self._step(
                self._stop("mission_finished"),
                ControlOwner.TERMINAL_EXECUTOR,
                stop=StopCause.MISSION_FINISHED,
            )

        return self._step(
            self._stop(self.last_event or "fault"),
            ControlOwner.TERMINAL_EXECUTOR,
            stop=self.fault_cause or StopCause.EXECUTOR_FAILED,
        )

    def fail(self, now: float, event: str, cause: StopCause) -> FallbackStep:
        if self.state != FallbackState.FAULT:
            self.fault_cause = cause
            self._enter(FallbackState.FAULT, now, event)
        cause = self.fault_cause or cause
        return self._step(
            self._stop(event),
            ControlOwner.TERMINAL_EXECUTOR,
            stop=cause,
            transition=TransitionEvent.EXECUTOR_ABORTED,
            event=event,
        )

    def _line_ready(self, fit: TrajectoryFit) -> bool:
        return self.cruise.can_take_moving_handoff(fit)

    def _enter(self, state: FallbackState, now: float, event: str) -> None:
        self.state = state
        self.state_started_at = now
        if state == FallbackState.PARK_REVERSE:
            self.parking_geometry_missing_frames = 0
        self.last_event = event

    def _step(
        self,
        command: MotionCommand,
        owner: ControlOwner,
        *,
        fan: bool = False,
        stop: StopCause | None = None,
        transition: TransitionEvent = TransitionEvent.NONE,
        event: str = "",
    ) -> FallbackStep:
        if fan and (command.v != 0.0 or command.w != 0.0):
            raise RuntimeError("fan and chassis motion cannot be requested together")
        return FallbackStep(
            self.state,
            command,
            owner,
            fan,
            stop,
            transition,
            event,
        )

    @staticmethod
    def _stop(reason: str) -> MotionCommand:
        return MotionCommand(0.0, 0.0, reason, RaceState.STOPPED)
