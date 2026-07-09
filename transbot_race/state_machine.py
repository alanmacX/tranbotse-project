from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import RaceConfig
from .vision import BranchFeature, LineFeatures


class RaceState(str, Enum):
    LINE_FOLLOW = "line_follow"
    GAP_BLIND = "gap_blind"
    TIMED_FORWARD = "timed_forward"
    TIMED_TURN = "timed_turn"
    REACQUIRE = "reacquire"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class MotionCommand:
    v: float
    w: float
    reason: str
    state: RaceState


class RaceStateMachine:
    """Single controller for line following and mutually exclusive special states."""

    def __init__(self, cfg: RaceConfig | None = None) -> None:
        self.cfg = cfg or RaceConfig()
        self.state = RaceState.LINE_FOLLOW
        self.state_started_at = 0.0
        self.last_err_norm = 0.0
        self.missing_count = 0
        self.corner_count = 0
        self.corner_direction: str | None = None
        self.active_turn_dir = 0.0
        self.reacquire_count = 0
        self.last_event = "init"

    def reset(self) -> None:
        self.__init__(self.cfg)

    def step(self, features: LineFeatures, now: float, obstacle: bool = False) -> MotionCommand:
        if obstacle:
            self._enter(RaceState.STOPPED, now, "obstacle")
            return MotionCommand(0.0, 0.0, "obstacle", self.state)

        if self.state == RaceState.STOPPED:
            return MotionCommand(0.0, 0.0, "stopped", self.state)

        if self.state == RaceState.TIMED_FORWARD:
            if now - self.state_started_at >= self.cfg.corner.forward_sec:
                self._enter(RaceState.TIMED_TURN, now, "turn_start")
                return MotionCommand(0.0, self.active_turn_dir * self.cfg.corner.turn_w, "turn_start", self.state)
            return MotionCommand(self.cfg.line.speed, 0.0, "timed_forward", self.state)

        if self.state == RaceState.TIMED_TURN:
            if now - self.state_started_at >= self.cfg.corner.turn_sec:
                self._enter(RaceState.REACQUIRE, now, "reacquire")
            return MotionCommand(0.0, self.active_turn_dir * self.cfg.corner.turn_w, "timed_turn", self.state)

        if self.state == RaceState.REACQUIRE:
            if features.found and abs(features.err_norm) <= self.cfg.corner.reacquire_err_norm:
                self.reacquire_count += 1
                if self.reacquire_count >= self.cfg.corner.reacquire_confirm_frames:
                    self._enter(RaceState.LINE_FOLLOW, now, "reacquired")
                    self.reacquire_count = 0
                    return self._line_follow(features)
            else:
                self.reacquire_count = 0
            return MotionCommand(0.0, self.active_turn_dir * self.cfg.corner.turn_w, "reacquire_turn", self.state)

        if not features.found:
            self.missing_count += 1
            if self.cfg.gap.enabled and self.missing_count >= self.cfg.gap.missing_frames:
                if self.state != RaceState.GAP_BLIND:
                    self._enter(RaceState.GAP_BLIND, now, "gap_start")
                if now - self.state_started_at <= self.cfg.gap.blind_sec:
                    w = -self.last_err_norm * self.cfg.line.max_w * self.cfg.gap.blind_turn_factor
                    return MotionCommand(self.cfg.line.speed * self.cfg.gap.blind_speed_factor, w, "gap_blind", self.state)
            w = -self.last_err_norm * self.cfg.gap.search_w
            return MotionCommand(0.0, w, "line_missing", self.state)

        self.missing_count = 0
        self.last_err_norm = features.err_norm
        if self.state == RaceState.GAP_BLIND:
            self._enter(RaceState.LINE_FOLLOW, now, "gap_recovered")

        branch = self._corner_branch(features)
        if branch is not None:
            if self.corner_direction == branch.direction:
                self.corner_count += 1
            else:
                self.corner_direction = branch.direction
                self.corner_count = 1
            if self.corner_count >= self.cfg.corner.confirm_frames:
                self.active_turn_dir = self._turn_dir(branch.direction)
                self._enter(RaceState.TIMED_FORWARD, now, f"{branch.direction}_corner")
                self.corner_count = 0
                return MotionCommand(self.cfg.line.speed, 0.0, "corner_forward", self.state)
        else:
            self.corner_count = 0
            self.corner_direction = None

        return self._line_follow(features)

    def _line_follow(self, features: LineFeatures) -> MotionCommand:
        sign = 1.0 if self.cfg.line.invert_turn else -1.0
        w = sign * self.cfg.line.kp * features.err_norm
        w = max(-self.cfg.line.max_w, min(self.cfg.line.max_w, w))
        slowdown = min(self.cfg.line.max_slowdown, abs(features.err_norm) * self.cfg.line.slow_on_error)
        v = self.cfg.line.speed * (1.0 - slowdown)
        return MotionCommand(v, w, "line_follow", self.state)

    def _corner_branch(self, features: LineFeatures) -> BranchFeature | None:
        mode = self.cfg.corner.mode
        if mode == "off":
            return None
        trigger_y = self.cfg.vision.trigger_y_frac
        candidates = []
        if mode in ("auto", "left") and features.branch_left and features.branch_left.y >= self._feature_height(features) * trigger_y:
            candidates.append(features.branch_left)
        if mode in ("auto", "right") and features.branch_right and features.branch_right.y >= self._feature_height(features) * trigger_y:
            candidates.append(features.branch_right)
        if not candidates:
            return None
        return max(candidates, key=lambda branch: branch.run.area)

    def _feature_height(self, features: LineFeatures) -> float:
        if not features.bands:
            return 1.0
        return float(max(band.y1 for band in features.bands))

    def _turn_dir(self, direction: str) -> float:
        if direction == "left":
            return self.cfg.corner.left_turn_dir
        return self.cfg.corner.right_turn_dir

    def _enter(self, state: RaceState, now: float, event: str) -> None:
        self.state = state
        self.state_started_at = now
        self.last_event = event

