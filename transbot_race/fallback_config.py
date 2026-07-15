from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

from .config import CameraConfig, TrackerConfig, VisionConfig, coerce_bool


@dataclass(slots=True)
class TerminalLineConfig:
    enabled: bool = True
    roi: tuple[float, float, float, float] = (0.08, 0.12, 0.92, 0.72)
    min_width_ratio: float = 0.34
    min_width_to_line: float = 3.0
    row_band_frac: float = 0.06
    horizontal_occupancy_min: float = 0.55
    stem_half_width_ratio: float = 1.8
    stem_occupancy_min: float = 0.18
    above_occupancy_max: float = 0.10
    max_wide_row_groups: int = 1
    confirm_frames: int = 4


@dataclass(slots=True)
class ParkingBayConfig:
    enabled: bool = True
    roi: tuple[float, float, float, float] = (0.50, 0.05, 0.98, 0.95)
    border_frac: float = 0.12
    edge_occupancy_min: float = 0.42
    required_edges: int = 3
    interior_occupancy_max: float = 0.28
    confirm_frames: int = 4
    confirm_timeout_sec: float = 1.0


@dataclass(slots=True)
class LeftTurnConfig:
    stop_hold_sec: float = 0.4
    w_radps: float = 0.16
    target_yaw_rad: float = 1.5707963267948966
    hard_timeout_sec: float = 14.0


@dataclass(slots=True)
class ParkingMotionConfig:
    geometry_missing_frames: int = 2
    reverse_v_mps: float = 0.018
    reverse_w_radps: float = 0.0
    reverse_sec: float = 1.5
    reverse_hard_max_sec: float = 3.0
    verify_hold_sec: float = 0.5


@dataclass(slots=True)
class FanConfig:
    run_sec: float = 2.0
    hard_max_sec: float = 5.0


@dataclass(slots=True)
class RuntimeConfig:
    startup_line_frames: int = 3
    max_v_mps: float = 0.06
    max_w_radps: float = 0.24
    loop_period_sec: float = 0.075
    camera_frame_timeout_sec: float = 0.5


@dataclass(slots=True)
class FallbackConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    terminal_line: TerminalLineConfig = field(default_factory=TerminalLineConfig)
    parking_bay: ParkingBayConfig = field(default_factory=ParkingBayConfig)
    left_turn: LeftTurnConfig = field(default_factory=LeftTurnConfig)
    parking: ParkingMotionConfig = field(default_factory=ParkingMotionConfig)
    fan: FanConfig = field(default_factory=FanConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _coerce_tuple(value: object) -> tuple:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"expected tuple/list, got {value!r}")
    return tuple(value)


def _update_dataclass(obj: object, values: dict) -> None:
    names = {item.name for item in fields(obj)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown config fields for {type(obj).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            _update_dataclass(current, value)
        elif isinstance(current, bool):
            setattr(obj, key, coerce_bool(value))
        elif isinstance(current, tuple):
            setattr(obj, key, _coerce_tuple(value))
        elif isinstance(current, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            setattr(obj, key, value)
        elif isinstance(current, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric")
            setattr(obj, key, float(value))
        elif isinstance(current, str):
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            setattr(obj, key, value)
        else:
            setattr(obj, key, value)


def _require_finite_config(value: object) -> None:
    if is_dataclass(value):
        for item in fields(value):
            _require_finite_config(getattr(value, item.name))
        return
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("config tuple values must be numeric")
            _require_finite_config(item)
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise ValueError("fallback config numeric values must be finite")


def _validate_roi(name: str, roi: tuple[float, ...]) -> None:
    if len(roi) != 4 or not all(0.0 <= value <= 1.0 for value in roi):
        raise ValueError(f"{name}.roi must contain four normalized values")
    x0, y0, x1, y1 = roi
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{name} ROI must have positive size")


def validate_fallback_config(cfg: FallbackConfig) -> None:
    _require_finite_config(cfg)
    cfg.camera.crop = tuple(int(value) for value in cfg.camera.crop)
    if len(cfg.camera.crop) != 4:
        raise ValueError("camera.crop must contain four values")
    x0, y0, x1, y1 = cfg.camera.crop
    if cfg.camera.frame_width <= 0 or cfg.camera.frame_height <= 0:
        raise ValueError("camera frame dimensions must be positive")
    if not (0 <= x0 < x1 <= cfg.camera.frame_width and 0 <= y0 < y1 <= cfg.camera.frame_height):
        raise ValueError("camera.crop must be within configured frame dimensions")
    if cfg.camera.expand_left_px < 0 or cfg.camera.expand_right_px < 0:
        raise ValueError("camera crop expansion cannot be negative")
    if cfg.vision.band_count <= 0 or y1 - y0 < cfg.vision.band_count * 12:
        raise ValueError("camera crop is too short for configured scan bands")
    if not 0 <= cfg.vision.threshold_min <= cfg.vision.threshold_max <= 255:
        raise ValueError("vision thresholds must be ordered within [0, 255]")
    if cfg.vision.fit_mode not in {"classic", "poly"}:
        raise ValueError("vision.fit_mode must be classic or poly")

    terminal = cfg.terminal_line
    _validate_roi("terminal_line", terminal.roi)
    if not 0.0 < terminal.min_width_ratio <= 1.0:
        raise ValueError("terminal line width ratio must be within (0, 1]")
    if terminal.min_width_to_line <= 1.0:
        raise ValueError("terminal line must be wider than the tracked line")
    if not 0.0 < terminal.row_band_frac < 0.25:
        raise ValueError("terminal row band fraction must be within (0, 0.25)")
    for name, value in (
        ("horizontal occupancy", terminal.horizontal_occupancy_min),
        ("stem occupancy", terminal.stem_occupancy_min),
        ("above occupancy", terminal.above_occupancy_max),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"terminal {name} must be within [0, 1]")
    if terminal.stem_half_width_ratio <= 0.0:
        raise ValueError("terminal stem width ratio must be positive")
    if terminal.max_wide_row_groups != 1 or terminal.confirm_frames <= 0:
        raise ValueError("terminal detector requires one wide row group and positive confirmation")

    bay = cfg.parking_bay
    _validate_roi("parking_bay", bay.roi)
    if not 0.0 < bay.border_frac < 0.5:
        raise ValueError("parking bay border fraction must be within (0, 0.5)")
    if not 0.0 <= bay.edge_occupancy_min <= 1.0:
        raise ValueError("parking bay edge occupancy must be within [0, 1]")
    if not 0.0 <= bay.interior_occupancy_max <= 1.0:
        raise ValueError("parking bay interior occupancy must be within [0, 1]")
    if bay.required_edges not in {3, 4} or bay.confirm_frames <= 0:
        raise ValueError("parking bay requires 3 or 4 edges and positive confirmation")
    if bay.confirm_timeout_sec <= 0.0:
        raise ValueError("parking bay confirmation timeout must be positive")

    turn = cfg.left_turn
    if turn.stop_hold_sec < 0.0:
        raise ValueError("left-turn stop hold cannot be negative")
    if not 0.0 < turn.w_radps <= cfg.runtime.max_w_radps:
        raise ValueError("left-turn angular speed must be positive and within runtime limit")
    if turn.target_yaw_rad <= 0.0 or turn.hard_timeout_sec <= 0.0:
        raise ValueError("left-turn calibration angle and hard timeout must be positive")
    if turn.target_yaw_rad / turn.w_radps >= turn.hard_timeout_sec:
        raise ValueError("left-turn calibrated duration must be below its hard timeout")

    parking = cfg.parking
    if parking.geometry_missing_frames < 0:
        raise ValueError("parking geometry missing-frame tolerance cannot be negative")
    if not 0.0 < parking.reverse_v_mps <= cfg.runtime.max_v_mps:
        raise ValueError("reverse speed must be positive and within the runtime limit")
    if not 0.0 < parking.reverse_sec < parking.reverse_hard_max_sec:
        raise ValueError("reverse duration must be positive and below its hard maximum")
    if parking.verify_hold_sec <= 0.0:
        raise ValueError("parking verification hold must be positive")
    if abs(parking.reverse_w_radps) > cfg.runtime.max_w_radps:
        raise ValueError("parking angular speed exceeds the runtime limit")

    if not 0.0 < cfg.fan.run_sec < cfg.fan.hard_max_sec:
        raise ValueError("fan run time must be positive and below its hard maximum")
    if (
        cfg.runtime.startup_line_frames <= 0
        or cfg.runtime.max_v_mps <= 0.0
        or cfg.runtime.max_w_radps <= 0.0
        or cfg.runtime.loop_period_sec <= 0.0
        or cfg.runtime.camera_frame_timeout_sec <= 0.0
    ):
        raise ValueError("runtime frame count, limits, periods, and timeouts must be positive")
    if not 0.0 <= cfg.tracker.conf_lost < cfg.tracker.conf_predict <= 1.0:
        raise ValueError("tracker confidence thresholds must be ordered within [0, 1]")
    if cfg.tracker.v_max < 0.0 or cfg.tracker.max_w < 0.0:
        raise ValueError("tracker command limits cannot be negative")
    if cfg.tracker.v_max > cfg.runtime.max_v_mps or cfg.tracker.max_w > cfg.runtime.max_w_radps:
        raise ValueError("tracker command limits exceed fallback runtime limits")


def load_fallback_config(path: Path) -> FallbackConfig:
    cfg = FallbackConfig()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("fallback config root must be an object")
    _update_dataclass(cfg, raw)
    validate_fallback_config(cfg)
    return cfg
