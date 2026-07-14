from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

from .config import CameraConfig, TrackerConfig, VisionConfig, coerce_bool


@dataclass(slots=True)
class ParkingTriggerConfig:
    enabled: bool = True
    roi: tuple[float, float, float, float] = (0.50, 0.05, 0.98, 0.95)
    border_frac: float = 0.12
    edge_occupancy_min: float = 0.42
    required_edges: int = 3
    interior_occupancy_max: float = 0.28
    confirm_frames: int = 4


@dataclass(slots=True)
class ParkingMotionConfig:
    trigger_timeout_sec: float = 1.0
    geometry_missing_frames: int = 2
    stage_v_mps: float = 0.0
    stage_w_radps: float = 0.0
    stage_sec: float = 0.4
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
    parking_trigger: ParkingTriggerConfig = field(default_factory=ParkingTriggerConfig)
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

    trigger = cfg.parking_trigger
    if len(trigger.roi) != 4 or not all(0.0 <= value <= 1.0 for value in trigger.roi):
        raise ValueError("parking_trigger.roi must contain four normalized values")
    rx0, ry0, rx1, ry1 = trigger.roi
    if rx1 <= rx0 or ry1 <= ry0:
        raise ValueError("parking trigger ROI must have positive size")
    if not 0.0 < trigger.border_frac < 0.5:
        raise ValueError("parking trigger border fraction must be within (0, 0.5)")
    if not 0.0 <= trigger.edge_occupancy_min <= 1.0:
        raise ValueError("parking edge occupancy must be within [0, 1]")
    if not 0.0 <= trigger.interior_occupancy_max <= 1.0:
        raise ValueError("parking interior occupancy must be within [0, 1]")
    if trigger.required_edges not in {3, 4} or trigger.confirm_frames <= 0:
        raise ValueError("parking trigger requires 3 or 4 edges and positive confirmation")

    parking = cfg.parking
    if parking.trigger_timeout_sec <= 0.0 or parking.stage_sec < 0.0:
        raise ValueError("parking trigger timeout must be positive and stage duration non-negative")
    if parking.geometry_missing_frames < 0:
        raise ValueError("parking geometry missing-frame tolerance cannot be negative")
    if not 0.0 < parking.reverse_v_mps <= cfg.runtime.max_v_mps:
        raise ValueError("reverse speed must be positive and within the runtime limit")
    if not 0.0 < parking.reverse_sec < parking.reverse_hard_max_sec:
        raise ValueError("reverse duration must be positive and below its hard maximum")
    if parking.verify_hold_sec <= 0.0:
        raise ValueError("parking verification hold must be positive")
    if abs(parking.stage_v_mps) > cfg.runtime.max_v_mps:
        raise ValueError("parking stage speed exceeds the runtime limit")
    if max(abs(parking.stage_w_radps), abs(parking.reverse_w_radps)) > cfg.runtime.max_w_radps:
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
