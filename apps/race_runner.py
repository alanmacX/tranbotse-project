#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path

import cv2 as cv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transbot_race.config import RaceConfig  # noqa: E402
from transbot_race.geometry import apply_occlusion, band_is_occluded  # noqa: E402
from transbot_race.path_memory import (  # noqa: E402
    CornerCommandDelay,
    PathStrategyStatus,
    read_motion_sample,
)
from transbot_race.obstacle import (  # noqa: E402
    ObstacleMonitor,
    draw_obstacle_overlay,
)
from transbot_race.state_machine import RaceStateMachine, command_summary  # noqa: E402
from transbot_race.vision import (  # noqa: E402
    _band_bounds,
    draw_debug_overlay,
    fit_line_trajectory,
    preprocess_blackline,
    scan_line_features,
)


def coerce_tuple(value: object) -> tuple:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"expected tuple/list value, got {value!r}")
    return tuple(tuple(item) if isinstance(item, list) else item for item in value)


def update_dataclass(obj: object, values: dict) -> None:
    if not is_dataclass(obj):
        return
    field_names = {field.name for field in fields(obj)}
    for key, value in values.items():
        if key not in field_names:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            update_dataclass(current, value)
        elif isinstance(current, tuple):
            setattr(obj, key, coerce_tuple(value))
        elif current is None and isinstance(value, list):
            setattr(obj, key, coerce_tuple(value))
        else:
            setattr(obj, key, value)


def load_config(path: Path) -> RaceConfig:
    cfg = RaceConfig()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            update_dataclass(cfg, json.load(f))
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: RaceConfig) -> None:
    cfg.camera.crop = tuple(int(item) for item in coerce_tuple(cfg.camera.crop))
    if len(cfg.camera.crop) != 4:
        raise ValueError(f"camera.crop must have 4 values, got {cfg.camera.crop!r}")
    x0, y0, x1, y1 = cfg.camera.crop
    crop_h = y1 - y0
    min_h = cfg.vision.band_count * 12
    if crop_h < min_h:
        raise ValueError(
            f"crop height {crop_h}px too small for band_count={cfg.vision.band_count} "
            f"(need >= {min_h}px). Check camera.crop={cfg.camera.crop}."
        )
    if x1 <= x0:
        raise ValueError(f"crop x1 must be > x0, got {cfg.camera.crop}")
    if cfg.path_memory.camera_to_axle_m < 0.0:
        raise ValueError("camera-to-axle distance cannot be negative")
    if cfg.path_memory.mode not in ("none", "corner_event"):
        raise ValueError(f"unsupported path-memory mode: {cfg.path_memory.mode}")
    if cfg.path_memory.corner_confirm_frames <= 0:
        raise ValueError("corner confirmation frame count must be positive")
    if cfg.path_memory.corner_replay_max_w <= 0.0 or cfg.path_memory.corner_turn_angle_rad <= 0.0:
        raise ValueError("corner turn angle and angular speed must be positive")
    if not 0.0 <= cfg.path_memory.corner_turn_speed_ratio <= 1.0:
        raise ValueError("corner turn speed ratio must be within [0, 1]")
    if not 0.0 <= cfg.path_memory.corner_reacquire_angle_rad <= cfg.path_memory.corner_turn_angle_rad:
        raise ValueError("corner reacquire angle must be within [0, turn angle]")
    if cfg.path_memory.corner_image_angle_gain <= 0.0 or cfg.path_memory.corner_search_extra_rad < 0.0:
        raise ValueError("corner angle gain must be positive and extra search angle non-negative")
    if cfg.path_memory.corner_reacquire_confirm_frames <= 0 or cfg.path_memory.corner_handoff_blend_frames <= 0:
        raise ValueError("corner reacquire and handoff frame counts must be positive")
    if cfg.path_memory.max_motion_dt_sec <= 0.0:
        raise ValueError("path-memory motion interval must be positive")


class DryBot:
    def __init__(self) -> None:
        self.v = 0.0
        self.w = 0.0

    def set_car_motion(self, v: float, w: float) -> None:
        self.v, self.w = float(v), float(w)
        print(f"DRY command v={v:.4f} w={w:.4f}")

    def get_motion_data(self) -> tuple[float, float]:
        return self.v, self.w

    def set_floodlight(self, value: int) -> None:
        print(f"DRY floodlight {value}")


def make_bot(dry_run: bool):
    if dry_run:
        return DryBot()
    sys.path.insert(0, "/home/pi/Transbot/py_install")
    from Transbot_Lib import Transbot  # type: ignore

    return Transbot()


def stop_chassis(bot, count: int = 20, delay: float = 0.04) -> None:
    for _ in range(count):
        bot.set_car_motion(0.0, 0.0)
        time.sleep(delay)


def crop_frame(frame, cfg: RaceConfig):
    x0, y0, x1, y1 = cfg.camera.crop
    ex0 = max(0, x0 - cfg.camera.expand_left_px)
    ex1 = min(cfg.camera.frame_width, x1 + cfg.camera.expand_right_px)
    track_center = ((x0 + x1) / 2.0) - ex0
    return frame[y0:y1, ex0:ex1], (ex0, y0, ex1, y1), track_center


def occluded_band_indices(height: int, width: int, cfg: RaceConfig) -> frozenset[int]:
    """Bands (in scan_line_features geometry) that fall in a static dead zone."""
    if not cfg.occlusion.enabled or not cfg.occlusion.rects:
        return frozenset()
    n = cfg.vision.band_count
    indices = []
    for index in range(n):
        y0, y1 = _band_bounds(index, height, n)
        if band_is_occluded(y0, y1, width, cfg.occlusion):
            indices.append(index)
    return frozenset(indices)


class DebugRecorder:
    """Optional runtime capture for field debugging."""

    def __init__(self, debug_dir: str, cfg: RaceConfig, args: argparse.Namespace) -> None:
        self.enabled = bool(debug_dir)
        self.root = Path(debug_dir) if debug_dir else None
        self.frame_period = max(0.1, float(args.debug_frame_period))
        self.jpeg_quality = max(40, min(95, int(args.debug_jpeg_quality)))
        self.last_frame_t = -1e9
        self.frame_count = 0
        self.started_at = time.time()
        self.telemetry = None

        if not self.enabled or self.root is None:
            return

        self.frames_dir = self.root / "frames"
        self.crops_dir = self.root / "crops"
        self.overlays_dir = self.root / "overlays"
        self.masks_dir = self.root / "masks"
        self.obstacles_dir = self.root / "obstacles"
        for path in (self.frames_dir, self.crops_dir, self.overlays_dir, self.masks_dir, self.obstacles_dir):
            path.mkdir(parents=True, exist_ok=True)

        with (self.root / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "max_sec": args.max_sec,
                    "period": args.period,
                    "log_period": args.log_period,
                    "frame_period": self.frame_period,
                    "camera": args.camera,
                    "config": asdict(cfg),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")

        self.telemetry = (self.root / "telemetry.jsonl").open("a", encoding="utf-8", buffering=1)
        print(f"debug_capture_dir={self.root}", file=sys.stderr, flush=True)

    def record(
        self, summary: dict, frame, crop, mask, features, cfg: RaceConfig, crop_center: float,
        raw_crop=None, obstacle_decision=None, raw_track_center: float | None = None,
        strategy_status: PathStrategyStatus | None = None,
    ) -> None:
        if not self.enabled or self.root is None:
            return

        if self.telemetry is not None:
            self.telemetry.write(json.dumps(summary, ensure_ascii=False) + "\n")

        elapsed = float(summary.get("t", 0.0))
        if elapsed - self.last_frame_t < self.frame_period:
            return

        stem = f"{self.frame_count:05d}_{int(elapsed * 1000):07d}"
        params = [int(cv.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        overlay = draw_debug_overlay(crop, features, cfg.vision.trigger_y_frac, crop_center=crop_center)
        if strategy_status is not None:
            label = f"{strategy_status.mode}: {strategy_status.reason}"
            if strategy_status.remaining_m > 0.0:
                label += f" {strategy_status.remaining_m:.3f}m"
            cv.rectangle(overlay, (0, 0), (min(overlay.shape[1], 300), 24), (0, 0, 0), -1)
            cv.putText(overlay, label, (6, 17), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv.LINE_AA)

        cv.imwrite(str(self.frames_dir / f"{stem}.jpg"), frame, params)
        cv.imwrite(str(self.crops_dir / f"{stem}.jpg"), crop, params)
        cv.imwrite(str(self.overlays_dir / f"{stem}.jpg"), overlay, params)
        cv.imwrite(str(self.masks_dir / f"{stem}.png"), mask)
        if raw_crop is not None and obstacle_decision is not None and raw_track_center is not None:
            obstacle_overlay = draw_obstacle_overlay(raw_crop, obstacle_decision, raw_track_center, cfg.obstacle)
            cv.imwrite(str(self.obstacles_dir / f"{stem}.jpg"), obstacle_overlay, params)

        self.last_frame_t = elapsed
        self.frame_count += 1

    def close(self) -> None:
        if not self.enabled or self.root is None:
            return
        if self.telemetry is not None:
            self.telemetry.close()
            self.telemetry = None
        with (self.root / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "duration_sec": round(time.time() - self.started_at, 3),
                    "frames_saved": self.frame_count,
                    "telemetry": "telemetry.jsonl",
                    "frames": "frames/",
                    "crops": "crops/",
                    "overlays": "overlays/",
                    "masks": "masks/",
                    "obstacles": "obstacles/",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    bot = make_bot(args.dry_run)
    sm = RaceStateMachine(cfg)
    corner_margin = CornerCommandDelay(cfg.path_memory)
    obstacle_monitor = ObstacleMonitor(cfg.obstacle)
    cap = cv.VideoCapture(args.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg.camera.frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cfg.camera.frame_height)
    if not cap.isOpened():
        raise RuntimeError(f"camera open failed: {args.camera}")
    debug = DebugRecorder(args.debug_dir, cfg, args)

    if hasattr(bot, "set_floodlight"):
        bot.set_floodlight(args.light)

    start = time.monotonic()
    last_log = 0.0
    occluded = frozenset()  # computed once from the first crop; static per run
    last_cmd_v = 0.0
    last_cmd_w = 0.0
    straight_streak = 0
    obstacle_armed_until = -1e9
    try:
        stop_chassis(bot, count=3, delay=0.03)
        while time.monotonic() - start < args.max_sec:
            ok, frame = cap.read()
            if not ok:
                bot.set_car_motion(0.0, 0.0)
                time.sleep(0.05)
                continue
            crop, (x0, y0, x1, y1), track_center = crop_frame(frame, cfg)
            raw_crop = crop.copy()
            raw_track_center = track_center
            strategy_mode = cfg.path_memory.mode if cfg.path_memory.enabled else "disabled"
            crop_w = crop.shape[1]
            mask = preprocess_blackline(crop, cfg.vision)
            mask = apply_occlusion(mask, cfg.occlusion)
            if not occluded and cfg.occlusion.enabled:
                occluded = occluded_band_indices(crop.shape[0], crop.shape[1], cfg)
            features = scan_line_features(mask, cfg.vision, crop_center=track_center)
            visual_fit = fit_line_trajectory(
                features,
                cfg.vision,
                crop_center=track_center,
                crop_width=crop_w,
                lookahead_frac=cfg.tracker.lookahead_frac,
                occluded_band_indices=occluded,
            )
            now = time.monotonic()
            motion = read_motion_sample(bot, last_cmd_v, last_cmd_w)
            stable_straight = bool(
                visual_fit.found
                and visual_fit.conf >= 0.65
                and visual_fit.n_bands >= 3
                and abs(visual_fit.theta) <= 0.16
                and abs(visual_fit.e0) <= 0.28
                and not visual_fit.disconnected
                and features.branch_left is None
                and features.branch_right is None
            )
            straight_streak = straight_streak + 1 if stable_straight else 0
            if straight_streak >= cfg.obstacle.stable_frames:
                obstacle_armed_until = now + cfg.obstacle.arm_hold_sec
            obstacle_armed = now <= obstacle_armed_until
            obstacle_decision = obstacle_monitor.update(raw_crop, raw_track_center, obstacle_armed)

            fit = visual_fit
            memory_status = PathStrategyStatus(False, strategy_mode, "disabled")
            if cfg.path_memory.enabled:
                if strategy_mode == "none":
                    memory_status = PathStrategyStatus(False, "none", "passthrough")
                elif strategy_mode == "corner_event":
                    memory_status = PathStrategyStatus(True, "corner_event", corner_margin.state)
            command = sm.step(fit, now=now, obstacle=obstacle_decision.stop_required)
            if strategy_mode == "corner_event" and not obstacle_decision.stop_required:
                delayed = corner_margin.step(
                    visual_fit, features, command.v, command.w, now,
                    motion.linear, motion.angular,
                )
                memory_status = delayed.status
                if delayed.v != command.v or delayed.w != command.w:
                    command = replace(
                        command,
                        v=delayed.v,
                        w=delayed.w,
                        reason=f"corner_{memory_status.reason}",
                    )
            if obstacle_decision.slow_required and command.v > 0.0:
                command = replace(
                    command,
                    v=min(command.v, cfg.tracker.v_max * cfg.obstacle.slow_speed_ratio),
                    reason="obstacle_approach_slow",
                )
            cmd_v, cmd_w = command.v, command.w
            bot.set_car_motion(cmd_v, cmd_w)
            last_cmd_v, last_cmd_w = cmd_v, cmd_w

            summary = command_summary(command, fit)
            summary["t"] = round(now - start, 2)
            summary["motion_v"] = round(motion.linear, 4)
            summary["motion_w"] = round(motion.angular, 4)
            summary["motion_source"] = motion.source
            summary["path_strategy"] = memory_status.mode
            summary["path_strategy_active"] = memory_status.active
            summary["path_strategy_reason"] = memory_status.reason
            summary["path_strategy_remaining_m"] = round(memory_status.remaining_m, 4)
            summary["path_strategy_intent_dir"] = memory_status.intent_dir
            summary["path_strategy_points"] = memory_status.point_count
            summary["path_strategy_target"] = memory_status.target
            summary["camera_to_axle_m"] = round(cfg.path_memory.camera_to_axle_m, 4)
            summary["obstacle_state"] = obstacle_decision.state.value
            summary["obstacle_conf"] = round(obstacle_decision.confidence, 3)
            summary["obstacle_armed"] = obstacle_armed
            candidate = obstacle_decision.evidence.candidate
            summary["obstacle_cue"] = None if candidate is None else candidate.cue
            summary["obstacle_bbox"] = None if candidate is None else candidate.bbox
            debug.record(
                summary, frame, crop, mask, features, cfg, track_center,
                raw_crop=raw_crop,
                obstacle_decision=obstacle_decision,
                raw_track_center=raw_track_center,
                strategy_status=memory_status,
            )
            if now - last_log >= args.log_period:
                print(json.dumps(summary, ensure_ascii=False))
                last_log = now

            if args.display:
                overlay = draw_debug_overlay(crop, features, cfg.vision.trigger_y_frac, crop_center=track_center)
                frame[y0:y1, x0:x1] = cv.resize(overlay, (x1 - x0, y1 - y0))
                cv.imshow("race_runner", frame)
                if cv.waitKey(1) & 0xFF == 27:
                    break
            time.sleep(args.period)
    finally:
        stop_chassis(bot)
        debug.close()
        cap.release()
        if args.display:
            cv.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the unified Transbot race tracker.")
    parser.add_argument("--config", default=str(ROOT / "configs/race_config.json"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-sec", type=float, default=60.0)
    parser.add_argument("--period", type=float, default=0.075)
    parser.add_argument("--log-period", type=float, default=0.4)
    parser.add_argument("--light", type=int, default=80)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-dir", default="", help="Save runtime frames and telemetry JSONL to this directory.")
    parser.add_argument("--debug-frame-period", type=float, default=0.5)
    parser.add_argument("--debug-jpeg-quality", type=int, default=82)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
