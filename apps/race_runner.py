#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import cv2 as cv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transbot_race.config import RaceConfig  # noqa: E402
from transbot_race.state_machine import RaceState, RaceStateMachine  # noqa: E402
from transbot_race.vision import draw_debug_overlay, preprocess_blackline, scan_line_features  # noqa: E402


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
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
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


class DryBot:
    def set_car_motion(self, v: float, w: float) -> None:
        print(f"DRY command v={v:.4f} w={w:.4f}")

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


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    bot = make_bot(args.dry_run)
    sm = RaceStateMachine(cfg)
    cap = cv.VideoCapture(args.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg.camera.frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cfg.camera.frame_height)
    if not cap.isOpened():
        raise RuntimeError(f"camera open failed: {args.camera}")

    if hasattr(bot, "set_floodlight"):
        bot.set_floodlight(args.light)

    start = time.monotonic()
    last_log = 0.0
    corner_started = False
    try:
        stop_chassis(bot, count=3, delay=0.03)
        while time.monotonic() - start < args.max_sec:
            ok, frame = cap.read()
            if not ok:
                bot.set_car_motion(0.0, 0.0)
                time.sleep(0.05)
                continue
            crop, (x0, y0, x1, y1), track_center = crop_frame(frame, cfg)
            mask = preprocess_blackline(crop, cfg.vision)
            features = scan_line_features(mask, cfg.vision, crop_center=track_center)
            command = sm.step(features, now=time.monotonic())
            if command.reason == "corner_forward" and args.force_turn_dir:
                sm.active_turn_dir = args.force_turn_dir
            cmd_v, cmd_w = command.v, command.w
            if args.force_turn_dir and command.state in (RaceState.TIMED_TURN, RaceState.REACQUIRE):
                cmd_w = args.force_turn_dir * cfg.corner.turn_w
            if command.reason in {"corner_forward", "turn_start", "timed_turn", "reacquire_turn"}:
                corner_started = True
            if args.stop_after_corner and corner_started and sm.last_event == "reacquired":
                stop_chassis(bot, count=8, delay=0.035)
                print(
                    json.dumps(
                        {
                            "t": round(time.monotonic() - start, 2),
                            "state": sm.state.value,
                            "reason": "corner_done_stop",
                            "found": features.found,
                            "err": round(features.err_norm, 3),
                        },
                        ensure_ascii=False,
                    )
                )
                break
            bot.set_car_motion(cmd_v, cmd_w)

            now = time.monotonic()
            if now - last_log >= args.log_period:
                print(
                    json.dumps(
                        {
                            "t": round(now - start, 2),
                            "state": sm.state.value,
                            "reason": command.reason,
                            "v": round(cmd_v, 4),
                            "w": round(cmd_w, 4),
                            "force_turn_dir": args.force_turn_dir,
                            "found": features.found,
                            "err": round(features.err_norm, 3),
                            "left": None if features.branch_left is None else round(features.branch_left.y, 1),
                            "right": None if features.branch_right is None else round(features.branch_right.y, 1),
                        },
                        ensure_ascii=False,
                    )
                )
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
        cap.release()
        if args.display:
            cv.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the integrated Transbot race controller.")
    parser.add_argument("--config", default=str(ROOT / "configs/race_config.json"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-sec", type=float, default=60.0)
    parser.add_argument("--period", type=float, default=0.075)
    parser.add_argument("--log-period", type=float, default=0.4)
    parser.add_argument("--light", type=int, default=80)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-corner", action="store_true")
    parser.add_argument("--force-turn-dir", type=float, default=0.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
