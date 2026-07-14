#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from queue import Empty, Full, Queue
import sys
import threading
import time
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transbot_race.control import CommandArbiter, StopCause  # noqa: E402
from transbot_race.fallback_config import FallbackConfig, load_fallback_config  # noqa: E402
from transbot_race.fallback_mission import FallbackMission, FallbackState  # noqa: E402
from transbot_race.fan import DryRunFan, Fan, UnavailableFan  # noqa: E402
from transbot_race.motor import MotorGateway  # noqa: E402
from transbot_race.parking import ParkingObservation, ParkingTriggerDetector  # noqa: E402
from transbot_race.vision import (  # noqa: E402
    TrajectoryFit,
    fit_line_trajectory,
    preprocess_blackline,
    scan_line_features,
)


class DryBot:
    def __init__(self) -> None:
        self.commands: list[tuple[float, float]] = []

    def set_car_motion(self, v: float, w: float) -> None:
        self.commands.append((float(v), float(w)))


def make_bot(dry_run: bool):
    if dry_run:
        return DryBot()
    sys.path.insert(0, "/home/pi/Transbot/py_install")
    from Transbot_Lib import Transbot  # type: ignore

    return Transbot()


def make_fan(dry_run: bool) -> Fan:
    return DryRunFan() if dry_run else UnavailableFan()


class CameraReader:
    def __init__(self, cap: cv.VideoCapture) -> None:
        self.cap = cap
        self.frames: Queue[tuple[bool, np.ndarray | None]] = Queue(maxsize=1)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name="fallback-camera", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def read(self, timeout: float) -> tuple[bool, np.ndarray | None]:
        try:
            return self.frames.get(timeout=timeout)
        except Empty:
            return False, None

    def close(self) -> None:
        self.stopped.set()
        self.cap.release()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self.stopped.is_set():
            ok, frame = self.cap.read()
            item = (bool(ok), frame if ok else None)
            try:
                self.frames.put_nowait(item)
            except Full:
                try:
                    self.frames.get_nowait()
                except Empty:
                    pass
                self.frames.put_nowait(item)
            if not ok:
                return


def crop_frame(frame: np.ndarray, cfg: FallbackConfig) -> tuple[np.ndarray, float]:
    x0, y0, x1, y1 = cfg.camera.crop
    ex0 = max(0, x0 - cfg.camera.expand_left_px)
    ex1 = min(cfg.camera.frame_width, x1 + cfg.camera.expand_right_px)
    track_center = ((x0 + x1) / 2.0) - ex0
    return frame[y0:y1, ex0:ex1], track_center


def analyze_frame(
    frame: np.ndarray,
    cfg: FallbackConfig,
    detector: ParkingTriggerDetector,
) -> tuple[TrajectoryFit, ParkingObservation]:
    crop, track_center = crop_frame(frame, cfg)
    if crop.size == 0:
        raise ValueError("configured camera crop is empty")
    track_mask = preprocess_blackline(crop, cfg.vision, anchor_x=track_center)
    features = scan_line_features(track_mask, cfg.vision, crop_center=track_center)
    fit = fit_line_trajectory(
        features,
        cfg.vision,
        crop_center=track_center,
        crop_width=crop.shape[1],
        lookahead_frac=cfg.tracker.lookahead_frac,
    )
    parking_mask = preprocess_blackline(crop, cfg.vision, anchor_x=None)
    parking = detector.analyze(parking_mask)
    return fit, parking


def apply_step(step, arbiter: CommandArbiter, motor: MotorGateway, fan: Fan):
    result = arbiter.resolve(
        step.command,
        owner=step.owner,
        stop_cause=step.stop_cause,
        transition_event=step.transition_event,
    )
    if step.fan_requested and (result.final.v != 0.0 or result.final.w != 0.0):
        raise RuntimeError("fan request rejected because chassis command is non-zero")
    try:
        motor.set_car_motion(result.final.v, result.final.w)
    except Exception:
        fan.off()
        raise
    if step.fan_requested:
        fan.on()
    else:
        fan.off()
    return result


def dry_run_inputs(index: int, cfg: FallbackConfig) -> tuple[TrajectoryFit, ParkingObservation]:
    fit = TrajectoryFit(found=True, conf=0.9, n_bands=6)
    trigger_at = cfg.runtime.startup_line_frames + 3
    candidate = index >= trigger_at
    confirmed = index >= trigger_at + cfg.parking_trigger.confirm_frames - 1
    parking = ParkingObservation(candidate, confirmed, 0.9 if candidate else 0.0, "dry_run")
    return fit, parking


def run_dry(args: argparse.Namespace, cfg: FallbackConfig) -> dict:
    detector = ParkingTriggerDetector(cfg.parking_trigger)
    del detector
    mission = FallbackMission(cfg, allow_unvalidated_parking=args.allow_unvalidated_parking)
    arbiter = CommandArbiter(max_v=cfg.runtime.max_v_mps, max_w=cfg.runtime.max_w_radps)
    motor = MotorGateway(
        DryBot(),
        role="fallback-dry-run",
        max_v=cfg.runtime.max_v_mps,
        max_w=cfg.runtime.max_w_radps,
        lease_path=args.lease_path,
    )
    fan = DryRunFan()
    now = 0.0
    index = 0
    states: deque[str] = deque(maxlen=512)
    try:
        while now <= args.max_sec:
            fit, parking = dry_run_inputs(index, cfg)
            step = mission.step(fit=fit, parking=parking, now=now)
            apply_step(step, arbiter, motor, fan)
            states.append(step.state.value)
            if mission.state in {FallbackState.FINISHED, FallbackState.FAULT}:
                break
            if mission.state == FallbackState.PARK_TRIGGER and not args.allow_unvalidated_parking:
                break
            index += 1
            now += cfg.runtime.loop_period_sec
        else:
            step = mission.fail(now, "runtime_timeout", StopCause.EXECUTOR_FAILED)
            apply_step(step, arbiter, motor, fan)
            states.append(step.state.value)
        return {
            "mode": "dry_run",
            "state": mission.state.value,
            "event": mission.last_event,
            "fan_events": fan.events,
            "states": list(states),
            "unvalidated_parking_unlocked": args.allow_unvalidated_parking,
        }
    finally:
        fan.close()
        motor.close(stop_delay=0.0)


def run_live(args: argparse.Namespace, cfg: FallbackConfig) -> dict:
    motor: MotorGateway | None = None
    fan: Fan | None = None
    cap: cv.VideoCapture | None = None
    reader: CameraReader | None = None
    try:
        bot = make_bot(False)
        motor = MotorGateway(
            bot,
            role="fallback-auto",
            max_v=cfg.runtime.max_v_mps,
            max_w=cfg.runtime.max_w_radps,
            lease_path=args.lease_path,
        )
        fan = make_fan(False)
        detector = ParkingTriggerDetector(cfg.parking_trigger)
        mission = FallbackMission(cfg, allow_unvalidated_parking=args.allow_unvalidated_parking)
        arbiter = CommandArbiter(max_v=cfg.runtime.max_v_mps, max_w=cfg.runtime.max_w_radps)
        cap = cv.VideoCapture(args.camera)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg.camera.frame_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cfg.camera.frame_height)
        if not cap.isOpened():
            raise RuntimeError(f"camera open failed: {args.camera}")
        reader = CameraReader(cap)
        reader.start()

        started = time.monotonic()
        while True:
            now = time.monotonic()
            if now - started > args.max_sec:
                step = mission.fail(now, "runtime_timeout", StopCause.EXECUTOR_FAILED)
                apply_step(step, arbiter, motor, fan)
                break
            remaining_sec = args.max_sec - (now - started)
            ok, frame = reader.read(min(cfg.runtime.camera_frame_timeout_sec, remaining_sec))
            now = time.monotonic()
            if not ok or frame is None:
                parking = ParkingObservation(False, False, 0.0, "camera_failure")
                step = mission.step(
                    fit=TrajectoryFit(found=False),
                    parking=parking,
                    now=now,
                    camera_ok=False,
                )
            else:
                fit, parking = analyze_frame(frame, cfg, detector)
                step = mission.step(fit=fit, parking=parking, now=now)
            result = apply_step(step, arbiter, motor, fan)
            print(json.dumps({
                "state": step.state.value,
                "event": step.event,
                "reason": result.final.reason,
                "v": result.final.v,
                "w": result.final.w,
                "parking": parking.reason,
                "fan": step.fan_requested,
            }, ensure_ascii=False))
            if mission.state in {FallbackState.FINISHED, FallbackState.FAULT}:
                break
            if mission.state == FallbackState.PARK_TRIGGER and not args.allow_unvalidated_parking:
                break
            time.sleep(cfg.runtime.loop_period_sec)
        return {"mode": "live", "state": mission.state.value, "event": mission.last_event}
    finally:
        try:
            if fan is not None:
                fan.close()
        finally:
            try:
                if motor is not None:
                    motor.close()
            finally:
                if reader is not None:
                    reader.close()
                elif cap is not None:
                    cap.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent fallback line/parking/fan runner")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fallback_course.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-sec", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unvalidated-parking",
        action="store_true",
        help="unlock low-speed timed parking calibration; not validated for unattended use",
    )
    parser.add_argument("--lease-path", type=Path, default=Path("/tmp/transbotse-fallback-motor.lock"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_fallback_config(args.config)
    if not math.isfinite(args.max_sec) or args.max_sec <= 0.0:
        raise ValueError("--max-sec must be finite and positive")
    summary = run_dry(args, cfg) if args.dry_run else run_live(args, cfg)
    print(json.dumps(summary, ensure_ascii=False))
    expected = FallbackState.FINISHED.value if args.allow_unvalidated_parking else FallbackState.PARK_TRIGGER.value
    return 0 if summary["state"] == expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
