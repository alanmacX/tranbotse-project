#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import base64
import json
import queue
import signal
import sys
import threading
import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path

import cv2 as cv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transbot_race.config import RaceConfig, coerce_bool  # noqa: E402
from transbot_race.capture_geometry import (  # noqa: E402
    draw_capture_geometry,
)
from transbot_race.frame_pipeline import FixedCourseFramePipeline  # noqa: E402
from transbot_race.path_memory import (  # noqa: E402
    PathStrategyStatus,
    read_motion_sample,
)
from transbot_race.mission import DetectorKind  # noqa: E402
from transbot_race.motor import MotorGateway  # noqa: E402
from transbot_race.state_machine import (  # noqa: E402
    MotionCommand,
    RaceState,
    command_summary,
)
from transbot_race.vision import (  # noqa: E402
    draw_debug_overlay,
)


_OPERATOR_CLEAR_REQUESTS = 0


def _request_operator_clear(_signum: int, _frame: object) -> None:
    global _OPERATOR_CLEAR_REQUESTS
    _OPERATOR_CLEAR_REQUESTS += 1


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
        elif isinstance(current, bool):
            setattr(obj, key, coerce_bool(value))
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
    if not 0 <= cfg.camera.control_bottom_trim_px < crop_h:
        raise ValueError(
            "camera.control_bottom_trim_px must be non-negative and smaller "
            "than the crop height"
        )
    control_h = crop_h - cfg.camera.control_bottom_trim_px
    min_h = cfg.vision.band_count * 12
    if control_h < min_h:
        raise ValueError(
            f"control crop height {control_h}px too small for "
            f"band_count={cfg.vision.band_count} "
            f"(need >= {min_h}px). Check camera.crop={cfg.camera.crop}."
        )
    if x1 <= x0:
        raise ValueError(f"crop x1 must be > x0, got {cfg.camera.crop}")
    if not 0 <= cfg.vision.percentile <= 100:
        raise ValueError("vision percentile must be within [0, 100]")
    if not 0 <= cfg.vision.threshold_min <= cfg.vision.threshold_max <= 255:
        raise ValueError("vision thresholds must satisfy 0 <= min <= max <= 255")
    if cfg.vision.component_min_thickness_px < 0.0:
        raise ValueError("vision component thickness cannot be negative")
    if cfg.path_memory.camera_to_axle_m < 0.0:
        raise ValueError("camera-to-axle distance cannot be negative")
    if cfg.path_memory.roundabout_margin_distance_m < 0.0:
        raise ValueError("roundabout margin distance cannot be negative")
    if cfg.path_memory.roundabout_entry_commit_v < 0.0:
        raise ValueError("roundabout entry commit speed cannot be negative")
    if cfg.path_memory.roundabout_entry_commit_w < 0.0:
        raise ValueError("roundabout entry commit turn rate cannot be negative")
    if cfg.path_memory.roundabout_entry_capture_frames < 1:
        raise ValueError("roundabout entry capture frames must be positive")
    if cfg.path_memory.roundabout_entry_commit_max_distance_m < 0.0:
        raise ValueError("roundabout entry commit distance cannot be negative")
    if cfg.path_memory.roundabout_entry_commit_max_frames < 1:
        raise ValueError("roundabout entry commit frame limit must be positive")
    if cfg.mission.ring_entry_direction not in (-1, 1):
        raise ValueError("mission ring-entry direction must be -1 (left) or +1 (right)")
    if cfg.mission.ring_exit_direction not in (-1, 1):
        raise ValueError("mission ring-exit direction must be -1 (left) or +1 (right)")
    if cfg.path_memory.corner_confirm_frames <= 0:
        raise ValueError("corner confirmation frame count must be positive")
    if cfg.path_memory.corner_replay_max_w <= 0.0 or cfg.path_memory.corner_turn_angle_rad <= 0.0:
        raise ValueError("corner turn angle and angular speed must be positive")
    if not 0.0 < cfg.path_memory.corner_max_turn_angle_rad <= 3.141592653589793:
        raise ValueError("corner maximum turn angle must be within (0, pi]")
    if cfg.path_memory.corner_max_turn_angle_rad < cfg.path_memory.corner_reacquire_angle_rad:
        raise ValueError("corner maximum turn angle must exceed reacquire angle")
    if cfg.path_memory.roundabout_replay_max_w <= 0.0:
        raise ValueError("roundabout angular speed must be positive")
    if cfg.tracker.max_w <= 0.0 or cfg.tracker.max_w_slew_rate <= 0.0:
        raise ValueError("tracker angular limits must be positive")
    if not 0.0 <= cfg.path_memory.corner_reacquire_angle_rad <= cfg.path_memory.corner_turn_angle_rad:
        raise ValueError("corner reacquire angle must be within [0, turn angle]")
    if cfg.path_memory.corner_command_yaw_scale <= 0.0:
        raise ValueError("corner command yaw scale must be positive")
    if cfg.path_memory.corner_reacquire_max_e <= 0.0 or cfg.path_memory.corner_reacquire_max_theta <= 0.0:
        raise ValueError("corner visual reacquire limits must be positive")
    if cfg.path_memory.corner_image_angle_gain <= 0.0 or cfg.path_memory.corner_search_extra_rad < 0.0:
        raise ValueError("corner angle gain must be positive and extra search angle non-negative")
    if cfg.path_memory.corner_reacquire_confirm_frames <= 0:
        raise ValueError("corner reacquire frame count must be positive")
    if cfg.path_memory.max_motion_dt_sec <= 0.0:
        raise ValueError("path-memory motion interval must be positive")
    if not 0.0 < cfg.path_memory.corner_gate_y_frac < 1.0:
        raise ValueError("corner geometry gate must be within (0, 1)")
    if cfg.path_memory.corner_gate_confirm_frames <= 0:
        raise ValueError("corner geometry gate confirmation must be positive")
    if cfg.path_memory.corner_approach_max_w < 0.0:
        raise ValueError("corner approach angular speed cannot be negative")
    if cfg.path_memory.corner_approach_missing_frames < 0:
        raise ValueError("corner approach missing-frame tolerance cannot be negative")
    if cfg.path_memory.corner_align_missing_frames < 0:
        raise ValueError("corner alignment missing-frame tolerance cannot be negative")
    if cfg.path_memory.corner_exit_confirm_frames <= 0:
        raise ValueError("corner exit confirmation must be positive")
    if cfg.path_memory.corner_exit_predict_sec <= 0.0:
        raise ValueError("corner exit prediction duration must be positive")
    if not 0.0 < cfg.path_memory.corner_exit_predict_v_ratio <= 1.0:
        raise ValueError("corner exit prediction speed ratio must be within (0, 1]")
    if cfg.path_memory.corner_align_v <= 0.0 or cfg.path_memory.corner_align_max_w <= 0.0:
        raise ValueError("corner alignment motion must be positive")
    if cfg.path_memory.corner_cruise_ready_frames <= 0:
        raise ValueError("corner cruise-ready confirmation must be positive")
    if cfg.path_memory.corner_cruise_ready_distance_m < 0.0:
        raise ValueError("corner cruise-ready distance cannot be negative")
    if cfg.path_memory.geometry_roi_top_offset_px < 0 or cfg.path_memory.geometry_chassis_trim_px <= 0:
        raise ValueError("capture geometry ROI values are invalid")
    for name, stage_geometry in (
        ("corner_geometry", cfg.corner_geometry),
        ("ring_entry_geometry", cfg.ring_entry_geometry),
        ("ring_exit_geometry", cfg.ring_exit_geometry),
    ):
        if not 0 <= stage_geometry.roi_top_offset_px <= cfg.path_memory.geometry_roi_top_offset_px:
            raise ValueError(f"{name}.roi_top_offset_px must fit inside common geometry ROI")
        if stage_geometry.chassis_trim_px <= 0 or stage_geometry.cadence_frames <= 0:
            raise ValueError(f"{name} trim/cadence values must be positive")


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


class LiveFramePublisher:
    """Rate-limited latest-frame publisher with encoding off the control loop."""

    def __init__(self, enabled: bool, period: float, jpeg_quality: int) -> None:
        self.enabled = enabled
        self.period = max(0.1, float(period))
        self.jpeg_quality = max(40, min(90, int(jpeg_quality)))
        self.last_publish_at = -1e9
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def publish(self, frame, now: float) -> None:
        if not self.enabled or now - self.last_publish_at < self.period:
            return
        item = (frame.copy(), float(now))
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)
        self.last_publish_at = now

    def close(self) -> None:
        if self._thread is None:
            return
        self.enabled = False
        self._stop.set()
        self._thread.join(timeout=2.0)
        if not self._thread.is_alive():
            self._thread = None

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            frame, now = item
            ok, encoded = cv.imencode(
                ".jpg",
                frame,
                [int(cv.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                continue
            payload = base64.b64encode(encoded).decode("ascii")
            print(f"LIVE_FRAME {now:.6f} {payload}", flush=True)


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
        self.image_queue: queue.Queue = queue.Queue(maxsize=2)
        self.image_thread = None
        self.image_sequence = 0
        self.image_drop_count = 0
        self.image_write_error_count = 0

        if not self.enabled or self.root is None:
            return

        self.frames_dir = self.root / "frames"
        self.crops_dir = self.root / "crops"
        self.overlays_dir = self.root / "overlays"
        self.masks_dir = self.root / "masks"
        self.geometry_dir = self.root / "geometry"
        for path in (self.frames_dir, self.crops_dir, self.overlays_dir, self.masks_dir, self.geometry_dir):
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
        self.image_thread = threading.Thread(target=self._write_images, daemon=True)
        self.image_thread.start()
        print(f"debug_capture_dir={self.root}", file=sys.stderr, flush=True)

    def should_capture(self, elapsed: float) -> bool:
        return bool(
            self.enabled
            and self.root is not None
            and float(elapsed) - self.last_frame_t >= self.frame_period
        )

    def record(
        self, summary: dict, frame, crop, mask, features, cfg: RaceConfig, crop_center: float,
        strategy_status: PathStrategyStatus | None = None,
        geometry_observation=None, geometry_debug=None,
    ) -> None:
        if not self.enabled or self.root is None:
            return

        if self.telemetry is not None:
            self.telemetry.write(json.dumps(summary, ensure_ascii=False) + "\n")

        elapsed = float(summary.get("t", 0.0))
        if not self.should_capture(elapsed):
            return

        if self.image_queue.full():
            try:
                self.image_queue.get_nowait()
                self.image_queue.task_done()
                self.image_drop_count += 1
            except queue.Empty:
                pass
        stem = f"{self.image_sequence:05d}_{int(elapsed * 1000):07d}"
        self.image_sequence += 1
        self.image_queue.put_nowait({
            "stem": stem,
            "frame": frame.copy(),
            "crop": crop.copy(),
            "mask": mask.copy(),
            "features": features,
            "cfg": cfg,
            "crop_center": crop_center,
            "strategy_status": strategy_status,
            "geometry_observation": geometry_observation,
            "geometry_debug": geometry_debug,
            "geometry_event": summary.get("geometry_event"),
            "course_session": summary.get("course_session"),
        })
        self.last_frame_t = elapsed

    def _write_images(self) -> None:
        params = [int(cv.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while True:
            item = self.image_queue.get()
            if item is None:
                self.image_queue.task_done()
                return
            try:
                stem = item["stem"]
                overlay = draw_debug_overlay(
                    item["crop"],
                    item["features"],
                    item["cfg"].vision.trigger_y_frac,
                    crop_center=item["crop_center"],
                )
                strategy_status = item["strategy_status"]
                if strategy_status is not None:
                    label = f"{strategy_status.mode}: {strategy_status.reason}"
                    if strategy_status.remaining_m > 0.0:
                        label += f" {strategy_status.remaining_m:.3f}m"
                    cv.rectangle(
                        overlay,
                        (0, 0),
                        (min(overlay.shape[1], 300), 24),
                        (0, 0, 0),
                        -1,
                    )
                    cv.putText(
                        overlay,
                        label,
                        (6, 17),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 220, 255),
                        1,
                        cv.LINE_AA,
                    )

                cv.imwrite(str(self.frames_dir / f"{stem}.jpg"), item["frame"], params)
                cv.imwrite(str(self.crops_dir / f"{stem}.jpg"), item["crop"], params)
                cv.imwrite(str(self.overlays_dir / f"{stem}.jpg"), overlay, params)
                cv.imwrite(str(self.masks_dir / f"{stem}.png"), item["mask"])
                if (
                    item["geometry_observation"] is not None
                    and item["geometry_debug"] is not None
                ):
                    geometry_overlay = draw_capture_geometry(
                        item["geometry_debug"],
                        item["geometry_observation"],
                        item["cfg"].path_memory.corner_gate_y_frac,
                        event_kind=item["geometry_event"],
                        session=item["course_session"],
                    )
                    cv.imwrite(
                        str(self.geometry_dir / f"{stem}.jpg"),
                        geometry_overlay,
                        params,
                    )
                self.frame_count += 1
            except Exception as exc:
                self.image_write_error_count += 1
                print(f"debug_image_write_error={exc!r}", file=sys.stderr, flush=True)
            finally:
                self.image_queue.task_done()

    def close(self) -> None:
        if not self.enabled or self.root is None:
            return
        self.enabled = False
        if self.telemetry is not None:
            self.telemetry.close()
            self.telemetry = None
        if self.image_thread is not None:
            try:
                self.image_queue.put_nowait(None)
            except queue.Full:
                try:
                    self.image_queue.get_nowait()
                    self.image_queue.task_done()
                    self.image_drop_count += 1
                except queue.Empty:
                    pass
                self.image_queue.put_nowait(None)
            self.image_thread.join(timeout=2.0)
            if not self.image_thread.is_alive():
                self.image_thread = None
        with (self.root / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "duration_sec": round(time.time() - self.started_at, 3),
                    "frames_saved": self.frame_count,
                    "image_drop_count": self.image_drop_count,
                    "image_write_error_count": self.image_write_error_count,
                    "telemetry": "telemetry.jsonl",
                    "frames": "frames/",
                    "crops": "crops/",
                    "overlays": "overlays/",
                    "masks": "masks/",
                    "geometry": "geometry/",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")


def run(args: argparse.Namespace) -> int:
    global _OPERATOR_CLEAR_REQUESTS
    cfg = load_config(Path(args.config))
    signal.signal(signal.SIGUSR1, _request_operator_clear)
    handled_clear_requests = _OPERATOR_CLEAR_REQUESTS
    bot = MotorGateway(
        make_bot(args.dry_run),
        role="auto",
        max_v=cfg.tracker.v_max,
        max_w=max(
            cfg.tracker.max_w,
            cfg.tracker.w_search,
            cfg.tracker.w_pivot,
            cfg.tracker.preview_turn_w,
            cfg.path_memory.corner_replay_max_w,
            cfg.path_memory.corner_align_max_w,
            cfg.path_memory.roundabout_replay_max_w,
        ),
    )
    frame_pipeline = FixedCourseFramePipeline(cfg)
    fixed_course = frame_pipeline.course
    mission = fixed_course.mission
    cap = cv.VideoCapture(args.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg.camera.frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cfg.camera.frame_height)
    if not cap.isOpened():
        raise RuntimeError(f"camera open failed: {args.camera}")
    debug = DebugRecorder(args.debug_dir, cfg, args)
    live_frames = LiveFramePublisher(
        args.publish_live_frames,
        args.live_frame_period,
        args.live_frame_jpeg_quality,
    )

    if hasattr(bot, "set_floodlight"):
        bot.set_floodlight(args.light)

    start = time.monotonic()
    last_log = 0.0
    last_cmd_v = 0.0
    last_cmd_w = 0.0
    previous_loop_total_ms = 0.0
    previous_deadline_miss = False
    previous_deadline_lag_ms = 0.0
    try:
        stop_chassis(bot, count=3, delay=0.03)
        next_deadline = time.monotonic()
        while time.monotonic() - start < args.max_sec:
            next_deadline += max(0.0, float(args.period))
            loop_started_at = time.monotonic()
            ok, frame = cap.read()
            capture_finished_at = time.monotonic()
            if not ok:
                bot.set_car_motion(0.0, 0.0)
                time.sleep(0.05)
                continue
            motion = read_motion_sample(bot, last_cmd_v, last_cmd_w)
            now = time.monotonic()
            capture_geometry_debug = debug.should_capture(now - start)
            pipeline_result = frame_pipeline.step(
                frame,
                now=now,
                motion=motion,
                last_command=MotionCommand(
                    last_cmd_v,
                    last_cmd_w,
                    "previous_final",
                    RaceState.TRACK,
                    None,
                ),
                capture_geometry_debug=capture_geometry_debug,
            )
            crop = pipeline_result.crop
            x0, y0, x1, y1 = pipeline_result.crop_rect
            track_center = pipeline_result.track_center
            mask = pipeline_result.mask
            features = pipeline_result.features
            visual_fit = pipeline_result.ordinary_fit
            fixed_step = pipeline_result.stage
            vision_finished_at = capture_finished_at + (
                pipeline_result.timings.ordinary_vision_ms / 1000.0
            )
            stage_started_at = vision_finished_at
            operator_clear_event = None
            if _OPERATOR_CLEAR_REQUESTS != handled_clear_requests:
                handled_clear_requests = _OPERATOR_CLEAR_REQUESTS
                if frame_pipeline.clear_route_loss():
                    operator_clear_event = "route_lost_cleared"
                else:
                    operator_clear_event = "clear_rejected"
            geometry_observation = fixed_step.geometry_observation
            geometry_decision = fixed_step.geometry_decision
            geometry_debug = fixed_step.geometry_debug
            geometry_finished_at = (
                stage_started_at + pipeline_result.timings.stage_ms / 1000.0
            )

            fit = fixed_step.fit
            memory_status = fixed_step.memory_status
            route_fit = fixed_step.route_fit
            mission_geometry_decision = geometry_decision
            barrier_step = fixed_step.barrier
            candidate_producer = fixed_step.candidate_producer
            ring_result = fixed_step.ring_result
            ring_executor = (
                getattr(fixed_course.component, "executor", None)
                if fixed_step.detector in {DetectorKind.RING_ENTRY, DetectorKind.RING_EXIT}
                else None
            )
            ring_travelled_for_log = (
                None
                if ring_executor is None
                else ring_executor.travelled_m
            )
            ring_clear_for_log = (
                None
                if ring_executor is None
                else ring_executor.clear_frames
            )

            command = fixed_step.candidate
            arbitration = pipeline_result.arbitration
            command = arbitration.final
            cmd_v, cmd_w = command.v, command.w
            bot.set_car_motion(cmd_v, cmd_w)
            control_finished_at = time.monotonic()
            last_cmd_v, last_cmd_w = cmd_v, cmd_w

            active_component = fixed_course.component
            corner_executor = (
                getattr(active_component, "executor", None)
                if fixed_step.detector == DetectorKind.CORNER
                else None
            )
            ring_executor = (
                getattr(active_component, "executor", None)
                if fixed_step.detector in {DetectorKind.RING_ENTRY, DetectorKind.RING_EXIT}
                else None
            )
            summary = command_summary(command, fit)
            summary["t"] = round(now - start, 2)
            summary["motion_v"] = round(motion.linear, 4)
            summary["motion_w"] = round(motion.angular, 4)
            summary["motion_source"] = motion.source
            summary["timing_capture_ms"] = round(
                (capture_finished_at - loop_started_at) * 1000.0, 3,
            )
            summary["timing_ordinary_vision_ms"] = round(
                (vision_finished_at - capture_finished_at) * 1000.0, 3,
            )
            summary["timing_geometry_ms"] = round(
                (geometry_finished_at - stage_started_at) * 1000.0, 3,
            )
            summary["timing_control_ms"] = round(
                (control_finished_at - geometry_finished_at) * 1000.0, 3,
            )
            summary["timing_control_path_ms"] = round(
                (control_finished_at - loop_started_at) * 1000.0, 3,
            )
            summary["timing_loop_total_ms"] = previous_loop_total_ms
            summary["deadline_miss"] = previous_deadline_miss
            summary["deadline_lag_ms"] = previous_deadline_lag_ms
            summary["path_strategy"] = memory_status.mode
            summary["mission_state"] = mission.session.value
            summary["mission_transition_event"] = (
                None
                if mission.last_transition_event is None
                else mission.last_transition_event.value
            )
            summary["candidate_producer"] = candidate_producer.value
            summary["control_owner"] = arbitration.owner.value
            summary["transition_barrier_state"] = fixed_step.barrier.barrier_state.value
            summary["stationary_method"] = (
                None
                if barrier_step is None
                else barrier_step.stationary_method.value
            )
            summary["stationary_samples"] = (
                0 if barrier_step is None else barrier_step.stationary_samples
            )
            summary["stationary_elapsed_sec"] = (
                0.0
                if barrier_step is None
                else round(barrier_step.stationary_elapsed_sec, 3)
            )
            summary["stationary_fallback_reason"] = (
                None if barrier_step is None else barrier_step.fallback_reason
            )
            summary["owner_epoch"] = arbitration.owner_epoch
            summary["executor_phase"] = fixed_step.executor_phase
            summary["safety_state"] = arbitration.safety_state.value
            summary["stop_cause"] = (
                None if arbitration.stop_cause is None else arbitration.stop_cause.value
            )
            summary["transition_event"] = arbitration.transition_event.value
            summary["safety_override"] = arbitration.safety_override
            summary["operator_clear_event"] = operator_clear_event
            summary["candidate_command"] = {
                "v": round(arbitration.candidate.v, 4),
                "w": round(arbitration.candidate.w, 4),
                "reason": arbitration.candidate.reason,
            }
            summary["final_command"] = {
                "v": round(arbitration.final.v, 4),
                "w": round(arbitration.final.w, 4),
                "reason": arbitration.final.reason,
            }
            summary["course_session"] = mission.session.value
            summary["session_detector"] = fixed_step.detector.value
            summary["session_executor"] = mission.executor_kind.value
            summary["path_strategy_active"] = memory_status.active
            summary["path_strategy_reason"] = memory_status.reason
            summary["path_strategy_remaining_m"] = round(memory_status.remaining_m, 4)
            summary["path_strategy_intent_dir"] = memory_status.intent_dir
            summary["path_strategy_points"] = memory_status.point_count
            summary["path_strategy_target"] = memory_status.target
            summary["turn_exit_latched"] = bool(
                corner_executor is not None
                and corner_executor.exit_latch.latched
            )
            summary["turn_exit_last_e"] = (
                round(corner_executor.exit_latch.last_e, 4)
                if corner_executor is not None
                and corner_executor.exit_latch.latched
                else None
            )
            summary["camera_to_axle_m"] = round(cfg.path_memory.camera_to_axle_m, 4)
            summary["turn_entry_delay_m"] = round(cfg.path_memory.camera_to_axle_m, 4)
            summary["geometry_raw_kind"] = (
                None if geometry_observation is None else geometry_observation.kind
            )
            summary["geometry_kind"] = (
                None if geometry_decision is None else geometry_decision.kind
            )
            summary["geometry_direction"] = None if geometry_observation is None else geometry_observation.direction
            summary["geometry_is_fork"] = None if geometry_observation is None else geometry_observation.is_fork
            summary["geometry_angle_deg"] = None if geometry_observation is None else round(
                geometry_observation.angle_rad * 180.0 / 3.141592653589793, 2,
            )
            summary["geometry_vertex_y_frac"] = (
                None if geometry_observation is None or geometry_observation.vertex_y_frac is None
                else round(geometry_observation.vertex_y_frac, 4)
            )
            summary["geometry_turn_onset_y_frac"] = summary["geometry_vertex_y_frac"]
            summary["geometry_incoming_e"] = (
                None if geometry_observation is None
                else round(geometry_observation.incoming_e, 4)
            )
            summary["geometry_incoming_theta"] = (
                None if geometry_observation is None
                else round(geometry_observation.incoming_theta, 4)
            )
            summary["geometry_decision"] = summary["geometry_kind"]
            summary["geometry_event"] = summary["geometry_kind"]
            summary["geometry_gate_accepted"] = mission_geometry_decision is not None
            summary["geometry_observation_age_frames"] = (
                fixed_step.geometry_age_frames
            )
            summary["geometry_controller_eligible"] = bool(
                mission_geometry_decision is not None
                or (ring_result is not None and ring_result.fit is not None)
            )
            summary["geometry_gate_reason"] = mission.last_gate_reason
            summary["geometry_candidate_directions"] = (
                [] if geometry_debug is None else list(geometry_debug.candidate_directions)
            )
            summary["ring_entry_state"] = (
                None
                if ring_result is None
                else "completed"
                if ring_result.completed
                else None if ring_executor is None else ring_executor.state
            )
            summary["ring_phase_event"] = (
                None if ring_result is None else ring_result.phase_event.value
            )
            summary["ring_observed_now"] = bool(
                geometry_observation is not None and route_fit is not None
            )
            summary["ring_route_memory_available"] = bool(
                ring_executor is not None and ring_executor.last_fit is not None
            )
            summary["ring_entry_selected_e_look"] = (
                None if route_fit is None else round(route_fit.e_look, 4)
            )
            summary["ring_entry_selected_theta"] = (
                None if route_fit is None else round(route_fit.theta, 4)
            )
            summary["ring_entry_travelled_m"] = (
                None if ring_travelled_for_log is None else round(ring_travelled_for_log, 4)
            )
            summary["ring_entry_clear_frames"] = (
                ring_clear_for_log
            )
            summary["roundabout_direction"] = (
                None if ring_executor is None else ring_executor.direction
            )
            summary["roundabout_margin_enabled"] = cfg.path_memory.roundabout_margin_enabled
            summary["ring_effective_margin_m"] = (
                0.0
                if ring_executor is None
                else round(ring_executor.margin_distance_m, 4)
            )
            summary["ring_margin_remaining_m"] = (
                round(ring_executor.margin_remaining_m, 4)
                if ring_executor is not None and ring_executor.state == "margin"
                else 0.0
            )
            summary["ring_route_control_active"] = bool(
                ring_result is not None
                and ring_result.fit is not None
                and ring_executor is not None
                and ring_executor.state in {"tracking", "inside", "exiting"}
            )
            summary["roundabout_turn_w"] = cfg.path_memory.roundabout_replay_max_w
            summary["debug_queue_depth"] = debug.image_queue.qsize() if debug.enabled else 0
            summary["debug_image_drop_count"] = debug.image_drop_count
            summary["debug_image_write_error_count"] = debug.image_write_error_count
            debug.record(
                summary, frame, crop, mask, features, cfg, track_center,
                strategy_status=memory_status,
                geometry_observation=geometry_observation,
                geometry_debug=geometry_debug,
            )
            if now - last_log >= args.log_period:
                print(json.dumps(summary, ensure_ascii=False))
                last_log = now
            live_frames.publish(frame, now)

            if args.display:
                overlay = draw_debug_overlay(crop, features, cfg.vision.trigger_y_frac, crop_center=track_center)
                frame[y0:y1, x0:x1] = cv.resize(overlay, (x1 - x0, y1 - y0))
                cv.imshow("race_runner", frame)
                if cv.waitKey(1) & 0xFF == 27:
                    break
            loop_finished_at = time.monotonic()
            previous_loop_total_ms = round(
                (loop_finished_at - loop_started_at) * 1000.0,
                3,
            )
            previous_deadline_miss = loop_finished_at > next_deadline
            previous_deadline_lag_ms = round(
                max(0.0, loop_finished_at - next_deadline) * 1000.0,
                3,
            )
            sleep_sec = next_deadline - loop_finished_at
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            else:
                # Do not accumulate permanent lag after one slow frame.
                next_deadline = loop_finished_at
    finally:
        stop_chassis(bot)
        live_frames.close()
        debug.close()
        cap.release()
        bot.close(stop_count=1, stop_delay=0.0)
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
    parser.add_argument("--publish-live-frames", action="store_true")
    parser.add_argument("--live-frame-period", type=float, default=0.15)
    parser.add_argument("--live-frame-jpeg-quality", type=int, default=72)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
