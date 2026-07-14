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

from transbot_race.config import RaceConfig, coerce_bool  # noqa: E402
from transbot_race.control import (  # noqa: E402
    CommandArbiter,
    ControlOwner,
    StopCause,
    TransitionEvent,
)
from transbot_race.capture_geometry import (  # noqa: E402
    CaptureGeometryFilter,
    CornerGeometryFilter,
    RingEntryGeometryFilter,
    analyze_capture_geometry,
    draw_capture_geometry,
)
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
from transbot_race.mission import (  # noqa: E402
    CourseSession,
    DetectorKind,
    ExecutorKind,
    FixedSessionMission,
    MissionEvent,
)
from transbot_race.motor import MotorGateway  # noqa: E402
from transbot_race.ring_entry import (  # noqa: E402
    RingEntryExecutor,
    RingEntryResult,
    RingPhaseEvent,
    selected_path_fit,
)
from transbot_race.state_machine import (  # noqa: E402
    MotionCommand,
    RaceState,
    RaceStateMachine,
    command_summary,
)
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
    min_h = cfg.vision.band_count * 12
    if crop_h < min_h:
        raise ValueError(
            f"crop height {crop_h}px too small for band_count={cfg.vision.band_count} "
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
    if cfg.path_memory.mode not in ("none", "corner_event", "fixed_sessions"):
        raise ValueError(f"unsupported path-memory mode: {cfg.path_memory.mode}")
    if cfg.mission.ring_entry_direction not in (-1, 1):
        raise ValueError("mission ring-entry direction must be -1 (left) or +1 (right)")
    if cfg.mission.ring_exit_direction not in (-1, 1):
        raise ValueError("mission ring-exit direction must be -1 (left) or +1 (right)")
    if cfg.mission.fork_branch not in {"left", "middle", "right"}:
        raise ValueError("mission.fork_branch must be left, middle, or right")
    if cfg.path_memory.mode == "fixed_sessions":
        # One route-direction source of truth in the new mode. The legacy
        # geometry extractor reads this older field internally.
        cfg.path_memory.roundabout_direction = cfg.mission.ring_entry_direction
    if cfg.path_memory.corner_confirm_frames <= 0:
        raise ValueError("corner confirmation frame count must be positive")
    if cfg.path_memory.corner_replay_max_w <= 0.0 or cfg.path_memory.corner_turn_angle_rad <= 0.0:
        raise ValueError("corner turn angle and angular speed must be positive")
    if not 0.0 < cfg.path_memory.corner_max_turn_angle_rad <= 3.141592653589793:
        raise ValueError("corner maximum turn angle must be within (0, pi]")
    if cfg.path_memory.corner_max_turn_angle_rad < cfg.path_memory.corner_reacquire_angle_rad:
        raise ValueError("corner maximum turn angle must exceed reacquire angle")
    if cfg.path_memory.roundabout_direction not in (-1, 1):
        raise ValueError("roundabout_direction must be -1 (left) or +1 (right)")
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


def _corner_control_allowed(
    strategy_mode: str,
    obstacle_stop_required: bool,
    state_machine: RaceStateMachine,
) -> bool:
    obstacle_stop_latched = (
        state_machine.state == RaceState.STOPPED
        and state_machine.last_event == "obstacle"
    )
    return bool(
        strategy_mode in {"corner_event", "fixed_sessions"}
        and not obstacle_stop_required
        and not obstacle_stop_latched
    )


def _corner_has_motor_ownership(state: str, pending_takeover: bool = False) -> bool:
    """Single arbitration rule for every corner shape, including forks."""
    return bool(state not in {"armed", "cooldown"} or pending_takeover)


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
        self.geometry_dir = self.root / "geometry"
        for path in (self.frames_dir, self.crops_dir, self.overlays_dir, self.masks_dir, self.obstacles_dir, self.geometry_dir):
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
        geometry_observation=None, geometry_debug=None,
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
        if geometry_observation is not None and geometry_debug is not None:
            geometry_overlay = draw_capture_geometry(
                geometry_debug,
                geometry_observation,
                cfg.path_memory.corner_gate_y_frac,
                event_kind=summary.get("geometry_event"),
                session=summary.get("course_session"),
            )
            cv.imwrite(str(self.geometry_dir / f"{stem}.jpg"), geometry_overlay, params)

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
                    "geometry": "geometry/",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
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
    sm = RaceStateMachine(cfg)
    corner_margin = CornerCommandDelay(
        cfg.path_memory,
        handoff_conf_min=cfg.tracker.conf_predict,
        tracker_cfg=cfg.tracker,
    )
    mission = FixedSessionMission(cfg.mission) if cfg.path_memory.mode == "fixed_sessions" else None
    ring_entry = RingEntryExecutor(
        cfg.mission.ring_entry_direction,
        margin_distance_m=cfg.path_memory.camera_to_axle_m,
        tracker_cfg=cfg.tracker,
    )
    legacy_geometry_filter = CaptureGeometryFilter(cfg.path_memory.corner_confirm_frames)
    session_geometry_filters = {
        DetectorKind.CORNER: CornerGeometryFilter(cfg.path_memory.corner_confirm_frames),
        DetectorKind.RING_ENTRY: RingEntryGeometryFilter(cfg.path_memory.corner_confirm_frames),
    }
    obstacle_monitor = ObstacleMonitor(cfg.obstacle)
    arbiter = CommandArbiter(
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
            mask = preprocess_blackline(crop, cfg.vision, anchor_x=track_center)
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

            geometry_observation = None
            geometry_decision = None
            geometry_debug = None
            if mission is not None:
                cfg.path_memory.roundabout_direction = (
                    cfg.mission.ring_exit_direction
                    if mission.session == CourseSession.RING_EXIT
                    else cfg.mission.ring_entry_direction
                )
            geometry_active = corner_margin.state in {"armed", "approach"}
            if (
                strategy_mode in {"corner_event", "fixed_sessions"}
                and cfg.path_memory.capture_geometry_enabled
                and geometry_active
                and (mission is None or mission.detector_enabled)
            ):
                geometry_observation, geometry_debug = analyze_capture_geometry(frame, cfg)
                active_geometry_filter = (
                    session_geometry_filters[mission.detector_kind]
                    if mission is not None else legacy_geometry_filter
                )
                geometry_decision = active_geometry_filter.update(geometry_observation)

            fit = visual_fit
            memory_status = PathStrategyStatus(False, strategy_mode, "disabled")
            if cfg.path_memory.enabled:
                if strategy_mode == "none":
                    memory_status = PathStrategyStatus(False, "none", "passthrough")
                elif strategy_mode == "corner_event":
                    memory_status = PathStrategyStatus(True, "corner_event", corner_margin.state)
                elif strategy_mode == "fixed_sessions" and mission is not None:
                    memory_status = PathStrategyStatus(
                        mission.detector_enabled,
                        "fixed_sessions",
                        mission.session.value,
                    )
            fork_straight_phase = bool(
                corner_margin.event_shape == "fork"
                and corner_margin.state in {"approach", "waiting"}
            )
            ring_session = bool(
                mission is not None
                and mission.executor_kind == ExecutorKind.RING_ENTRY
            )
            route_fit = selected_path_fit(
                geometry_debug,
                geometry_observation,
                frame_center_x=x0 + track_center,
                control_width=crop_w,
            ) if ring_session else None
            effective_geometry_decision = geometry_decision
            if mission is not None:
                effective_geometry_decision = mission.gate(
                    effective_geometry_decision,
                    geometry_observation,
                    visual_fit,
                    incoming_ready=sm.can_take_moving_handoff(visual_fit),
                )
            mission_geometry_decision = effective_geometry_decision
            ring_result = None
            ring_travelled_for_log = None
            ring_clear_for_log = None
            if ring_session:
                ring_entry.set_direction(cfg.path_memory.roundabout_direction)
                if obstacle_decision.stop_required and ring_entry.state != "waiting":
                    ring_result = RingEntryResult(
                        ring_entry.last_fit,
                        "ring_executor_held_by_obstacle",
                    )
                else:
                    ring_result = ring_entry.step(
                        geometry_observation,
                        route_fit,
                        now=now,
                        linear=motion.linear,
                        accepted_entry=effective_geometry_decision is not None,
                        # A raw far-field candidate is observation only. Motor
                        # ownership begins exclusively after the mission gate.
                        confirmed_entry=False,
                        cruise_fit=visual_fit,
                        incoming_v=last_cmd_v,
                        incoming_w=last_cmd_w,
                    )
                ring_travelled_for_log = ring_entry.travelled_m
                ring_clear_for_log = ring_entry.clear_frames

            if not ring_session and corner_margin.state == "armed" and sm.state == RaceState.STOPPED:
                # A stopped cruise controller may observe geometry, but must
                # first recover a trustworthy line before a new event can take
                # motor ownership.
                effective_geometry_decision = None
            pending_corner_takeover = bool(
                not ring_session
                and corner_margin.will_accept_geometry(effective_geometry_decision)
            )
            corner_owns_chassis = _corner_has_motor_ownership(
                corner_margin.state,
                pending_corner_takeover,
            ) if not ring_session else False
            if ring_session and ring_result is not None:
                fit = ring_result.fit or visual_fit
                if obstacle_decision.stop_required and ring_entry.state != "waiting":
                    command = MotionCommand(
                        last_cmd_v, last_cmd_w, ring_result.reason, RaceState.TRACK, None,
                    )
                elif ring_result.fit is not None:
                    ring_v, ring_w = ring_entry.control(
                        ring_result.fit,
                        now=now,
                        v_max=cfg.tracker.v_max,
                        k_pursuit=cfg.tracker.k_pursuit,
                        k_theta=cfg.tracker.k_theta,
                        max_w=cfg.path_memory.roundabout_replay_max_w,
                        approach_max_w=cfg.path_memory.corner_approach_max_w,
                        invert_turn=cfg.tracker.invert_turn,
                    )
                    command = MotionCommand(
                        ring_v,
                        ring_w,
                        ring_result.reason,
                        RaceState.TRACK,
                        None,
                    )
                elif ring_entry.state == "waiting":
                    command = sm.step(visual_fit, now=now, obstacle=False)
                elif ring_entry.state == "margin":
                    command = MotionCommand(
                        ring_entry.margin_v,
                        ring_entry.margin_w,
                        ring_result.reason,
                        RaceState.TRACK,
                        None,
                    )
                else:
                    # Once a route has been committed, silently switching back
                    # to the generic near-line tracker may select the other
                    # branch. Stop if the selected skeleton is lost too long.
                    command = MotionCommand(
                        0.0,
                        0.0,
                        ring_result.reason,
                        RaceState.STOPPED,
                        None,
                    )
                memory_status = PathStrategyStatus(
                    True,
                    "fixed_sessions",
                    ring_result.reason,
                    max(0.0, ring_entry.min_distance_m - ring_entry.travelled_m),
                    cfg.mission.ring_entry_direction,
                    ring_entry.clear_frames,
                )
                if ring_entry.state == "margin":
                    memory_status = replace(
                        memory_status,
                        remaining_m=ring_entry.margin_remaining_m,
                    )
                if (
                    ring_result.phase_event
                    in {RingPhaseEvent.ENTRY_ESTABLISHED, RingPhaseEvent.EXECUTOR_COMPLETED}
                    and mission is not None
                ):
                    next_session = mission.transition(MissionEvent.PHASE_COMPLETED)
                    for session_filter in session_geometry_filters.values():
                        session_filter.reset()
                    legacy_geometry_filter.reset()
                    if next_session == CourseSession.RING_EXIT:
                        # Entry completion is only a topology phase boundary.
                        # Keep the same filtered path and controller command
                        # while circulating; generic cruise cannot own a ring.
                        memory_status = replace(
                            memory_status,
                            reason="ring_inside_tracking",
                        )
                    else:
                        if ring_result.fit is not None:
                            command = sm.reacquire_from(ring_result.fit, now)
                        ring_entry.reset()
            elif corner_owns_chassis:
                # Do not even advance the cruise TRACK/LOST/PIVOT state while
                # a turn is active.  Its output used to be overwritten later,
                # but its hidden state still timed out or flipped search/pivot
                # direction, then leaked back at handoff.
                command = MotionCommand(
                    last_cmd_v if obstacle_decision.stop_required else 0.0,
                    last_cmd_w if obstacle_decision.stop_required else 0.0,
                    "corner_executor_held" if obstacle_decision.stop_required else "corner_owned",
                    sm.state,
                    None,
                )
            else:
                command = sm.step(
                    fit, now=now, obstacle=False,
                )
            if (
                not ring_session
                and
                _corner_control_allowed(strategy_mode, obstacle_decision.stop_required, sm)
                and (mission is None or mission.detector_enabled)
            ):
                corner_state_before = corner_margin.state
                delayed = corner_margin.step(
                    visual_fit, features, command.v, command.w, now,
                    motion.linear, motion.angular,
                    geometry=geometry_observation,
                    geometry_decision=effective_geometry_decision,
                    invert_turn=cfg.tracker.invert_turn,
                    angular_scale=(
                        cfg.path_memory.corner_command_yaw_scale
                        if motion.source == "command_fallback"
                        else 1.0
                    ),
                )
                memory_status = delayed.status
                if mission is not None:
                    memory_status = replace(memory_status, mode="fixed_sessions")
                    if (
                        mission.executor_kind == ExecutorKind.CORNER
                        and corner_state_before == "armed"
                        and corner_margin.state == "armed"
                    ):
                        if mission_geometry_decision is not None:
                            fixed_reason = (
                                "corner_waiting_cruise_recovery"
                                if sm.state == RaceState.STOPPED
                                else "corner_waiting_control_prerequisite"
                            )
                        elif geometry_decision is not None:
                            fixed_reason = mission.last_gate_reason
                        elif (
                            geometry_observation is not None
                            and geometry_observation.kind != "straight_or_unknown"
                        ):
                            fixed_reason = f"corner_observing_{geometry_observation.kind}"
                        else:
                            fixed_reason = "corner_waiting_geometry"
                        memory_status = replace(memory_status, reason=fixed_reason)
                if (
                    memory_status.transition_event == TransitionEvent.HANDOFF_READY
                    or (corner_state_before != "armed" and corner_margin.state == "armed")
                ):
                    for session_filter in session_geometry_filters.values():
                        session_filter.reset()
                    legacy_geometry_filter.reset()
                if memory_status.transition_event == TransitionEvent.HANDOFF_READY:
                    command = sm.reacquire_from(visual_fit, now)
                    if mission is not None:
                        mission.transition(MissionEvent.PHASE_COMPLETED)
                        # Session progress itself prevents duplicate triggers.
                        # Begin the next detector fresh instead of cooldown.
                        corner_margin = CornerCommandDelay(
                            cfg.path_memory,
                            handoff_conf_min=cfg.tracker.conf_predict,
                            tracker_cfg=cfg.tracker,
                        )
                        for session_filter in session_geometry_filters.values():
                            session_filter.reset()
                        legacy_geometry_filter.reset()
                elif corner_state_before == "approach" and corner_margin.state == "armed":
                    # Approach rejection is an ownership hand-back, not a
                    # continuation of the cruise filter that was frozen before
                    # the event.  Rebuild it atomically from the current view.
                    sm.reset()
                    command = sm.step(
                        visual_fit,
                        now=now,
                        obstacle=obstacle_decision.stop_required,
                    )
                elif delayed.v != command.v or delayed.w != command.w:
                    command = replace(
                        command,
                        v=delayed.v,
                        w=delayed.w,
                        reason=f"corner_{memory_status.reason}",
                    )
            if ring_session and ring_entry.state != "waiting":
                control_owner = ControlOwner.RING_EXECUTOR
            elif corner_owns_chassis:
                control_owner = ControlOwner.CORNER_EXECUTOR
            else:
                control_owner = ControlOwner.CRUISE

            stop_cause = None
            transition_event = memory_status.transition_event
            if mission is not None and mission.session == CourseSession.FINISHED:
                stop_cause = StopCause.MISSION_FINISHED
            elif obstacle_decision.stop_required:
                stop_cause = StopCause.OBSTACLE
                if control_owner != ControlOwner.CRUISE:
                    transition_event = TransitionEvent.EXECUTOR_HELD
            elif (
                control_owner == ControlOwner.RING_EXECUTOR
                and ring_entry.state in {"tracking", "inside", "exiting"}
                and ring_result is not None
                and ring_result.fit is None
            ):
                stop_cause = StopCause.ROUTE_LOST
                transition_event = TransitionEvent.ROUTE_LOST
            elif control_owner == ControlOwner.CORNER_EXECUTOR and corner_margin.state in {
                "failed", "failed_locked",
            }:
                stop_cause = StopCause.EXECUTOR_FAILED
            elif sm.state == RaceState.STOPPED and sm.last_event == "search_timeout":
                stop_cause = StopCause.SEARCH_TIMEOUT

            arbitration = arbiter.resolve(
                command,
                owner=control_owner,
                stop_cause=stop_cause,
                slow_v_limit=(
                    cfg.tracker.v_max * cfg.obstacle.slow_speed_ratio
                    if obstacle_decision.slow_required else None
                ),
                transition_event=transition_event,
            )
            command = arbitration.final
            cmd_v, cmd_w = command.v, command.w
            bot.set_car_motion(cmd_v, cmd_w)
            last_cmd_v, last_cmd_w = cmd_v, cmd_w

            summary = command_summary(command, fit)
            summary["t"] = round(now - start, 2)
            summary["motion_v"] = round(motion.linear, 4)
            summary["motion_w"] = round(motion.angular, 4)
            summary["motion_source"] = motion.source
            summary["path_strategy"] = memory_status.mode
            summary["mission_state"] = None if mission is None else mission.session.value
            summary["mission_transition_event"] = (
                None
                if mission is None or mission.last_transition_event is None
                else mission.last_transition_event.value
            )
            summary["control_owner"] = arbitration.owner.value
            summary["owner_epoch"] = arbitration.owner_epoch
            summary["executor_phase"] = (
                ring_entry.state
                if arbitration.owner == ControlOwner.RING_EXECUTOR
                else corner_margin.state
                if arbitration.owner == ControlOwner.CORNER_EXECUTOR
                else sm.state.value
            )
            summary["safety_state"] = arbitration.safety_state.value
            summary["stop_cause"] = (
                None if arbitration.stop_cause is None else arbitration.stop_cause.value
            )
            summary["transition_event"] = arbitration.transition_event.value
            summary["safety_override"] = arbitration.safety_override
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
            summary["course_session"] = None if mission is None else mission.session.value
            summary["session_detector"] = (
                None if mission is None else mission.detector_kind.value
            )
            summary["session_executor"] = (
                None if mission is None else mission.executor_kind.value
            )
            summary["path_strategy_active"] = memory_status.active
            summary["path_strategy_reason"] = memory_status.reason
            summary["path_strategy_remaining_m"] = round(memory_status.remaining_m, 4)
            summary["path_strategy_intent_dir"] = memory_status.intent_dir
            summary["path_strategy_points"] = memory_status.point_count
            summary["path_strategy_target"] = memory_status.target
            summary["turn_exit_latched"] = corner_margin.exit_latch.latched
            summary["turn_exit_last_e"] = (
                round(corner_margin.exit_latch.last_e, 4)
                if corner_margin.exit_latch.latched else None
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
            summary["geometry_controller_eligible"] = bool(
                pending_corner_takeover
                or (ring_result is not None and ring_result.fit is not None)
            )
            summary["geometry_gate_reason"] = (
                None if mission is None else mission.last_gate_reason
            )
            summary["geometry_candidate_directions"] = (
                [] if geometry_debug is None else list(geometry_debug.candidate_directions)
            )
            summary["ring_entry_state"] = (
                None
                if ring_result is None
                else "completed" if ring_result.completed else ring_entry.state
            )
            summary["ring_phase_event"] = (
                None if ring_result is None else ring_result.phase_event.value
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
            summary["fork_straight_phase"] = fork_straight_phase
            summary["roundabout_direction"] = cfg.path_memory.roundabout_direction
            summary["roundabout_margin_enabled"] = cfg.path_memory.roundabout_margin_enabled
            summary["roundabout_turn_w"] = cfg.path_memory.roundabout_replay_max_w
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
                geometry_observation=geometry_observation,
                geometry_debug=geometry_debug,
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
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
