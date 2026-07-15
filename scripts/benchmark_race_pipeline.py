#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics
import sys

import cv2 as cv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.race_runner import load_config
from transbot_race.frame_pipeline import FixedCourseFramePipeline
from transbot_race.path_memory import MotionSample
from transbot_race.state_machine import MotionCommand, RaceState


def stats(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}

    def percentile(fraction: float) -> float:
        return ordered[int(round((len(ordered) - 1) * fraction))]

    return {
        "mean": round(statistics.mean(ordered), 3),
        "p50": round(percentile(0.50), 3),
        "p95": round(percentile(0.95), 3),
        "p99": round(percentile(0.99), 3),
        "max": round(ordered[-1], 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Mac/Pi fixed-course benchmark")
    parser.add_argument("--config", default=str(ROOT / "configs/race_config.json"))
    parser.add_argument(
        "--frames",
        default=str(ROOT / "artifacts/live_debug/20260714-160851_final/frames/*.jpg"),
        help="glob for recorded input frames",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--period-ms", type=float, default=75.0)
    args = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(args.frames))]
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no frames matched {args.frames!r}")

    cfg = load_config(Path(args.config))
    pipeline = FixedCourseFramePipeline(cfg)
    last = MotionCommand(0.0, 0.0, "benchmark_start", RaceState.STOPPED)
    timings = {"ordinary": [], "stage": [], "arbiter": [], "total": []}
    detector_counts: dict[str, int] = {}
    valid = 0
    for index, path in enumerate(paths):
        frame = cv.imread(str(path))
        if frame is None:
            raise SystemExit(f"cannot decode {path}")
        result = pipeline.step(
            frame,
            now=index * args.period_ms / 1000.0,
            motion=MotionSample(last.v, last.w, "command_fallback"),
            last_command=last,
        )
        valid += int(result.ordinary_fit.control_valid)
        detector = result.stage.detector.value
        detector_counts[detector] = detector_counts.get(detector, 0) + 1
        timings["ordinary"].append(result.timings.ordinary_vision_ms)
        timings["stage"].append(result.timings.stage_ms)
        timings["arbiter"].append(result.timings.arbiter_ms)
        timings["total"].append(result.timings.total_ms)
        last = result.arbitration.final

    payload = {
        "frames": len(paths),
        "ordinary_control_valid": valid,
        "detector_counts": detector_counts,
        "period_ms": args.period_ms,
        "deadline_misses": sum(value > args.period_ms for value in timings["total"]),
        "timings_ms": {name: stats(values) for name, values in timings.items()},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
