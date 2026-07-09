# Transbot SE Course Project

This repository stores the working Transbot SE debug tools, tested parameters,
and course-state-machine experiments for the July 2026 race-track task.

## Current Baseline

- Main local web tuner: `apps/robot_tune_app.py`
- Current tested state: `configs/robot_tune_state.json`
- Useful standalone scripts: `scripts/`
- Race document: `docs/race/赛道说明-final.pdf`
- Selected debug outputs: `artifacts/baseline/`

## Integrated Race Logic

New clean logic lives in:

- `transbot_race/vision.py`: black-line preprocessing, scan-line features, and
  `fit_line_trajectory()` (continuous e0/theta/kappa/conf estimate).
- `transbot_race/state_machine.py`: unified continuous tracker with only three
  states (TRACK / LOST / STOPPED). Corners of any angle, dashed lines and
  roundabouts emerge from one control law plus a confidence filter, not from
  per-case states or timed open-loop maneuvers.
- `transbot_race/config.py`: typed configuration (single `tracker` section).
- `apps/race_debug_app.py`: merged debug frontend for saved-image analysis plus
  optional live deploy/start/stop/video controls.
- `apps/race_runner.py`: integrated on-robot runner for camera + Transbot motion.
- `tests/test_race_state_machine.py` and `tests/test_trajectory_fit.py`: unit
  tests for the tracker (straight/offset/pivot/gap/lost) and the fitter.

The known-good tuner in `apps/robot_tune_app.py` is treated as the preserved
baseline. New race behavior should be integrated through `transbot_race/*`,
`apps/race_debug_app.py`, and `apps/race_runner.py` so the proven straight-line
workflow remains untouched.

Run the no-SSH debug frontend:

```bash
python3 apps/race_debug_app.py
```

Open:

```text
http://127.0.0.1:8776
```

Run the integrated controller on the robot, once SSH/live access is available
and the code is copied to the robot:

```bash
python3 apps/race_runner.py --config configs/race_config.json --max-sec 60
```

For import/config smoke tests without touching motors:

```bash
python3 apps/race_runner.py --dry-run --max-sec 1
```

## Known Status

- Straight-line navigation is the most reliable component.
- Curves, circles and corners are all handled by the same continuous
  trajectory tracker (line-center + heading + curvature feedback).
- Sharp bends (including right angles) trigger the `pivot` sub-mode of TRACK:
  near-zero speed + strong steering until re-aligned. There is no separate
  corner state and no timed open-loop turn.
- Dashed / missing line is carried by the confidence filter's predict mode;
  only a sustained loss drops into LOST (search sweep, then STOPPED on timeout).
- No SSH or live robot access is assumed for the next refactor phase.
- Remaining tuning is field calibration only (see
  `docs/architecture/unified_tracker_plan.md`, Phase 4).

## Run Local Tuning App

```bash
python3 apps/robot_tune_app.py
```

Open:

```text
http://127.0.0.1:8765
```

The app expects the robot to be reachable through the local SSH alias
`yahboom` when live actions are used.

## Test

Use the bundled or local Python environment with OpenCV installed:

```bash
python3 -m unittest discover -s tests -v
```
