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
- `transbot_race/state_machine.py`: ordinary line-following controller with
  TRACK / LOST / STOPPED behavior. It does not own corner or roundabout phases.
- `transbot_race/mission.py`: fixed course phase and typed mission transitions.
- `transbot_race/control.py`: the single command arbiter, typed control owner,
  stop latch, safety override, and owner epoch.
- `transbot_race/path_memory.py` and `transbot_race/ring_entry.py`: dedicated
  corner and roundabout executors with typed handoff/phase events.
- `transbot_race/motor.py`: bounded motor gateway and process-wide exclusive
  lease shared by automatic and manual control.
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

The current normal-flow config uses the fixed course order (`corner ->
ring_entry -> ring_exit -> finished`). Ring entry and exit use fixed route
choices; obstacle, color-sign, and direction-sign recognition are not run.
Only the current session's event detector may request control. The arbiter
selects one owner and emits one candidate/final command pair per tick. There is
no runtime switch back to the legacy global detector.

## Recorded Manual Fallback

Run this directly on the robot when the automatic runner needs to be bypassed:

```bash
python3 apps/manual_drive_app.py --host 0.0.0.0 --port 8780
```

Open `http://<robot-ip>:8780`. Each click performs exactly one bounded forward,
backward, left-angle, or right-angle step and stops. Raw camera frames, before
and after snapshots, requested commands, measured/fallback motion samples and
action results are saved under `artifacts/manual_runs/`.

Automatic and manual runners use the same process lease. If either runner is
active, the other refuses motor access before sending a non-zero command.

For a no-motor smoke test:

```bash
python3 apps/manual_drive_app.py --dry-run --port 8780
```

Dry-run mode does not open the host computer's camera. Pass
`--allow-local-camera` only when a local camera test is explicitly desired.

## Known Status

- Straight-line navigation is the most reliable component.
- Corners and roundabouts are handled by dedicated executors; generic cruise
  is frozen while another owner controls the chassis.
- Geometry votes must be consecutive, and irreversible exit identity requires
  temporal confirmation even for a strong observation.
- Dashed / missing line is carried by the confidence filter's predict mode;
  only a sustained loss drops into LOST (search sweep, then STOPPED on timeout).
- `FINISHED`, selected-route loss, executor failure, invalid commands, and
  operator/process failures use typed stop causes. Non-recoverable causes stay
  latched until an explicit clear.
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
python3 -m pytest -q
```
