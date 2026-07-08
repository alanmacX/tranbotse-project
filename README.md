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

- `transbot_race/vision.py`: black-line preprocessing and scan-line features.
- `transbot_race/state_machine.py`: one owner for line follow, gap blind drive,
  timed turns, and line reacquisition.
- `transbot_race/config.py`: typed configuration.
- `apps/race_debug_app.py`: no-SSH debug frontend for saved images.
- `tests/test_race_state_machine.py`: unit tests for straight line, left/right
  corners, dashed-line recovery, and thin-noise rejection.

Run the no-SSH debug frontend:

```bash
python3 apps/race_debug_app.py
```

Open:

```text
http://127.0.0.1:8776
```

## Known Status

- Straight-line navigation is the most reliable component.
- Curves and circles can be handled by the same local line-center tracking loop.
- Right-angle detection is now symmetric for left/right branches in the clean
  state machine.
- Dashed-line behavior is represented by `GAP_BLIND`.
- No SSH or live robot access is assumed for the next refactor phase.

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
