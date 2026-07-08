# Transbot SE Course Project

This repository stores the working Transbot SE debug tools, tested parameters,
and course-state-machine experiments for the July 2026 race-track task.

## Current Baseline

- Main local web tuner: `apps/robot_tune_app.py`
- Current tested state: `configs/robot_tune_state.json`
- Useful standalone scripts: `scripts/`
- Race document: `docs/race/赛道说明-final.pdf`
- Selected debug outputs: `artifacts/baseline/`

## Known Status

- Straight-line navigation is the most reliable component.
- Curves and circles can be handled by the same local line-center tracking loop.
- Right-angle turning works but still needs a cleaner bidirectional state machine.
- Left-turn detection is not yet merged.
- Dashed-line handling has not been formalized yet.
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

