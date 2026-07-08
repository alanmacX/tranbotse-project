# Baseline Snapshot

This snapshot preserves the useful code and parameters from the local
`/Users/macalan/Documents/bot` workspace before the integrated race logic
refactor.

## Preserved Files

- `apps/robot_tune_app.py`: local web app for camera, arm, manual motion, line
  following, and right-angle experiments.
- `configs/robot_tune_state.json`: latest local tuning state.
- `scripts/lite_blackline_mask.py`: lightweight black-line mask debugger.
- `scripts/tiny_line_seg.py`: tiny segmentation experiment for line masks.
- `scripts/hand_gimbal_demo.py`: hand-tracking gimbal demo.
- `scripts/yolo_bottle_grab_demo.py`: YOLO bottle approach/grab prototype.
- `scripts/black_adapter_grab_demo.py`: black-adapter approach/grab prototype.
- `docs/transbot_tonight_plan_notes.md`: planning and debug notes.
- `artifacts/baseline/`: selected images and JSON outputs from working tests.

## Important Parameters From Current State

```json
{
  "cam1": 90,
  "cam2": 17,
  "j1": 225,
  "j2": 30,
  "j3": 45,
  "crop": [311, 336, 432, 398],
  "line_speed": 0.06,
  "line_invert": false,
  "corner_forward_sec": 5.0,
  "corner_turn_sec": 2.55,
  "corner_trigger_frac": 0.35
}
```

## Refactor Guardrail

This commit is intentionally a preservation point. The next work should happen
after this baseline commit so changes can be reviewed against the last known
working tools and parameters.

