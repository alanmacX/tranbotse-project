# Race State Machine Architecture

This document maps the race-track PDF requirements to a single controller
architecture. The goal is to keep straight-line tracking as the stable base
behavior and let special cases take control only through explicit events.

## Course Requirements From The PDF

Track one is a visual line-following and intelligent scheduling course:

- Obstacle: stop or avoid while following the black line.
- Right angle: detect a 90-degree turn and use a special maneuver.
- Dashed / missing line: blind-drive briefly, then recover line following.
- Roundabout: enter, follow the circular track, exit at the assigned branch.
- Material detection: stop or slow down in the gray zone and classify red,
  green, or blue.
- Fork: choose unloading zone A/B/C based on the detected color.
- Return: drive back without repeating the visual tasks, but keep the same
  roundabout direction and handle obstacle logic again.

## Design Rule

The car has one active controller at a time:

1. Emergency / obstacle stop
2. Timed special maneuver, such as a right-angle turn
3. Gap blind drive
4. Normal line following

This priority prevents "left brain fights right brain" behavior. For example,
corner detection cannot keep firing while a timed turn is already active.

## Normal Line Following

Normal tracking uses scan-line features:

- Split the crop into horizontal bands.
- Detect black runs in each band.
- Choose the run closest to the configured line center for line following.
- Compute normalized error from the near bands.
- Apply proportional steering.

Curves and round paths do not need a special "circle detector" at first. A
circle is locally just a curve, so line-center tracking follows it as long as
the curvature is within the controller's range.

## Right And Left Corners

Corner detection is now symmetric:

- Right corner: wide branch extends right from the line center.
- Left corner: wide branch extends left from the line center.
- Auto mode picks the larger confirmed branch.

The branch must cross the trigger line before it can start a timed maneuver.
This makes the trigger point stable and avoids reacting to the first distant
glimpse of a branch.

## Dashed / Missing Line

When the line disappears for several frames:

- Enter `GAP_BLIND`.
- Continue forward with reduced speed.
- Keep a small correction from the last known error.
- Return to line following when the line is found again.

This prevents the car from spinning in place inside the dashed section.

## Roundabout

Initial strategy:

- Do not detect a circle as a separate primitive.
- Keep normal line following through the circular arc.
- Add roundabout entry/exit guards later only where branch decisions matter.

The state machine can later add `ROUNDABOUT_ACTIVE` if testing shows that the
fork-like entry or exit creates ambiguous branch events.

## Color And Fork

Color detection should be gated by course state, not always active:

- Only run HSV color classification in the material zone.
- Save `target_color`.
- At the fork, select branch A/B/C based on `target_color`.

This avoids color blocks or room objects influencing early line-follow states.

## Current Implementation

Core files:

- `transbot_race/vision.py`
- `transbot_race/state_machine.py`
- `transbot_race/config.py`
- `apps/race_debug_app.py`

Legacy baseline:

- `apps/robot_tune_app.py`

The legacy app is preserved but should not be the place for new race logic.
Use the package modules for new behavior and keep tests in `tests/`.

