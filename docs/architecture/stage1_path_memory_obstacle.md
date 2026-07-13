# Stage 1: Margin Strategy Experiments and Obstacle Stop

The debug dashboard exposes three mutually exclusive margin strategies. The
selected mode and all parameters are copied into every debug capture.

## No margin

`none` is the control baseline. It passes the current `visual_fit` directly to
the existing tracker without corner gating, ground projection, or path memory.

## Corner event

`corner_event` keeps the original raw-crop tracker. A unilateral branch,
preview, or confirmed heading signal latches a turn direction. The command
output stays straight until forward travel reaches `camera_to_axle_m`, while
the first few final steering commands are recorded. Those scalar commands are
then replayed with a safety limit before returning to live control. It never
queues an old `TrajectoryFit`. Simultaneous left and right branches are treated
as an undecided junction and do not latch a direction.

## IPM axle path

`ipm_axle` uses the rectangle calibration to warp the crop into a metric bird
view. The selected centerline is converted to axle-frame points, advanced with
linear/angular motion, and tracked with a metric lookahead target.

## Local path pursuit

`local_pursuit` leaves the crop, mask, and visual extraction unchanged. Only
the selected centerline points are projected through the rectangle calibration
to local ground coordinates. A rolling path is re-expressed in the current
robot frame with odometry and supplies a pure-pursuit target.

Both metric modes require four rectangle clicks in this order: bottom-left,
top-left, top-right, bottom-right. Calibration maps crop pixels directly to
ground `(forward, left)` metres. IPM warps the image; local pursuit does not.

## Obstacle priority

Obstacle detection always uses the raw crop and remains independent of the
selected margin strategy:

```text
obstacle STOP > obstacle slowdown > selected margin strategy > visual tracker
```
