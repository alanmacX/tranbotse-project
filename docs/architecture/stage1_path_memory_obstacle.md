# Stage 1: Axle-Frame Path Memory and Obstacle Stop

## Path memory

The existing vision extractor remains responsible for finding the line. When a
metric bird's-eye calibration is active, its selected `features.path` is mapped
to `(forward, left)` coordinates relative to the chassis turning center.

Recent path snapshots are propagated with measured linear and angular velocity.
The snapshot reaching closest to the axle supplies a metric lookahead target and
is converted back to the existing `TrajectoryFit` interface. No corner detector
or timed corner maneuver is introduced.

The effective longitudinal offset is:

```text
camera_to_axle_physical_m + tracking_offset_m
```

Increasing it delays the turn. `lookahead_m` controls tracking smoothness and is
kept separate from the physical offset.

Path memory automatically bypasses itself unless both it and the perspective
transform are enabled. This preserves the original tracker before calibration.

## Metric calibration

The A4 calibration requires the measured ground distance from the camera's
vertical projection to the paper's near edge. This anchors the BEV longitudinal
origin; paper dimensions provide scale. Click order is bottom-left, top-left,
top-right, bottom-right.

## Obstacle monitor

Obstacle detection uses the raw perspective crop, never the black-line mask or
the BEV image. It combines chroma, dark-object, neutral horizontal-edge, and line
occlusion evidence inside the predicted straight corridor.

The detector is armed only after stable straight tracking. A first candidate is
`SUSPECT`; two-of-three confirmation is required for slowdown or stop. A short
candidate-local hold keeps detection alive after the object occludes the line,
without leaving the detector armed through a later corner.

Command priority is:

```text
obstacle STOP > obstacle slowdown > buffered path > ordinary visual fit
```

Stage 1 stops for a confirmed near obstacle and does not attempt a go-around.
