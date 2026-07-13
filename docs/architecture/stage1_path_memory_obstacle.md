# Stage 1: Distance-Delayed Steering and Obstacle Stop

## Camera-to-axle margin

The vision pipeline works directly on the configured camera crop. Each valid
`TrajectoryFit` is queued with this remaining travel distance:

```text
camera_to_axle_m
```

Commanded or measured forward velocity integrates travelled distance. The
queued heading, lookahead error, and curvature are released only after that
distance reaches zero. Increasing the value delays the turn; decreasing it
advances the turn.

Near-field lateral error remains live while the queue fills, so the chassis can
still correct its position on a straight line. This method needs no ground-plane
transform, paper calibration, or separate corner state.

Telemetry exposes `path_memory_reason`, `path_memory_queue`, and
`path_memory_remaining_m`. A normal startup reports `filling`, then
`distance_delay` after the first fit reaches the axle.

## Obstacle monitor

Obstacle detection uses the raw crop, independently of the black-line mask. It
combines chroma, dark-object, neutral horizontal-edge, and line-occlusion cues
inside the predicted straight corridor.

The detector is armed only after stable straight tracking. A first candidate is
`SUSPECT`; repeated confirmation is required for slowdown or stop. Command
priority is:

```text
obstacle STOP > obstacle slowdown > distance-delayed steering > visual fit
```

Stage 1 stops for a confirmed near obstacle and does not attempt a go-around.
