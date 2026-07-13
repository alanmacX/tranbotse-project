# Turn Controller and Obstacle Stop

The dashboard exposes a cruise-only baseline and a closed-loop turn controller.
The selected mode and all parameters are copied into every debug capture.

## Cruise only

`none` is the control baseline. It passes the current `visual_fit` directly to
the existing tracker without corner gating, ground projection, or path memory.

## Closed-loop turn

`corner_event` treats a persistent rounded curve and a sharp corner as one turn
event. Geometry is measured from the chassis-anchored skeleton. The trigger
point is the first sustained departure from the incoming tangent (curvature
onset), not the noisiest/highest-curvature pixel. When that onset crosses the
near-field image gate, `camera_to_axle_m` is integrated from odometry. Despite
its legacy config name, this value means **gate-to-drive-axle longitudinal
compensation**; it is not an image margin and does not start at far detection.

After a turn is latched, the incoming-line approach command is frozen. The turn
controller exclusively owns the chassis through approach, pivot, visual
exit-line confirmation and bounded recovery. The cruise TRACK/PIVOT/LOST state machine is
not advanced in the background. A partially visible curve angle is never used
as the total turn angle; the configured angle is only a blind-search safety
envelope, and reliable first exit-line confirmation ends the maneuver earlier.

Exit acquisition uses a first-entry latch. A right-turn exit must first enter
from image-right toward the centre; a left turn is its mirror. Four coherent
near-field bands may latch even when a disconnected far fragment exists. Once
latched, the line identity cannot be replaced. If it is lost beyond the short
hold tolerance, the controller fail-stops instead of continuing around and
binding to the return leg.

The latch edge immediately commands `v=0,w=0`, cancelling the coarse pivot.
If the same line remains locally continuous on the next frame, control passes
directly to the normal cruise follower. There is no second corner-alignment
controller and no requirement that the exit already be centred or parallel.

Before the first valid line acquisition the chassis remains stationary. LOST
search rotation is enabled only after the tracker has acquired a line once, so
camera warm-up or a temporarily blank mask cannot rotate the chassis before
margin detection runs.

## Obstacle priority

Obstacle detection always uses the raw crop and remains independent of the
selected margin strategy:

```text
obstacle STOP > obstacle slowdown > selected margin strategy > visual tracker
```
