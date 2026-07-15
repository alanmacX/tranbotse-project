import numpy as np

from transbot_race.config import RaceConfig
from transbot_race.frame_pipeline import FixedCourseFramePipeline
from transbot_race.path_memory import MotionSample
from transbot_race.state_machine import MotionCommand, RaceState


def track_frame():
    frame = np.full((480, 640, 3), 225, dtype=np.uint8)
    frame[265:443, 359:371] = 12
    return frame


def track_frame_with_chassis_edge():
    frame = np.full((480, 640, 3), 225, dtype=np.uint8)
    frame[265:420, 359:371] = 12
    frame[420:442, 280:440] = 12
    return frame


def pipeline():
    cfg = RaceConfig()
    return FixedCourseFramePipeline(cfg)


def previous(v=0.0, w=0.0):
    return MotionCommand(v, w, "previous", RaceState.TRACK)


def test_single_frame_pipeline_enforces_one_producer_owner_and_arbiter_input():
    result = pipeline().step(
        track_frame(),
        now=0.0,
        motion=MotionSample(0.0, 0.0, "measured"),
        last_command=previous(),
    )

    assert result.ordinary_fit.control_valid
    assert result.stage.candidate_producer.value == result.stage.control_owner.value
    assert result.arbitration.owner == result.stage.control_owner
    assert result.arbitration.candidate == result.stage.candidate
    assert result.timings.total_ms >= result.timings.ordinary_vision_ms


def test_ordinary_control_crop_excludes_chassis_edge_before_first_acquisition():
    result = pipeline().step(
        track_frame_with_chassis_edge(),
        now=0.0,
        motion=MotionSample(0.0, 0.0, "measured"),
        last_command=previous(),
    )

    assert result.crop.shape[0] == 155
    assert result.ordinary_fit.n_bands >= 3
    assert not result.ordinary_fit.disconnected
    assert result.stage.candidate.reason != "await_first_line"
    assert result.arbitration.final.v > 0.0
