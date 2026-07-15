import glob
from pathlib import Path

import cv2 as cv
import pytest

from apps.race_runner import load_config
from transbot_race.capture_geometry import analyze_capture_geometry
from transbot_race.config import StageGeometryConfig


ROOT = Path(__file__).resolve().parents[1]
FRAME_GLOB = ROOT / "artifacts/live_debug/20260714-160851_final/frames/*.jpg"


def test_corner_trimmed_roi_preserves_formal_frame_geometry_decisions():
    paths = sorted(glob.glob(str(FRAME_GLOB)))
    if not paths:
        pytest.skip("formal frame corpus is unavailable")
    cfg = load_config(ROOT / "configs/race_config.json")
    common = StageGeometryConfig(
        cfg.path_memory.geometry_roi_top_offset_px,
        cfg.path_memory.geometry_chassis_trim_px,
    )
    for path in paths:
        frame = cv.imread(path)
        baseline, baseline_debug = analyze_capture_geometry(
            frame,
            cfg,
            include_debug_images=False,
            geometry_cfg=common,
        )
        optimized, optimized_debug = analyze_capture_geometry(
            frame,
            cfg,
            include_debug_images=False,
            geometry_cfg=cfg.corner_geometry,
        )
        for field in baseline.__dataclass_fields__:
            if field == "vertex":
                continue
            assert getattr(optimized, field) == getattr(baseline, field), (
                Path(path).name,
                field,
            )
        if baseline.vertex is None:
            assert optimized.vertex is None
        else:
            assert optimized.vertex is not None
            assert optimized.vertex[0] == baseline.vertex[0]
            assert optimized.vertex[1] + optimized_debug.roi_y0 == (
                baseline.vertex[1] + baseline_debug.roi_y0
            )
