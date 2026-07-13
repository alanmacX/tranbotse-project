import unittest

from apps.race_debug_app import _deep_update_cfg
from transbot_race.config import RaceConfig


class DebugConfigTests(unittest.TestCase):
    def test_path_memory_values_are_applied(self):
        cfg = RaceConfig()
        _deep_update_cfg(cfg, {
            "path_memory": {
                "camera_to_axle_physical_m": 0.135,
                "tracking_offset_m": -0.015,
                "lookahead_m": 0.08,
            }
        })
        self.assertAlmostEqual(cfg.path_memory.camera_to_axle_physical_m, 0.135)
        self.assertAlmostEqual(cfg.path_memory.tracking_offset_m, -0.015)
        self.assertAlmostEqual(cfg.path_memory.effective_camera_to_axle_m, 0.12)

    def test_homography_list_loads_as_numeric_tuple(self):
        cfg = RaceConfig()
        values = [float(index) for index in range(9)]
        _deep_update_cfg(cfg, {"perspective": {"homography": values}})
        self.assertEqual(cfg.perspective.homography, tuple(values))

    def test_nested_occlusion_rects_stay_nested(self):
        cfg = RaceConfig()
        _deep_update_cfg(cfg, {"occlusion": {"rects": [[1, 2, 3, 4]]}})
        self.assertEqual(cfg.occlusion.rects, ((1, 2, 3, 4),))


if __name__ == "__main__":
    unittest.main()
