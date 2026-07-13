import unittest

from apps.race_debug_app import _deep_update_cfg
from transbot_race.config import RaceConfig


class DebugConfigTests(unittest.TestCase):
    def test_no_margin_and_obstacle_switch_are_applied(self):
        cfg = RaceConfig()
        _deep_update_cfg(cfg, {"path_memory": {"mode": "none"}, "obstacle": {"enabled": False}})
        self.assertEqual(cfg.path_memory.mode, "none")
        self.assertFalse(cfg.obstacle.enabled)

    def test_path_memory_values_are_applied(self):
        cfg = RaceConfig()
        _deep_update_cfg(cfg, {
            "path_memory": {
                "camera_to_axle_m": 0.12,
                "mode": "local_pursuit",
                "path_max_points": 64,
            }
        })
        self.assertAlmostEqual(cfg.path_memory.camera_to_axle_m, 0.12)
        self.assertEqual(cfg.path_memory.mode, "local_pursuit")
        self.assertEqual(cfg.path_memory.path_max_points, 64)

    def test_ground_homography_loads_as_tuple(self):
        cfg = RaceConfig()
        values = [float(index) for index in range(9)]
        _deep_update_cfg(cfg, {"ground_projection": {"homography": values}})
        self.assertEqual(cfg.ground_projection.homography, tuple(values))

    def test_nested_occlusion_rects_stay_nested(self):
        cfg = RaceConfig()
        _deep_update_cfg(cfg, {"occlusion": {"rects": [[1, 2, 3, 4]]}})
        self.assertEqual(cfg.occlusion.rects, ((1, 2, 3, 4),))


if __name__ == "__main__":
    unittest.main()
