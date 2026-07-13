import unittest

from apps.race_debug_app import _deep_update_cfg
from transbot_race.config import RaceConfig


class DebugConfigTests(unittest.TestCase):
    def test_path_memory_values_are_applied(self):
        cfg = RaceConfig()
        _deep_update_cfg(cfg, {
            "path_memory": {
                "camera_to_axle_m": 0.12,
                "max_queue_frames": 64,
            }
        })
        self.assertAlmostEqual(cfg.path_memory.camera_to_axle_m, 0.12)
        self.assertEqual(cfg.path_memory.max_queue_frames, 64)

    def test_nested_occlusion_rects_stay_nested(self):
        cfg = RaceConfig()
        _deep_update_cfg(cfg, {"occlusion": {"rects": [[1, 2, 3, 4]]}})
        self.assertEqual(cfg.occlusion.rects, ((1, 2, 3, 4),))


if __name__ == "__main__":
    unittest.main()
