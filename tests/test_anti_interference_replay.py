import unittest
from pathlib import Path

import cv2 as cv

from transbot_race.config import RaceConfig
from transbot_race.vision import fit_line_trajectory, scan_line_features


FIXTURES = Path(__file__).parent / "fixtures" / "anti_interference"


class AntiInterferenceReplayTests(unittest.TestCase):
    def fit_fixture(self, name):
        mask = cv.imread(str(FIXTURES / name), cv.IMREAD_GRAYSCALE)
        self.assertIsNotNone(mask)
        cfg = RaceConfig()
        features = scan_line_features(mask, cfg.vision, crop_center=85.0)
        return fit_line_trajectory(
            features,
            cfg.vision,
            crop_center=85.0,
            crop_width=mask.shape[1],
        )

    def test_confirmed_reflection_paths_are_observable_but_not_controllable(self):
        names = (
            "false_far_20260714-174902_00040_0023470.png",
            "false_far_20260714-174902_00041_0024050.png",
        )
        for name in names:
            with self.subTest(name=name):
                fit = self.fit_fixture(name)
                self.assertTrue(fit.found)
                self.assertGreaterEqual(fit.nearest_band_index, 3)
                self.assertFalse(fit.has_near_support)
                self.assertFalse(fit.control_valid)
                self.assertFalse(fit.quadratic)
                self.assertGreater(abs(fit.theta), 0.9)
                self.assertGreater(fit.e0, -0.95)

    def test_same_run_near_track_remains_controllable(self):
        fit = self.fit_fixture(
            "valid_near_20260714-174902_00008_0005620.png",
        )
        self.assertTrue(fit.control_valid)
        self.assertEqual(fit.nearest_band_index, 2)
        self.assertLess(abs(fit.e0), 0.1)
        self.assertLess(abs(fit.theta), 0.1)


if __name__ == "__main__":
    unittest.main()
