import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.manual_drive_app import CameraThread, Recorder, StepController


class ManualDriveParsingTests(unittest.TestCase):
    def setUp(self):
        self.controller = StepController.__new__(StepController)

    def test_forward_is_distance_bounded(self):
        action = self.controller.parse({"action": "forward", "distance": 99, "v": 2})
        self.assertEqual(action.target, 0.5)
        self.assertEqual(action.v, 0.08)

    def test_rotation_converts_and_bounds_degrees(self):
        action = self.controller.parse({"action": "right", "angle_deg": 999, "w": 9})
        self.assertAlmostEqual(action.target, 3.141592653589793)
        self.assertEqual(action.w, 0.4)

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.parse({"action": "coast"})


class ManualCameraSafetyTests(unittest.TestCase):
    def test_dry_run_does_not_open_local_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                camera=0, dry_run=True, allow_local_camera=False,
                width=320, height=240, record_fps=1.0,
            )
            recorder = Recorder(Path(tmp), args)
            camera = CameraThread(args, recorder)
            with patch("apps.manual_drive_app.cv.VideoCapture") as capture:
                camera.start()
                for _ in range(20):
                    if camera.latest_jpeg is not None:
                        break
                    camera.stop_event.wait(0.01)
                camera.stop_event.set()
                camera.join(timeout=1)
                capture.assert_not_called()
                self.assertIsNotNone(camera.latest_jpeg)
            recorder.close()


if __name__ == "__main__":
    unittest.main()
