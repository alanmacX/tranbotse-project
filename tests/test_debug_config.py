import argparse
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2 as cv
import numpy as np

import apps.race_debug_app as debug_app
from apps.race_debug_app import (
    _archive_member_paths,
    _clear_live_frame,
    _current_run_dir,
    _deep_update_cfg,
    _pull_remote_debug,
    _publish_live_frame,
    _run_summaries,
)
from apps.race_runner import DebugRecorder, LiveFramePublisher
from transbot_race.config import RaceConfig
from transbot_race.path_memory import PathStrategyStatus
from transbot_race.vision import LineFeatures


class DebugConfigTests(unittest.TestCase):
    def test_path_memory_values_are_applied(self):
        cfg = RaceConfig()
        _deep_update_cfg(cfg, {
            "path_memory": {
                "camera_to_axle_m": 0.12,
            }
        })
        self.assertAlmostEqual(cfg.path_memory.camera_to_axle_m, 0.12)

    def test_ground_homography_loads_as_tuple(self):
        cfg = RaceConfig()
        values = [float(index) for index in range(9)]
        _deep_update_cfg(cfg, {"ground_projection": {"homography": values}})
        self.assertEqual(cfg.ground_projection.homography, tuple(values))

    def test_mission_route_is_applied(self):
        cfg = RaceConfig()
        _deep_update_cfg(cfg, {"mission": {"ring_entry_direction": -1}})
        self.assertEqual(cfg.mission.ring_entry_direction, -1)

    def test_nested_occlusion_rects_stay_nested(self):
        cfg = RaceConfig()
        _deep_update_cfg(cfg, {"occlusion": {"rects": [[1, 2, 3, 4]]}})
        self.assertEqual(cfg.occlusion.rects, ((1, 2, 3, 4),))


class LiveFramePublisherTests(unittest.TestCase):
    def test_publisher_emits_decodable_rate_limited_frame(self):
        publisher = LiveFramePublisher(True, period=0.2, jpeg_quality=72)
        frame = np.full((24, 32, 3), 127, dtype=np.uint8)
        output = io.StringIO()
        with patch("sys.stdout", output):
            publisher.publish(frame, now=1.0)
            publisher.publish(frame, now=1.1)
            publisher.close()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        prefix, timestamp, payload = lines[0].split(" ", 2)
        self.assertEqual(prefix, "LIVE_FRAME")
        self.assertEqual(timestamp, "1.000000")
        import base64

        image = cv.imdecode(
            np.frombuffer(base64.b64decode(payload), dtype=np.uint8),
            cv.IMREAD_COLOR,
        )
        self.assertEqual(image.shape[:2], (24, 32))
        self.assertIsNone(publisher._thread)


class DebugRecorderTests(unittest.TestCase):
    def test_recorder_writes_telemetry_and_drains_image_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                debug_frame_period=0.1,
                debug_jpeg_quality=82,
                max_sec=1.0,
                period=0.075,
                log_period=0.4,
                camera=0,
            )
            cfg = RaceConfig()
            recorder = DebugRecorder(tmp, cfg, args)
            frame = np.full((48, 64, 3), 127, dtype=np.uint8)
            crop = frame[8:40, 8:56]
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            recorder.record(
                {"t": 0.0, "geometry_event": None, "course_session": "corner"},
                frame,
                crop,
                mask,
                LineFeatures(found=False),
                cfg,
                crop_center=24.0,
                strategy_status=PathStrategyStatus(False, "fixed_sessions", "test"),
            )
            recorder.close()

            root = Path(tmp)
            rows = (root / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["frames_saved"], 1)
            self.assertEqual(manifest["image_drop_count"], 0)
            self.assertEqual(manifest["image_write_error_count"], 0)
            frame_paths = list((root / "frames").glob("*.jpg"))
            self.assertEqual(len(frame_paths), 1)
            self.assertIsNotNone(cv.imread(str(frame_paths[0])))
            self.assertIsNone(recorder.image_thread)


class LiveFrameStateTests(unittest.TestCase):
    def test_clear_live_frame_releases_stale_runner_image(self):
        _publish_live_frame(b"jpeg", 1.0)
        self.assertEqual(debug_app.LIVE_FRAME_JPEG, b"jpeg")
        _clear_live_frame()
        self.assertIsNone(debug_app.LIVE_FRAME_JPEG)
        self.assertEqual(debug_app.LIVE_FRAME_TIMESTAMP, 0.0)
        self.assertEqual(debug_app.LIVE_FRAME_RECEIVED_AT, 0.0)


class DebugPullTests(unittest.TestCase):
    def test_remote_check_and_pack_are_single_ssh_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = [
                subprocess.CompletedProcess([], 0, "", ""),
                subprocess.CompletedProcess([], 0, b"archive", b""),
                subprocess.CompletedProcess([], 0, b"", b""),
            ]
            with patch("apps.race_debug_app.ROOT", Path(tmp)), patch(
                "apps.race_debug_app.subprocess.run",
                side_effect=completed,
            ) as run:
                result = _pull_remote_debug(
                    "robot",
                    "artifacts/live_debug/run-a",
                )

            self.assertEqual(
                result,
                Path(tmp) / "artifacts/live_debug/run-a",
            )
            self.assertEqual(
                run.call_args_list[0].args[0],
                [
                    "ssh",
                    "robot",
                    "test -d /home/pi/tranbotse-project/artifacts/live_debug/run-a",
                ],
            )
            self.assertEqual(run.call_args_list[1].args[0][:2], ["ssh", "robot"])
            self.assertNotIn("sh", run.call_args_list[1].args[0])


class DebugArchiveSafetyTests(unittest.TestCase):
    def test_run_list_is_sorted_and_excludes_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "live_debug"
            first = root / "run-a"
            second = root / "run-b"
            first.mkdir(parents=True)
            second.mkdir()
            (second / "manifest.json").write_text(
                json.dumps({"frames_saved": 12}),
                encoding="utf-8",
            )
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (root / "run-link").symlink_to(outside, target_is_directory=True)
            with patch("apps.race_debug_app._live_debug_root", return_value=root.resolve()):
                runs = _run_summaries()
            self.assertEqual({item["name"] for item in runs}, {"run-a", "run-b"})
            self.assertEqual(
                next(item for item in runs if item["name"] == "run-b")["frames_saved"],
                12,
            )

    def test_current_run_stays_below_live_debug_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "live_debug"
            run = root / "run-a"
            run.mkdir(parents=True)
            with patch("apps.race_debug_app._live_debug_root", return_value=root.resolve()), patch(
                "apps.race_debug_app.ROOT", Path(tmp)
            ), patch("apps.race_debug_app.CURRENT_RUN_REL", "live_debug/run-a"):
                self.assertEqual(_current_run_dir(), run.resolve())

    def test_current_run_fallback_uses_timestamped_name_not_pull_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "live_debug"
            older = root / "20260715-110040_final"
            newer = root / "20260715-111556_final"
            newer.mkdir(parents=True)
            older.mkdir()
            with patch("apps.race_debug_app._live_debug_root", return_value=root.resolve()), patch(
                "apps.race_debug_app.ROOT", Path(tmp)
            ), patch("apps.race_debug_app.CURRENT_RUN_REL", None):
                self.assertEqual(_current_run_dir(), newer.resolve())

    def test_archive_rejects_outside_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "live_debug"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            with patch("apps.race_debug_app._live_debug_root", return_value=root.resolve()):
                with self.assertRaises(ValueError):
                    _archive_member_paths(outside)

    def test_current_run_rejects_symlink_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "live_debug"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "run-link").symlink_to(outside, target_is_directory=True)
            with patch("apps.race_debug_app._live_debug_root", return_value=root.resolve()), patch(
                "apps.race_debug_app.ROOT", Path(tmp)
            ), patch("apps.race_debug_app.CURRENT_RUN_REL", "live_debug/run-link"):
                self.assertIsNone(_current_run_dir())

    def test_archive_rejects_symlink_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "live_debug"
            run = root / "run-a"
            run.mkdir(parents=True)
            outside = Path(tmp) / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            (run / "escape").symlink_to(outside)
            with patch("apps.race_debug_app._live_debug_root", return_value=root.resolve()):
                with self.assertRaises(ValueError):
                    _archive_member_paths(run)


class MainWebContractTests(unittest.TestCase):
    def test_main_ui_consumes_runner_owned_video_and_typed_status(self):
        script = (debug_app.WEB_ROOT / "app.js").read_text(encoding="utf-8")
        page = (debug_app.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('el("video").src = `/video?', script)
        self.assertNotIn("VideoCapture", script + page)
        self.assertIn("roundaboutMarginDistance", script + page)
        for field in (
            "mission_state",
            "session_detector",
            "executor_phase",
            "candidate_producer",
            "control_owner",
            "transition_barrier_state",
            "safety_state",
            "stop_cause",
            "candidate_command",
            "final_command",
            "timing_geometry_ms",
        ):
            self.assertIn(field, script)


if __name__ == "__main__":
    unittest.main()
