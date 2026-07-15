import bisect
import hashlib
import json
import unittest
from pathlib import Path

import cv2 as cv

from apps.race_runner import update_dataclass
from transbot_race.config import RaceConfig
from transbot_race.frame_pipeline import FixedCourseFramePipeline
from transbot_race.path_memory import MotionSample
from transbot_race.state_machine import MotionCommand, RaceState


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST_PATH = FIXTURES / "formal_run_manifest.json"
RUN_ROOT = ROOT / "artifacts" / "live_debug"
REQUIRED_TELEMETRY_FIELDS = {
    "mission_state",
    "geometry_raw_kind",
    "geometry_decision",
    "geometry_gate_accepted",
    "executor_phase",
    "control_owner",
    "candidate_command",
    "safety_state",
    "final_command",
    "transition_event",
}


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class FormalFixtureManifestTests(unittest.TestCase):
    def test_manifest_covers_all_six_formal_runs(self):
        manifest = load_manifest()
        self.assertEqual(manifest["schema_version"], 1)
        runs = manifest["formal_runs"]
        self.assertEqual(len(runs), 6)
        self.assertEqual(sum(run["telemetry_rows"] for run in runs), 2116)
        self.assertEqual(sum(run["saved_points"] for run in runs), 542)
        self.assertTrue(all(run["key_frames"] for run in runs))

    def test_ci_fixture_checksums_and_sources_are_stable(self):
        manifest = load_manifest()
        known_runs = {run["run"] for run in manifest["formal_runs"]}
        for fixture in manifest["ci_fixtures"]:
            with self.subTest(path=fixture["path"]):
                path = FIXTURES / fixture["path"]
                self.assertTrue(path.is_file())
                self.assertIn(fixture["source_run"], known_runs)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, fixture["sha256"])
                self.assertIsNotNone(cv.imread(str(path), cv.IMREAD_GRAYSCALE))


class OptionalFullFormalRunReplayTests(unittest.TestCase):
    def test_available_full_runs_replay_through_production_controller(self):
        manifest = load_manifest()
        replayed = 0
        for expected in manifest["formal_runs"]:
            run_dir = RUN_ROOT / expected["run"]
            required = [run_dir / "meta.json", run_dir / "telemetry.jsonl", run_dir / "frames"]
            if not all(path.exists() for path in required):
                continue
            replayed += 1
            with self.subTest(run=expected["run"]):
                self._replay_controller(run_dir, expected)
        if replayed == 0:
            self.skipTest("no complete formal run bundles are available locally")

    def test_available_full_runs_have_complete_decodable_artifacts(self):
        manifest = load_manifest()
        replayed = 0
        missing = []
        for expected in manifest["formal_runs"]:
            run_dir = RUN_ROOT / expected["run"]
            required = [run_dir / "manifest.json", run_dir / "meta.json", run_dir / "telemetry.jsonl"]
            if not all(path.is_file() for path in required):
                missing.append(expected["run"])
                continue
            replayed += 1
            with self.subTest(run=expected["run"]):
                self._verify_full_run(run_dir, expected)
        if replayed == 0:
            self.skipTest("no complete formal run bundles are available locally")
        if missing:
            print("optional formal runs unavailable: " + ", ".join(missing))

    def _replay_controller(self, run_dir, expected):
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        cfg = RaceConfig()
        update_dataclass(cfg, meta["config"])
        pipeline = FixedCourseFramePipeline(cfg)
        telemetry = [
            json.loads(line)
            for line in (run_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        telemetry_times = [round(float(row["t"]) * 1000) for row in telemetry]
        frame_paths = sorted((run_dir / "frames").glob("*.jpg"))
        self.assertEqual(len(frame_paths), expected["saved_points"])
        last_command = MotionCommand(0.0, 0.0, "replay_start", RaceState.STOPPED)
        for frame_path in frame_paths:
            elapsed_ms = int(frame_path.stem.rsplit("_", 1)[1])
            row_index = bisect.bisect_left(telemetry_times, elapsed_ms)
            row_index = min(len(telemetry) - 1, row_index)
            if row_index and abs(telemetry_times[row_index - 1] - elapsed_ms) < abs(telemetry_times[row_index] - elapsed_ms):
                row_index -= 1
            row = telemetry[row_index]
            frame = cv.imread(str(frame_path))
            self.assertIsNotNone(frame, frame_path)
            result = pipeline.step(
                frame,
                now=elapsed_ms / 1000.0,
                motion=MotionSample(
                    float(row.get("motion_v", 0.0)),
                    float(row.get("motion_w", 0.0)),
                    str(row.get("motion_source", "command_fallback")),
                ),
                last_command=last_command,
            )
            step = result.stage
            arbitration = result.arbitration
            self.assertEqual(step.candidate_producer.value, step.control_owner.value)
            self.assertEqual(arbitration.owner, step.control_owner)
            self.assertTrue(all(map(lambda value: abs(value) < float("inf"), (
                step.candidate.v,
                step.candidate.w,
                arbitration.final.v,
                arbitration.final.w,
            ))))
            if step.barrier.barrier_state.value == "active":
                self.assertEqual(step.detector, pipeline.course.mission.detector_kind)
            else:
                self.assertIn(
                    step.detector,
                    {pipeline.course.mission.detector_kind, type(step.detector).NONE},
                )
            last_command = arbitration.final

    def _verify_full_run(self, run_dir, expected):
        capture_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(capture_manifest["frames_saved"], expected["saved_points"])
        rows = [
            json.loads(line)
            for line in (run_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), expected["telemetry_rows"])
        self.assertTrue(all(REQUIRED_TELEMETRY_FIELDS <= row.keys() for row in rows))
        telemetry_ms = [round(float(row["t"]) * 1000) for row in rows]

        image_specs = {
            "frames": ".jpg",
            "crops": ".jpg",
            "masks": ".png",
            "overlays": ".jpg",
        }
        stems = None
        for directory, suffix in image_specs.items():
            paths = sorted((run_dir / directory).glob(f"*{suffix}"))
            self.assertEqual(len(paths), expected["saved_points"])
            current_stems = [path.stem for path in paths]
            stems = current_stems if stems is None else stems
            self.assertEqual(current_stems, stems)
            for path in paths:
                self.assertIsNotNone(cv.imread(str(path), cv.IMREAD_UNCHANGED), path)
        geometry = sorted((run_dir / "geometry").glob("*.jpg"))
        self.assertEqual(len(geometry), expected["geometry_points"])
        self.assertTrue(all(cv.imread(str(path)) is not None for path in geometry))

        for stem in stems:
            elapsed_ms = int(stem.rsplit("_", 1)[1])
            self.assertLessEqual(min(abs(elapsed_ms - value) for value in telemetry_ms), 1)


if __name__ == "__main__":
    unittest.main()
