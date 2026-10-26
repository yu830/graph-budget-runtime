"""Synthetic CPU integration checks for orchestration boundaries, never GPU evidence."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from graphbudget.benchmark import load_workload
from graphbudget.experiment import run_case, measure_candidates


class ReachedGPUBoundary(Exception):
    pass


@contextmanager
def no_gpu():
    raise ReachedGPUBoundary
    yield


def workload():
    return {"schema_version": 1, "workload_id": "synthetic-contract", "purpose": "calibration",
            "requests": [{"request_id": "r1", "input_ids": [1, 2],
                          "max_new_tokens": 4, "planned_at_s": 0}]}


class ExperimentTests(unittest.TestCase):
    def test_case_accepts_the_measurement_controllers_loaded_workload(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "case"
            with patch("graphbudget.experiment.gpu_lease", no_gpu), self.assertRaises(ReachedGPUBoundary):
                run_case([], load_workload(workload()), output, capture_sizes=[1])
            command = json.loads((output / "invocation.json").read_text())
            self.assertEqual(command["command"][-2:], ["--cuda-graph-bs", "1"])
            self.assertEqual(len(command["workload_sha256"]), 64)

    def test_failed_candidates_retained_in_precreated_empty_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "attempt"
            output.mkdir()
            load = root / "load.json"
            load.write_text(json.dumps(workload()))
            config = {"calibration_workload": str(load),
                      "candidates": [{"id": "cap1", "capture_sizes": [1]}],
                      "server_args": [], "plan_server_args": {},
                      "budget": {"max_startup_seconds": 10}}
            source = root / "experiment.json"
            source.write_text(json.dumps(config))
            def failure(_args, _load, path, **kwargs):
                path.mkdir()
                case = {"status": "failed", "reason": "synthetic OOM fixture", "run_id": "cpu"}
                (path / "case.json").write_text(json.dumps(case))
                return case
            with patch("graphbudget.experiment.run_case", side_effect=failure):
                measurements = measure_candidates(source, output)
            self.assertEqual(measurements[0]["status"], "failed")
            self.assertIn("synthetic OOM", measurements[0]["reason"])
            self.assertIsNone(json.loads((output / "measurements.json").read_text())["identity"])
            with self.assertRaisesRegex(ValueError, "new or empty"):
                measure_candidates(source, output)


if __name__ == "__main__":
    unittest.main()
