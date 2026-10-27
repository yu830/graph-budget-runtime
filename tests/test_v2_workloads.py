"""Check model/concurrency arguments and the held-out workload boundary on CPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_v2_workloads
import profile_metadata
import run_metadata_case


class V2WorkloadTests(unittest.TestCase):
    def test_extra_workloads_are_distinct_and_deterministic(self):
        source = ROOT / "examples/workloads/calibration.json"
        first = make_v2_workloads.make_workloads(source)
        self.assertEqual(first, make_v2_workloads.make_workloads(source))
        self.assertEqual(len(first["interactive"]["requests"]), 8)
        self.assertEqual(len(first["acceptance"]["requests"]), 24)
        with tempfile.TemporaryDirectory() as tmp:
            primary, regression = run_metadata_case.make_workloads(Path(tmp))
        used = {tuple(x["input_ids"]) for workload in (primary, regression)
                for x in workload["requests"]}
        self.assertTrue(all(tuple(x["input_ids"]) not in used
                            for x in first["acceptance"]["requests"]))
        self.assertEqual(first["acceptance"]["purpose"], "heldout_acceptance")

    def test_server_args_use_requested_capture_ceiling_and_capacity(self):
        with patch.dict("os.environ", {}, clear=True):
            args = profile_metadata.server_args("/tmp/model/revision", 8, 2048, 31234)
        self.assertEqual(args[args.index("--model-path") + 1], "/tmp/model/revision")
        self.assertEqual(args[args.index("--revision") + 1], "revision")
        self.assertEqual(args[args.index("--max-running-requests") + 1], "8")
        self.assertEqual(args[args.index("--max-total-tokens") + 1], "2048")
        start = args.index("--cuda-graph-bs") + 1
        self.assertEqual(args[start:args.index("--host")], ["1", "2", "4", "8"])
        self.assertEqual(args[args.index("--port") + 1], "31234")
        with self.assertRaises(ValueError):
            profile_metadata.server_args("/tmp/model/revision", 17, 2048)


if __name__ == "__main__":
    unittest.main()
