"""CPU protocol tests; no CUDA import, model launch or GPU work."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_paired_decode.py"
SPEC = importlib.util.spec_from_file_location("paired_decode", SCRIPT)
paired = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paired)


class PairedDecodeTests(unittest.TestCase):
    def test_fixed_order_and_smoke_boundary(self):
        cases = paired.schedule()
        self.assertEqual("".join(c["arm"] for c in cases), "BBAA BCCBBCCBBC BB".replace(" ", ""))
        self.assertEqual([c["pair"] for c in cases if c["stage"] == "paired"],
                         [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        self.assertEqual([c["mode"] for c in paired.schedule(True)],
                         ["upstream_gemm", "v2_residual"])

    def test_focused_order_has_independent_controls_and_no_A(self):
        cases = paired.schedule(False, "bc-focused-v1")
        self.assertEqual("".join(c["arm"] for c in cases), "BBBCCBBCCBBCBB")
        self.assertEqual([c["stage"] for c in cases[:2]], ["bb_before", "bb_before"])
        self.assertEqual([c["stage"] for c in cases[-2:]], ["bb_after", "bb_after"])
        self.assertEqual(len([c for c in cases if c["stage"] == "paired"]), 10)
        self.assertNotIn("A", [c["arm"] for c in cases])

    def test_focused_rule_requires_both_BB_controls_and_exceeds_noise(self):
        pairs = [(100, 110)] * 4 + [(100, 99)]
        self.assertFalse(paired.evaluate_bc_focused(pairs, 0.01, None, None)["candidate_rule_met"])
        self.assertTrue(paired.evaluate_bc_focused(pairs, 0.01, 0.02, 0.01)["candidate_rule_met"])
        noisy = paired.evaluate_bc_focused(pairs, 0.01, 0.11, 0.01)
        self.assertFalse(noisy["candidate_rule_met"])
        self.assertIn("NOISE_OR_DRIFT_AT_LEAST_CANDIDATE_EFFECT", noisy["review_flags"])
        drift = paired.evaluate_bc_focused([(100, 120)] * 5, 0.01, 0.02, 0.06)
        self.assertIn("POST_BB_OR_DRIFT_EXCEEDS_5_PERCENT", drift["review_flags"])
        self.assertNotIn("aa_max_absolute_change", drift)
        self.assertFalse(paired.evaluate_bc_focused([(100, 120)] * 5, 0.06, 0.01, 0.01)["candidate_rule_met"])
        self.assertFalse(paired.evaluate_bc_focused([(100, 110)] * 3 + [(100, 99)] * 2,
                                                    0.01, 0.02, 0.01)["candidate_rule_met"])

    def test_rule_requires_five_pairs_four_positive_and_strict_threshold(self):
        self.assertFalse(paired.evaluate_pairs([(100, 105)] * 5, [0.01])["candidate_rule_met"])
        self.assertTrue(paired.evaluate_pairs([(100, 106)] * 5, [0.01])["candidate_rule_met"])
        self.assertFalse(paired.evaluate_pairs([(100, 110)] * 4, [0.01])["candidate_rule_met"])
        self.assertFalse(paired.evaluate_pairs([(100, 112)] * 3 + [(100, 99)] * 2,
                                               [0.01])["candidate_rule_met"])
        self.assertFalse(paired.evaluate_pairs([(100, 110)] * 5, [0.11])["candidate_rule_met"])

    def test_control_noise_or_regression_requires_review_even_when_rule_met(self):
        report = paired.evaluate_pairs([(100, 110)] * 4 + [(100, 99)], [0.01], [0.12])
        self.assertTrue(report["candidate_rule_met"])
        self.assertIn("NOISE_OR_DRIFT_AT_LEAST_CANDIDATE_EFFECT", report["review_flags"])

    def test_identity_only_ignores_the_declared_switches(self):
        a = {"runtime_env": {**paired.SWITCHES["A"], "OMP_NUM_THREADS": "4"}, "cuda": "12.8"}
        b = {"runtime_env": {**paired.SWITCHES["B"], "OMP_NUM_THREADS": "4"}, "cuda": "12.8"}
        self.assertEqual(paired.comparable_identity(a, "A"), paired.comparable_identity(b, "B"))
        b["runtime_env"]["OMP_NUM_THREADS"] = "8"
        self.assertNotEqual(paired.comparable_identity(a, "A"), paired.comparable_identity(b, "B"))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            paired.comparable_identity(a, "B")

    def test_new_source_cannot_appear_after_freeze(self):
        protocol = {"git": {"head": "abc", "files_sha256": {"old.py": "x"}}, "workloads": {}}
        current = {"head": "abc", "files_sha256": {"old.py": "x", "new.py": "y"}}
        with mock.patch.object(paired, "git_snapshot", return_value=current):
            with self.assertRaisesRegex(ValueError, "source changed"):
                paired.assert_frozen(protocol)

    def test_real_failed_child_keeps_logs_counts_and_stops_before_next_case(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cases = paired.schedule(True)
            case_dir = output / "cases" / cases[0]["id"]
            code = ("import json,sys; from pathlib import Path; "
                    f"p=Path({str(case_dir)!r}); p.mkdir(parents=True); "
                    "(p/'primary-0.json').write_text(json.dumps({'summary':"
                    "{'failed_requests':2,'timed_out_requests':1,'not_started_requests':0}})); "
                    "print('retained stdout'); print('retained stderr',file=sys.stderr); sys.exit(7)")
            protocol = {"cases": cases, "smoke": True, "workloads": {"primary": {}, "holdout": {}},
                        "settings": {"gpu_lock": str(output / "gpu.lock"),
                                     "triton_cache_dir": str(output / "cache"), "case_pythonpath": ""}}
            with mock.patch.object(paired, "assert_frozen"), \
                 mock.patch.object(paired, "case_command", return_value=[sys.executable, "-c", code]), \
                 mock.patch.object(paired, "verify_case") as verify:
                records = paired.execute_schedule(protocol, output)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["exit_code"], 7)
            self.assertEqual(records[0]["status"], "FAILED")
            self.assertIn("retained stdout", Path(records[0]["stdout"]).read_text())
            self.assertIn("retained stderr", Path(records[0]["stderr"]).read_text())
            self.assertTrue((output / "executions" / (cases[0]["id"] + ".json")).is_file())
            verify.assert_not_called()
            counts = paired.request_failure_counts(protocol, records)
            self.assertEqual(counts["failed_requests"], 2)
            self.assertEqual(counts["timed_out_requests"], 1)
            self.assertEqual(counts["missing_or_invalid_run_files"], 1)
            report = paired.summarize(protocol, records)
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["failed_case_count"], 1)
            self.assertEqual(report["unstarted_cases"], 1)

    def test_independent_bb_gate_stops_before_formal_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cases = paired.schedule()
            protocol = {"cases": cases, "smoke": False, "workloads": {"primary": {}},
                        "settings": {"gpu_lock": str(output / "gpu.lock"),
                                     "triton_cache_dir": str(output / "cache"), "case_pythonpath": ""}}
            summaries = [({}, {"primary": {paired.METRIC: 100.0}}),
                         ({}, {"primary": {paired.METRIC: 89.0}})]
            with mock.patch.object(paired, "assert_frozen"), \
                 mock.patch.object(paired, "case_command", return_value=[sys.executable, "-c", "pass"]), \
                 mock.patch.object(paired, "verify_case", side_effect=summaries):
                records = paired.execute_schedule(protocol, output)
            gate = paired.read_json(output / "bb-gate.json")
            self.assertEqual(gate["status"], "CONTROL_FAILED")
            self.assertAlmostEqual(gate["relative_change"], -0.11)
            self.assertEqual([r["stage"] for r in records], ["bb_before", "bb_before"])
            self.assertEqual(len(list((output / "executions").glob("*.json"))), 2)
            with mock.patch.object(paired, "request_failure_counts",
                                   return_value={"missing_or_invalid_run_files": 0}):
                self.assertEqual(paired.summarize(protocol, records)["status"], "CONTROL_FAILED")

    def test_focused_gate_stops_before_any_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cases = paired.schedule(False, "bc-focused-v1")
            protocol = {"cases": cases, "smoke": False, "measurement_protocol": "bc-focused-v1",
                        "workloads": {"primary": {}},
                        "settings": {"gpu_lock": str(output / "gpu.lock"),
                                     "triton_cache_dir": str(output / "cache"), "case_pythonpath": ""}}
            summaries = [({}, {"primary": {paired.METRIC: 100.0}}),
                         ({}, {"primary": {paired.METRIC: 106.0}})]
            with mock.patch.object(paired, "assert_frozen"), \
                 mock.patch.object(paired, "case_command", return_value=[sys.executable, "-c", "pass"]), \
                 mock.patch.object(paired, "verify_case", side_effect=summaries):
                records = paired.execute_schedule(protocol, output)
            self.assertEqual(len(records), 2)
            self.assertEqual(paired.read_json(output / "bb-gate.json")["status"], "CONTROL_FAILED")
            self.assertTrue(all(record["stage"] == "bb_before" for record in records))

    def test_frozen_workloads_include_all_requests_and_explicit_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(model_path="/models/snapshots/" + "a" * 40,
                                      max_running_requests=16, max_total_tokens=4096,
                                      port=30277, gpu_lock="/tmp/test-paired.lock", smoke=False)
            snapshot = {"head": "abc", "status": "", "files_sha256": {}}
            with mock.patch.object(paired, "model_identity", return_value={"revision": "a" * 40}):
                protocol = paired.prepare_protocol(args, Path(directory), snapshot)
            self.assertEqual({k: v["requests"] for k, v in protocol["workloads"].items()},
                             {"correctness": 43, "primary": 256, "holdout": 34,
                              "interactive": 8, "acceptance": 24})
            self.assertEqual(protocol["workloads"]["correctness"]["workers"], 43)
            self.assertEqual(protocol["workloads"]["interactive"]["workers"], 1)
            self.assertEqual(protocol["workloads"]["acceptance"]["workers"], 16)
            for item in protocol["workloads"].values():
                self.assertEqual(paired.file_digest(item["path"]), item["file_sha256"])
            self.assertEqual(len(protocol["cases"]), 16)
            self.assertTrue(all(Path(p).is_absolute() for p in
                                protocol["settings"]["case_pythonpath"].split(paired.os.pathsep)))
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("sglang", sys.modules)

    def test_smoke_does_not_consume_acceptance_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(model_path="/models/snapshots/" + "a" * 40,
                                      max_running_requests=16, max_total_tokens=4096,
                                      port=30277, gpu_lock="/tmp/test-paired.lock", smoke=True)
            snapshot = {"head": "abc", "status": "", "files_sha256": {}}
            with mock.patch.object(paired, "model_identity", return_value={"revision": "a" * 40}):
                protocol = paired.prepare_protocol(args, Path(directory), snapshot)
            self.assertNotIn("acceptance", protocol["workloads"])
            self.assertEqual(protocol["workloads"]["primary"]["requests"], 64)

    def test_focused_protocol_freezes_full_warmup_and_BB_noise_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(model_path="/models/snapshots/" + "a" * 40,
                                      max_running_requests=16, max_total_tokens=4096,
                                      port=30277, gpu_lock="/tmp/test-paired.lock", smoke=False,
                                      protocol="bc-focused-v1")
            snapshot = {"head": "abc", "status": "", "files_sha256": {}}
            with mock.patch.object(paired, "model_identity", return_value={"revision": "a" * 40}):
                protocol = paired.prepare_protocol(args, Path(directory), snapshot)
            self.assertEqual(protocol["measurement_protocol"], "bc-focused-v1")
            self.assertEqual(len(protocol["cases"]), 14)
            self.assertTrue(protocol["settings"]["primary_shaped_warmup"])
            self.assertEqual(protocol["workloads"]["primary"]["requests"], 256)
            self.assertIn("B/B before", protocol["candidate_rule"]["median_gain_must_exceed"])
            self.assertNotIn("A/A", protocol["candidate_rule"]["median_gain_must_exceed"])
            command = paired.case_command(protocol["cases"][0], Path(directory), protocol, None)
            self.assertIn("--primary-shaped-warmup", command)

    def test_summary_pairs_B_and_C_even_when_C_runs_first(self):
        protocol = {"cases": paired.schedule(), "smoke": False,
                    "workloads": {name: {} for name in
                                  ("correctness", "primary", "holdout", "interactive", "acceptance")}}
        records = []
        for case in protocol["cases"]:
            rate = {"A": 80, "B": 100, "C": 110}[case["arm"]]
            summary = {paired.METRIC: rate, "ttft_s": {"p50": 0.1},
                       "request_tpot_s": {"p50": 0.01}}
            records.append({**case, "status": "PASS",
                            "workloads": {label: summary for label in protocol["workloads"]}})
        with mock.patch.object(paired, "request_failure_counts",
                               return_value={"missing_or_invalid_run_files": 0}):
            result = paired.summarize(protocol, records)
        self.assertEqual(result["status"], "PRIMARY_RULE_MET")
        self.assertTrue(result["primary_rule"]["candidate_rule_met"])
        self.assertEqual(result["primary_rule"]["positive_pairs"], 5)
        self.assertEqual([row["B"] for row in result["pairs"]], [100] * 5)
        self.assertAlmostEqual(result["primary_rule"]["median_relative_gain"], 0.1)
        self.assertEqual(result["overall_acceptance"], "NOT_ASSESSED")

    def test_focused_summary_needs_matching_precontrol_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cases = paired.schedule(False, "bc-focused-v1")
            protocol = {"cases": cases, "smoke": False, "measurement_protocol": "bc-focused-v1",
                        "workloads": {"primary": {}}}
            records = []
            for case in cases:
                rate = 100 if case["arm"] == "B" else 110
                summary = {paired.METRIC: rate, "ttft_s": {"p50": 0.1, "p95": 0.2},
                           "token_itl_s": {"p50": 0.01, "p95": 0.02},
                           "request_tpot_s": {"p50": 0.01}, "makespan_s": 1.0}
                records.append({**case, "status": "PASS",
                                "directory": str(output / "cases" / case["id"]),
                                "workloads": {"primary": summary}})
            gate = {"status": "PASS", "case_ids": [cases[0]["id"], cases[1]["id"]],
                    "throughput": [100.0, 100.0]}
            paired.write_json(output / "bb-gate.json", gate)
            with mock.patch.object(paired, "request_failure_counts",
                                   return_value={"missing_or_invalid_run_files": 0}):
                self.assertEqual(paired.summarize(protocol, records)["status"], "PRIMARY_RULE_MET")
                (output / "bb-gate.json").unlink()
                report = paired.summarize(protocol, records)
                self.assertEqual(report["status"], "NEEDS_REVIEW")
                self.assertIn("MISSING_INDEPENDENT_BB_GATE", report["review_flags"])
                paired.write_json(output / "bb-gate.json", gate)
                records[3]["workloads"]["primary"]["ttft_s"].pop("p95")
                report = paired.summarize(protocol, records)
                self.assertEqual(report["status"], "NEEDS_REVIEW")
                self.assertIn("primary:ttft_s.p95:MISSING_OR_INVALID", report["review_flags"])

    def test_focused_rejects_changed_workload_or_blocking_cuda(self):
        for extra in (["--max-running-requests", "4"], ["--max-total-tokens", "2048"],
                      ["--transport", "inherit"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as raised:
                paired.main(["--output", "/unused", "--model-path", "/unused",
                             "--protocol", "bc-focused-v1", *extra])
            self.assertEqual(raised.exception.code, 2)
        with mock.patch.dict(paired.os.environ, {"CUDA_LAUNCH_BLOCKING": "1"}):
            with self.assertRaises(SystemExit):
                paired.main(["--output", "/unused", "--model-path", "/unused",
                             "--protocol", "bc-focused-v1"])

    def test_actual_case_validation_rejects_changed_workers_and_missing_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory)
            sg = {"base_commit": "base", "patch_sha256": "patch",
                  "srt_tree_sha256": "tree", "patched_files": {"srt/x.py": "x"}}
            identity = {"graphbudget_sha256": "gb", "sglang": sg, "model": {"revision": "model"},
                        "runtime_env": paired.SWITCHES["B"]}
            work = {"requests": [{"request_id": "q", "input_ids": [1], "max_new_tokens": 1}]}
            run = {"kind": "graphbudget_benchmark_run",
                   "run": {"run_id": "test", "max_workers": 1}, "workload": work,
                   "requests": [{"workload_request_id": "q", "status": "success", "output_ids": [42]}],
                   "summary": {"completed_requests": 1, paired.METRIC: 10.0}}
            protocol = {"expected_source": {"graphbudget_sha256": "gb", "sglang": sg},
                        "model_identity": {"revision": "model"},
                        "workloads": {"primary": {"workload_sha256": paired.digest(work),
                                                 "requests": 1, "workers": 1}}}
            paired.write_json(case_dir / "runtime/ready.json", {"identity": identity})
            paired.write_json(case_dir / "result.json", {"status": "PASS"})
            paired.write_json(case_dir / "primary-0.json", run)
            paired.verify_case({"arm": "B"}, case_dir, protocol, None, None)
            run["run"]["max_workers"] = 2
            paired.write_json(case_dir / "primary-0.json", run)
            with self.assertRaisesRegex(ValueError, "client concurrency"):
                paired.verify_case({"arm": "B"}, case_dir, protocol, None, None)
            run["run"]["max_workers"] = 1
            run["requests"] = []
            paired.write_json(case_dir / "primary-0.json", run)
            with self.assertRaisesRegex(ValueError, "Exact request validation"):
                paired.verify_case({"arm": "B"}, case_dir, protocol, None, None)

    def test_focused_case_rejects_warmup_output_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory)
            sg = {"base_commit": "base", "patch_sha256": "patch",
                  "srt_tree_sha256": "tree", "patched_files": {}}
            identity = {"graphbudget_sha256": "gb", "sglang": sg, "model": {"revision": "model"},
                        "runtime_env": paired.SWITCHES["B"]}
            work = {"requests": [{"request_id": "q", "input_ids": [1], "max_new_tokens": 1}]}
            transport = {"effective_mode": "direct"}
            run = {"kind": "graphbudget_benchmark_run",
                   "run": {"run_id": "test", "max_workers": 1, "boot_id": "boot",
                           "transport": transport, "reset_cache_requested": True},
                   "protocol": {"request_diagnostics": False},
                   "cache_reset": {"status": "success", "transport_peer": "target_loopback_port"},
                   "workload": work,
                   "requests": [{"workload_request_id": "q", "status": "success", "output_ids": [42],
                                 "transport_peer": "target_loopback_port"}],
                   "summary": {"completed_requests": 1, paired.METRIC: 10.0}}
            protocol = {"expected_source": {"graphbudget_sha256": "gb", "sglang": sg},
                        "model_identity": {"revision": "model"}, "boot_id": "boot",
                        "transport": transport, "settings": {"primary_shaped_warmup": True},
                        "initial_warmup": {"workload_sha256": paired.digest(work), "requests": 1, "workers": 1},
                        "workloads": {"primary": {"workload_sha256": paired.digest(work),
                                                   "requests": 1, "workers": 1}}}
            paired.write_json(case_dir / "runtime/ready.json", {"identity": identity})
            paired.write_json(case_dir / "result.json", {"status": "PASS"})
            paired.write_json(case_dir / "warmup.json", run)
            paired.write_json(case_dir / "primary-warmup.json", run)
            paired.write_json(case_dir / "primary-0.json", run)
            paired.verify_case({"arm": "B"}, case_dir, protocol, None, None)
            run["requests"][0]["output_ids"] = [43]
            paired.write_json(case_dir / "primary-0.json", run)
            with self.assertRaisesRegex(ValueError, "First warmup differs"):
                paired.verify_case({"arm": "B"}, case_dir, protocol, None, None)

    def test_formal_run_refuses_dirty_checkout_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-output"
            with mock.patch.object(paired, "git_snapshot", return_value={"status": "?? new.py\n"}):
                with self.assertRaises(SystemExit) as raised:
                    paired.main(["--output", str(output), "--model-path", "/unused"])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
