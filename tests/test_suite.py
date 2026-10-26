"""CPU-only checks for the phase orchestration script.

These tests validate frozen workload/config construction and observed-trace
guards.  They intentionally do not start SGLang, access CUDA, or claim a GPU
measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from graphbudget.benchmark import load_workload
from scripts import run_core


class _FakeTokenizer:
    def __call__(self, prompt, **_kwargs):
        count = max(2, min(64, len(prompt.split()) // 2))
        return {"input_ids": [100 + index for index in range(count)]}


class SuiteScriptTests(unittest.TestCase):
    def test_frozen_prompt_schedules_have_distinct_stable_id_scopes(self) -> None:
        blueprints = run_core._prompt_blueprints()
        self.assertTrue(blueprints["calibration"])
        self.assertTrue(all(item["request_id"].startswith("cal-") for item in blueprints["calibration"]))
        self.assertTrue(all(item["request_id"].startswith("normal-") for item in blueprints["heldout_normal"]))
        self.assertTrue(all(item["request_id"].startswith("boundary-") for item in blueprints["heldout_boundary"]))
        self.assertTrue(all(item["request_id"].startswith("mixed-") for item in blueprints["heldout_mixed"]))
        for entries in blueprints.values():
            self.assertEqual(len({entry["request_id"] for entry in entries}), len(entries))
            self.assertEqual(
                [entry["planned_at_s"] for entry in entries],
                sorted(entry["planned_at_s"] for entry in entries),
            )
            self.assertLessEqual(max(entry["burst_size"] for entry in entries), 16)

    def test_freeze_workloads_uses_real_tokenizer_shape_and_derives_kv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = {
                "path": str(root / "model"),
                "revision": "a" * 40,
                "manifest_sha256": "b" * 64,
            }
            paths = run_core._freeze_workloads(
                _FakeTokenizer(),
                model=model,
                revision=model["revision"],
                prepare_dir=root / "prepare",
            )
            self.assertEqual(
                set(paths), {"calibration", "heldout_normal", "heldout_boundary", "heldout_mixed"}
            )
            for path in paths.values():
                workload = load_workload(path)
                self.assertTrue(workload.token_ids_validated_against_model)
                self.assertTrue(workload.requests)
                self.assertTrue(all(request.input_ids[0] >= 100 for request in workload.requests))
            kv = run_core._derive_min_kv_tokens(paths)
            self.assertEqual(kv["max_concurrent_requests"], 16)
            self.assertIn(kv["max_total_tokens"], (4096, 8192))
            self.assertGreaterEqual(kv["min_kv_tokens"], 16)

    def test_server_args_freeze_backend_and_shared_constraints(self) -> None:
        args, config = run_core._server_args(
            model_path=Path("C:/model"),
            revision="a" * 40,
            backend="flashinfer",
            max_total_tokens=4096,
        )
        self.assertIn("--disable-overlap-schedule", args)
        self.assertIn("--stream-output", args)
        self.assertIn("--attention-backend", args)
        self.assertEqual(args[args.index("--attention-backend") + 1], "flashinfer")
        self.assertNotIn("--disable-radix-cache", args)
        self.assertEqual(config["max_running_requests"], 16)
        self.assertEqual(config["max_total_tokens"], 4096)
        self.assertEqual(config["context_length"], 4096)

    def test_trace_guard_reports_observed_pairs_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp)
            (case / "case.json").write_text(
                json.dumps(
                    {
                        "run_id": "server-run",
                        "benchmark": {
                            "run": {"run_id": "timed-run"},
                            "requests": [
                                {
                                    "client_request_id": "graphbudget/timed-run/req-1",
                                    "status": "success",
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (case / "paths_after.json").write_text(
                json.dumps(
                    {
                        "dropped_steps": 0,
                        "counts": [{"used_path": "graph", "reason": "capture_ceiling"}],
                        "steps": [
                            {
                                "request_ids": ["graphbudget/timed-run-warmup/req-1"],
                                "batch_size": 99,
                                "padded_size": 100,
                                "used_path": "graph",
                                "reason": "warmup-only",
                            },
                            {
                                "request_ids": ["graphbudget/timed-run/req-1"],
                                "batch_size": 3,
                                "padded_size": 4,
                                "used_path": "graph",
                                "reason": "capture_ceiling",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            coverage = run_core._trace_coverage(case, "synthetic")
            self.assertTrue(coverage["coverage_is_observed"])
            self.assertEqual(coverage["observed_batch_sizes"], [3])
            self.assertEqual(coverage["observed_padded_sizes"], [4])
            self.assertEqual(coverage["observed_batch_padded_pairs"], [[3, 4]])
            self.assertEqual(coverage["timed_completed_request_count"], 1)

    def test_cli_exposes_explicit_phase_and_resume_controls(self) -> None:
        parser = run_core.build_parser()
        args = parser.parse_args(
            [
                "--model-path",
                "C:/model",
                "--revision",
                "a" * 40,
                "--backend",
                "triton",
                "--output",
                "C:/suite",
                "--phase",
                "all",
                "--resume",
            ]
        )
        self.assertEqual(args.phase, "all")
        self.assertTrue(args.resume)

    def test_prepare_stage_state_can_point_at_its_own_immutable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / "prepare"
            (stage / "workloads").mkdir(parents=True)
            experiment = {"kind": "graphbudget_experiment"}
            (stage / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
            workload = stage / "workloads" / "calibration.json"
            workload.write_text(json.dumps({"kind": "workload"}), encoding="utf-8")
            manifest = {
                "prepare_artifacts": {
                    "experiment_sha256": run_core._sha256_file(stage / "experiment.json"),
                    "workloads": {"calibration": run_core._sha256_file(workload)},
                }
            }
            (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            run_core._mark_stage(stage, status="ok", attempt=stage, details={"manifest": manifest})
            self.assertEqual(run_core._completed_attempt(stage), stage.resolve())
            workload.write_text("tampered", encoding="utf-8")
            self.assertIsNone(run_core._completed_attempt(stage))

    def test_completed_stage_hash_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare = root / "prepare"
            prepare.mkdir()
            experiment = prepare / "experiment.json"
            experiment.write_text(json.dumps({"kind": "experiment"}), encoding="utf-8")
            manifest = {
                "prepare_artifacts": {
                    "experiment_sha256": run_core._sha256_file(experiment),
                    "workloads": {},
                }
            }
            (prepare / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            run_core._mark_stage(prepare, status="ok", attempt=prepare, details={"manifest": manifest})

            stage = root / "measure"
            attempt = stage / "attempt-001"
            attempt.mkdir(parents=True)
            artifact = attempt / "measurements.json"
            artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
            run_core._mark_stage(
                stage,
                status="ok",
                attempt=attempt,
                details={
                    "measurements_sha256": run_core._sha256_file(artifact),
                    "experiment_sha256": run_core._sha256_file(experiment),
                },
            )
            self.assertEqual(run_core._completed_attempt(stage), attempt.resolve())
            artifact.write_text(json.dumps({"ok": False}), encoding="utf-8")
            self.assertIsNone(run_core._completed_attempt(stage))

    def test_cost_fallback_is_reported_not_attributed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp)
            (case / "case.json").write_text(
                json.dumps(
                    {
                        "ready": {
                            "policy_mode": "cost",
                            "policy_status": "native_fallback: profile disabled",
                        },
                        "timed_path_counts": [
                            {"used_path": "graph", "count": 5},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            evidence = run_core._case_path_evidence(case)
            self.assertEqual(evidence["status"], "not_attributable")
            self.assertIn("native_fallback", evidence["reason"])

    def test_loaded_cost_with_timed_eager_is_attributable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp)
            (case / "case.json").write_text(
                json.dumps(
                    {
                        "ready": {"policy_mode": "cost", "policy_status": "loaded"},
                        "timed_path_counts": [
                            {"eligible": True, "selected_path": "eager", "used_path": "eager",
                             "reason": "eager_faster_beyond_noise", "count": 2},
                            {"used_path": "graph", "count": 3},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            evidence = run_core._case_path_evidence(case)
            self.assertEqual(evidence["status"], "attributable")

    def test_overflow_eager_is_not_a_cost_policy_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp)
            (case / "case.json").write_text(json.dumps({
                "ready": {"policy_mode": "cost", "policy_status": "loaded"},
                "timed_path_counts": [{"eligible": False, "selected_path": "eager",
                    "used_path": "eager", "reason": "capture_ceiling", "count": 5}]
            }), encoding="utf-8")
            self.assertEqual(run_core._case_path_evidence(case)["status"], "not_attributable")

    def test_source_drift_stales_dependents_but_preserves_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare = root / "prepare"
            prepare.mkdir()
            model = {
                "path": str(root / "model"),
                "revision": "a" * 40,
                "manifest_sha256": "model-hash",
            }
            manifest = {
                "backend": "triton",
                "model": model,
                "graphbudget_commit": "old-gb",
                "sglang_commit": "old-sg",
                "graphbudget_source_sha256": "old-gb-tree",
                "sglang_tree_sha256": "old-sg-tree",
                "prepare_artifacts": {"experiment_sha256": "unused", "workloads": {}},
            }
            (prepare / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            measure = root / "measure"
            run_core._mark_stage(measure, status="ok", attempt=measure, details={})
            args = type(
                "Args",
                (),
                {
                    "output": str(root),
                    "resume": True,
                    "backend": "triton",
                    "model_path": model["path"],
                    "revision": model["revision"],
                    "tokenizer_path": None,
                    "sglang_checkout": None,
                },
            )()
            with patch.object(run_core, "_validate_model_snapshot", return_value=model), patch.object(
                run_core,
                "_source_snapshot",
                return_value={
                    "graphbudget_commit": "new-gb",
                    "sglang_commit": "new-sg",
                    "graphbudget_source_sha256": "new-gb-tree",
                    "sglang_tree_sha256": "new-sg-tree",
                },
            ):
                run_core._ensure_output_root(args)
            self.assertEqual(run_core._stage_state(measure)["status"], "stale")

    def test_matching_worker_identity_preserves_measurement_attempt(self) -> None:
        identity = {
            "graphbudget_sha256": "gb-current",
            "sglang": {
                "base_commit": "sg-current",
                "patch_sha256": "patch-current",
                "srt_tree_sha256": "tree-current",
            },
        }
        current = {
            "graphbudget_identity_sha256": "gb-current",
            "sglang_commit": "sg-current",
            "sglang_patch_sha256": "patch-current",
            "sglang_tree_sha256": "tree-current",
        }
        self.assertTrue(run_core._source_matches(identity, current))

    def test_completed_case_reuse_checks_workload_source_and_flags(self) -> None:
        # Synthetic receipt validates recovery guards only, never GPU execution.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workload = root / "workload.json"
            workload.write_text(json.dumps({"workload_id":"cpu", "purpose":"test",
                "requests":[{"request_id":"one", "input_ids":[1,2], "max_new_tokens":2}]}))
            frozen = run_core.load_workload(workload).to_dict()
            output = root / "case"
            output.mkdir()
            (output / "case.json").write_text(json.dumps({"status":"ok", "run_id":"r",
                "ready":{"fingerprint":"a"*64},
                "benchmark":{"workload":frozen,"run":{"run_id":"r"}}}))
            (output / "warmup.json").write_text('{"synthetic_cpu_fixture":true}')
            (output / "invocation.json").write_text(json.dumps({"run_id":"r", "command":[
                "python","-m","graphbudget.cli","launch","--policy","native","--observe","counts",
                "--","--fixture","--host","127.0.0.1","--port","30150"]}))
            experiment = {"server_args":["--fixture"],"port":30150,
                          "_expected_runtime_fingerprint":"a"*64}
            with patch.object(run_core, "gpu_lease", side_effect=AssertionError("recovery is read-only")):
                self.assertEqual(run_core._run_case(experiment,workload=workload,output=output)["status"],"ok")
                with self.assertRaises(run_core.SuiteError):
                    run_core._run_case({**experiment,"_expected_runtime_fingerprint":"b"*64},workload=workload,output=output)
                with self.assertRaises(run_core.SuiteError):
                    run_core._run_case(experiment,workload=workload,output=output,policy="cost")
                changed = json.loads(workload.read_text())
                changed["requests"][0]["input_ids"] = [3,4]
                workload.write_text(json.dumps(changed))
                with self.assertRaises(run_core.SuiteError):
                    run_core._run_case(experiment,workload=workload,output=output)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
