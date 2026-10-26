"""CPU-only synthetic fixtures; no row here is a real GPU measurement."""

import unittest

from graphbudget.plan import (
    InvalidPlanError,
    MissingMeasurementError,
    NoFeasiblePlanError,
    StalePlanError,
    SYNTHETIC_CPU_PROVENANCE,
    dump_plan,
    fingerprint,
    load_plan,
    select_plan,
)


def _identity(**overrides):
    value = {
        "sglang": {
            "base_commit": "sglang-base",
            "patch_sha256": "patch-sha",
            "srt_tree_sha256": "srt-sha",
            "patched_files": ["model_runner.py"],
        },
        "graphbudget_sha256": "graphbudget-sha",
        "model": {
            "revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
            "files_sha256": {"config.json": "config-sha"},
            "manifest_sha256": "manifest-sha",
        },
        "gpu": {
            "name": "RTX 5070 Laptop",
            "compute_capability": "12.0",
            "total_bytes": 8 * 1024**3,
            "driver": "580.0",
        },
        "dependencies": {"torch": "2.9.1", "sglang": "0.5.9"},
        "python": "3.12.14",
        "cuda": "12.8",
        "dtype": "bfloat16",
        "attention_backend": "flashinfer",
        "config": {
            "mem_fraction_static": 0.8,
            "max_total_tokens": 2048,
            "max_running_requests": 8,
            "context_length": 4096,
            "page_size": 1,
            "disable_radix_cache": True,
            "disable_overlap_schedule": True,
            "disable_cuda_graph_padding": False,
            "chunked_prefill_size": 0,
            "sampling_backend": "pytorch",
            "random_seed": 1,
        },
    }
    value.update(overrides)
    return value


def _budget(**overrides):
    value = {
        "allowed_device_bytes": 900,
        "min_kv_tokens": 512,
        "max_startup_seconds": 3.0,
    }
    value.update(overrides)
    return value


def _candidate(candidate_id, *, used=800, kv=1024, startup=2.0, calibration=1.0, **overrides):
    value = {
        "id": candidate_id,
        "capture_sizes": [2, 4, 8],
        "server_args": {
            "mem_fraction_static": 0.8,
            "max_total_tokens": 2048,
            "max_running_requests": 8,
            "context_length": 4096,
        },
        "ready": {
            "device_total_bytes": 1000,
            "device_used_bytes": used,
            "device_free_bytes": 1000 - used,
            "kv_tokens": kv,
            "startup_seconds": startup,
            "capture_seconds": 0.5,
        },
        "runtime_peak_device_used_bytes": used,
        "calibration_seconds": calibration,
        "status": "ok",
        "reason": "",
        "evidence": {
            "provenance": "measured",
            "run_id": f"run-{candidate_id}",
            "actual_fingerprint": fingerprint(_identity()),
            "requested_capture_sizes": [2, 4, 8],
            "actual_capture_sizes": [2, 4, 8],
        },
    }
    value.update(overrides)
    return value


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.identity = _identity()
        self.budget = _budget()

    def test_identity_digest_is_canonical_and_strict(self):
        digest = fingerprint(self.identity)
        changed = _identity(dtype="float16")
        self.assertNotEqual(digest, fingerprint(changed))
        with self.assertRaisesRegex(InvalidPlanError, "required keys"):
            fingerprint({"model": {"revision": "x"}})

    def test_whole_device_used_budget_ignores_allocator_metadata(self):
        over = _candidate("over", used=950)
        over["evidence"]["torch_allocated_bytes"] = 1
        good = _candidate("good", used=850, calibration=2.0)
        plan = select_plan(self.identity, self.budget, [over, good])
        self.assertEqual(plan["selected"]["id"], "good")
        self.assertIn("ready device used", plan["rejections"][0]["reasons"][0])

    def test_pareto_and_rejections_are_preserved(self):
        low_memory = _candidate("memory", used=600, calibration=2.0)
        low_latency = _candidate("latency", used=850, calibration=1.0)
        too_slow = _candidate("slow", startup=5.0)
        plan = select_plan(self.identity, self.budget, [low_memory, low_latency, too_slow])
        self.assertEqual(set(plan["pareto_ids"]), {"memory", "latency"})
        self.assertEqual(plan["selected"]["id"], "latency")
        self.assertEqual(plan["rejections"][0]["id"], "slow")

    def test_synthetic_fixture_requires_explicit_cpu_override(self):
        fixture = _candidate(
            "fixture",
            evidence={
                "provenance": SYNTHETIC_CPU_PROVENANCE,
                "run_id": "fixture",
                "actual_fingerprint": fingerprint(self.identity),
                "requested_capture_sizes": [2, 4, 8],
                "actual_capture_sizes": [2, 4, 8],
            },
        )
        with self.assertRaises(MissingMeasurementError) as raised:
            select_plan(self.identity, self.budget, [fixture])
        self.assertTrue(any("synthetic CPU fixture" in reason for reason in raised.exception.rejections[0]["reasons"]))
        plan = select_plan(self.identity, self.budget, [fixture], allow_synthetic=True)
        with self.assertRaises(InvalidPlanError):
            load_plan(plan)

    def test_missing_real_fields_fail_closed(self):
        missing = _candidate("missing")
        del missing["ready"]["device_used_bytes"]
        with self.assertRaises(MissingMeasurementError) as raised:
            select_plan(self.identity, self.budget, [missing])
        self.assertIn("missing ready.device_used_bytes", raised.exception.rejections[0]["reasons"])

    def test_measurement_boundaries_and_identity_config_are_checked(self):
        runtime_below_ready = _candidate("runtime-below", used=800, runtime_peak_device_used_bytes=700)
        with self.assertRaises(MissingMeasurementError) as raised:
            select_plan(self.identity, self.budget, [runtime_below_ready])
        self.assertIn("runtime peak used is below ready device used", raised.exception.rejections[0]["reasons"])

        capture_after_start = _candidate("capture-after", used=800)
        capture_after_start["ready"]["capture_seconds"] = 3.1
        with self.assertRaises(MissingMeasurementError) as raised:
            select_plan(self.identity, self.budget, [capture_after_start])
        self.assertIn("capture_seconds exceeds", " ".join(raised.exception.rejections[0]["reasons"]))

        mismatched_config = _candidate("config-mismatch")
        mismatched_config["server_args"]["max_total_tokens"] = 1024
        with self.assertRaises(MissingMeasurementError) as raised:
            select_plan(self.identity, self.budget, [mismatched_config])
        self.assertTrue(any("identity.config" in reason for reason in raised.exception.rejections[0]["reasons"]))

        no_fingerprint = _candidate("no-fingerprint")
        del no_fingerprint["evidence"]["actual_fingerprint"]
        with self.assertRaises(MissingMeasurementError) as raised:
            select_plan(self.identity, self.budget, [no_fingerprint])
        self.assertIn("actual_fingerprint", " ".join(raised.exception.rejections[0]["reasons"]))

    def test_malformed_capture_sizes_reject_without_unhashable_error(self):
        malformed = _candidate("malformed")
        malformed["capture_sizes"] = [[2], 4]
        with self.assertRaises(MissingMeasurementError) as raised:
            select_plan(self.identity, self.budget, [malformed])
        self.assertIn("positive integers", " ".join(raised.exception.rejections[0]["reasons"]))

    def test_candidate_count_is_bounded(self):
        with self.assertRaisesRegex(InvalidPlanError, "at most"):
            select_plan(self.identity, self.budget, [_candidate(str(i)) for i in range(6)])

    def test_plan_round_trip_and_stale_checks(self):
        plan = select_plan(self.identity, self.budget, [_candidate("a")])
        loaded = load_plan(dump_plan(plan), actual_fingerprint=fingerprint(self.identity), actual_budget=self.budget)
        self.assertEqual(loaded, plan)
        stale_identity = _identity(dtype="float16")
        with self.assertRaises(StalePlanError):
            load_plan(plan, actual_fingerprint=fingerprint(stale_identity))
        with self.assertRaises(StalePlanError):
            load_plan(plan, actual_budget=_budget(allowed_device_bytes=899))


if __name__ == "__main__":
    unittest.main()
