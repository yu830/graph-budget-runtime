"""Real RuntimeState startup on CPU with explicit external-observation stubs.

These tests exercise profile I/O, validation, compilation and runtime selection.
They do not load a model, observe native source identity, or execute GPU work.
"""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from graphbudget.plan import StalePlanError, dump_plan, fingerprint, select_plan
from graphbudget.policy import decide_policy
from graphbudget.runtime import RuntimeState
from test_plan import _budget, _candidate, _identity
from test_policy import _context, _policy
from test_runtime import batch


class ProfileFallbackTests(unittest.TestCase):
    @contextmanager
    def runtime(self, profile, *, plan=None, observation_error=None):
        observed_identity = _identity()
        args = NS(device="cuda", tp_size=1, pp_size=1, dp_size=1,
                  speculative_algorithm=None, quantization=None, dtype="bfloat16",
                  disable_overlap_schedule=True, disaggregation_mode="null",
                  cuda_graph_bs=[2, 4, 8], disable_cuda_graph_padding=False,
                  load_format="auto", torchao_config="", cpu_offload_gb=0,
                  model_loader_extra_config={}, json_model_override_args="{}")
        runner = NS(server_args=args, gpu_id=0, graph_runner=NS(capture_bs=[2, 4, 8]),
                    model_config=NS(hf_config=NS(model_type="qwen2")))
        with tempfile.TemporaryDirectory() as td:
            profile_path = Path(td) / "profile.json"
            # json.loads accepts NaN/Infinity; test that field validation rejects them.
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            env = {"GRAPHBUDGET_STARTED": "0", "GRAPHBUDGET_RUN_ID": "c2-cpu-stub",
                   "GRAPHBUDGET_OUTPUT": td, "GRAPHBUDGET_POLICY": "cost",
                   "GRAPHBUDGET_PROFILE": str(profile_path), "GRAPHBUDGET_OBSERVE": "diagnostic"}
            if plan is not None:
                plan_path = Path(td) / "plan.json"
                plan_path.write_text(dump_plan(plan), encoding="utf-8")
                env["GRAPHBUDGET_PLAN"] = str(plan_path)
            with patch.dict(os.environ, env, clear=True), \
                    patch.dict(sys.modules, {"torch": NS()}), \
                    patch("graphbudget.runtime.identity.collect", return_value=observed_identity,
                          side_effect=observation_error), \
                    patch.object(RuntimeState, "memory", return_value={"cpu_stub": True}), \
                    patch("graphbudget.runtime.atexit.register"):
                yield RuntimeState(runner)

    def test_bad_optional_rows_construct_and_select_native_with_reason(self):
        domains = {
            "provenance": (True, None, [], {}, 1, "unknown"),
            "graph_ns": (True, None, [], 10**400, float("inf"), float("nan")),
            "eager_ns": (True, None, [], 10**400, float("inf"), float("nan")),
            "noise_ns": (True, None, [], 10**400, float("inf"), float("nan")),
        }
        for field, values in domains.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    profile = _policy(_identity())
                    profile["entries"][0][field] = value
                    with self.runtime(profile) as state:
                        self.assertEqual(state.policy_status, "loaded")
                        decision = decide_policy(state.policy, _context(), True, state.fingerprint)
                        self.assertEqual(decision["selected_path"], "native")
                        if field == "provenance":
                            expected_reason = "profile_not_measured"
                        elif value is None:
                            expected_reason = ("profile_noise_missing" if field == "noise_ns"
                                               else "profile_cost_missing")
                        else:
                            expected_reason = "profile_entry_invalid"
                        self.assertEqual(decision["reason"], expected_reason)
                        # Native resolves to the upstream eligible path, with its reason retained.
                        self.assertTrue(state.begin(batch(2, [1024, 1024]), True))
                        self.assertEqual(state.pending["reason"], decision["reason"])

    def test_valid_optional_profile_behavior_is_unchanged(self):
        profile = _policy(_identity())
        with self.runtime(profile) as state:
            self.assertEqual(state.policy_status, "loaded")
            self.assertFalse(state.begin(batch(2, [1024, 1024]), True))
            self.assertEqual(state.pending["reason"], "eager_faster_beyond_noise")

    def test_invalid_whole_profile_falls_back_with_diagnostic(self):
        profile = _policy(_identity(), capture_sizes=[])
        with self.runtime(profile) as state:
            self.assertIsNone(state.policy)
            self.assertIn("native_fallback", state.policy_status)
            self.assertIn("capture_sizes", state.policy_status)
            self.assertTrue(state.begin(batch(2, [1024, 1024]), True))

    def test_bad_optional_profile_does_not_hide_hard_plan_refusal(self):
        profile = _policy(_identity())
        profile["entries"][0]["provenance"] = []
        stale_identity = copy.deepcopy(_identity())
        stale_identity["graphbudget_sha256"] = "different-source"
        candidate = _candidate("stale")
        candidate["evidence"]["actual_fingerprint"] = fingerprint(stale_identity)
        plan = select_plan(stale_identity, _budget(), [candidate])
        with self.assertRaises(StalePlanError), self.runtime(profile, plan=plan):
            self.fail("stale hard plan must reject startup")

    def test_source_and_model_observation_errors_are_not_optional_profile_errors(self):
        profile = _policy(_identity())
        profile["entries"][0]["graph_ns"] = 10**400
        for message in ("pinned source drift", "model snapshot invalid"):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message), \
                    self.runtime(profile, observation_error=ValueError(message)):
                self.fail("identity observation failure must reject startup")


if __name__ == "__main__":
    unittest.main()
