"""Loading-boundary regressions; external hardware/model observations are CPU stubs."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from graphbudget import identity
from graphbudget.plan import load_plan, StalePlanError
from graphbudget.runtime import apply_before_init, check_supported

ROOT = Path(__file__).resolve().parents[1]


def args():
    original = json.loads((ROOT / "results/2026-10-26/plan.json").read_text())
    return NS(**original["fingerprint"]["identity"]["config"], device="cuda", tp_size=1,
              pp_size=1, dp_size=1, speculative_algorithm=None, quantization=None,
              dtype="bfloat16", disaggregation_mode="null", model_path="cpu-stub-snapshot",
              tokenizer_path=None, revision="a"*40, attention_backend="triton",
              prefill_attention_backend=None, decode_attention_backend=None,
              load_format="auto", model_impl="auto", torchao_config="", cpu_offload_gb=0,
              model_loader_extra_config="{}", json_model_override_args="{}")


class IdentityTests(unittest.TestCase):
    def test_defaults_and_equivalent_loading_representations(self):
        baseline = args()
        check_supported(baseline, NS(hf_config=NS(model_type="qwen2")))
        alternate = args()
        alternate.model_loader_extra_config = {}
        alternate.json_model_override_args = " { } "
        alternate.custom_weight_loader = []
        alternate.cpu_offload_gb = 0.0
        alternate.offload_group_size = 0
        alternate.torchao_config = None
        self.assertEqual(identity.execution_config(baseline), identity.execution_config(alternate))

    def test_unsupported_execution_rejected_at_every_entry(self):
        invalid = {
            "load_format": ["dummy", "remote", "custom", []],
            "model_impl": ["transformers", "mindspore"],
            "torchao_config": ["int8wo", [], False],
            "cpu_offload_gb": [1, True, "0", float("nan")],
            "offload_group_size": [1],
            "custom_weight_loader": [["module.loader"]],
            "model_loader_extra_config": ['{"num_threads":2}', "[]", "null", "bad"],
            "json_model_override_args": ['{"hidden_size":2}', [], {}, None],
            "decrypted_config_file": ["alternate-config.json"],
            "decrypted_draft_config_file": ["alternate-config.json"],
            "forward_hooks": [[{"module": "custom"}]],
            "trust_remote_code": [True],
            "quantization": ["fp8"],
            "modelopt_quant": ["fp8"],
            "enable_fp32_lm_head": [True],
            "delete_ckpt_after_loading": [True],
            "remote_instance_weight_loader_start_seed_via_transfer_engine": [True],
        }
        with patch.dict(os.environ, {}, clear=True):
            for key, values in invalid.items():
                for value in values:
                    with self.subTest(key=key, value=value):
                        candidate = args()
                        setattr(candidate, key, value)
                        # Must reject without a plan, and before any GPU import.
                        for call in (lambda: apply_before_init(candidate),
                                     lambda: check_supported(candidate, NS(hf_config=NS(model_type="qwen2"))),
                                     lambda: identity.collect(candidate)):
                            with self.assertRaisesRegex(ValueError, key):
                                call()

    def test_collect_binds_supported_loader_settings_not_port(self):
        gpu = NS(name="CPU STUB", major=12, minor=0, total_memory=8*1024**3)
        torch = NS(cuda=NS(get_device_properties=lambda _: gpu), version=NS(cuda="stub"))
        with patch.dict(sys.modules, {"sglang": NS(__file__="cpu-stub.py"), "torch": torch}), \
             patch.object(identity, "source_identity", return_value={"source": "CPU STUB"}), \
             patch.object(identity, "model_identity", return_value={"weights": "CPU STUB"}), \
             patch.object(identity.importlib.metadata, "version", return_value="CPU STUB"), \
             patch.object(identity.subprocess, "check_output", return_value="driver,uuid"):
            candidate = args()
            baseline = identity.collect(candidate)
            candidate.port = 39999
            candidate.log_level = "debug"
            self.assertEqual(baseline, identity.collect(candidate))
            for key, value in (("load_format", "safetensors"), ("weight_loader_disable_mmap", True)):
                changed = args()
                setattr(changed, key, value)
                actual = identity.collect(changed)
                self.assertNotEqual(identity.digest(baseline), identity.digest(actual))
                self.assertEqual(actual["execution"][key], value)
            self.assertEqual(baseline["identity_schema_version"], 2)

    def test_legacy_hard_plan_not_relabeled_as_current(self):
        original = json.loads((ROOT / "results/2026-10-26/plan.json").read_text())
        current = copy.deepcopy(original["fingerprint"]["identity"])
        current["identity_schema_version"] = 2
        current["execution"] = identity.execution_config(args())
        with self.assertRaises(StalePlanError):
            load_plan(original, actual_fingerprint=current)

    @unittest.skipUnless(os.environ.get("GRAPHBUDGET_SGLANG_SOURCE"), "pinned upstream source required")
    def test_actual_server_args_rejects_before_upstream_normalization(self):
        source = Path(os.environ["GRAPHBUDGET_SGLANG_SOURCE"]) / "python/sglang/srt/server_args.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ServerArgs")
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"))
        scope = {"os": os}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), scope)
        candidate = args()
        candidate.load_format = "dummy"
        candidate._handle_load_balance_method = lambda: self.fail("upstream normalization reached")
        with patch.dict(os.environ, {"GRAPHBUDGET_ACTIVE": "1"}):
            with self.assertRaisesRegex(ValueError, "load_format"):
                scope["__post_init__"](candidate)


if __name__ == "__main__":
    unittest.main()
