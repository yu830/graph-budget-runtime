"""CPU contract tests. Fake execution is explicitly not CUDA/model evidence."""
import ast
from collections import Counter
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import time
import unittest
from unittest.mock import patch

from graphbudget.runtime import RuntimeState, apply_before_init
from graphbudget.policy import compile_policy
from graphbudget.plan import select_plan, dump_plan
from test_plan import _identity, _budget, _candidate


class FakeCPU:
    device = NS(type="cpu")

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def state(policy=None, diagnostic=True):
    s = RuntimeState.__new__(RuntimeState)
    s.runner = NS(graph_runner=NS(capture_bs=[1, 2, 4, 8]))
    s.args = NS(disable_cuda_graph_padding=False)
    s.run_id = "synthetic-runtime-unit"
    s.fingerprint = "a" * 64
    s.policy = compile_policy(policy, s.fingerprint) if policy else None
    s.force_eager = False
    s.test_only = False
    s.calibrate = False
    s.diagnostic = diagnostic
    s.counts = Counter()
    s.steps = []
    s.dropped_steps = 0
    s.step_id = 0
    s.pending = None
    return s


def batch(bs=3, lengths=None):
    return NS(batch_size=bs, forward_mode=NS(is_decode=lambda: True),
              seq_lens_cpu=FakeCPU(lengths or [12] * bs), graphbudget_request_ids=["test"] * bs)


class RuntimeTests(unittest.TestCase):
    def test_native_is_resolved_and_actual_call_required(self):
        s = state()
        self.assertTrue(s.begin(batch(), True))
        self.assertEqual(s.pending["selected_path"], "graph")
        with self.assertRaisesRegex(RuntimeError, "path mismatch"):
            s.finish()
        s.replayed(4)
        s.finish()
        self.assertEqual(s.steps[0]["actual_capture_size"], 4)
        self.assertEqual(s.steps[0]["used_path"], "graph")

    def test_refusal_and_ceiling_have_different_reasons(self):
        s = state()
        s.force_eager = s.test_only = True
        self.assertFalse(s.begin(batch(), True))
        s.eager_returned()
        s.finish()
        self.assertTrue(s.steps[-1]["eligible"])
        self.assertEqual(s.steps[-1]["reason"], "forced_eager_test")
        self.assertFalse(s.begin(batch(9), False))
        s.eager_returned()
        s.finish()
        self.assertFalse(s.steps[-1]["eligible"])
        self.assertEqual(s.steps[-1]["reason"], "capture_ceiling")

    def test_no_gpu_tensor_read_and_heterogeneous_native(self):
        s = state()
        b = batch()
        b.seq_lens_cpu = NS(device=NS(type="cuda"), tolist=lambda: self.fail("D2H is forbidden"))
        self.assertTrue(s.begin(b, True))
        self.assertIsNone(s.pending["context"])
        self.assertTrue(s.begin(batch(lengths=[12, 13, 12]), True))
        self.assertIsNone(s.pending["context"])

    def test_capture_size_mismatch_detected(self):
        s = state()
        s.begin(batch(), True)
        s.replayed(8)
        with self.assertRaisesRegex(RuntimeError, "capture size mismatch"):
            s.finish()

    def test_plan_is_applied_before_initialization(self):
        ident = _identity()
        candidate = _candidate("example")
        # Test helper is synthetic CPU data that simulates receipt validation.
        plan = select_plan(ident, _budget(), [candidate])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "plan.json"
            path.write_text(dump_plan(plan))
            args = NS(disable_cuda_graph=False, cuda_graph_bs=None)
            with patch.dict(os.environ, {"GRAPHBUDGET_PLAN": str(path)}):
                apply_before_init(args)
            self.assertEqual(args.cuda_graph_bs, candidate["capture_sizes"])
            self.assertEqual(args.max_total_tokens, candidate["server_args"]["max_total_tokens"])

    def test_ready_hard_budget_fails_before_service(self):
        s = state()
        s.identity = _identity()
        s.ready_record = None
        s.started = time.perf_counter()
        s.capture_seconds = 0.1
        s.capture_before = s.capture_after = None
        s.pre_model = {}
        s.policy_status = "native_fallback: missing profile"
        s.args.page_size = 1
        s.runner.max_total_num_tokens = 1024
        s.memory = lambda: {"device_used_bytes": 900}
        s.plan = {"selected": {"capture_sizes": [1, 2, 4, 8]},
                  "budget": {"allowed_device_bytes": 800, "min_kv_tokens": 512,
                             "max_startup_seconds": 100}}
        with tempfile.TemporaryDirectory() as td:
            s.output = Path(td)
            with patch("graphbudget.runtime.identity.local_details", return_value={}), \
                    self.assertRaisesRegex(RuntimeError, "hard budget"):
                s.ready()
            self.assertEqual(json.loads((s.output / "ready.json").read_text())["status"],
                             "rejected_before_service")


@unittest.skipUnless(os.environ.get("GRAPHBUDGET_SGLANG_SOURCE"), "set GRAPHBUDGET_SGLANG_SOURCE for pinned-source control-flow checks")
class PinnedSourceTests(unittest.TestCase):
    """Execute the actual patched _forward_raw AST on CPU sentinel collaborators.
    This checks branch/preparation order, never claims physical graph execution.
    """

    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["GRAPHBUDGET_SGLANG_SOURCE"])
        path = cls.root / "python/sglang/srt/model_executor/model_runner.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        runner_class = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner")
        function = copy.deepcopy(next(n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_raw"))
        for arg in function.args.args:
            arg.annotation = None
        function.returns = None
        scope = {"ModelRunnerOutput": lambda **kwargs: NS(**kwargs)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), scope)
        cls.forward = staticmethod(scope["_forward_raw"])

    def test_real_dispatch_order_native_and_policy_refusal(self):
        for forced in (False, True):
            events = []
            s = state()
            s.force_eager = s.test_only = forced
            b = batch()
            b.forward_mode.is_cuda_graph = lambda: True
            b.global_num_tokens_cpu = b.num_token_non_padded = None
            b.prepare_attn_tp_scatter_input = lambda _: events.append("eager_prepare")
            def replay(*args, **kwargs):
                events.append("replay_prepare_and_call")
                s.replayed(4)
                return "tokens"
            runner = NS(device="cuda", graphbudget=s,
                        graph_runner=NS(can_run=lambda _: True, replay=replay),
                        forward_decode=lambda *a, **k: events.append("eager_forward") or "tokens")
            out = self.forward(runner, b, False, None)
            self.assertEqual(out.can_run_graph, not forced)
            self.assertEqual(events, ["eager_prepare", "eager_forward"] if forced else ["replay_prepare_and_call"])
            self.assertEqual(s.steps[0]["used_path"], "eager" if forced else "graph")

    def test_off_path_does_not_call_policy_or_eager_prepare(self):
        b = batch()
        b.forward_mode.is_cuda_graph = lambda: True
        runner = NS(device="cuda", graphbudget=None,
                    graph_runner=NS(can_run=lambda _: True, replay=lambda *a, **k: "stock"))
        out = self.forward(runner, b, False, None)
        self.assertTrue(out.can_run_graph)
        self.assertEqual(out.logits_output, "stock")

    def test_source_hook_positions_and_real_replay_marker(self):
        runner = (self.root / "python/sglang/srt/model_executor/model_runner.py").read_text(encoding="utf-8")
        self.assertLess(runner.index("RuntimeState(self)"), runner.index("self.initialize(min_per_gpu_memory)"))
        self.assertLess(runner.index("self.init_memory_pool(min_per_gpu_memory)"), runner.index("self.init_device_graphs()"))
        args = (self.root / "python/sglang/srt/server_args.py").read_text(encoding="utf-8")
        self.assertLess(args.index("apply_before_init(self)"), args.index("self._handle_gpu_memory_settings(gpu_mem)"))
        graph = (self.root / "python/sglang/srt/model_executor/cuda_graph_runner.py").read_text(encoding="utf-8")
        self.assertIn("self.graphs[graph_key].replay()\n        if self.model_runner.graphbudget", graph)

    def test_entire_runtime_source_is_bound(self):
        from graphbudget.identity import source_identity, BASE_SHA
        identity = source_identity(self.root / "python/sglang/__init__.py")
        self.assertEqual(identity["base_commit"], BASE_SHA)

    def test_management_barrier_exports_only_when_idle(self):
        path = self.root / "python/sglang/srt/managers/scheduler.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "flush_cache"))
        events = []
        scope = {"torch": NS(cuda=NS(empty_cache=lambda: events.append("upstream_empty_cache"))),
                 "logger": NS(info=lambda *_: None), "logging": NS(warning=lambda *_: None)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
        scheduler = NS(_is_no_request=lambda: True, tree_cache=NS(reset=lambda: None),
                       req_to_token_pool=NS(clear=lambda: None),
                       token_to_kv_pool_allocator=NS(clear=lambda: None),
                       grammar_manager=NS(clear=lambda: None), reset_metrics=lambda: None,
                       draft_worker=None, tp_worker=NS(model_runner=NS(graphbudget=NS(flush=lambda: events.append("export")))))
        self.assertTrue(scope["flush_cache"](scheduler))
        self.assertEqual(events, ["upstream_empty_cache", "export"])
        scheduler._is_no_request = lambda: False
        scheduler.waiting_queue = [1]
        scheduler.running_batch = NS(reqs=[])
        events.clear()
        self.assertFalse(scope["flush_cache"](scheduler))
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
