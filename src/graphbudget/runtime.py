"""Small opt-in adapter for the pinned SGLang patch.

Only calibration mode synchronizes per step. Production policy uses CPU
metadata and a frozen lookup. Diagnostic steps are buffered; the scheduler's
idle cache-flush barrier exports them outside timed requests. This module is
not imported on the native path.
"""

import atexit
from bisect import bisect_left
from collections import Counter
import json
import os
from pathlib import Path
import time

from . import identity
from .plan import load_plan
from .policy import load_policy, compile_policy, decide_policy


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def apply_before_init(args):
    """Called as the first statement of ServerArgs.__post_init__."""
    # Also applies without a plan. Reject loader/override modes before upstream
    # normalization, ModelConfig construction, distributed init or weight load.
    identity.execution_config(args)
    plan_path = os.environ.get("GRAPHBUDGET_PLAN")
    if not plan_path:
        return
    plan = load_plan(Path(plan_path).read_text(encoding="utf-8"))
    selected = plan["selected"]
    if args.disable_cuda_graph:
        raise ValueError("A graph plan cannot be combined with --disable-cuda-graph")
    # Explicit plan authority precedes default calculation and all allocations.
    args.cuda_graph_bs = list(selected["capture_sizes"])
    args.cuda_graph_max_bs = max(args.cuda_graph_bs)
    for key, value in selected["server_args"].items():
        setattr(args, key, value)


def check_supported(args, model_config):
    identity.execution_config(args)
    unsupported = [key for key in (
        "enable_dp_attention", "enable_two_batch_overlap", "enable_pdmux",
        "enable_piecewise_cuda_graph", "enable_torch_compile", "enable_lora",
        "enable_memory_saver", "enable_mixed_chunk", "enable_hierarchical_cache",
        "enable_return_hidden_states", "enable_symm_mem",
    ) if getattr(args, key, False)]
    if (args.device != "cuda" or args.tp_size != 1 or args.pp_size != 1 or
            args.dp_size != 1 or args.speculative_algorithm is not None or
            args.quantization is not None or args.dtype not in ("bfloat16", "bf16") or
            not args.disable_overlap_schedule or args.disaggregation_mode != "null"):
        unsupported.append("requires CUDA BF16 dense TP=PP=DP=1 ordinary decode without overlap")
    # The first adapter is deliberately restricted to the intended dense model.
    if getattr(model_config.hf_config, "model_type", None) != "qwen2":
        unsupported.append("only dense Qwen2/Qwen2.5 supported in this adapter")
    if unsupported:
        raise ValueError("Unsupported GraphBudget v1 configuration: " + ", ".join(unsupported))


class RuntimeState:
    def __init__(self, runner):
        import torch
        self.torch = torch
        self.runner = runner
        self.args = runner.server_args
        check_supported(self.args, runner.model_config)
        self.started = float(os.environ["GRAPHBUDGET_STARTED"])
        self.run_id = os.environ["GRAPHBUDGET_RUN_ID"]
        self.output = Path(os.environ["GRAPHBUDGET_OUTPUT"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.identity = identity.collect(self.args, runner.gpu_id)
        self.fingerprint = identity.digest(self.identity)
        self.plan = None
        if plan_path := os.environ.get("GRAPHBUDGET_PLAN"):
            self.plan = load_plan(Path(plan_path).read_text(), actual_fingerprint=self.identity)
            expected = self.plan["selected"]["capture_sizes"]
            if list(self.args.cuda_graph_bs) != expected:
                raise ValueError("SGLang normalized the frozen capture set differently")
        self.policy = None
        self.policy_status = "disabled"
        mode = os.environ.get("GRAPHBUDGET_POLICY", "native")
        if mode not in ("native", "cost", "test_eager"):
            raise ValueError("Unknown GraphBudget policy mode")
        self.force_eager = mode == "test_eager"
        self.test_only = os.environ.get("GRAPHBUDGET_TEST_ONLY") == "1"
        if self.force_eager and not self.test_only:
            raise ValueError("Forced eager is restricted to explicitly marked functional/calibration tests")
        if mode == "cost":
            try:
                self.policy = load_policy(Path(os.environ["GRAPHBUDGET_PROFILE"]).read_text())
                if self.policy["fingerprint"] != self.fingerprint:
                    raise ValueError("cost profile identity mismatch")
                if self.policy["capture_sizes"] != list(self.args.cuda_graph_bs):
                    raise ValueError("cost profile capture set mismatch")
                self.policy = compile_policy(self.policy, self.fingerprint)
                self.policy_status = self.policy["status"]
                if self.policy.get("fallback_reason"):
                    self.policy_status += ": " + self.policy["fallback_reason"]
                if self.policy["status"] != "loaded":
                    self.policy = None
            except (OSError, KeyError, ValueError) as error:
                # Only costs fail open to native. Hard plan/budget errors never do.
                self.policy = None
                self.policy_status = "native_fallback: " + str(error)
        self.calibrate = os.environ.get("GRAPHBUDGET_CALIBRATE") == "1"
        self.diagnostic = os.environ.get("GRAPHBUDGET_OBSERVE", "counts") == "diagnostic"
        if self.calibrate and not self.diagnostic:
            raise ValueError("Calibration requires diagnostic observation")
        self.counts = Counter()
        self.steps = []
        self.dropped_steps = 0
        self.step_id = 0
        self.pending = None
        self.capture_seconds = 0.0
        self.capture_before = None
        self.capture_after = None
        self.pre_model = self.memory()
        self.ready_record = None
        atexit.register(self.flush)

    def memory(self):
        free, total = self.torch.cuda.mem_get_info(self.runner.gpu_id)
        return {"device_total_bytes": total, "device_free_bytes": free,
                "device_used_bytes": total - free,
                "torch_allocated_bytes": self.torch.cuda.memory_allocated(self.runner.gpu_id),
                "torch_reserved_bytes": self.torch.cuda.memory_reserved(self.runner.gpu_id),
                "torch_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.runner.gpu_id),
                "torch_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(self.runner.gpu_id)}

    def capture_begin(self):
        self.capture_before = self.memory()
        self.capture_started = time.perf_counter()

    def capture_end(self):
        self.capture_seconds = time.perf_counter() - self.capture_started
        self.capture_after = self.memory()

    def ready(self):
        graph = self.runner.graph_runner
        sizes = list(graph.capture_bs) if graph else []
        memory = self.memory()
        kv_tokens = self.runner.max_total_num_tokens
        record = {"schema_version": 1, "run_id": self.run_id, "pid": os.getpid(),
                  "identity": self.identity, "fingerprint": self.fingerprint,
                  "local_details": identity.local_details(), "capture_sizes": sizes,
                  "kv_tokens": kv_tokens, "kv_blocks": kv_tokens // self.args.page_size,
                  "kv_sizing": "pre_capture", "startup_seconds": time.perf_counter() - self.started,
                  "capture_seconds": self.capture_seconds, "memory": memory,
                  "pre_model": self.pre_model, "capture_before": self.capture_before,
                  "capture_after": self.capture_after, "policy_status": self.policy_status,
                  "policy_mode": os.environ.get("GRAPHBUDGET_POLICY", "native"),
                  "calibration": self.calibrate, "diagnostic": self.diagnostic,
                  "status": "worker_initialized"}
        record["worker_init_seconds"] = record["startup_seconds"]
        record["startup_metric"] = "worker_initialized; HTTP deadline is enforced by lifecycle.json"
        self.ready_record = record
        if self.plan:
            budget = self.plan["budget"]
            errors = []
            if sizes != self.plan["selected"]["capture_sizes"]:
                errors.append("actual capture set differs")
            if kv_tokens < budget["min_kv_tokens"]:
                errors.append("actual KV below required minimum")
            if memory["device_used_bytes"] > budget["allowed_device_bytes"]:
                errors.append("device usage exceeds hard budget")
            if record["startup_seconds"] > budget["max_startup_seconds"]:
                errors.append("startup exceeds hard budget")
            if errors:
                record.update(status="rejected_before_service", errors=errors)
                write_json(self.output / "ready.json", record)
                raise RuntimeError("; ".join(errors))
        write_json(self.output / "ready.json", record)

    def begin(self, batch, eligible, skip_attn_backend_init=False):
        if not batch.forward_mode.is_decode():
            self.pending = None
            return eligible
        self.step_id += 1
        graph = self.runner.graph_runner
        padded = None
        if eligible and graph:
            index = bisect_left(graph.capture_bs, batch.batch_size)
            if index < len(graph.capture_bs):
                padded = graph.capture_bs[index]
        context = None
        # Never read GPU seq_lens, call GPU .item(), or copy D2H to make a decision.
        cpu_lengths = batch.seq_lens_cpu
        if ((self.policy is not None or self.calibrate or self.diagnostic) and
                not skip_attn_backend_init and cpu_lengths is not None and
                cpu_lengths.device.type == "cpu"):
            values = cpu_lengths.tolist()
            if (len(values) == batch.batch_size and values and
                    all(type(v) is int and v == values[0] and v > 0 for v in values)):
                context = {"batch_size": batch.batch_size, "padded_size": padded,
                           "context_length": values[0]}
        decision = decide_policy(
            self.policy, context, bool(eligible), actual_fingerprint=self.fingerprint,
            force_eager=self.force_eager and not skip_attn_backend_init,
            test_only_force_eager=self.test_only)
        if decision["selected_path"] == "native":
            decision["selected_path"] = "graph" if eligible else "eager"
        reason = decision["reason"]
        if not eligible:
            if not graph:
                reason = "graph_disabled"
            elif batch.batch_size > max(graph.capture_bs):
                reason = "capture_ceiling"
            elif self.args.disable_cuda_graph_padding and batch.batch_size not in graph.capture_bs:
                reason = "uncaptured_exact_size"
            else:
                reason = "upstream_eligibility"
        self.pending = {"run_id": self.run_id, "step_id": self.step_id,
                        "batch_size": batch.batch_size, "padded_size": padded,
                        "eligible": bool(eligible), "selected_path": decision["selected_path"],
                        "used_path": None, "actual_capture_size": None, "reason": reason,
                        "context": context, "forced_test": self.force_eager,
                        "request_ids": getattr(batch, "graphbudget_request_ids", None)}
        if self.calibrate:
            self.torch.cuda.synchronize(self.runner.gpu_id)
            self.step_started = time.perf_counter_ns()
        return decision["selected_path"] == "graph"

    def replayed(self, size):
        # Called only immediately after self.graphs[graph_key].replay() returns.
        if self.pending is not None:
            self.pending.update(used_path="graph", actual_capture_size=size)

    def eager_returned(self):
        # Called only after forward_decode has returned on the real eager branch.
        if self.pending is not None:
            self.pending["used_path"] = "eager"

    def finish(self):
        if self.pending is None:
            return
        record = self.pending
        if self.calibrate:
            self.torch.cuda.synchronize(self.runner.gpu_id)
            record["complete_forward_ns"] = time.perf_counter_ns() - self.step_started
        if record["selected_path"] != record["used_path"]:
            raise RuntimeError("Selected/actual runtime path mismatch")
        if record["used_path"] == "graph" and record["actual_capture_size"] != record["padded_size"]:
            raise RuntimeError("Chosen/actual capture size mismatch")
        key = (record["eligible"], record["selected_path"], record["used_path"],
               record["batch_size"], record["actual_capture_size"], record["reason"])
        self.counts[key] += 1
        if self.diagnostic:
            if len(self.steps) < 200000:
                self.steps.append(record)
            else:
                self.dropped_steps += 1
        self.pending = None

    def flush(self):
        # Called synchronously from the scheduler's idle cache-flush barrier.
        counts = [dict(zip(("eligible", "selected_path", "used_path", "batch_size",
                            "actual_capture_size", "reason"), key), count=value)
                  for key, value in self.counts.items()]
        write_json(self.output / "paths.json", {
            "schema_version": 1, "run_id": self.run_id, "fingerprint": self.fingerprint,
            "pid": os.getpid(), "counts": counts, "steps": self.steps,
            "dropped_steps": self.dropped_steps, "calibration": self.calibrate,
            "diagnostic": self.diagnostic, "provenance": "patched_runtime_calls",
        })
