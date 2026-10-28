#!/usr/bin/env python3
"""Compare complete native and patched residual RMSNorm chains in separate CUDA graphs."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import statistics
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from graphbudget.experiment import gpu_lease
from graphbudget.identity import file_digest
from graphbudget.runtime import write_json
from graphbudget.transport import boot_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu-lock", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "RUNNING",
        "scope": "operator-chain graph replay only; not model serving throughput",
        "boot_id": boot_id(),
        "script_sha256": file_digest(__file__),
        "protocol": {
            "widths": [896, 1536], "rows": [1, 4, 16], "seed": 28472,
            "blocks": 7, "replays_per_measurement": 100,
            "order": "BCCB then CBBC, alternating",
            "warmup_replays_per_arm": 100,
            "controls": "two B samples before and after the interleaved blocks",
            "mechanism_gate": "896x16 median latency reduction > max(5%, absolute BB controls and BB drift), at least 5/7 blocks improved, all bitwise checks pass",
            "event_scope": "100 graph replays on the device timeline; host submission gaps may be included",
        },
        "shapes": [],
    }

    def save():
        write_json(output / "result.json", result)

    save()
    try:
        with open(args.gpu_lock, "a") as lock, gpu_lease():
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # This switch is read during the SGLang module import. Both
            # graphs keep their captured operations for the whole experiment.
            os.environ["GRAPHBUDGET_RESIDUAL_NORM"] = "1"
            import torch
            from safetensors import safe_open
            import graphbudget.fused_norm as fused
            import sglang.srt.layers.layernorm as layernorm

            device = torch.cuda.get_device_properties(0)
            if (device.major, device.minor) != (12, 0) or device.name != "NVIDIA GeForce RTX 5070 Laptop GPU":
                raise RuntimeError(f"Unsupported device for this candidate: {device.name}")
            if layernorm._GRAPHBUDGET_RESIDUAL_NORM != "1":
                raise RuntimeError("Patched residual-norm flag was not selected at import")
            result["environment"] = {
                "torch": torch.__version__, "cuda": torch.version.cuda, "device": device.name,
                "capability": [device.major, device.minor],
                "fused_norm_sha256": file_digest(fused.__file__),
                "layernorm_sha256": file_digest(layernorm.__file__),
                "model_weights_sha256": file_digest(Path(args.model_path) / "model.safetensors"),
                "selected_residual_flag": layernorm._GRAPHBUDGET_RESIDUAL_NORM,
            }
            torch.manual_seed(28472)
            torch.set_grad_enabled(False)

            def equal(left, right):
                return all(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
                           for a, b in zip(left, right))

            def capture(fn):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(16):
                        fn()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = fn()
                graph.replay()
                torch.cuda.synchronize()
                return graph, outputs

            def measure(graph):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                wall_start = time.perf_counter()
                start.record()
                for _ in range(100):
                    graph.replay()
                end.record()
                end.synchronize()
                return {"event_us_per_replay": start.elapsed_time(end) * 10,
                        "wall_us_per_replay": (time.perf_counter() - wall_start) * 10000}

            weights_path = Path(args.model_path) / "model.safetensors"
            with safe_open(str(weights_path), framework="pt", device="cpu") as weights:
                model_weight = weights.get_tensor("model.layers.0.input_layernorm.weight").to("cuda")
            if model_weight.numel() != 896 or model_weight.dtype != torch.bfloat16:
                raise RuntimeError("Unexpected model weight shape or dtype")

            with torch.inference_mode(), \
                 patch.object(layernorm, "is_batch_invariant_mode_enabled", return_value=True), \
                 patch.object(layernorm, "get_global_server_args", return_value=SimpleNamespace(rl_on_policy_target="")):
                for n in (896, 1536):
                    for m in (1, 4, 16):
                        item = {"M": m, "N": n, "status": "RUNNING", "blocks": []}
                        result["shapes"].append(item)
                        save()
                        layer = layernorm.RMSNorm(n, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
                        layer.weight.copy_(model_weight if n == 896 else torch.randn_like(layer.weight))
                        x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
                        residual = torch.randn_like(x)
                        native = lambda: layer.forward_native(x, residual)
                        candidate = lambda: layer.forward_cuda(x, residual)
                        original = (x.clone(), residual.clone())
                        expected = native()
                        with patch.object(layernorm, "_graphbudget_residual_norm",
                                          wraps=fused.residual_rms_norm) as called:
                            actual = candidate()
                            if called.call_count != 1:
                                raise AssertionError("Patched C dispatch was not selected")
                        torch.cuda.synchronize()
                        if not equal(expected, actual):
                            raise AssertionError(f"eager bitwise mismatch at {m}x{n}")
                        graph_b, output_b = capture(native)
                        with patch.object(layernorm, "_graphbudget_residual_norm",
                                          wraps=fused.residual_rms_norm) as captured:
                            graph_c, output_c = capture(candidate)
                            if captured.call_count < 1:
                                raise AssertionError("Patched C was not reached during capture")
                        if graph_b is graph_c or not equal(output_b, output_c) or not equal(expected, output_b):
                            raise AssertionError(f"independent graph mismatch at {m}x{n}")
                        if not torch.equal(x, original[0]) or not torch.equal(residual, original[1]):
                            raise AssertionError("Capture changed input tensors")
                        replay_checks = []
                        for index in range(2):
                            x.add_(0.25 * (index + 1))
                            residual.mul_(1.125)
                            frozen_inputs = (x.clone(), residual.clone())
                            expected = native()
                            graph_b.replay()
                            graph_c.replay()
                            torch.cuda.synchronize()
                            if not equal(expected, output_b) or not equal(expected, output_c):
                                raise AssertionError(f"changed-input graph mismatch at {m}x{n}")
                            if not torch.equal(x, frozen_inputs[0]) or not torch.equal(residual, frozen_inputs[1]):
                                raise AssertionError("Replay changed input tensors")
                            replay_checks.append({"changed_input_replay": index + 1, "bitwise_equal": True})
                        for _ in range(100):
                            graph_b.replay()
                            graph_c.replay()
                        torch.cuda.synchronize()
                        before = [measure(graph_b), measure(graph_b)]
                        for block in range(7):
                            order = "BCCB" if block % 2 == 0 else "CBBC"
                            samples = [(arm, measure(graph_b if arm == "B" else graph_c)) for arm in order]
                            b = statistics.mean(value["event_us_per_replay"] for arm, value in samples if arm == "B")
                            c = statistics.mean(value["event_us_per_replay"] for arm, value in samples if arm == "C")
                            item["blocks"].append({"block": block, "order": order, "samples": samples,
                                                   "B_us": b, "C_us": c, "relative_latency_change": c / b - 1})
                        after = [measure(graph_b), measure(graph_b)]
                        if not equal(output_b, output_c):
                            raise AssertionError("Outputs differ after timed replay")
                        b0, b1 = [v["event_us_per_replay"] for v in before]
                        b2, b3 = [v["event_us_per_replay"] for v in after]
                        noise = max(abs(b1 / b0 - 1), abs(b3 / b2 - 1),
                                    abs(statistics.median([b2, b3]) / statistics.median([b0, b1]) - 1))
                        change = statistics.median(v["relative_latency_change"] for v in item["blocks"])
                        item.update(status="PASS", dispatch_verified=True, separate_graphs=True,
                                    changed_input_checks=replay_checks, inputs_unchanged=True,
                                    B_us_median=statistics.median(v["B_us"] for v in item["blocks"]),
                                    C_us_median=statistics.median(v["C_us"] for v in item["blocks"]),
                                    median_relative_latency_change=change,
                                    improved_blocks=sum(v["relative_latency_change"] < 0 for v in item["blocks"]),
                                    BB_before=before, BB_after=after, max_bb_noise=noise)
                        save()
                        print(f"{m}x{n}: B={item['B_us_median']:.3f}us C={item['C_us_median']:.3f}us "
                              f"median_change={change:+.2%} BB_noise={noise:.2%}", flush=True)
                        del graph_b, graph_c, output_b, output_c, expected, actual, layer, x, residual
                        torch.cuda.synchronize()
            primary = next(row for row in result["shapes"] if row["M"] == 16 and row["N"] == 896)
            reduction = -primary["median_relative_latency_change"]
            floor = max(0.05, primary["max_bb_noise"])
            result["mechanism_gate"] = {
                "status": "SUPPORTED" if reduction > floor and primary["improved_blocks"] >= 5 else "NOT_DISTINGUISHED",
                "scope": "896x16 operator chain only",
                "median_latency_reduction": reduction, "noise_floor": floor,
                "improved_blocks": primary["improved_blocks"],
            }
            result["status"] = "PASS"
            save()
    except BaseException as error:
        result.update(status="FAIL", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        save()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
