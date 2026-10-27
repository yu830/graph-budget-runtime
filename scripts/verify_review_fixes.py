"""Bounded reliability smoke, not a new performance evaluation.

Use the pinned existing runtime. Measures ONE predeclared configuration, then
checks plan application, missing/stale/corrupt optional costs, the public CLI,
and exact outputs. Fault tables are invalid test inputs, never measured costs.
All outputs go to a new directory; failures and original results are retained.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

from graphbudget.benchmark import compare_exact_tokens, run_benchmark
from graphbudget.experiment import gpu_lease, measure_candidates, run_case
from graphbudget.identity import code_digest, digest, file_digest
from graphbudget.lifecycle import OwnedProcess, stop_owned
from graphbudget.plan import select_plan
from graphbudget.runtime import write_json


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def live_group(pgid):
    """Read-only Linux ownership check; never signal a discovered process."""
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def direct_case(server_args, workload, output, plan, profile, port):
    """Exercise public launch (its supervisor, not the experiment helper)."""
    output.mkdir()
    runtime = output / "runtime"
    run_id = uuid.uuid4().hex
    command = [sys.executable, "-m", "graphbudget.cli", "launch", "--output", str(runtime),
               "--run-id", run_id, "--plan", str(plan), "--policy", "cost", "--profile", str(profile),
               "--observe", "diagnostic", "--startup-timeout", "600", "--", *server_args,
               "--host", "127.0.0.1", "--port", str(port)]
    write_json(output / "invocation.json", {"command": command, "run_id": run_id})
    process = None
    result = {"run_id": run_id, "status": "failed", "reason": "incomplete"}
    with gpu_lease(), (output / "server.log").open("w") as log:
        try:
            process = OwnedProcess(command, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 620
            record_path = runtime / "lifecycle.json"
            while not record_path.exists():
                require(process.poll() is None, "Public CLI exited before lifecycle receipt")
                require(time.monotonic() < deadline, "Public CLI did not produce a bounded startup receipt")
                time.sleep(.1)
            lifecycle = read(record_path)
            require(lifecycle["status"] == "http_ready", f"Direct startup failed: {lifecycle}")
            ready = read(runtime / "ready.json")
            require(lifecycle["run_id"] == ready["run_id"] == run_id, "Direct run ownership mismatch")
            bench = run_benchmark(lifecycle["url"], workload, output=output / "requests.json",
                                  run_id=run_id, timeout_s=120, reset_cache=True)
            require(bench["summary"]["completed_requests"] == bench["summary"]["planned_requests"],
                    "Public CLI requests failed")
            request = urllib.request.Request(lifecycle["url"] + "/flush_cache", method="POST")
            with urllib.request.urlopen(request, timeout=10) as response:
                response.read()
            paths = read(runtime / "paths.json")
            write_json(output / "paths_after.json", paths)
            result.update(status="ok", reason="", ready=ready, http_readiness=lifecycle, benchmark=bench,
                          entry="public_cli_launch")
        except Exception as error:
            result["reason"] = f"{type(error).__name__}: {error}"
        finally:
            stop_owned(process)
            result["supervisor_exit_code"] = process.returncode if process else None
            if "http_readiness" in result:
                result["owned_server_group_cleaned"] = not live_group(result["http_readiness"]["server_pid"])
                if not result["owned_server_group_cleaned"]:
                    result.update(status="failed", reason="Owned server group survived public CLI shutdown")
            write_json(output / "case.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=30170)
    parser.add_argument("--reference", help="Optional original 2026-10-26 calibration requests.json for exact comparison")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(not output.exists(), "Use a new output directory; previous evidence is immutable")
    output.mkdir(parents=True)
    os.environ["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"] = "0"
    workload = ROOT / "examples/workloads/calibration.json"
    plan_settings = {"mem_fraction_static": .85, "max_total_tokens": 4096,
                     "max_running_requests": 16, "context_length": 4096}
    server_args = ["--model-path", args.model_path, "--tokenizer-path", args.model_path,
                   "--revision", args.revision, "--dtype", "bfloat16", "--device", "cuda",
                   "--tensor-parallel-size", "1", "--attention-backend", "triton",
                   "--disable-overlap-schedule", "--stream-output", "--random-seed", "0",
                   "--enable-deterministic-inference"]
    for key, value in plan_settings.items():
        server_args += ["--" + key.replace("_", "-"), str(value)]
    config = {"server_args": server_args, "plan_server_args": plan_settings,
              "calibration_workload": str(workload), "port": args.port,
              "candidates": [{"id": "review-smoke-only", "capture_sizes": [1, 2, 4, 8, 12, 16]}],
              "budget": {"allowed_device_bytes": 7*1024**3, "min_kv_tokens": 1296, "max_startup_seconds": 600}}
    modules = {p.name: code_digest(p) for p in sorted((ROOT / "src/graphbudget").glob("*.py"))}
    summary = {"schema_version": 1, "status": "RUNNING", "scope": "reliability smoke, one measured candidate; no cost recalibration or performance adoption",
               "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
               "runtime_code_sha256": digest(modules), "module_sha256": modules,
               "workload_sha256": file_digest(workload), "comparisons": {}, "cases": {}}
    write_json(output / "summary.json", summary)
    write_json(output / "experiment.json", config)
    try:
        print("measure: one configuration", flush=True)
        measure_candidates(output / "experiment.json", output / "measure")
        measurements = read(output / "measure/measurements.json")
        plan = select_plan(measurements["identity"], config["budget"], measurements["candidates"])
        plan_path = output / "plan.json"
        write_json(plan_path, plan)
        measured = read(output / "measure/candidate-0/case.json")
        summary["fingerprint"] = measured["ready"]["fingerprint"]
        summary["cases"]["measurement"] = {"status": measured["status"], "run_id": measured["run_id"]}
        baseline = measured["benchmark"]
        if args.reference:
            historical = compare_exact_tokens(read(args.reference), baseline)
            write_json(output / "historical-exact.json", historical)
            require(historical["equivalent"], "Historical exact output comparison failed")
            summary["historical_exact"] = historical
        native = run_case(server_args, workload, output / "native-plan", plan=plan_path,
                          diagnostic=True, port=args.port)
        require(native["status"] == "ok", f"Native plan failed: {native['reason']}")
        contexts = {tuple(row["context"][k] for k in ("batch_size", "padded_size", "context_length"))
                    for row in read(output / "native-plan/paths_after.json")["steps"]
                    if row.get("context") and row["eligible"]}
        require(contexts, "No actual homogeneous context observed for corrupt-profile injection")
        # The following fields are deliberately fabricated fault inputs. They
        # must never be described or exported as a measured cost calibration.
        fault = {"schema_version": 1, "fingerprint": summary["fingerprint"],
                 "capture_sizes": plan["selected"]["capture_sizes"], "enabled": True,
                 "fixture_kind": "INVALID_FAULT_INJECTION_NOT_MEASURED_COSTS", "entries": [
                     dict(zip(("batch_size", "padded_size", "context_length"), key),
                          graph_ns=1, eager_ns=1, noise_ns=0, samples=3, provenance=[])
                     for key in sorted(contexts)]}
        write_json(output / "fault-provenance.json", fault)
        overflow = copy.deepcopy(fault)
        for row in overflow["entries"]:
            row.update(provenance="measured", graph_ns=10**400)
        write_json(output / "fault-overflow.json", overflow)
        cases = {"native-plan": native}
        for name, profile in (("missing-profile", output / "absent.json"),
                              ("stale-profile", ROOT / "results/2026-10-26/profile.json"),
                              ("bad-provenance", output / "fault-provenance.json")):
            print(name, flush=True)
            cases[name] = run_case(server_args, workload, output / name, plan=plan_path,
                                   policy="cost", profile=profile, diagnostic=True, port=args.port)
            require(cases[name]["status"] == "ok", f"{name} failed: {cases[name]['reason']}")
        print("direct CLI: bad overflow profile", flush=True)
        cases["direct-overflow"] = direct_case(server_args, workload, output / "direct-overflow",
                                               plan_path, output / "fault-overflow.json", args.port)
        for name, case in cases.items():
            require(case["status"] == "ok", f"{name} failed: {case['reason']}")
            require(case["ready"]["fingerprint"] == summary["fingerprint"], "Runtime identity drift")
            comparison = compare_exact_tokens(baseline, case["benchmark"])
            write_json(output / name / "exact-comparison.json", comparison)
            require(comparison["equivalent"], f"Exact outputs differ: {name}")
            trace = read(output / name / "paths_after.json")
            timed = [r for r in trace["steps"] if r.get("request_ids") and
                     all(f"/{case['run_id']}/" in rid for rid in r["request_ids"])]
            require(timed and all(r["selected_path"] == r["used_path"] for r in timed), "Missing/mismatched actual path")
            require(all(r["used_path"] == ("graph" if r["eligible"] else "eager") for r in timed), "Fault input changed native dispatch")
            reasons = sorted({r["reason"] for r in timed})
            if name == "bad-provenance":
                require("profile_not_measured" in reasons, "Bad provenance never exercised")
            if name == "direct-overflow":
                require("profile_entry_invalid" in reasons, "Overflow timing never exercised")
            if name in ("missing-profile", "stale-profile"):
                require(case["ready"]["policy_status"].startswith("native_fallback:"), "Profile did not fail soft")
            require(not live_group(case["http_readiness"]["server_pid"]), "Owned process group still live")
            summary["comparisons"][name] = comparison
            summary["cases"][name] = {"status": case["status"], "run_id": case["run_id"],
                                       "policy_status": case["ready"]["policy_status"], "reasons": reasons,
                                       "timed_decode_steps": len(timed), "http_readiness": case["http_readiness"],
                                       "case_sha256": file_digest(output / name / "case.json")}
        summary.update(status="PASS", performance="NOT_RERUN_NO_NEW_GAIN_CLAIM")
    except Exception as error:
        summary.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "fingerprint": summary["fingerprint"],
                      "exact_comparisons": len(summary["comparisons"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
