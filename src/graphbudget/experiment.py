"""Serial fresh-process experiments; calibration and held-out testing stay separate."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid

from .benchmark import run_benchmark, compare_exact_tokens, load_workload, Workload
from .identity import digest, file_digest
from .runtime import write_json


@contextmanager
def gpu_lease():
    """Serialize this project's GPU experiments (does not reserve other apps)."""
    import fcntl
    path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / f"graphbudget-{os.getuid()}.lock"
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another GraphBudget GPU experiment owns the lease") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _stop(process):
    if process is None:
        return
    # The launcher may have exited while its own workers still occupy the GPU.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        raise


def _flush(process, output, ready, url):
    pid = ready["pid"]
    if os.getpgid(pid) != process.pid:
        raise RuntimeError("Worker PID is outside the experiment process group")
    path = output / "runtime/paths.json"
    before = path.stat().st_mtime_ns if path.exists() else None
    # The patched idle management barrier flushes observations synchronously.
    # This maintenance request is outside all timed requests and preserves KV
    # backing addresses. No dependence on Python signal delivery while in C++.
    request = urllib.request.Request(url + "/flush_cache", method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()
    end = time.monotonic() + 5
    while time.monotonic() < end:
        if path.exists() and path.stat().st_mtime_ns != before:
            return json.loads(path.read_text())
        if process.poll() is not None:
            raise RuntimeError("Server exited before observation flush")
        time.sleep(0.05)
    raise RuntimeError("Worker did not flush observations")


def run_case(server_args, workload, output, *, capture_sizes=None, plan=None,
             profile=None, policy="native", diagnostic=False, calibrate=False,
             startup_timeout=600, request_timeout=120, port=30150, warmup=True):
    """One server, one warmup, one workload, raw outputs, and no hidden retries."""
    output = Path(output).resolve()
    workload = workload if isinstance(workload, Workload) else load_workload(workload)
    if output.exists():
        raise ValueError(f"Refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)
    run_id = uuid.uuid4().hex
    command = [sys.executable, "-m", "graphbudget.cli", "launch", "--output",
               str(output / "runtime"), "--run-id", run_id, "--policy", policy,
               "--observe", "diagnostic" if diagnostic else "counts"]
    if policy == "test_eager":
        command += ["--test-only"]
    if calibrate:
        command += ["--calibrate"]
    if plan:
        command += ["--plan", str(Path(plan).resolve())]
    if profile:
        command += ["--profile", str(Path(profile).resolve())]
    command += ["--", *server_args, "--host", "127.0.0.1", "--port", str(port)]
    if capture_sizes is not None:
        command += ["--cuda-graph-bs", *map(str, capture_sizes)]
    write_json(output / "invocation.json", {"command": command, "run_id": run_id,
                                             "workload_sha256": digest(workload.to_dict())})
    process = sampler = None
    result = {"run_id": run_id, "status": "failed", "reason": "incomplete"}
    with gpu_lease(), (output / "server.log").open("w") as log, (output / "device.csv").open("w") as telemetry:
        try:
            start = time.perf_counter()
            sampler = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits", "-lms", "100"],
                stdout=telemetry, stderr=subprocess.STDOUT, start_new_session=True)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; see server.log")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                time.sleep(0.25)
            else:
                raise TimeoutError("Server startup deadline exceeded")
            startup_seconds = time.perf_counter() - start
            ready = json.loads((output / "runtime/ready.json").read_text())
            if ready["run_id"] != run_id:
                raise RuntimeError("Ready record belongs to another process")
            if plan:
                hard_budget = json.loads(Path(plan).read_text())["budget"]
                if startup_seconds > hard_budget["max_startup_seconds"]:
                    raise RuntimeError("HTTP-ready startup exceeds frozen budget")
            if warmup:
                warmed = run_benchmark(url, workload, output=output / "warmup.json",
                                       timeout_s=request_timeout, run_id=run_id + "-warmup", reset_cache=True)
                if warmed["summary"]["completed_requests"] != warmed["summary"]["planned_requests"]:
                    raise RuntimeError("Warmup requests failed; do not measure an unwarmed server")
            before = _flush(process, output, ready, url)
            write_json(output / "paths_before.json", before)
            bench = run_benchmark(url, workload, output=output / "requests.json",
                                  timeout_s=request_timeout, run_id=run_id, reset_cache=True)
            after = _flush(process, output, ready, url)
            write_json(output / "paths_after.json", after)
            fields = ("eligible", "selected_path", "used_path", "batch_size", "actual_capture_size", "reason")
            previous = {tuple(row[k] for k in fields): row["count"] for row in before["counts"]}
            timed_counts = []
            for row in after["counts"]:
                count = row["count"] - previous.get(tuple(row[k] for k in fields), 0)
                if count < 0:
                    raise RuntimeError("Runtime counters moved backwards")
                if count:
                    timed_counts.append({**row, "count": count})
            write_json(output / "timed_paths.json", {"run_id": run_id, "fingerprint": ready["fingerprint"],
                                                     "counts": timed_counts,
                                                     "scope": "after-minus-before timed requests; excludes warmup"})
            if bench["summary"]["completed_requests"] != bench["summary"]["planned_requests"]:
                raise RuntimeError("Some planned requests failed or timed out")
            result.update(status="ok", reason="", startup_seconds=startup_seconds,
                          ready=ready, benchmark=bench, diagnostic=diagnostic, calibration=calibrate,
                          timed_path_counts=timed_counts)
        except Exception as error:
            result.update(reason=f"{type(error).__name__}: {error}")
        finally:
            try:
                _stop(process)
            finally:
                _stop(sampler)
    samples = []
    for line in (output / "device.csv").read_text().splitlines():
        parts = line.split(",")
        if len(parts) == 5:
            try:
                samples.append(int(float(parts[1]) * 1024**2))
            except ValueError:
                pass
    result["observed_peak_device_used_bytes"] = max(samples) if samples else None
    result["device_peak_scope"] = "100ms sampling of whole device over startup/warmup/test; not an absolute CUDA peak"
    write_json(output / "case.json", result)
    return result


def measure_candidates(experiment, output):
    config = json.loads(Path(experiment).read_text())
    workload = load_workload(config["calibration_workload"])
    if workload.purpose != "calibration":
        raise ValueError("Selection requires a separate calibration workload")
    candidates = config["candidates"]
    if not 1 <= len(candidates) <= 5:
        raise ValueError("Use at most five predeclared candidates")
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Measurement output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_experiment.json", config)
    measurements = []
    actual_identity = None
    for index, candidate in enumerate(candidates):
        case_dir = output / f"candidate-{index}"
        case = run_case(config["server_args"], workload, case_dir,
                        capture_sizes=candidate["capture_sizes"],
                        startup_timeout=config["budget"]["max_startup_seconds"],
                        port=config.get("port", 30150))
        measurement = {"id": candidate["id"], "capture_sizes": candidate["capture_sizes"],
                       "server_args": config["plan_server_args"], "status": case["status"],
                       "reason": case["reason"], "evidence": {"provenance": "measured",
                       "run_id": case["run_id"], "case_sha256": file_digest(case_dir / "case.json"),
                       "raw_directory": str(case_dir)}}
        if case["status"] == "ok":
            ready = case["ready"]
            if actual_identity is None:
                actual_identity = ready["identity"]
            if ready["identity"] != actual_identity:
                raise ValueError("Candidate identity changed during calibration")
            if ready["capture_sizes"] != candidate["capture_sizes"]:
                measurement.update(status="unsupported", reason="Actual capture sizes differ from requested")
            measurement.update(ready={**ready["memory"], "kv_tokens": ready["kv_tokens"],
                                      "startup_seconds": case["startup_seconds"],
                                      "capture_seconds": ready["capture_seconds"]},
                               calibration_seconds=case["benchmark"]["summary"]["makespan_s"],
                               runtime_peak_device_used_bytes=(
                                   max(case["observed_peak_device_used_bytes"], ready["memory"]["device_used_bytes"])
                                   if case["observed_peak_device_used_bytes"] is not None else None))
            measurement["evidence"].update(fingerprint=ready["fingerprint"],
                                            actual_capture_sizes=ready["capture_sizes"],
                                            requested_capture_sizes=candidate["capture_sizes"])
        measurements.append(measurement)
        write_json(output / "measurements.json", {"identity": actual_identity, "budget": config["budget"],
                                                  "candidates": measurements})
    return measurements


def build_cost_profile(graph_dirs, eager_dirs):
    """Use independent, logically matched autoregressive trajectories, never replay
    two alternatives on an already-modified KV state. Synchronization is confined
    to these diagnostic runs. Raw sources remain mandatory and digest-bound.
    """
    if len(graph_dirs) != len(eager_dirs) or len(graph_dirs) < 3:
        raise ValueError("Need at least three paired fresh graph/eager calibration runs")
    costs = {}
    sources = []
    common = None
    capture_sizes = None
    for graph_dir, eager_dir in zip(graph_dirs, eager_dirs):
        paths = [Path(graph_dir), Path(eager_dir)]
        cases = [json.loads((p / "case.json").read_text()) for p in paths]
        if not compare_exact_tokens(cases[0]["benchmark"], cases[1]["benchmark"])["equivalent"]:
            raise ValueError("Graph/eager exact-token equivalence failed")
        for i, case in enumerate(cases):
            ready = case["ready"]
            if case["status"] != "ok" or not case["calibration"] or not case["diagnostic"]:
                raise ValueError("Cost rows require successful diagnostic calibration")
            if case["benchmark"]["workload"]["purpose"] != "calibration":
                raise ValueError("Held-out data cannot train the policy")
            signature = (ready["fingerprint"], ready["kv_tokens"], ready["capture_sizes"])
            if common is None:
                common = signature
                capture_sizes = ready["capture_sizes"]
            if signature != common:
                raise ValueError("Identity, KV capacity or capture set differs between paths")
            sources.append({"case_sha256": file_digest(paths[i] / "case.json"),
                            "paths_sha256": file_digest(paths[i] / "paths_after.json"),
                            "run_id": case["run_id"]})
        paired = []
        for p, case in zip(paths, cases):
            trace = json.loads((p / "paths_after.json").read_text())
            if trace["dropped_steps"] or trace["run_id"] != case["run_id"]:
                raise ValueError("Incomplete or mismatched native trace")
            if trace.get("provenance") != "patched_runtime_calls" or trace["fingerprint"] != common[0]:
                raise ValueError("Cost input must bind the actual native-call trace to the same worker identity")
            rows = {}
            for row in trace["steps"]:
                rids = row.get("request_ids")
                if not row.get("context") or not rids or not row["eligible"]:
                    continue
                # Request IDs include the run ID; retain stable workload identity.
                if not all(f"/{case['run_id']}/" in rid for rid in rids):
                    continue
                stable_rids = tuple(sorted(rid.rsplit("/", 1)[-1] for rid in rids))
                ctx = row["context"]
                key = (ctx["batch_size"], ctx["padded_size"], ctx["context_length"])
                logical_key = (key, stable_rids)
                if logical_key in rows:
                    raise ValueError("Repeated logical KV state; cannot identify an equivalent sample")
                rows[logical_key] = row
            paired.append(rows)
        for logical_key in paired[0].keys() & paired[1].keys():
            g, e = paired[0][logical_key], paired[1][logical_key]
            if g["used_path"] != "graph" or e["used_path"] != "eager":
                continue
            if not e["forced_test"] or g["forced_test"]:
                raise ValueError("Unexpected path/calibration mode")
            key = logical_key[0]
            values = costs.setdefault(key, [[], []])
            values[0].append(g["complete_forward_ns"])
            values[1].append(e["complete_forward_ns"])
    entries = []
    for key, (g, e) in sorted(costs.items()):
        if len(g) != len(e):
            raise ValueError("Internal error: unpaired calibration samples")
        if len(g) < 3:
            continue
        graph_ns, eager_ns = statistics.median(g), statistics.median(e)
        entries.append(dict(zip(("batch_size", "padded_size", "context_length"), key),
                            graph_ns=graph_ns, eager_ns=eager_ns,
                            noise_ns=max(max(g)-min(g), max(e)-min(e), 0.02*max(graph_ns,eager_ns)),
                            samples=len(g), provenance="measured"))
    return {"schema_version": 1, "fingerprint": common[0], "capture_sizes": capture_sizes,
            "enabled": True, "entries": entries, "sources": sources,
            "timing_scope": "synchronized complete _forward_raw incl padding/copy/metadata; diagnostic only",
            "noise_rule": "max(full graph range, full eager range, 2 percent of slower median)"}
