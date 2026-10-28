"""Fresh-process A/A, B/B and five B/C decode pairs with frozen inputs.

Run with an installed GraphBudget/patched SGLang environment and this checkout:
python /absolute/checkout/scripts/run_paired_decode.py --output /new/output \
  --model-path /models/snapshots/REVISION
A=native, B=upstream_gemm, C=v2_residual. This script never adopts a candidate.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from graphbudget.benchmark import compare_exact_tokens, load_workload
from graphbudget.identity import code_digest, digest, file_digest, model_identity
from graphbudget.runtime import write_json
from graphbudget.transport import HTTPTransport, boot_id

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"A": "native", "B": "upstream_gemm", "C": "v2_residual"}
SWITCHES = {
    "A": {"GRAPHBUDGET_SMALL_M_GEMM": "0", "GRAPHBUDGET_RESIDUAL_NORM": "0"},
    "B": {"GRAPHBUDGET_SMALL_M_GEMM": "upstream_42145", "GRAPHBUDGET_RESIDUAL_NORM": "0"},
    "C": {"GRAPHBUDGET_SMALL_M_GEMM": "upstream_42145", "GRAPHBUDGET_RESIDUAL_NORM": "1"},
}
METRIC = "effective_output_tokens_per_s"


def read_json(path):
    return json.loads(Path(path).read_text())


def schedule(smoke=False, protocol_name="legacy-v1"):
    if protocol_name == "bc-focused-v1" and not smoke:
        groups = [("bb_before", 0, "BB"),
                  *(("paired", i + 1, order) for i, order in enumerate(("BC", "CB", "BC", "CB", "BC"))),
                  ("bb_after", 0, "BB")]
    else:
        groups = [("smoke", 0, "BC")] if smoke else [
            ("bb_before", 0, "BB"), ("aa_before", 0, "AA"),
            *(("paired", i + 1, order) for i, order in enumerate(("BC", "CB", "BC", "CB", "BC"))),
            ("bb_after", 0, "BB"),
        ]
    cases = []
    for stage, pair, order in groups:
        for position, arm in enumerate(order):
            cases.append({"id": f"{len(cases) + 1:02}-{stage}-{pair}-{arm}",
                          "stage": stage, "pair": pair, "position": position,
                          "arm": arm, "mode": ARMS[arm]})
    return cases


def git_snapshot():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    files = {name: file_digest(ROOT / name) for name in sorted(set(paths))
             if name and (ROOT / name).is_file()}
    return {"head": git("rev-parse", "HEAD").strip(),
            "status": git("status", "--porcelain=v1", "--untracked-files=all"),
            "files_sha256": files}


def assert_frozen(protocol):
    if protocol.get("boot_id") is not None and boot_id() != protocol["boot_id"]:
        raise ValueError("WSL boot changed after protocol freeze")
    current = git_snapshot()
    if current["head"] != protocol["git"]["head"]:
        raise ValueError("Git HEAD changed after protocol freeze")
    if current["files_sha256"] != protocol["git"]["files_sha256"]:
        raise ValueError("Tracked/untracked source changed after protocol freeze")
    for item in protocol["workloads"].values():
        if file_digest(item["path"]) != item["file_sha256"]:
            raise ValueError(f"Frozen input changed: {item['path']}")
    if protocol.get("initial_warmup") and file_digest(protocol["initial_warmup"]["path"]) != protocol["initial_warmup"]["file_sha256"]:
        raise ValueError("Frozen initial warmup input changed")


def prepare_protocol(args, output, snapshot):
    # Reuse the case runner's owning input generator, rather than a second recipe.
    # Explicit sibling resolution also works with PYTHONSAFEPATH=1.
    sys.path.insert(0, str(ROOT / "scripts"))
    from profile_metadata import server_args, workload as make_initial_warmup
    from run_metadata_case import make_workloads as make_base_workloads
    from make_v2_workloads import make_workloads as make_extra_workloads

    inputs = output / "inputs"
    inputs.mkdir()
    primary, holdout = make_base_workloads(inputs)
    if args.smoke:
        primary["requests"] = primary["requests"][:64]
        primary["workload_id"] = "metadata-screen-64x128-v1"
        primary["purpose"] = "calibration"
    extra = make_extra_workloads(ROOT / "examples/workloads/calibration.json")
    # A pre-formal smoke must not consume the unseen acceptance workload.
    if args.smoke:
        extra.pop("acceptance")
    workloads = {"correctness": read_json(ROOT / "examples/workloads/calibration.json"),
                 "primary": primary, "holdout": holdout, **extra}
    frozen_workloads = {}
    for label, data in workloads.items():
        normalized = load_workload(data).to_dict()
        path = inputs / f"{label}.json"
        write_json(path, normalized)
        frozen_workloads[label] = {
            "path": str(path), "file_sha256": file_digest(path),
            "workload_sha256": digest(normalized), "requests": len(normalized["requests"]),
            "workers": (len(normalized["requests"]) if label == "correctness" else
                        1 if label == "interactive" else 16 if label == "acceptance" else 32),
        }
    manifest = read_json(ROOT / "src/graphbudget/patches/manifest.json")
    patch_hash = file_digest(ROOT / "src/graphbudget/patches/sglang-v0.5.9.patch")
    if patch_hash != manifest["patch_sha256"]:
        raise ValueError("Packaged SGLang patch differs from its manifest")
    module_hash = digest({p.name: code_digest(p) for p in sorted((ROOT / "src/graphbudget").glob("*.py"))})
    model = Path(args.model_path).resolve()
    inherited_paths = os.environ.get("PYTHONPATH", "")
    case_paths = [str(ROOT / "scripts")]
    if inherited_paths:
        case_paths.extend(str(Path(entry or ".").resolve()) for entry in inherited_paths.split(os.pathsep))
    protocol_name = getattr(args, "protocol", "legacy-v1")
    focused = protocol_name == "bc-focused-v1"
    initial_warmup = None
    if focused:
        normalized = load_workload(make_initial_warmup(4, 64)).to_dict()
        path = inputs / "initial-warmup.json"
        write_json(path, normalized)
        initial_warmup = {"path": str(path), "file_sha256": file_digest(path),
                          "workload_sha256": digest(normalized), "requests": 4, "workers": 4}
    return {
        "schema_version": 1, "kind": "graphbudget_frozen_paired_decode_protocol",
        "measurement_protocol": protocol_name,
        "initial_warmup": initial_warmup,
        "boot_id": boot_id(),
        "smoke": args.smoke, "git": snapshot, "python_executable": sys.executable,
        "python_version": sys.version, "model_path": str(model),
        "model_identity": model_identity(model, model.name),
        "expected_source": {"graphbudget_sha256": module_hash, "sglang": manifest},
        "patch_sha256": patch_hash, "workloads": frozen_workloads,
        "cases": schedule(args.smoke, protocol_name),
        "settings": {"max_running_requests": args.max_running_requests,
                     "max_total_tokens": args.max_total_tokens, "port": args.port,
                     "rounds_per_case": 1,
                     "primary_shaped_warmup": focused,
                     "warmup_rule": ("4x64 plus 43 calibration plus one 256x128 primary-shaped workload, identical in every arm"
                                     if focused else "4x64 plus 43 calibration"),
                     "server_arguments": server_args(str(model), args.max_running_requests,
                                                     args.max_total_tokens, args.port),
                     "gpu_lock": str(Path(args.gpu_lock).resolve()),
                     "triton_cache_dir": str(output / "triton-cache"),
                     "pythonpath": os.environ.get("PYTHONPATH", ""),
                     "case_pythonpath": os.pathsep.join(case_paths),
                     "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "")},
        "allowed_identity_differences": list(SWITCHES["A"]), "arm_switches": SWITCHES,
        "primary_metric": METRIC,
        "transport": HTTPTransport(f"http://127.0.0.1:{args.port}",
                                   getattr(args, "transport", "direct")).metadata,
        "bb_gate": {"max_absolute_relative_change": 0.05,
                    "independent_of_formal_pairs": True,
                    "rule": ("stop before B/C if independent B/B exceeds 5%" if focused
                             else "stop before A/A and B/C if independent B/B exceeds 5%")},
        "candidate_rule": ({"paired_rounds": 5, "minimum_positive_pairs": 4,
                            "median_gain_must_exceed": "max(0.05, abs(B/B before), abs(B/B after), abs(before-to-after B drift))",
                            "post_control_rule": "B/B after or before-to-after B drift above 5% requires review",
                            "minimum_relative_gain": 0.05}
                           if focused else
                           {"paired_rounds": 5, "minimum_positive_pairs": 4,
                            "median_gain_must_exceed": "max(0.05, max(abs(A/A relative change)))",
                            "minimum_relative_gain": 0.05}),
        "review_rules": ["noise or B/B drift at least as large as candidate effect",
                         "consistent latency regressions above a 5% screening threshold"],
        "acceptance_scope": "Primary throughput rule only; overall acceptance is never automatic.",
    }


def comparable_identity(identity, arm):
    value = copy.deepcopy(identity)
    runtime_env = value["runtime_env"]
    for key, expected in SWITCHES[arm].items():
        if runtime_env.get(key) != expected:
            raise ValueError(f"Arm {arm} has unexpected {key}: {runtime_env.get(key)!r}")
        del runtime_env[key]
    return value


def case_command(case, output, protocol, reference_dir):
    settings = protocol["settings"]
    command = [sys.executable, str(ROOT / "scripts/run_metadata_case.py"),
               "--mode", case["mode"], "--rounds", "1",
               "--output", str(output / "cases" / case["id"]),
               "--model-path", protocol["model_path"],
               "--max-running-requests", str(settings["max_running_requests"]),
               "--max-total-tokens", str(settings["max_total_tokens"]),
               "--port", str(settings["port"])]
    command += ["--transport", protocol["transport"]["effective_mode"]]
    if settings.get("primary_shaped_warmup"):
        command.append("--primary-shaped-warmup")
    if protocol["smoke"]:
        command.append("--screen")
    if reference_dir is not None:
        command += ["--reference-dir", str(reference_dir)]
    for label in ("interactive", "acceptance"):
        if label in protocol["workloads"]:
            item = protocol["workloads"][label]
            command += ["--extra-workload", label, item["path"], str(item["workers"])]
    return command


def verify_case(case, case_dir, protocol, reference_dir, reference_identity):
    ready = read_json(case_dir / "runtime/ready.json")
    identity = ready["identity"]
    expected = protocol["expected_source"]
    if identity["graphbudget_sha256"] != expected["graphbudget_sha256"]:
        raise ValueError("Loaded GraphBudget differs from the frozen checkout")
    for key in ("base_commit", "patch_sha256", "srt_tree_sha256", "patched_files"):
        if identity["sglang"][key] != expected["sglang"][key]:
            raise ValueError(f"Loaded SGLang differs from frozen manifest: {key}")
    if identity["model"] != protocol["model_identity"]:
        raise ValueError("Loaded model differs from frozen input")
    comparable = comparable_identity(identity, case["arm"])
    if reference_identity is not None and comparable != reference_identity:
        raise ValueError("Ready identity differs beyond the two declared arm switches")
    if read_json(case_dir / "result.json").get("status") != "PASS":
        raise ValueError("Case runner did not record PASS")
    results, comparisons = {}, {}
    if protocol.get("settings", {}).get("primary_shaped_warmup"):
        initial = read_json(case_dir / "warmup.json")
        initial_spec = protocol["initial_warmup"]
        if (digest(initial["workload"]) != initial_spec["workload_sha256"]
                or initial["run"]["max_workers"] != initial_spec["workers"]
                or initial["run"]["boot_id"] != protocol["boot_id"]
                or initial["run"]["transport"] != protocol["transport"]
                or initial["protocol"]["request_diagnostics"]
                or initial["cache_reset"]["status"] != "success"):
            raise ValueError("Initial warmup differs from frozen inputs, transport or boot")
        if protocol["transport"]["effective_mode"] == "direct":
            peers = {request["transport_peer"] for request in initial["requests"]}
            peers.add(initial["cache_reset"]["transport_peer"])
            if peers - {"target_loopback_port", "UNKNOWN"}:
                raise ValueError(f"Initial warmup observed an unexpected peer: {peers}")
        initial_reference = initial if reference_dir is None else read_json(reference_dir / "warmup.json")
        comparison = compare_exact_tokens(initial_reference, initial)
        comparisons["initial_warmup"] = comparison
        if (not comparison["equivalent"] or comparison["request_count"] != initial_spec["requests"]
                or initial["summary"]["completed_requests"] != initial_spec["requests"]):
            raise ValueError("Initial warmup exact request validation failed")
        warmup = read_json(case_dir / "primary-warmup.json")
        workload = protocol["workloads"]["primary"]
        if digest(warmup["workload"]) != workload["workload_sha256"]:
            raise ValueError("Primary-shaped warmup inputs differ from the frozen workload")
        if warmup["run"]["max_workers"] != workload["workers"]:
            raise ValueError("Primary-shaped warmup client concurrency differs")
        if warmup["run"]["boot_id"] != protocol["boot_id"]:
            raise ValueError("Primary-shaped warmup boot differs")
        if warmup["run"]["transport"] != protocol["transport"] or warmup["protocol"]["request_diagnostics"]:
            raise ValueError("Primary-shaped warmup transport or diagnostics differ")
        if warmup["cache_reset"]["status"] != "success":
            raise ValueError("Primary-shaped warmup cache reset failed")
        if protocol["transport"]["effective_mode"] == "direct":
            peers = {request["transport_peer"] for request in warmup["requests"]}
            peers.add(warmup["cache_reset"]["transport_peer"])
            if peers - {"target_loopback_port", "UNKNOWN"}:
                raise ValueError(f"Primary-shaped warmup observed an unexpected peer: {peers}")
        warmup_reference = (warmup if reference_dir is None
                            else read_json(reference_dir / "primary-warmup.json"))
        comparison = compare_exact_tokens(warmup_reference, warmup)
        comparisons["primary_warmup"] = comparison
        if (not comparison["equivalent"] or comparison["request_count"] != workload["requests"]
                or warmup["summary"]["completed_requests"] != workload["requests"]):
            write_json(case_dir / "paired-exact-comparisons.json", comparisons)
            raise ValueError("Primary-shaped warmup exact request validation failed")
        if reference_dir is None:
            measured = read_json(case_dir / "primary-0.json")
            if not compare_exact_tokens(warmup, measured)["equivalent"]:
                raise ValueError("First warmup differs from its measured primary outputs")
    for label, workload in protocol["workloads"].items():
        name = "correctness.json" if label == "correctness" else f"{label}-0.json"
        run = read_json(case_dir / name)
        if "transport" in protocol:
            if run["run"]["transport"] != protocol["transport"]:
                raise ValueError(f"Actual {label} transport differs from frozen protocol")
            if run["protocol"]["request_diagnostics"]:
                raise ValueError("Diagnostic instrumentation is forbidden in formal timing")
            if run["run"]["boot_id"] != protocol["boot_id"]:
                raise ValueError("Case boot differs from frozen protocol")
            if run["run"]["reset_cache_requested"] and run["cache_reset"]["status"] != "success":
                raise ValueError(f"{label} cache reset failed")
            if protocol["transport"]["effective_mode"] == "direct":
                peers = {request["transport_peer"] for request in run["requests"]}
                if run["run"]["reset_cache_requested"]:
                    peers.add(run["cache_reset"]["transport_peer"])
                if peers - {"target_loopback_port", "UNKNOWN"}:
                    raise ValueError(f"Direct transport observed an unexpected peer: {peers}")
        if digest(run["workload"]) != workload["workload_sha256"]:
            raise ValueError(f"Actual {label} inputs differ from the frozen workload")
        if run["run"]["max_workers"] != workload["workers"]:
            raise ValueError(f"Actual {label} client concurrency differs from frozen settings")
        reference = run if reference_dir is None else read_json(reference_dir / name)
        comparison = compare_exact_tokens(reference, run)
        comparisons[label] = comparison
        summary = run["summary"]
        positive_metric(summary, METRIC)
        results[label] = summary
        if (not comparison["equivalent"] or comparison["request_count"] != workload["requests"]
                or summary["completed_requests"] != workload["requests"]):
            write_json(case_dir / "paired-exact-comparisons.json", comparisons)
            raise ValueError(f"Exact request validation failed for {label}")
    write_json(case_dir / "paired-exact-comparisons.json", comparisons)
    return comparable, results


def execute_schedule(protocol, output):
    records, reference_dir, reference_identity = [], None, None
    for case in protocol["cases"]:
        record = {**case, "directory": str(output / "cases" / case["id"]),
                  "command": case_command(case, output, protocol, reference_dir),
                  "exit_code": None, "status": "RUNNING"}
        log_dir = output / "logs"
        log_dir.mkdir(exist_ok=True)
        record.update(stdout=str(log_dir / f"{case['id']}.stdout.log"),
                      stderr=str(log_dir / f"{case['id']}.stderr.log"))
        started = time.monotonic()
        print(f"START {case['id']} mode={case['mode']}", flush=True)
        try:
            assert_frozen(protocol)
            env = dict(os.environ)
            env.update(SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM="0",
                       GRAPHBUDGET_SHARED_GPU_LOCK=protocol["settings"]["gpu_lock"],
                       TRITON_CACHE_DIR=protocol["settings"]["triton_cache_dir"],
                       PYTHONPATH=protocol["settings"]["case_pythonpath"])
            write_json(output / "executions" / f"{case['id']}.json", record)
            with open(record["stdout"], "wb") as stdout, open(record["stderr"], "wb") as stderr:
                completed = subprocess.run(record["command"], env=env, stdout=stdout,
                                           stderr=stderr, check=False, cwd=output)
            record["exit_code"] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f"Case exited with code {completed.returncode}")
            comparable, results = verify_case(case, Path(record["directory"]), protocol,
                                              reference_dir, reference_identity)
            assert_frozen(protocol)
            if reference_dir is None:
                reference_dir, reference_identity = Path(record["directory"]), comparable
                write_json(output / "reference-identity.json", reference_identity)
            record.update(status="PASS", workloads=results)
        except (Exception, KeyboardInterrupt) as error:
            record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            records.append(record)
            write_json(output / "executions" / f"{case['id']}.json", record)
        print(f"END {case['id']} status={record['status']} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
        if record["status"] != "PASS":
            break
        if case["stage"] == "bb_before" and case["position"] == 1 and not protocol["smoke"]:
            first, second = records[-2:]
            left = positive_metric(first["workloads"]["primary"], METRIC)
            right = positive_metric(second["workloads"]["primary"], METRIC)
            change = right / left - 1
            gate = {"status": "PASS" if abs(change) <= 0.05 else "CONTROL_FAILED",
                    "case_ids": [first["id"], second["id"]],
                    "throughput": [left, right], "relative_change": change,
                    "max_absolute_relative_change": 0.05,
                    "scope": "same boot, independent B/B; all requests retained"}
            write_json(output / "bb-gate.json", gate)
            print(f"INDEPENDENT B/B gate={gate['status']} change={change:+.2%}", flush=True)
            if gate["status"] != "PASS":
                break
    return records


def positive_metric(summary, metric):
    value = summary.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Missing or invalid positive metric: {metric}")
    return float(value)


def evaluate_pairs(pairs, aa_changes, bb_changes=(), drift=0.0):
    gains = [right / left - 1 for left, right in pairs]
    noise = max((abs(value) for value in aa_changes), default=0.0)
    median = statistics.median(gains) if gains else None
    # Ratios such as 105 / 100 must not turn an exact 5% boundary into a pass.
    above_threshold = median is not None and median - max(0.05, noise) > 1e-12
    met = len(gains) == 5 and bool(aa_changes) and above_threshold and sum(g > 0 for g in gains) >= 4
    flags = []
    all_noise = max([noise, abs(drift), *(abs(value) for value in bb_changes)])
    if median is not None and all_noise >= abs(median):
        flags.append("NOISE_OR_DRIFT_AT_LEAST_CANDIDATE_EFFECT")
    return {"relative_gains": gains, "median_relative_gain": median,
            "positive_pairs": sum(g > 0 for g in gains), "aa_max_absolute_change": noise,
            "required_relative_gain": max(0.05, noise), "candidate_rule_met": met,
            "review_flags": flags}


def evaluate_bc_focused(pairs, bb_before, bb_after, drift):
    gains = [right / left - 1 for left, right in pairs]
    median = statistics.median(gains) if gains else None
    controls_available = all(value is not None for value in (bb_before, bb_after, drift))
    noise = max(abs(value) for value in (bb_before, bb_after, drift)) if controls_available else None
    floor = max(0.05, noise) if noise is not None else None
    met = (len(gains) == 5 and controls_available and abs(bb_before) <= 0.05 and
           sum(value > 0 for value in gains) >= 4 and median - floor > 1e-12)
    flags = []
    if bb_before is not None and abs(bb_before) > 0.05:
        flags.append("PRE_BB_EXCEEDS_5_PERCENT")
    if median is not None and noise is not None and noise >= abs(median):
        flags.append("NOISE_OR_DRIFT_AT_LEAST_CANDIDATE_EFFECT")
    if controls_available and (abs(bb_after) > 0.05 or abs(drift) > 0.05):
        flags.append("POST_BB_OR_DRIFT_EXCEEDS_5_PERCENT")
    return {"relative_gains": gains, "median_relative_gain": median,
            "positive_pairs": sum(value > 0 for value in gains),
            "bb_noise_max_absolute_change": noise,
            "required_relative_gain": floor,
            "candidate_rule_met": met, "review_flags": flags}


def request_failure_counts(protocol, records):
    counts = {"failed_requests": 0, "timed_out_requests": 0, "not_started_requests": 0,
              "missing_or_invalid_run_files": 0}
    for record in records:
        names = ["correctness.json" if label == "correctness" else f"{label}-0.json"
                 for label in protocol["workloads"]]
        if protocol.get("settings", {}).get("primary_shaped_warmup"):
            names.extend(("warmup.json", "primary-warmup.json"))
        for name in names:
            try:
                summary = read_json(Path(record["directory"]) / name)["summary"]
                for key in ("failed_requests", "timed_out_requests", "not_started_requests"):
                    counts[key] += summary[key]
            except (OSError, ValueError, KeyError, TypeError):
                counts["missing_or_invalid_run_files"] += 1
    return counts


def summarize(protocol, records):
    passed = [r for r in records if r["status"] == "PASS"]
    def metric(record, label="primary"):
        return positive_metric(record["workloads"][label], METRIC)
    def control(stage):
        rows = [r for r in passed if r["stage"] == stage]
        return {"case_ids": [r["id"] for r in rows],
                "throughput": [metric(r) for r in rows],
                "relative_change": metric(rows[1]) / metric(rows[0]) - 1 if len(rows) == 2 else None}
    focused = protocol.get("measurement_protocol") == "bc-focused-v1"
    stages = ("bb_before", "bb_after") if focused else ("aa_before", "bb_before", "bb_after")
    controls = {stage: control(stage) for stage in stages}
    paired, summaries, flags = [], {}, []
    for number in range(1, 6):
        rows = {r["arm"]: r for r in passed if r["stage"] == "paired" and r["pair"] == number}
        if set(rows) == {"B", "C"}:
            paired.append({"pair": number, "B_case": rows["B"]["id"], "C_case": rows["C"]["id"],
                           "B": metric(rows["B"]), "C": metric(rows["C"]),
                           "relative_gain": metric(rows["C"]) / metric(rows["B"]) - 1})
    bb = [controls[s]["relative_change"] for s in ("bb_before", "bb_after")
          if controls[s]["relative_change"] is not None]
    before, after = controls["bb_before"]["throughput"], controls["bb_after"]["throughput"]
    drift = statistics.median(after) / statistics.median(before) - 1 if len(before) == len(after) == 2 else None
    if focused:
        rule = evaluate_bc_focused([(p["B"], p["C"]) for p in paired],
                                   controls["bb_before"]["relative_change"],
                                   controls["bb_after"]["relative_change"], drift)
    else:
        aa = [controls["aa_before"]["relative_change"]] if controls["aa_before"]["relative_change"] is not None else []
        rule = evaluate_pairs([(p["B"], p["C"]) for p in paired], aa, bb, drift or 0.0)
    latency_metrics = (("ttft_s", "p50"), ("ttft_s", "p95"),
                       ("token_itl_s", "p50"), ("token_itl_s", "p95"),
                       ("request_tpot_s", "p50"), ("makespan_s", None))
    for label in ("primary", "holdout", "interactive", "acceptance"):
        if label not in protocol["workloads"]:
            continue
        throughput_changes = []
        latency_changes = {f"{name}.{percentile}" if percentile else name: []
                           for name, percentile in latency_metrics}
        absolute = []
        for pair in paired:
            b = next(r for r in passed if r["id"] == pair["B_case"])["workloads"][label]
            c = next(r for r in passed if r["id"] == pair["C_case"])["workloads"][label]
            throughput_changes.append(positive_metric(c, METRIC) / positive_metric(b, METRIC) - 1)
            absolute.append({"pair": pair["pair"], "B": b, "C": c})
            for name, percentile in latency_metrics:
                key = f"{name}.{percentile}" if percentile else name
                left = (b.get(name) or {}).get(percentile) if percentile else b.get(name)
                right = (c.get(name) or {}).get(percentile) if percentile else c.get(name)
                valid = (isinstance(left, (int, float)) and not isinstance(left, bool) and left > 0
                         and isinstance(right, (int, float)) and not isinstance(right, bool))
                if focused and valid:
                    valid = math.isfinite(left) and math.isfinite(right) and right > 0
                if valid:
                    latency_changes[key].append(right / left - 1)
        summaries[label] = {"throughput_relative_changes": throughput_changes,
                            "median_throughput_relative_change": statistics.median(throughput_changes) if throughput_changes else None,
                            "latency_relative_changes": latency_changes,
                            "median_latency_relative_changes": {
                                name: statistics.median(values) if values else None
                                for name, values in latency_changes.items()},
                            "absolute_pair_metrics": absolute}
        for name, values in latency_changes.items():
            if focused and len(paired) == 5 and len(values) != 5:
                flags.append(f"{label}:{name}:MISSING_OR_INVALID")
            # A screening flag, not a universal latency acceptance threshold.
            if len(values) == 5 and statistics.median(values) > 0.05 and sum(v > 0 for v in values) >= 4:
                flags.append(f"{label}:{name}:CONSISTENT_LATENCY_REGRESSION_REVIEW")
    complete = len(records) == len(protocol["cases"]) and len(passed) == len(records)
    failed = [r["id"] for r in records if r["status"] != "PASS"]
    flags = rule["review_flags"] + flags
    failure_counts = request_failure_counts(protocol, records)
    if failure_counts["missing_or_invalid_run_files"]:
        flags.append("INCOMPLETE_RAW_REQUEST_EVIDENCE")
    directory = records[0].get("directory") if records else None
    gate_path = Path(directory).parents[1] / "bb-gate.json" if directory else None
    gate = read_json(gate_path) if gate_path and gate_path.exists() else None
    if focused and complete:
        if gate is None:
            flags.append("MISSING_INDEPENDENT_BB_GATE")
        elif (gate.get("status") != "PASS"
              or gate["case_ids"] != controls["bb_before"]["case_ids"]
              or gate["throughput"] != controls["bb_before"]["throughput"]):
            flags.append("INDEPENDENT_BB_GATE_MISMATCH")
    status = ("CONTROL_FAILED" if gate and gate["status"] == "CONTROL_FAILED" else
              "FAILED" if not complete else "SMOKE_ONLY" if protocol["smoke"] else
              "NEEDS_REVIEW" if flags else "PRIMARY_RULE_MET" if rule["candidate_rule_met"] else "PRIMARY_RULE_NOT_MET")
    return {"schema_version": 1, "status": status, "overall_acceptance": "NOT_ASSESSED",
            "smoke": protocol["smoke"], "planned_cases": len(protocol["cases"]),
            "completed_cases": len(passed), "failed_case_count": len(failed), "failed_cases": failed,
            "unstarted_cases": len(protocol["cases"]) - len(records), "pairs": paired,
            "controls": controls, "bb_gate": gate, "bb_before_to_after_relative_drift": drift,
            "primary_rule": rule, "workloads": summaries, "review_flags": flags,
            "request_failure_counts": failure_counts,
            "failure_count_scope": "Includes retained raw files from failed cases; missing files are counted explicitly.",
            "cases": records, "note": "All rounds retained; no outlier exclusion or automatic overall acceptance."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--port", type=int, default=30277)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol", choices=("legacy-v1", "bc-focused-v1"), default="legacy-v1")
    parser.add_argument("--transport", choices=("direct", "inherit"), default="direct")
    parser.add_argument("--gpu-lock", default=os.environ.get(
        "GRAPHBUDGET_SHARED_GPU_LOCK",
        str(Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / f"graphbudget-shared-gpu-{os.getuid()}.lock")))
    args = parser.parse_args(argv)
    if args.smoke and args.protocol != "legacy-v1":
        parser.error("bc-focused-v1 is a fixed formal protocol and cannot be combined with --smoke")
    if args.protocol == "bc-focused-v1" and os.environ.get("CUDA_LAUNCH_BLOCKING") not in (None, "", "0"):
        parser.error("bc-focused-v1 performance runs require CUDA_LAUNCH_BLOCKING to be disabled")
    if args.protocol == "bc-focused-v1" and (
        args.max_running_requests != 16 or args.max_total_tokens != 4096 or args.transport != "direct"
    ):
        parser.error("bc-focused-v1 requires max-running-requests=16, max-total-tokens=4096 and direct transport")
    if not 1 <= args.max_running_requests <= 16 or args.max_total_tokens < 1 or not 1 <= args.port <= 65535:
        parser.error("max-running-requests must be 1..16, tokens positive, and port 1..65535")
    output = Path(args.output).resolve()
    if not boot_id():
        parser.error("Same-boot measurement requires a readable Linux boot ID")
    if output.is_relative_to(ROOT):
        ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--quiet", str(output)]).returncode == 0
        if not ignored:
            parser.error("An output inside the checkout must be Git-ignored (for example work/paired-run)")
    snapshot = git_snapshot()
    if snapshot["status"] and not args.smoke:
        parser.error("Formal runs require a clean tracked/untracked checkout; commit or isolate source changes first")
    output.mkdir(parents=True, exist_ok=False)
    protocol = prepare_protocol(args, output, snapshot)
    write_json(output / "protocol.json", protocol)
    records = execute_schedule(protocol, output)
    result = summarize(protocol, records)
    write_json(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "output": str(output),
                      "failed_case_count": result["failed_case_count"]}))
    return 0 if result["status"] not in {"FAILED", "CONTROL_FAILED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
