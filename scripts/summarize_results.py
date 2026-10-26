"""Derive small, path-redacted results from a completed frozen experiment.

This checks/derives records, not physical execution. Original requests, native
paths and device samples remain the evidence. No metric is inferred from a cap.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphbudget.benchmark import compare_exact_tokens
from graphbudget.identity import code_digest, digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(run, name, filename):
    state = read(run / name / "stage_state.json")
    if state["status"] != "ok":
        raise ValueError(f"{name} is not complete")
    directory = (run / name / state["attempt"]).resolve()
    if not directory.is_relative_to((run / name).resolve()):
        raise ValueError("Stage path escaped the experiment")
    path = directory / filename
    if sha(path) != state[filename.split(".")[0] + "_sha256"]:
        raise ValueError(f"{name} artifact changed")
    return directory, read(path)


def delta(a, b):
    return 100 * (b / a - 1) if a not in (None, 0) and b is not None else None


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def metrics(case):
    r = case["benchmark"]["summary"]
    return {"tokens_per_s": r["effective_output_tokens_per_s"],
            "ttft_p95_ms": r["ttft_s"]["p95"] * 1000,
            "tpot_p95_ms": r["request_tpot_s"]["p95"] * 1000,
            "itl_p95_ms": (r["token_itl_s"]["p95"] * 1000
                           if r["token_itl_s"]["p95"] is not None else None),
            "itl_unavailable_requests": r["token_itl_unavailable_request_count"],
            "completed_requests": r["completed_requests"], "output_tokens": r["successful_output_tokens"],
            "failed_requests": r["failed_requests"], "timed_out_requests": r["timed_out_requests"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory")
    measured_dir, measured = stage(run, "measure", "measurements.json")
    selected_dir, plan = stage(run, "select", "plan.json")
    profile_dir, profile = stage(run, "profile", "profile.json")
    validation_dir, validation = stage(run, "validate", "validation.json")
    performance_dir, performance = stage(run, "performance", "performance.json")
    modules = {p.name: code_digest(p) for p in sorted((ROOT / "src/graphbudget").glob("*.py"))}
    if digest(modules) != plan["fingerprint"]["identity"]["graphbudget_sha256"]:
        raise ValueError("Current runtime source differs from the measured identity")
    for name, expected in modules.items():
        committed = subprocess.check_output(["git", "show", f"HEAD:src/graphbudget/{name}"], cwd=ROOT)
        if hashlib.sha256(committed.replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError("Runtime source is not represented by the reported Git commit")
    manifest = read(ROOT / "src/graphbudget/patches/manifest.json")
    if manifest["patch_sha256"] != plan["fingerprint"]["identity"]["sglang"]["patch_sha256"]:
        raise ValueError("Packaged upstream patch differs from the measured version")
    if any(not c["equivalent"] for c in validation["exact_token_comparisons"]):
        raise ValueError("Exact-token validation failed")
    rows = {}
    fingerprint = plan["fingerprint"]["sha256"]
    source_hashes = {}
    violations = []
    def case(path):
        value = read(path / "case.json")
        if value["status"] != "ok" or value["ready"]["fingerprint"] != fingerprint:
            raise ValueError(f"Failed or stale case: {path}")
        bench = value["benchmark"]["summary"]
        if bench["completed_requests"] != bench["planned_requests"]:
            raise ValueError("Requests were lost")
        budget = plan["budget"]
        if (value["observed_peak_device_used_bytes"] > budget["allowed_device_bytes"]
                or value["ready"]["kv_tokens"] < budget["min_kv_tokens"]
                or value["startup_seconds"] > budget["max_startup_seconds"]):
            violations.append(str(path.relative_to(run)))
        for name in ("case.json", "requests.json", "paths_before.json", "paths_after.json", "timed_paths.json", "device.csv"):
            file = path / name
            source_hashes[file.relative_to(run).as_posix()] = sha(file)
        return value
    for workload in ("heldout_normal", "heldout_boundary", "heldout_mixed"):
        rounds = []
        for number in range(1, 6):
            root = performance_dir / "paired" / workload / f"round-{number:02d}"
            a, b = case(root / "T"), case(root / "C")
            if not compare_exact_tokens(a["benchmark"], b["benchmark"])["equivalent"]:
                raise ValueError("Paired output mismatch")
            if a["ready"]["capture_sizes"] != b["ready"]["capture_sizes"] or a["ready"]["kv_tokens"] != b["ready"]["kv_tokens"]:
                raise ValueError("T/C resources differ")
            left, right = metrics(a), metrics(b)
            rejects = sum(r["count"] for r in b["timed_path_counts"]
                          if r["eligible"] and r["used_path"] == "eager" and r["reason"] == "eager_faster_beyond_noise")
            rounds.append({"round": number, "T": left, "C": right,
                           "relative_delta_percent": {key: delta(left[key], right[key])
                               for key in ("tokens_per_s", "ttft_p95_ms", "tpot_p95_ms", "itl_p95_ms")},
                           "cost_rejections": rejects})
        rows[workload] = {"rounds": rounds,
            "median_paired_delta_percent": {key: median([r["relative_delta_percent"][key]
                for r in rounds if r["relative_delta_percent"][key] is not None])
                for key in ("tokens_per_s", "ttft_p95_ms", "tpot_p95_ms", "itl_p95_ms")},
            "cost_rejections": sum(r["cost_rejections"] for r in rounds)}
    aa = []
    for number in range(1, 6):
        root = performance_dir / "aa/heldout_boundary" / f"round-{number:02d}"
        a, b = case(root / "a"), case(root / "b")
        if not compare_exact_tokens(a["benchmark"], b["benchmark"])["equivalent"]:
            raise ValueError("A/A exact-token mismatch")
        aa.append({"round": number, "A1": metrics(a), "A2": metrics(b),
                   "throughput_delta_percent": delta(metrics(a)["tokens_per_s"], metrics(b)["tokens_per_s"])})
    summary = {"schema_version": 1, "fingerprint": fingerprint,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Qwen2.5-0.5B BF16, TP=1, SM120, pinned deterministic Triton mode",
        "selected_capture_sizes": plan["selected"]["capture_sizes"], "kv_tokens": plan["selected"]["ready"]["kv_tokens"],
        "profile_keys": len(profile["entries"]), "profile_eager_wins": sum(e["eager_ns"] + e["noise_ns"] < e["graph_ns"] for e in profile["entries"]),
        "aa_scope": "heldout_boundary only", "aa_rounds": aa,
        "aa_max_absolute_throughput_delta_percent": max(abs(r["throughput_delta_percent"]) for r in aa),
        "paired": rows, "resource_constraint_violations": violations,
        "performance_gain": "NOT_ESTABLISHED", "adoption": "KEEP_NATIVE_DISPATCH",
        "interpretation": "P and T use the same frozen capture set. No cost-driven eligible rejection was observed. Timing deltas do not establish a useful cost-policy improvement.",
        "metric_scope": "Client request wall time; ITL only for individually observed tokens. No P99 claim. NVML peaks are sampled whole-device usage, separate from PyTorch allocation."}
    if any(r["cost_rejections"] for r in rows.values()):
        raise ValueError("Cost path changes observed: review adoption interpretation before exporting")
    for directory in (measured_dir, selected_dir, profile_dir, validation_dir, performance_dir):
        for file in sorted(directory.rglob("*")):
            if file.is_file() and file.suffix in (".json", ".csv", ".log"):
                source_hashes[file.relative_to(run).as_posix()] = sha(file)
    performance = json.loads(json.dumps(performance))
    performance["kind"] = "graphbudget_performance_summary"
    for row in performance["exploratory_single_runs"]:
        bench = row["benchmark"]
        row["benchmark"] = {"run": bench["run"], "summary": bench["summary"],
                            "workload_id": bench["workload"]["workload_id"]}
    def redact(value):
        if isinstance(value, dict):
            return {k: redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        if isinstance(value, str):
            marker = "/work/core-deterministic"
            if marker in value:
                return "${RUN}" + value[value.index(marker) + len(marker):]
        return value
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary.json", summary), ("plan.json", plan), ("profile.json", profile),
                       ("validation.json", validation), ("performance.json", performance)):
        (output / name).write_text(json.dumps(redact(data), indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "RAW_MANIFEST.json").write_text(json.dumps(source_hashes, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
