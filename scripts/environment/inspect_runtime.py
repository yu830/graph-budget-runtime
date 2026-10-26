#!/usr/bin/env python3
"""Record a read-only effective dependency/runtime lock.

This intentionally uses package metadata and ``find_spec`` only.  It does not
import torch, Triton, SGLang, or any CUDA module and therefore must not create
a CUDA context while another process owns the GPU.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


OVERLAY_REQUIREMENTS = (
    "compressed-tensors",
    "gguf",
    "einops",
    "cuda-python",
    "cuda-bindings",
    "cuda-pathfinder",
    "jsonschema",
    "jsonschema-specifications",
    "referencing",
    "rpds-py",
    "loguru",
    "xgrammar",
)

MODULE_HINTS = {
    "compressed-tensors": "compressed_tensors",
    "gguf": "gguf",
    "einops": "einops",
    "cuda-python": "cuda",
    "cuda-bindings": "cuda",
    "cuda-pathfinder": "cuda_pathfinder",
    "jsonschema": "jsonschema",
    "jsonschema-specifications": "jsonschema_specifications",
    "referencing": "referencing",
    "rpds-py": "rpds",
    "loguru": "loguru",
    "xgrammar": "xgrammar",
    "torch": "torch",
    "triton": "triton",
    "sglang": "sglang",
    "sgl-kernel": "sgl_kernel",
    "flashinfer-python": "flashinfer",
    "flashinfer-cubin": "flashinfer",
    "transformers": "transformers",
}


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def as_posix(value: str | Path) -> str:
    return str(value).replace("\\", "/")


def path_key(value: str | Path) -> str:
    try:
        return str(Path(value).resolve())
    except (OSError, RuntimeError):
        return str(value)


def path_prefix(path: str, root: str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def command(args: list[str], cwd: str | None = None) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 else None


def git_identity(path: str | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "exists": False}
    root = Path(path)
    result: dict[str, Any] = {"path": as_posix(root), "exists": root.exists()}
    if not root.exists():
        return result
    result["head"] = command(["git", "-C", str(root), "rev-parse", "HEAD"])
    result["status"] = command(["git", "-C", str(root), "status", "--short"])
    try:
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary"],
            check=False,
            capture_output=True,
            timeout=30,
        )
        result["diff_sha256"] = hashlib.sha256(diff.stdout).hexdigest()
        result["diff_bytes"] = len(diff.stdout)
    except (OSError, subprocess.SubprocessError):
        result["diff_sha256"] = None
    return result


def find_spec_record(module_name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return {"module": module_name, "status": "ERROR", "error": str(exc)}
    if spec is None:
        return {"module": module_name, "status": "ABSENT"}
    locations = [as_posix(x) for x in (spec.submodule_search_locations or [])]
    return {
        "module": module_name,
        "status": "FOUND",
        "origin": as_posix(spec.origin) if spec.origin else None,
        "locations": locations,
    }


def distribution_location(dist: metadata.Distribution) -> str:
    return path_key(dist.locate_file(""))


def collect_distributions(sys_path: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for index, entry in enumerate(sys_path):
        if not entry:
            continue
        try:
            distributions = metadata.distributions(path=[entry])
        except (OSError, ValueError):
            continue
        for dist in distributions:
            name = dist.metadata.get("Name") or dist.metadata.get("name")
            if not name:
                continue
            location = distribution_location(dist)
            key = (canonical(name), str(dist.version), location)
            if key in seen:
                continue
            seen.add(key)
            grouped.setdefault(canonical(name), []).append(
                {
                    "name": name,
                    "version": dist.version,
                    "location": as_posix(location),
                    "sys_path_index": index,
                    "requires": list(dist.requires or []),
                }
            )
    for candidates in grouped.values():
        candidates.sort(key=lambda item: (item["sys_path_index"], item["location"]))
    return grouped


def choose_effective(
    package_name: str,
    candidates: list[dict[str, Any]],
    module_record: dict[str, Any],
) -> dict[str, Any] | None:
    if not candidates:
        return None
    observed_paths = []
    origin = module_record.get("origin")
    if origin and origin not in {"built-in", "frozen"}:
        observed_paths.append(origin)
    observed_paths.extend(module_record.get("locations", []))
    for candidate in candidates:
        if any(path_prefix(path, candidate["location"]) for path in observed_paths):
            selected = dict(candidate)
            selected["effective"] = True
            selected["loaded_from"] = observed_paths[0] if observed_paths else None
            selected["source_override"] = False
            return selected
    source_root = os.environ.get("SGLANG_SOURCE")
    if (
        package_name == "sglang"
        and source_root
        and observed_paths
        and any(path_prefix(path, source_root) for path in observed_paths)
    ):
        selected = dict(candidates[0])
        selected["effective"] = True
        selected["loaded_from"] = observed_paths[0]
        selected["source_override"] = True
        selected["source_root"] = as_posix(source_root)
        return selected
    selected = dict(candidates[0])
    selected["effective"] = True
    selected["loaded_from"] = observed_paths[0] if observed_paths else None
    selected["source_override"] = False
    return selected


def ready_evidence(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": as_posix(path), "status": "UNREADABLE"}
    keep = ("schema_version", "run_id", "identity", "fingerprint", "local_details")
    return {
        "path": as_posix(path),
        "status": "READ",
        **{key: data.get(key) for key in keep if key in data},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument(
        "--pinned-output",
        help="Optional sanitized path-free package/constraint JSON",
    )
    parser.add_argument("--ready-json", help="Optional prior ready.json evidence")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    sys_path = [str(Path(item).resolve()) if item else "" for item in sys.path]
    pythonpath = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    distributions = collect_distributions(sys.path)

    effective: dict[str, dict[str, Any] | None] = {}
    shadowed: dict[str, list[dict[str, Any]]] = {}
    module_records: dict[str, dict[str, Any]] = {}
    package_names = set(OVERLAY_REQUIREMENTS) | set(MODULE_HINTS)
    for package_name in sorted(package_names):
        module_name = MODULE_HINTS.get(package_name, package_name.replace("-", "_"))
        module_record = find_spec_record(module_name)
        module_records[package_name] = module_record
        candidates = distributions.get(canonical(package_name), [])
        selected = choose_effective(package_name, candidates, module_record)
        effective[package_name] = selected
        if selected:
            shadowed[package_name] = [
                candidate
                for candidate in candidates
                if candidate is not selected
                and (
                    candidate["version"] != selected["version"]
                    or candidate["location"] != selected["location"]
                )
            ]

    overlay_metadata = {
        name: effective.get(name, {"status": "ABSENT"}) for name in OVERLAY_REQUIREMENTS
    }
    output: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "collection_mode": "metadata_and_find_spec_only_no_torch_import",
        "python": {
            "executable": as_posix(sys.executable),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "environment_paths": {
            "repo_root": as_posix(repo_root),
            "graphbudget_env": os.environ.get("GRAPHBUDGET_ENV"),
            "sglang_overlay": os.environ.get("SGLANG_OVERLAY"),
            "torch_env": os.environ.get("TORCH_ENV"),
            "sglang_source": os.environ.get("SGLANG_SOURCE"),
            "pythonpath": [as_posix(item) for item in pythonpath],
            "sys_path": [as_posix(item) for item in sys_path],
        },
        "identities": {
            "graphbudget": git_identity(str(repo_root)),
            "sglang_source": git_identity(os.environ.get("SGLANG_SOURCE")),
        },
        "module_resolution": module_records,
        "effective_packages": effective,
        "shadowed_package_candidates": shadowed,
        "all_distributions": distributions,
        "overlay_metadata_requires": overlay_metadata,
        "prior_ready_evidence": ready_evidence(args.ready_json),
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.pinned_output:
        runtime_names = (
            "torch",
            "triton",
            "sglang",
            "sgl-kernel",
            "flashinfer-python",
            "flashinfer-cubin",
            "transformers",
        )
        runtime = {
            name: {
                "version": (effective.get(name) or {}).get("version"),
                "source_override": (effective.get(name) or {}).get(
                    "source_override", False
                ),
            }
            for name in runtime_names
        }
        pinned = {
            "schema_version": 1,
            "collection_mode": "metadata_and_find_spec_only_no_torch_import",
            "framework": {
                "name": "sglang",
                "release": "0.5.9",
                "base_commit": (output["identities"]["sglang_source"] or {}).get(
                    "head"
                ),
                "patch_sha256": (output["identities"]["sglang_source"] or {}).get(
                    "diff_sha256"
                ),
            },
            "install_policy": "overlay packages are exact-pinned and installed with --no-deps --only-binary=:all:; Torch/Triton/CUDA/FlashInfer/SGLang stay inherited or source-overridden",
            "runtime": runtime,
            "overlay_packages": {
                name: {
                    "version": (effective.get(name) or {}).get("version"),
                    "requires_dist": (effective.get(name) or {}).get("requires", []),
                }
                for name in OVERLAY_REQUIREMENTS
            },
        }
        pinned_destination = Path(args.pinned_output)
        pinned_destination.parent.mkdir(parents=True, exist_ok=True)
        pinned_destination.write_text(
            json.dumps(pinned, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"output": as_posix(destination), "packages": len(effective)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
