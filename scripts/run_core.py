"""Practical GraphBudget validation orchestration for the pinned SGLang runtime.

This is intentionally a small, phase-oriented runner rather than a general
experiment platform.  It freezes model/tokenizer identity and real tokenized
workloads before any server is started, then delegates one fresh server
process at a time to :mod:`graphbudget.experiment`.

Typical use in the prepared SGLang environment is::

    python scripts/run_core.py --model-path /models/qwen-snapshot \
      --revision <40-char-model-revision> --backend triton \
      --output work/suite --phase prepare
    python scripts/run_core.py --model-path /models/qwen-snapshot \
      --revision <40-char-model-revision> --backend triton \
      --output work/suite --phase all --resume

The script does not start a server during import or preparation.  GPU phases
are serial and each case has a new output directory.  Existing raw case logs
are never overwritten; ``--resume`` only reuses a phase whose immutable
artifact is complete or starts a new numbered attempt for an incomplete phase.
Failures are recorded as failed stage state and stop dependent phases.  The
script does not turn empty measurements, absent path traces, or token
mismatches into PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from itertools import combinations
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# ``experiment.run_case`` starts ``python -m graphbudget.cli`` in a child
# process.  Make the source checkout importable without a global install.
_python_path_parts = [str(SRC)]
if os.environ.get("PYTHONPATH"):
    _python_path_parts.append(os.environ["PYTHONPATH"])
os.environ["PYTHONPATH"] = os.pathsep.join(_python_path_parts)

from graphbudget.benchmark import (  # noqa: E402
    compare_exact_tokens,
    load_workload,
    summarize_paired_rounds,
)
from graphbudget.experiment import (  # noqa: E402
    build_cost_profile,
    measure_candidates,
    run_case,
    gpu_lease,
)
from graphbudget.identity import model_identity, code_digest, source_identity  # noqa: E402
from graphbudget.plan import select_plan  # noqa: E402
from graphbudget.runtime import write_json  # noqa: E402


SCHEMA_VERSION = 1
MODEL_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_RUNNING_REQUESTS = 16
MAX_TOTAL_TOKENS_CHOICES = (4096, 8192)
CAPTURE_CANDIDATES = (
    ("cap-1-2-4", [1, 2, 4]),
    ("cap-1-2-4-8", [1, 2, 4, 8]),
    ("cap-1-2-4-8-12-16", [1, 2, 4, 8, 12, 16]),
)
VARIANTS = ("U", "E", "P", "T", "C")
HELDOUT_WORKLOAD_NAMES = ("heldout_normal", "heldout_boundary", "heldout_mixed")


class SuiteError(RuntimeError):
    """A phase cannot make a trustworthy next artifact."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_tree(
    root: Path,
    *,
    pattern: str = "*.py",
    normalize_crlf: bool = False,
) -> Optional[str]:
    """Hash a source tree by relative path and file content.

    This is a CPU-side resume identity check; it deliberately does not import
    the source tree or start the runtime.
    """

    root = Path(root).resolve()
    if not root.is_dir():
        return None
    files = sorted(path for path in root.rglob(pattern) if path.is_file())
    entries = {}
    for path in files:
        content = path.read_bytes()
        if normalize_crlf:
            content = content.replace(b"\r\n", b"\n")
        entries[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(content).hexdigest()
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    checkout = (
        Path(args.sglang_checkout).expanduser().resolve()
        if getattr(args, "sglang_checkout", None)
        else None
    )
    manifest_path = SRC / "graphbudget" / "patches" / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    module_entries = {
        path.name: code_digest(path)
        for path in sorted(SRC.glob("graphbudget/*.py"))
        if path.is_file()
    }
    graphbudget_identity_payload = json.dumps(
        module_entries, sort_keys=True, separators=(",", ":")
    )
    checkout_tree = (
        source_identity(checkout / "python" / "sglang" / "__init__.py")["srt_tree_sha256"]
        if checkout is not None
        else None
    )
    return {
        "graphbudget_commit": _git_revision(ROOT),
        "graphbudget_source_sha256": _sha256_tree(SRC),
        "graphbudget_identity_sha256": hashlib.sha256(
            graphbudget_identity_payload.encode("utf-8")
        ).hexdigest(),
        "sglang_commit": (
            _git_revision(checkout) if checkout is not None else os.environ.get("SGLANG_REVISION")
        ),
        "sglang_patch_sha256": manifest.get("patch_sha256"),
        "sglang_tree_sha256": checkout_tree,
    }


def _git_revision(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    """Write an immutable JSON artifact, accepting only byte-equivalent data."""

    path = Path(path)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SuiteError(f"Existing artifact is not readable JSON: {path}") from error
        if existing != dict(value):
            raise SuiteError(f"Refusing to overwrite immutable artifact: {path}")
        return
    write_json(path, value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SuiteError(f"Cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise SuiteError(f"Expected JSON object: {path}")
    return value


def _new_attempt(stage_root: Path) -> Path:
    stage_root.mkdir(parents=True, exist_ok=True)
    indexes = []
    for child in stage_root.glob("attempt-*"):
        try:
            indexes.append(int(child.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return stage_root / f"attempt-{max(indexes, default=0) + 1:03d}"


def _stage_state_path(stage_root: Path) -> Path:
    return stage_root / "stage_state.json"


def _stage_state(stage_root: Path) -> Optional[dict[str, Any]]:
    path = _stage_state_path(stage_root)
    return _read_json(path) if path.exists() else None


_STAGE_ARTIFACTS = {
    "measure": {"measurements_sha256": "measurements.json"},
    "select": {"plan_sha256": "plan.json"},
    "profile": {"profile_sha256": "profile.json"},
    "validate": {"validation_sha256": "validation.json"},
    "performance": {"performance_sha256": "performance.json"},
}
_STAGE_INPUT_ARTIFACTS = {
    "measure": {"experiment_sha256": ("prepare", "experiment.json")},
    "select": {
        "measurements_sha256": ("measure", "measurements.json"),
        "experiment_sha256": ("prepare", "experiment.json"),
    },
    "profile": {
        "plan_sha256": ("select", "plan.json"),
        "experiment_sha256": ("prepare", "experiment.json"),
    },
    "validate": {
        "plan_sha256": ("select", "plan.json"),
        "profile_sha256": ("profile", "profile.json"),
        "experiment_sha256": ("prepare", "experiment.json"),
    },
    "performance": {
        "plan_sha256": ("select", "plan.json"),
        "profile_sha256": ("profile", "profile.json"),
        "experiment_sha256": ("prepare", "experiment.json"),
    },
}


def _stage_artifact_from_state(stage_root: Path, stage_name: str, filename: str) -> Optional[Path]:
    upstream_root = stage_root.parent / stage_name
    upstream_state = _stage_state(upstream_root)
    if not upstream_state or upstream_state.get("status") != "ok":
        return None
    attempt = upstream_state.get("attempt")
    if not isinstance(attempt, str):
        return None
    path = (upstream_root / attempt).resolve()
    root = upstream_root.resolve()
    if path != root and root not in path.parents:
        return None
    artifact = path / filename
    return artifact if artifact.is_file() else None


def _verify_stage_artifacts(stage_root: Path, attempt: Path, state: Mapping[str, Any]) -> bool:
    """Validate the immutable outputs named by a completed stage state."""

    if stage_root.name == "prepare":
        manifest_path = attempt / "manifest.json"
        experiment_path = attempt / "experiment.json"
        if not manifest_path.is_file() or not experiment_path.is_file():
            return False
        try:
            manifest = _read_json(manifest_path)
            recorded_manifest = state.get("manifest")
            if isinstance(recorded_manifest, Mapping) and dict(recorded_manifest) != manifest:
                return False
            artifacts = manifest.get("prepare_artifacts")
            if not isinstance(artifacts, Mapping):
                return False
            if artifacts.get("experiment_sha256") != _sha256_file(experiment_path):
                return False
            workloads = artifacts.get("workloads")
            if not isinstance(workloads, Mapping):
                return False
            for relative, expected in workloads.items():
                path = attempt / "workloads" / f"{relative}.json"
                if not path.is_file() or expected != _sha256_file(path):
                    return False
            return True
        except (OSError, ValueError, TypeError):
            return False

    expected = _STAGE_ARTIFACTS.get(stage_root.name)
    if not expected:
        return False
    for state_key, relative in expected.items():
        recorded = state.get(state_key)
        path = attempt / relative
        if not isinstance(recorded, str) or not path.is_file():
            return False
        try:
            if recorded != _sha256_file(path):
                return False
        except OSError:
            return False
    for state_key, (upstream_stage, filename) in _STAGE_INPUT_ARTIFACTS.get(
        stage_root.name, {}
    ).items():
        recorded = state.get(state_key)
        # Older successful measurement attempts predate input-hash fields.  Their
        # own immutable output hash plus the worker identity is enough to reuse
        # those raw measurements; derived stages must be rebuilt when their
        # upstream binding is absent.
        if recorded is None:
            if stage_root.name == "measure":
                continue
            return False
        artifact = _stage_artifact_from_state(stage_root, upstream_stage, filename)
        if not isinstance(recorded, str) or artifact is None:
            return False
        try:
            if recorded != _sha256_file(artifact):
                return False
        except OSError:
            return False
    return True


def _completed_attempt(stage_root: Path) -> Optional[Path]:
    state = _stage_state(stage_root)
    if not state or state.get("status") != "ok":
        return None
    attempt = state.get("attempt")
    if not isinstance(attempt, str):
        return None
    root = stage_root.resolve()
    path = (stage_root / attempt).resolve()
    if path != root and root not in path.parents:
        raise SuiteError(f"Stage state escapes its output directory: {stage_root}")
    if not path.exists():
        raise SuiteError(f"Stage marked complete but attempt is missing: {path}")
    if not _verify_stage_artifacts(stage_root, path, state):
        # Keep the old attempt and let the phase create a new numbered one.
        # A completed state is not evidence by itself.
        return None
    return path


def _mark_stage(
    stage_root: Path,
    *,
    status: str,
    attempt: Optional[Path],
    details: Optional[Mapping[str, Any]] = None,
) -> None:
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage_root.name,
        "status": status,
        "attempt": str(attempt.relative_to(stage_root)) if attempt else None,
        "updated_at_utc": _utc_now(),
    }
    if details:
        state.update(dict(details))
    # Stage state is the one mutable progress record.  Raw case artifacts are
    # immutable and are never replaced on a resume.
    write_json(_stage_state_path(stage_root), state)


def _source_signature(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project only source fields; scripts/run_core.py edits stay out of it."""

    if "sglang" in value and isinstance(value.get("sglang"), Mapping):
        sglang = value["sglang"]
        return {
            "graphbudget_identity_sha256": value.get("graphbudget_sha256"),
            "sglang_commit": sglang.get("base_commit"),
            "sglang_patch_sha256": sglang.get("patch_sha256"),
            "sglang_tree_sha256": sglang.get("srt_tree_sha256"),
        }
    return {
        key: value.get(key)
        for key in (
            "graphbudget_identity_sha256",
            "sglang_commit",
            "sglang_patch_sha256",
            "sglang_tree_sha256",
        )
    }


def _source_matches(expected: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Compare only source fields observed in both snapshots."""

    expected_signature = _source_signature(expected)
    current_signature = _source_signature(current)
    for key, current_value in current_signature.items():
        if current_value is not None and expected_signature.get(key) is not None:
            if current_value != expected_signature.get(key):
                return False
    return True


def _case_identity_source_signature(case_path: Path) -> Optional[dict[str, Any]]:
    try:
        case = _read_json(case_path)
    except SuiteError:
        return None
    identity = case.get("identity")
    if not isinstance(identity, Mapping):
        ready = case.get("ready")
        identity = ready.get("identity") if isinstance(ready, Mapping) else None
    return _source_signature(identity) if isinstance(identity, Mapping) else None


def _stage_source_signature(stage_name: str, attempt: Path) -> Optional[dict[str, Any]]:
    # Executed worker identity is stronger than a controller's pre-launch scan.
    if stage_name == "measure":
        try:
            measurements = _read_json(attempt / "measurements.json")
        except SuiteError:
            return None
        identity = measurements.get("identity")
        return _source_signature(identity) if isinstance(identity, Mapping) else None
    if stage_name == "select":
        try:
            plan = _read_json(attempt / "plan.json")
            identity = plan.get("fingerprint", {}).get("identity")
        except SuiteError:
            return None
        return _source_signature(identity) if isinstance(identity, Mapping) else None
    for case_path in sorted(attempt.rglob("case.json")):
        signature = _case_identity_source_signature(case_path)
        if signature is not None:
            return signature
    state = _stage_state(attempt.parent)
    if state and isinstance(state.get("source_snapshot"), Mapping):
        return _source_signature(state["source_snapshot"])
    return None


def _invalidate_dependent_stages(
    output: Path,
    reason: str,
    *,
    current_source: Optional[Mapping[str, Any]] = None,
) -> None:
    """Invalidate downstream GPU stages while preserving every raw attempt."""

    for stage_name in ("measure", "select", "profile", "validate", "performance"):
        stage_root = output / stage_name
        state = _stage_state(stage_root)
        if not state or state.get("status") != "ok":
            continue
        attempt_value = state.get("attempt")
        existing_attempt = (
            (stage_root / attempt_value).resolve()
            if isinstance(attempt_value, str)
            else None
        )
        if existing_attempt is not None:
            root = stage_root.resolve()
            if existing_attempt != root and root not in existing_attempt.parents:
                raise SuiteError(f"Stage state escapes its output directory: {stage_root}")
        if current_source is not None and existing_attempt is not None:
            current_signature = _source_signature(current_source)
            existing_signature = _stage_source_signature(stage_name, existing_attempt)
            if existing_signature is not None and _source_matches(existing_signature, current_signature):
                continue
        attempt: Optional[Path] = None
        if isinstance(attempt_value, str):
            attempt = (stage_root / attempt_value).resolve()
            root = stage_root.resolve()
            if attempt != root and root not in attempt.parents:
                raise SuiteError(f"Stage state escapes its output directory: {stage_root}")
        _mark_stage(
            stage_root,
            status="stale",
            attempt=attempt,
            details={"stale_reason": reason},
        )


def _validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or not MODEL_REVISION_RE.fullmatch(revision):
        raise SuiteError("--revision must be an exact 40-character hexadecimal model revision")
    return revision


def _validate_model_snapshot(model_path: str, revision: str) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise SuiteError(f"--model-path must be a local immutable snapshot directory: {path}")
    try:
        identity = model_identity(str(path), revision)
    except (OSError, ValueError) as error:
        raise SuiteError(f"Model snapshot validation failed: {error}") from error
    return {
        "path": str(path),
        "revision": revision,
        "manifest_sha256": identity["manifest_sha256"],
        "file_count": len(identity["files_sha256"]),
    }


def _load_local_tokenizer(model_path: Path, revision: str, tokenizer_path: Optional[str]):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SuiteError(
            "transformers is required in the selected SGLang environment to freeze token IDs"
        ) from error
    source = Path(tokenizer_path).expanduser().resolve() if tokenizer_path else model_path
    if source != model_path:
        raise SuiteError("v1 requires model and tokenizer from the same local snapshot")
    kwargs: dict[str, Any] = {"local_files_only": True}
    if revision:
        kwargs["revision"] = revision
    try:
        return AutoTokenizer.from_pretrained(str(source), **kwargs)
    except Exception as error:  # tokenizer libraries expose several exception types
        raise SuiteError(
            "Could not load the local tokenizer; prepare does not download model data: "
            + str(error)
        ) from error


def _tokenize(tokenizer: Any, prompt: str) -> list[int]:
    try:
        encoded = tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)
        values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    except Exception as error:
        raise SuiteError(f"Tokenizer failed for a prepared prompt: {error}") from error
    if values and isinstance(values[0], list):
        values = values[0]
    if not isinstance(values, list) or not values:
        raise SuiteError("Tokenizer returned an empty/non-list input_ids sequence")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in values):
        raise SuiteError("Tokenizer returned invalid input IDs")
    return list(values)


def _prompt_blueprints() -> dict[str, list[dict[str, Any]]]:
    """Return deterministic prompt schedules with bursts near graph buckets."""

    calibration: list[dict[str, Any]] = []
    burst_sizes = (1, 2, 4, 8, 12, 16)
    serial = 0
    for group_index, burst_size in enumerate(burst_sizes):
        for position in range(burst_size):
            repeat = 1 + (position + group_index) % 5
            calibration.append(
                {
                    "request_id": f"cal-{serial:03d}",
                    "prompt": (
                        f"GraphBudget calibration group {burst_size}, item {position}. "
                        "Describe the fixed decode experiment and keep the answer precise. "
                        + "The same prompt shape is intentionally repeated. " * repeat
                    ),
                    "planned_at_s": round(group_index * 0.25, 3),
                    "max_new_tokens": 8 + (position + group_index) % 3 * 4,
                    "probe": "calibration_adjacent_capture_bucket",
                    "burst_size": burst_size,
                }
            )
            serial += 1

    normal = []
    for index in range(16):
        normal.append(
            {
                "request_id": f"normal-{index:03d}",
                "prompt": (
                    f"Held-out normal request {index}. Summarize a small systems observation "
                    "in one clear paragraph. "
                    + "Use stable wording for deterministic decode. " * (1 + index % 3)
                ),
                "planned_at_s": round(index * 0.18, 3),
                "max_new_tokens": (8, 12, 16)[index % 3],
                "probe": "normal_low_concurrency",
                "burst_size": 1,
            }
        )

    boundary = []
    serial = 0
    for group_index, burst_size in enumerate((3, 7, 8, 9, 12, 15, 16)):
        for position in range(burst_size):
            boundary.append(
                {
                    "request_id": f"boundary-{serial:03d}",
                    "prompt": (
                        f"Held-out boundary burst {burst_size}, item {position}. "
                        "Return a fixed-length technical explanation with no headings. "
                        + "Boundary padding probe word. " * (1 + (position % 4))
                    ),
                    "planned_at_s": round(group_index * 0.22, 3),
                    "max_new_tokens": (8, 12, 16)[(group_index + position) % 3],
                    "probe": "boundary_padding_overflow_probe",
                    "burst_size": burst_size,
                }
            )
            serial += 1

    mixed = []
    for index in range(20):
        repeat = (index * 3) % 9
        mixed.append(
            {
                "request_id": f"mixed-{index:03d}",
                "prompt": (
                    f"Held-out mixed request {index}. Explain one measurable effect. "
                    + "Context variation token. " * repeat
                ),
                "planned_at_s": round(index * 0.075, 3),
                "max_new_tokens": (8, 12, 16)[(index * 2) % 3],
                "probe": "mixed_prompt_and_output_lengths",
                "burst_size": 1 + index % 4,
            }
        )
    return {
        "calibration": calibration,
        "heldout_normal": normal,
        "heldout_boundary": boundary,
        "heldout_mixed": mixed,
    }


def _freeze_workloads(
    tokenizer: Any,
    *,
    model: Mapping[str, Any],
    revision: str,
    prepare_dir: Path,
) -> dict[str, Path]:
    workload_paths: dict[str, Path] = {}
    blueprints = _prompt_blueprints()
    for name, entries in blueprints.items():
        requests = []
        for blueprint in entries:
            input_ids = _tokenize(tokenizer, blueprint["prompt"])
            max_new_tokens = int(blueprint["max_new_tokens"])
            if len(input_ids) + max_new_tokens > 4096:
                raise SuiteError(
                    f"{name}/{blueprint['request_id']} exceeds fixed context_length=4096"
                )
            requests.append(
                {
                    "request_id": blueprint["request_id"],
                    "input_ids": input_ids,
                    "max_new_tokens": max_new_tokens,
                    "planned_at_s": blueprint["planned_at_s"],
                    "metadata": {
                        "probe": blueprint["probe"],
                        "burst_size": blueprint["burst_size"],
                        "prompt_tokens": len(input_ids),
                        "target_output_tokens": max_new_tokens,
                        "tokenizer_revision": revision,
                    },
                }
            )
        purpose = "calibration" if name == "calibration" else name
        workload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "graphbudget_workload",
            "workload_id": f"{name}-tokenized-v1",
            "purpose": purpose,
            "description": (
                "Frozen real-tokenizer workload; request IDs and input IDs are immutable "
                "after prepare."
            ),
            "tokenizer": {
                "model_path": model["path"],
                "revision": revision,
                "model_manifest_sha256": model["manifest_sha256"],
            },
            "token_ids_note": "Token IDs were produced by the local tokenizer at prepare time; no tiny literal placeholder IDs.",
            "token_ids_validated_against_model": True,
            "requests": requests,
        }
        path = prepare_dir / "workloads" / f"{name}.json"
        _write_once(path, workload)
        # Validate the frozen representation before any experiment can consume it.
        load_workload(path)
        workload_paths[name] = path
    return workload_paths


def _derive_min_kv_tokens(workload_paths: Mapping[str, Path]) -> dict[str, int]:
    maximum = 0
    request_count = 0
    for path in workload_paths.values():
        workload = load_workload(path)
        request_count += len(workload.requests)
        maximum = max(
            maximum,
            max(
                len(request.input_ids) + request.max_new_tokens
                for request in workload.requests
            ),
        )
    max_running = MAX_RUNNING_REQUESTS
    minimum = max_running * maximum
    max_total = next(
        (value for value in MAX_TOTAL_TOKENS_CHOICES if value >= minimum), None
    )
    if max_total is None:
        raise SuiteError(
            f"Frozen max_total_tokens choices {MAX_TOTAL_TOKENS_CHOICES} cannot hold "
            f"16 concurrent requests at target length {maximum}"
        )
    return {
        "max_concurrent_requests": max_running,
        "max_prompt_plus_output_tokens": maximum,
        "min_kv_tokens": minimum,
        "max_total_tokens": max_total,
        "workload_request_count": request_count,
    }


def _server_args(
    *,
    model_path: Path,
    revision: str,
    backend: str,
    max_total_tokens: int,
) -> tuple[list[str], dict[str, Any]]:
    args = [
        "--model-path",
        str(model_path),
        "--tokenizer-path",
        str(model_path),
        "--revision",
        revision,
        "--dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--tensor-parallel-size",
        "1",
        "--attention-backend",
        backend,
        "--disable-overlap-schedule",
        "--stream-output",
        "--random-seed",
        "0",
        "--max-running-requests",
        str(MAX_RUNNING_REQUESTS),
        "--max-total-tokens",
        str(max_total_tokens),
        "--context-length",
        "4096",
        "--mem-fraction-static",
        "0.85",
        # No --disable-radix-cache: every baseline uses the same enabled radix
        # cache and the benchmark makes cache-reset scope explicit.
    ]
    config = {
        "mem_fraction_static": 0.85,
        "max_total_tokens": max_total_tokens,
        "max_running_requests": MAX_RUNNING_REQUESTS,
        "context_length": 4096,
    }
    return args, config


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    prepare_dir = output / "prepare"
    prepare_state = _stage_state(prepare_dir)
    state = _completed_attempt(prepare_dir)
    if state is not None:
        return _read_json(prepare_dir / "experiment.json")
    if prepare_state and prepare_state.get("status") == "ok":
        raise SuiteError(
            "prepare artifacts failed immutable verification; use a new output directory"
        )
    revision = _validate_revision(args.revision)
    model = _validate_model_snapshot(args.model_path, revision)
    model_path = Path(model["path"])
    tokenizer = _load_local_tokenizer(model_path, revision, args.tokenizer_path)
    prepare_dir.mkdir(parents=True, exist_ok=True)
    workload_paths = _freeze_workloads(
        tokenizer, model=model, revision=revision, prepare_dir=prepare_dir
    )
    kv = _derive_min_kv_tokens(workload_paths)
    server_args, plan_server_args = _server_args(
        model_path=model_path,
        revision=revision,
        backend=args.backend,
        max_total_tokens=kv["max_total_tokens"],
    )
    if getattr(args, "deterministic", False):
        server_args.append("--enable-deterministic-inference")
    source = _source_snapshot(args)
    experiment = {
        "schema_version": SCHEMA_VERSION,
        "kind": "graphbudget_experiment",
        "model": model,
        "backend": args.backend,
        "dtype": "bfloat16",
        "tp_size": 1,
        "deterministic_inference": bool(getattr(args, "deterministic", False)),
        "batch_invariant_deepgemm": os.environ.get("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"),
        "radix_cache": "enabled_and_explicitly_reset_between_cases",
        "server_args": server_args,
        "plan_server_args": plan_server_args,
        "calibration_workload": str(workload_paths["calibration"]),
        "heldout_workloads": {
            name: str(workload_paths[name]) for name in HELDOUT_WORKLOAD_NAMES
        },
        "candidates": [
            {"id": candidate_id, "capture_sizes": sizes}
            for candidate_id, sizes in CAPTURE_CANDIDATES
        ],
        "budget": {
            "allowed_device_bytes": 7 * 1024**3,
            "min_kv_tokens": kv["min_kv_tokens"],
            "max_startup_seconds": float(args.startup_timeout),
        },
        "kv_derivation": kv,
        "port": int(args.port),
        "request_timeout_s": float(args.request_timeout),
        "source": {
            **source,
        },
        "frozen_before_measurement": True,
    }
    _write_once(prepare_dir / "experiment.json", experiment)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "graphbudget_suite_manifest",
        "created_at_utc": _utc_now(),
        "graphbudget_commit": experiment["source"]["graphbudget_commit"],
        "graphbudget_identity_sha256": experiment["source"]["graphbudget_identity_sha256"],
        "sglang_commit": experiment["source"]["sglang_commit"],
        "sglang_patch_sha256": experiment["source"]["sglang_patch_sha256"],
        "graphbudget_source_sha256": experiment["source"]["graphbudget_source_sha256"],
        "sglang_tree_sha256": experiment["source"]["sglang_tree_sha256"],
        "model": model,
        "backend": args.backend,
        "dtype": "bfloat16",
        "tp_size": 1,
        "deterministic_inference": bool(getattr(args, "deterministic", False)),
        "batch_invariant_deepgemm": os.environ.get("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"),
        "prepare_artifacts": {
            "experiment_sha256": _sha256_file(prepare_dir / "experiment.json"),
            "workloads": {
                name: _sha256_file(path) for name, path in workload_paths.items()
            },
        },
        "constraints": {
            "max_running_requests": MAX_RUNNING_REQUESTS,
            "allowed_device_bytes": 7 * 1024**3,
            "max_total_tokens": kv["max_total_tokens"],
            "min_kv_tokens": kv["min_kv_tokens"],
        },
    }
    _write_once(prepare_dir / "manifest.json", manifest)
    _mark_stage(prepare_dir, status="ok", attempt=prepare_dir, details={"manifest": manifest})
    return experiment


def _require_experiment(output: Path) -> dict[str, Any]:
    prepare_dir = output / "prepare"
    if _completed_attempt(prepare_dir) is None:
        raise SuiteError("prepare phase is not complete; run --phase prepare first")
    return _read_json(prepare_dir / "experiment.json")


def _stage_source(experiment: Mapping[str, Any]) -> Mapping[str, Any]:
    source = experiment.get("_current_source", experiment.get("source", {}))
    return source if isinstance(source, Mapping) else {}


def _phase_measure(output: Path, experiment: Mapping[str, Any]) -> Path:
    stage = output / "measure"
    completed = _completed_attempt(stage)
    if completed is not None:
        return completed
    attempt = _new_attempt(stage)
    _mark_stage(stage, status="running", attempt=attempt)
    try:
        measurements = measure_candidates(output / "prepare" / "experiment.json", attempt)
        expected = {candidate_id for candidate_id, _ in CAPTURE_CANDIDATES}
        actual = {str(row.get("id")) for row in measurements}
        if actual != expected or len(measurements) != len(CAPTURE_CANDIDATES):
            raise SuiteError("measurement did not retain exactly the three frozen candidates")
        if not any(row.get("status") == "ok" for row in measurements):
            raise SuiteError("All candidate measurements failed; inspect retained raw cases")
        _mark_stage(
            stage,
            status="ok",
            attempt=attempt,
            details={
                "measurements_sha256": _sha256_file(attempt / "measurements.json"),
                "experiment_sha256": _sha256_file(output / "prepare" / "experiment.json"),
                "source_snapshot": dict(_stage_source(experiment)),
            },
        )
        return attempt
    except Exception as error:
        _mark_stage(stage, status="failed", attempt=attempt, details={"error": str(error)})
        raise


def _phase_select(output: Path, experiment: Mapping[str, Any]) -> Path:
    stage = output / "select"
    completed = _completed_attempt(stage)
    if completed is not None:
        return completed / "plan.json"
    measure_attempt = _completed_attempt(output / "measure")
    if measure_attempt is None:
        raise SuiteError("measurement phase is not complete")
    attempt = _new_attempt(stage)
    _mark_stage(stage, status="running", attempt=attempt)
    try:
        measurements = _read_json(measure_attempt / "measurements.json")
        identity = measurements.get("identity")
        candidates = measurements.get("candidates")
        if not isinstance(identity, Mapping) or not isinstance(candidates, list):
            raise SuiteError("measurements are missing identity/candidates")
        plan = select_plan(identity, experiment["budget"], candidates)
        plan_path = attempt / "plan.json"
        _write_once(plan_path, plan)
        _mark_stage(
            stage,
            status="ok",
            attempt=attempt,
            details={
                "plan_sha256": _sha256_file(plan_path),
                "measurements_sha256": _sha256_file(measure_attempt / "measurements.json"),
                "experiment_sha256": _sha256_file(output / "prepare" / "experiment.json"),
                "source_snapshot": dict(_stage_source(experiment)),
            },
        )
        return plan_path
    except Exception as error:
        _mark_stage(stage, status="failed", attempt=attempt, details={"error": str(error)})
        raise


def _load_plan_path(output: Path) -> Path:
    stage = _completed_attempt(output / "select")
    if stage is None:
        raise SuiteError("select phase is not complete")
    path = stage / "plan.json"
    if not path.is_file():
        raise SuiteError("select phase has no plan.json")
    return path


def _run_case(
    experiment: Mapping[str, Any],
    *,
    workload: Path,
    output: Path,
    policy: str = "native",
    capture_sizes: Optional[Sequence[int]] = None,
    plan: Optional[Path] = None,
    profile: Optional[Path] = None,
    server_args: Optional[Sequence[str]] = None,
    diagnostic: bool = False,
    calibrate: bool = False,
    warmup: bool = True,
) -> dict[str, Any]:
    effective_args = list(server_args) if server_args is not None else experiment["server_args"]
    if output.exists():
        case_path = output / "case.json"
        invocation = _read_json(output / "invocation.json")
        if case_path.is_file():
            case = _read_json(case_path)
            command = invocation["command"]
            split = command.index("--")
            head, tail = command[:split], command[split + 1:]
            expected_tail = [*effective_args, "--host", "127.0.0.1", "--port", str(experiment["port"])]
            if capture_sizes is not None:
                expected_tail += ["--cuda-graph-bs", *map(str, capture_sizes)]
            expected = {"--policy": policy, "--observe": "diagnostic" if diagnostic else "counts"}
            for name, value in (("--plan", plan), ("--profile", profile)):
                expected[name] = str(Path(value).resolve()) if value is not None else None
            actual = {name: head[head.index(name) + 1] if name in head else None for name in expected}
            ready = case.get("ready", {})
            if (case.get("status") != "ok" or tail != expected_tail or actual != expected
                    or ("--calibrate" in head) != calibrate
                    or (output / "warmup.json").is_file() != warmup
                    or ready.get("fingerprint") != experiment.get("_expected_runtime_fingerprint")
                    or case.get("benchmark", {}).get("workload") != load_workload(workload).to_dict()
                    or case.get("run_id") != invocation.get("run_id")
                    or case.get("benchmark", {}).get("run", {}).get("run_id") != case.get("run_id")):
                raise SuiteError(f"Existing case does not match the frozen requested run: {output}")
            return case
        # A refused lease creates only invocation.json, before any process or
        # measurement is started. Preserve that command and use a fresh path.
        if output.is_symlink() or {p.name for p in output.iterdir()} != {"invocation.json"}:
            raise SuiteError(f"Incomplete executed case requires a new attempt: {output}")
        archived = output.with_name(output.name + f".lease-rejected-{time.time_ns()}")
        output.rename(archived)
    deadline = time.monotonic() + 60
    while True:
        try:
            with gpu_lease():
                pass
            break
        except RuntimeError as error:
            if "owns the lease" not in str(error) or time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
    return run_case(
        effective_args,
        workload,
        output,
        capture_sizes=capture_sizes,
        plan=plan,
        profile=profile,
        policy=policy,
        diagnostic=diagnostic,
        calibrate=calibrate,
        startup_timeout=float(experiment["budget"]["max_startup_seconds"]),
        request_timeout=float(experiment["request_timeout_s"]),
        port=int(experiment["port"]),
        warmup=warmup,
    )


def _timed_request_ids(case: Mapping[str, Any]) -> set[str]:
    benchmark = case.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return set()
    requests = benchmark.get("requests")
    if not isinstance(requests, list):
        return set()
    run = benchmark.get("run")
    timed_run_id = run.get("run_id") if isinstance(run, Mapping) else None
    prefix = f"graphbudget/{timed_run_id}/" if isinstance(timed_run_id, str) else None
    return {
        str(request["client_request_id"])
        for request in requests
        if isinstance(request, Mapping)
        and request.get("status") == "success"
        and isinstance(request.get("client_request_id"), str)
        and (prefix is None or request["client_request_id"].startswith(prefix))
    }


def _timed_steps(trace: Mapping[str, Any], case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    timed_ids = _timed_request_ids(case)
    if not timed_ids:
        return []
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return []
    benchmark = case.get("benchmark")
    run = benchmark.get("run") if isinstance(benchmark, Mapping) else None
    timed_run_id = run.get("run_id") if isinstance(run, Mapping) else None
    prefix = f"graphbudget/{timed_run_id}/" if isinstance(timed_run_id, str) else None
    selected: list[Mapping[str, Any]] = []
    for row in steps:
        if not isinstance(row, Mapping):
            continue
        request_ids = row.get("request_ids")
        if not isinstance(request_ids, list) or not request_ids:
            continue
        if all(
            isinstance(request_id, str)
            and request_id in timed_ids
            and (prefix is None or request_id.startswith(prefix))
            for request_id in request_ids
        ):
            selected.append(row)
    return selected


def _timed_path_counts(case: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prefer the controller's counter delta; otherwise derive from filtered steps."""

    for key in ("timed_path_counts", "path_counts_delta", "timed_counts", "path_counts_timed"):
        value = case.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, Mapping)]
    grouped: dict[tuple[Any, ...], int] = {}
    fields = (
        "eligible",
        "selected_path",
        "used_path",
        "batch_size",
        "actual_capture_size",
        "reason",
    )
    for row in steps:
        key = tuple(row.get(field) for field in fields)
        grouped[key] = grouped.get(key, 0) + 1
    return [dict(zip(fields, key), count=count) for key, count in grouped.items()]


def _trace_coverage(case_dir: Path, workload_name: str) -> dict[str, Any]:
    case = _read_json(case_dir / "case.json")
    trace_path = case_dir / "paths_after.json"
    if not trace_path.is_file():
        raise SuiteError(f"Missing native path trace for {workload_name}: {trace_path}")
    trace = _read_json(trace_path)
    steps = _timed_steps(trace, case)
    counts = _timed_path_counts(case, steps)
    dropped_steps = case.get("timed_dropped_steps", case.get("dropped_steps"))
    if dropped_steps is None:
        dropped_steps = trace.get("dropped_steps", 0)
    if dropped_steps:
        raise SuiteError(f"Diagnostic trace dropped steps for {workload_name}")
    if not isinstance(steps, list) or not steps or not isinstance(counts, list) or not counts:
        raise SuiteError(
            f"No observed runtime path coverage for {workload_name}; refusing to assume it"
        )
    reasons = sorted({str(row.get("reason")) for row in steps if row.get("reason") is not None})
    used_paths = sorted({str(row.get("used_path")) for row in steps})
    batch_sizes = sorted(
        {int(row["batch_size"]) for row in steps if isinstance(row.get("batch_size"), int)}
    )
    padded_sizes = sorted(
        {
            int(row["padded_size"])
            for row in steps
            if isinstance(row.get("padded_size"), int)
        }
    )
    batch_padded_pairs = sorted(
        {
            (int(row["batch_size"]), int(row["padded_size"]))
            for row in steps
            if isinstance(row.get("batch_size"), int)
            and isinstance(row.get("padded_size"), int)
        }
    )
    return {
        "workload": workload_name,
        "observed_step_count": len(steps),
        "observed_count_rows": len(counts),
        "timed_completed_request_count": len(_timed_request_ids(case)),
        "observed_reasons": reasons,
        "observed_used_paths": used_paths,
        "observed_batch_sizes": batch_sizes,
        "observed_padded_sizes": padded_sizes,
        "observed_batch_padded_pairs": [list(pair) for pair in batch_padded_pairs],
        "dropped_steps": dropped_steps,
        "coverage_is_observed": True,
    }


def _variant_specs(
    experiment: Mapping[str, Any], plan_path: Path, profile_path: Path
) -> dict[str, dict[str, Any]]:
    selected_sizes = _read_json(plan_path)["selected"]["capture_sizes"]
    base = list(experiment["server_args"])
    eager_args = base + ["--disable-cuda-graph"]
    return {
        "U": {"server_args": base, "policy": "native", "capture_sizes": None, "plan": None, "profile": None},
        "E": {"server_args": eager_args, "policy": "native", "capture_sizes": None, "plan": None, "profile": None},
        "P": {"server_args": base, "policy": "native", "capture_sizes": None, "plan": plan_path, "profile": None},
        "T": {"server_args": base, "policy": "native", "capture_sizes": selected_sizes, "plan": None, "profile": None},
        "C": {"server_args": base, "policy": "cost", "capture_sizes": None, "plan": plan_path, "profile": profile_path},
    }


def _phase_profile(output: Path, experiment: Mapping[str, Any]) -> Path:
    stage = output / "profile"
    completed = _completed_attempt(stage)
    if completed is not None:
        return completed / "profile.json"
    plan_path = _load_plan_path(output)
    selected_sizes = _read_json(plan_path)["selected"]["capture_sizes"]
    attempt = _new_attempt(stage)
    _mark_stage(stage, status="running", attempt=attempt)
    graph_dirs: list[Path] = []
    eager_dirs: list[Path] = []
    try:
        workload = Path(experiment["calibration_workload"])
        for round_index in range(3):
            graph_dir = attempt / f"round-{round_index + 1:02d}" / "graph"
            eager_dir = attempt / f"round-{round_index + 1:02d}" / "eager"
            graph = _run_case(
                experiment,
                workload=workload,
                output=graph_dir,
                policy="native",
                capture_sizes=selected_sizes,
                diagnostic=True,
                calibrate=True,
            )
            eager = _run_case(
                experiment,
                workload=workload,
                output=eager_dir,
                policy="test_eager",
                capture_sizes=selected_sizes,
                diagnostic=True,
                calibrate=True,
            )
            graph_dirs.append(graph_dir)
            eager_dirs.append(eager_dir)
            if graph.get("status") != "ok" or eager.get("status") != "ok":
                raise SuiteError(
                    f"Diagnostic calibration failed in round {round_index + 1}: "
                    f"graph={graph.get('reason')} eager={eager.get('reason')}"
                )
        profile = build_cost_profile(graph_dirs, eager_dirs)
        if not profile.get("entries"):
            profile = dict(profile)
            profile["enabled"] = False
            profile["negative_support"] = True
            profile["negative_support_reason"] = "No matched measured graph/eager rows reached the minimum sample count"
        profile_path = attempt / "profile.json"
        _write_once(profile_path, profile)
        _mark_stage(
            stage,
            status="ok",
            attempt=attempt,
            details={
                "profile_sha256": _sha256_file(profile_path),
                "entry_count": len(profile.get("entries", [])),
                "enabled": bool(profile.get("enabled")),
                "plan_sha256": _sha256_file(plan_path),
                "experiment_sha256": _sha256_file(output / "prepare" / "experiment.json"),
                "source_snapshot": dict(_stage_source(experiment)),
            },
        )
        return profile_path
    except Exception as error:
        _mark_stage(stage, status="failed", attempt=attempt, details={"error": str(error)})
        raise


def _phase_validate(output: Path, experiment: Mapping[str, Any]) -> Path:
    stage = output / "validate"
    completed = _completed_attempt(stage)
    if completed is not None:
        return completed / "validation.json"
    plan_path = _load_plan_path(output)
    profile_stage = _completed_attempt(output / "profile")
    if profile_stage is None:
        raise SuiteError("profile phase is not complete")
    profile_path = profile_stage / "profile.json"
    attempt = _new_attempt(stage)
    _mark_stage(stage, status="running", attempt=attempt)
    try:
        workloads = {
            "calibration": Path(experiment["calibration_workload"]),
            **{name: Path(path) for name, path in experiment["heldout_workloads"].items()},
        }
        specs = _variant_specs(experiment, plan_path, profile_path)
        cases: dict[str, dict[str, dict[str, Any]]] = {}
        coverage: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            cases[variant] = {}
            variant_spec = specs[variant]
            for workload_name, workload_path in workloads.items():
                case_dir = attempt / variant / workload_name
                case = _run_case(
                    experiment,
                    workload=workload_path,
                    output=case_dir,
                    policy=variant_spec["policy"],
                    capture_sizes=variant_spec["capture_sizes"],
                    plan=variant_spec["plan"],
                    profile=variant_spec["profile"],
                    server_args=variant_spec["server_args"],
                    diagnostic=True,
                )
                if case.get("status") != "ok":
                    raise SuiteError(
                        f"Validation case failed: {variant}/{workload_name}: {case.get('reason')}"
                    )
                cases[variant][workload_name] = case
                coverage[f"{variant}/{workload_name}"] = _trace_coverage(
                    case_dir, workload_name
                )

        # A selected plan may legitimately include 16, so it is not valid to
        # demand that P itself overflow.  Use an explicit cap-4 native control
        # on the burst workload to make overflow evidence an observed, separate
        # diagnostic rather than an assumption about the selected plan.
        overflow_dir = attempt / "overflow-control" / "heldout_boundary"
        overflow_case = _run_case(
            experiment,
            workload=workloads["heldout_boundary"],
            output=overflow_dir,
            policy="native",
            capture_sizes=[1, 2, 4],
            server_args=specs["U"]["server_args"],
            diagnostic=True,
        )
        if overflow_case.get("status") != "ok":
            raise SuiteError(f"Overflow control case failed: {overflow_case.get('reason')}")
        coverage["overflow-control/heldout_boundary"] = _trace_coverage(
            overflow_dir, "heldout_boundary"
        )

        comparisons = []
        for workload_name in workloads:
            for left, right in combinations(VARIANTS, 2):
                comparison = compare_exact_tokens(
                    cases[left][workload_name]["benchmark"],
                    cases[right][workload_name]["benchmark"],
                )
                row = {
                    "workload": workload_name,
                    "left": left,
                    "right": right,
                    "equivalent": comparison["equivalent"],
                    "comparison": comparison,
                }
                comparisons.append(row)
                if not comparison["equivalent"]:
                    raise SuiteError(
                        f"Exact token mismatch: {left}/{right}/{workload_name}"
                    )
        overflow_comparison = compare_exact_tokens(
            cases["P"]["heldout_boundary"]["benchmark"],
            overflow_case["benchmark"],
        )
        if not overflow_comparison["equivalent"]:
            raise SuiteError("Exact token mismatch between P and cap-4 overflow control")

        # These are assertions about observed traces, not assumptions from the
        # requested workload labels.  If a future runtime cannot expose a
        # padding or overflow reason, validation remains incomplete.
        overflow_coverage = coverage.get("overflow-control/heldout_boundary", {})
        if not any(
            reason in {"capture_ceiling", "uncaptured_exact_size"}
            for reason in overflow_coverage.get("observed_reasons", [])
        ):
            raise SuiteError(
                "Cap-4 overflow control did not observe capture-ceiling/uncaptured fallback "
                f"(reasons={overflow_coverage.get('observed_reasons')})"
            )
        mixed_coverage = overflow_coverage
        if not any(
            padded > batch
            for batch, padded in mixed_coverage.get("observed_batch_padded_pairs", [])
        ):
            raise SuiteError("Cap-4 boundary control did not observe any padded batch size")

        validation = {
            "schema_version": SCHEMA_VERSION,
            "kind": "graphbudget_validation_suite",
            "status": "ok",
            "variants": list(VARIANTS),
            "cases": {
                variant: {
                    name: str(attempt / variant / name)
                    for name in workloads
                }
                for variant in VARIANTS
            },
            "control_cases": {
                "overflow_cap4_native": str(overflow_dir),
            },
            "coverage": coverage,
            "exact_token_comparisons": comparisons,
            "control_exact_token_comparisons": [
                {
                    "left": "P",
                    "right": "overflow-control-cap4-native",
                    "workload": "heldout_boundary",
                    "comparison": overflow_comparison,
                }
            ],
            "coverage_assertion": "derived only from observed paths_after.json; missing boundary/padding evidence fails validation",
        }
        validation_path = attempt / "validation.json"
        _write_once(validation_path, validation)
        _mark_stage(
            stage,
            status="ok",
            attempt=attempt,
            details={
                "validation_sha256": _sha256_file(validation_path),
                "plan_sha256": _sha256_file(plan_path),
                "profile_sha256": _sha256_file(profile_path),
                "experiment_sha256": _sha256_file(output / "prepare" / "experiment.json"),
                "source_snapshot": dict(_stage_source(experiment)),
            },
        )
        return validation_path
    except Exception as error:
        _mark_stage(stage, status="failed", attempt=attempt, details={"error": str(error)})
        raise


def _case_benchmark(case_dir: Path) -> Mapping[str, Any]:
    case = _read_json(case_dir / "case.json")
    if case.get("status") != "ok" or not isinstance(case.get("benchmark"), Mapping):
        raise SuiteError(f"Case is not a successful benchmark: {case_dir}")
    return case["benchmark"]


def _case_path_evidence(case_dir: Path) -> dict[str, Any]:
    """Summarize timed counters without treating cumulative warmup counts as timed."""

    case = _read_json(case_dir / "case.json")
    ready = case.get("ready") if isinstance(case.get("ready"), Mapping) else {}
    counts = None
    for key in ("timed_path_counts", "path_counts_delta", "timed_counts", "path_counts_timed"):
        if isinstance(case.get(key), list):
            counts = [dict(row) for row in case[key] if isinstance(row, Mapping)]
            break
    policy_status = ready.get("policy_status")
    policy_mode = ready.get("policy_mode")
    used_paths = sorted(
        {
            str(row.get("used_path"))
            for row in (counts or [])
            if row.get("used_path") is not None
        }
    )
    if counts is None:
        return {
            "status": "not_available",
            "reason": "timed path counter delta was not recorded",
            "policy_mode": policy_mode,
            "policy_status": policy_status,
            "timed_path_counts": [],
            "used_paths": used_paths,
        }
    if policy_mode != "cost":
        attribution_status = "not_attributable"
        reason = "case is not a cost-policy run"
    elif policy_status != "loaded":
        attribution_status = "not_attributable"
        reason = f"cost policy status is {policy_status!r}"
    elif not any(row.get("eligible") is True and row.get("used_path") == "eager"
                 and row.get("selected_path") == "eager"
                 and row.get("reason") == "eager_faster_beyond_noise"
                 and row.get("count", 0) > 0 for row in counts):
        attribution_status = "not_attributable"
        reason = "no measured cost rejection of an eligible graph; overflow is upstream fallback"
    else:
        attribution_status = "attributable"
        reason = "timed eligible graph rejected by the measured cost rule"
    return {
        "status": attribution_status,
        "reason": reason,
        "policy_mode": policy_mode,
        "policy_status": policy_status,
        "timed_path_counts": counts,
        "used_paths": used_paths,
    }


def _exploratory_single_runs(
    attempt: Path,
    experiment: Mapping[str, Any],
    workloads: Mapping[str, Path],
    specs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run one clean U/E/P case per held-out workload, separately from T/C."""

    rows: list[dict[str, Any]] = []
    for workload_name, workload_path in workloads.items():
        for variant in ("U", "E", "P"):
            case_dir = attempt / "exploratory" / workload_name / variant
            row: dict[str, Any] = {
                "variant": variant,
                "workload": workload_name,
                "raw_directory": str(case_dir),
            }
            try:
                case = _run_case(
                    experiment,
                    workload=workload_path,
                    output=case_dir,
                    policy=specs[variant]["policy"],
                    capture_sizes=specs[variant]["capture_sizes"],
                    plan=specs[variant]["plan"],
                    profile=specs[variant]["profile"],
                    server_args=specs[variant]["server_args"],
                    diagnostic=False,
                )
                row.update(
                    status=case.get("status"),
                    reason=case.get("reason"),
                    path_evidence=_case_path_evidence(case_dir),
                )
                if case.get("status") == "ok":
                    row["benchmark"] = case["benchmark"]
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            rows.append(row)
    return rows


def _phase_performance(
    output: Path,
    experiment: Mapping[str, Any],
    *,
    rounds: int,
) -> Path:
    if rounds < 5:
        raise SuiteError("Performance phase requires at least five paired rounds")
    stage = output / "performance"
    completed = _completed_attempt(stage)
    completed_state = _stage_state(stage)
    if completed is not None and completed_state and completed_state.get("rounds") == rounds:
        return completed / "performance.json"
    plan_path = _load_plan_path(output)
    profile_stage = _completed_attempt(output / "profile")
    if profile_stage is None:
        raise SuiteError("profile phase is not complete")
    profile_path = profile_stage / "profile.json"
    attempt = _new_attempt(stage)
    prior = _stage_state(stage)
    if (prior and prior.get("status") == "failed"
            and "owns the lease" in prior.get("error", "")
            and re.fullmatch(r"attempt-\d+", prior.get("attempt", ""))):
        attempt = stage / prior["attempt"]
    experiment = {**experiment, "_expected_runtime_fingerprint": _read_json(plan_path)["fingerprint"]["sha256"]}
    _mark_stage(stage, status="running", attempt=attempt)
    try:
        workloads = {
            name: Path(path) for name, path in experiment["heldout_workloads"].items()
        }
        specs = _variant_specs(experiment, plan_path, profile_path)
        exploratory = _exploratory_single_runs(attempt, experiment, workloads, specs)
        aa_scope = "heldout_boundary"
        aa_pairs_scope = []
        aa_workload_path = workloads[aa_scope]
        # Collect all A/A rounds first on the most meaningful stress workload.
        # T/C rounds below then start only after the noise baseline exists.
        for round_index in range(rounds):
            aa_a_dir = attempt / "aa" / aa_scope / f"round-{round_index + 1:02d}" / "a"
            aa_b_dir = attempt / "aa" / aa_scope / f"round-{round_index + 1:02d}" / "b"
            aa_a = _run_case(
                experiment,
                workload=aa_workload_path,
                output=aa_a_dir,
                policy="native",
                capture_sizes=specs["T"]["capture_sizes"],
                server_args=specs["T"]["server_args"],
                diagnostic=False,
            )
            aa_b = _run_case(
                experiment,
                workload=aa_workload_path,
                output=aa_b_dir,
                policy="native",
                capture_sizes=specs["T"]["capture_sizes"],
                server_args=specs["T"]["server_args"],
                diagnostic=False,
            )
            if aa_a.get("status") != "ok" or aa_b.get("status") != "ok":
                raise SuiteError(f"A/A case failed in {aa_scope} round {round_index + 1}")
            aa_bench_a = _case_benchmark(aa_a_dir)
            aa_bench_b = _case_benchmark(aa_b_dir)
            aa_cmp = compare_exact_tokens(aa_bench_a, aa_bench_b)
            if not aa_cmp["equivalent"]:
                raise SuiteError(f"A/A exact-token mismatch in {aa_scope} round {round_index + 1}")
            aa_pairs_scope.append({"baseline": aa_bench_a, "candidate": aa_bench_b})

        output_rows: dict[str, Any] = {}
        for workload_name, workload_path in workloads.items():
            paired = []
            pair_exact = []
            path_evidence_rows = []
            for round_index in range(rounds):
                order = ("T", "C") if round_index % 2 == 0 else ("C", "T")
                benches: dict[str, Mapping[str, Any]] = {}
                path_evidence: dict[str, Any] = {}
                for variant in order:
                    case_dir = attempt / "paired" / workload_name / f"round-{round_index + 1:02d}" / variant
                    case = _run_case(
                        experiment,
                        workload=workload_path,
                        output=case_dir,
                        policy=specs[variant]["policy"],
                        capture_sizes=specs[variant]["capture_sizes"],
                        plan=specs[variant]["plan"],
                        profile=specs[variant]["profile"],
                        server_args=specs[variant]["server_args"],
                        diagnostic=False,
                    )
                    if case.get("status") != "ok":
                        raise SuiteError(
                            f"Paired case failed in {workload_name} round {round_index + 1}: {variant}"
                        )
                    benches[variant] = _case_benchmark(case_dir)
                    path_evidence[variant] = _case_path_evidence(case_dir)
                comparison = compare_exact_tokens(benches["T"], benches["C"])
                if not comparison["equivalent"]:
                    raise SuiteError(f"T/C exact-token mismatch in {workload_name} round {round_index + 1}")
                paired.append({"baseline": benches["T"], "candidate": benches["C"]})
                pair_exact.append(comparison)
                path_evidence_rows.append(
                    {"round": round_index + 1, "cases": path_evidence}
                )

            aa_pairs = aa_pairs_scope if workload_name == aa_scope else None
            summary = summarize_paired_rounds(paired, aa_pairs=aa_pairs)
            if summary.get("status") != "OK":
                raise SuiteError(f"Paired summary is not statistically ready for {workload_name}")
            if workload_name == aa_scope and summary.get("aa_noise", {}).get("status") != "OK":
                raise SuiteError(f"A/A noise summary is not ready for {workload_name}")
            output_rows[workload_name] = {
                "paired_round_count": len(paired),
                "aa_round_count": len(aa_pairs_scope) if workload_name == aa_scope else 0,
                "aa_scope": workload_name == aa_scope,
                "summary": summary,
                "exact_token_comparisons": pair_exact,
                "path_evidence": path_evidence_rows,
                "cost_attribution": {
                    "status": (
                        "attributable"
                        if all(
                            row["cases"].get("C", {}).get("status") == "attributable"
                            for row in path_evidence_rows
                        )
                        else "not_attributable"
                    ),
                    "reason": (
                        "all timed C cases observed eager with loaded cost policy"
                        if all(
                            row["cases"].get("C", {}).get("status") == "attributable"
                            for row in path_evidence_rows
                        )
                        else "at least one C case lacked loaded/eager timed-path evidence"
                    ),
                },
                "raw_directory": str(attempt),
            }

        performance = {
            "schema_version": SCHEMA_VERSION,
            "kind": "graphbudget_performance_suite",
            "status": "ok",
            "rounds": rounds,
            "aa_scope": aa_scope,
            "exploratory_single_runs": exploratory,
            "workloads": output_rows,
            "notes": [
                "A/A noise was collected before T/C paired rounds.",
                "Raw case directories remain the authority for request events and paths.",
                "No P99 or adoption claim is made by this orchestration layer.",
            ],
        }
        performance_path = attempt / "performance.json"
        _write_once(performance_path, performance)
        _mark_stage(
            stage,
            status="ok",
            attempt=attempt,
            details={
                "performance_sha256": _sha256_file(performance_path),
                "plan_sha256": _sha256_file(plan_path),
                "profile_sha256": _sha256_file(profile_path),
                "experiment_sha256": _sha256_file(output / "prepare" / "experiment.json"),
                "source_snapshot": dict(_stage_source(experiment)),
                "rounds": rounds,
            },
        )
        return performance_path
    except Exception as error:
        _mark_stage(stage, status="failed", attempt=attempt, details={"error": str(error)})
        raise


def _ensure_output_root(args: argparse.Namespace) -> Path:
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise SuiteError("Output exists; pass --resume to reuse immutable preparation/raw artifacts")
        manifest = output / "prepare" / "manifest.json"
        if not manifest.is_file():
            raise SuiteError("--resume requires an existing prepare/manifest.json")
        current = _read_json(manifest)
        if current.get("backend") != args.backend:
            raise SuiteError("Resume backend differs from frozen manifest")
        if current.get("deterministic_inference", False) != bool(getattr(args, "deterministic", False)):
            raise SuiteError("Resume deterministic mode differs from frozen manifest")
        if current.get("batch_invariant_deepgemm") != os.environ.get("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"):
            raise SuiteError("Resume batch-invariant matmul backend differs from frozen manifest")
        model_path = str(Path(args.model_path).expanduser().resolve())
        if current.get("model", {}).get("path") != model_path:
            raise SuiteError("Resume model path differs from frozen manifest")
        revision = _validate_revision(args.revision)
        if current.get("model", {}).get("revision") != revision:
            raise SuiteError("Resume model revision differs from frozen manifest")
        if args.tokenizer_path and str(Path(args.tokenizer_path).expanduser().resolve()) != model_path:
            raise SuiteError("v1 requires the resume tokenizer path to equal the model snapshot")

        # Re-hash the local snapshot on resume.  A path/revision pair is not
        # enough to bind prepared token IDs to the same files.
        current_model = _validate_model_snapshot(model_path, revision)
        if current_model != current.get("model"):
            raise SuiteError("Resume model snapshot manifest differs from prepare")

        source_now = _source_snapshot(args)
        # Preparation binds inputs; each GPU stage binds its executed source.
        # Compare stages even if an older prepare manifest has no source hash.
        _invalidate_dependent_stages(
            output,
            "runtime source differs from the executed stage; raw artifacts retained",
            current_source=source_now,
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "deterministic", False):
        # Upstream DeepGEMM rejects SM120 in this pinned build; its existing
        # Triton implementation preserves the batch-invariance contract.
        os.environ["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"] = "0"
    output = _ensure_output_root(args)
    phase = args.phase
    results: dict[str, Any] = {"output": str(output), "phase": phase}
    if phase in ("prepare", "all"):
        experiment = _prepare(args)
        results["prepare"] = "ok"
    else:
        experiment = _require_experiment(output)
    experiment = dict(experiment)
    experiment["_current_source"] = _source_snapshot(args)
    if phase in ("measure", "all"):
        measure_attempt = _phase_measure(output, experiment)
        results["measure"] = str(measure_attempt)
    if phase in ("select", "all"):
        plan_path = _phase_select(output, experiment)
        results["select"] = str(plan_path)
    if phase in ("profile", "all"):
        profile_path = _phase_profile(output, experiment)
        results["profile"] = str(profile_path)
    if phase in ("validate", "all"):
        validation_path = _phase_validate(output, experiment)
        results["validate"] = str(validation_path)
    if phase in ("performance", "all"):
        performance_path = _phase_performance(output, experiment, rounds=args.rounds)
        results["performance"] = str(performance_path)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--backend", choices=("triton", "flashinfer"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase",
        choices=("prepare", "measure", "select", "profile", "validate", "performance", "all"),
        default="prepare",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use upstream batch-invariant Triton matmul for strict SM120 token checks; frozen for every variant")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--sglang-checkout")
    parser.add_argument("--port", type=int, default=30150)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--rounds", type=int, default=5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _run(args)
    except (SuiteError, ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 2
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
