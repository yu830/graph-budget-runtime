"""Small, fail-closed GraphBudget plan/profile contract.

The public API is dict-first so a launcher can load a frozen JSON plan without
depending on a framework or a project-specific object graph.  Every candidate
is measured as one complete capture configuration.  Device budget checks use
the same-sample ``device_total_bytes``/``device_used_bytes`` values only; no
allocator, reserved, or host-memory counters are added.  Candidate evidence
binds the actual fingerprint and both requested and observed capture sets.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


PLAN_SCHEMA_VERSION = 1
MEASURED_PROVENANCE = "measured"
SYNTHETIC_CPU_PROVENANCE = "synthetic_cpu_fixture"
VALID_PROVENANCE = frozenset({MEASURED_PROVENANCE, SYNTHETIC_CPU_PROVENANCE})
VALID_STATUS = frozenset({"ok", "failed", "oom", "unsupported"})
ALLOWED_SERVER_ARGS = frozenset(
    {
        "mem_fraction_static",
        "max_total_tokens",
        "max_running_requests",
        "context_length",
    }
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# These groups match the v1 launcher identity.  Richer identity fields are
# retained verbatim, so changing any source/model/GPU/dependency/config detail
# changes the digest even when it is not named here.
REQUIRED_IDENTITY_KEYS = (
    "sglang",
    "graphbudget_sha256",
    "model",
    "gpu",
    "dependencies",
    "python",
    "cuda",
    "dtype",
    "attention_backend",
    "config",
)


class GraphBudgetError(ValueError):
    """Base error for malformed, stale, or unusable plan data."""


class InvalidPlanError(GraphBudgetError):
    """The plan/identity/budget shape is invalid."""


class StalePlanError(GraphBudgetError):
    """The frozen identity or budget does not match the active runtime."""


class NoFeasiblePlanError(GraphBudgetError):
    """All bounded candidates were rejected; reasons remain attached."""

    def __init__(self, message: str, rejections: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(message)
        self.rejections = tuple(deepcopy(dict(item)) for item in rejections)


class MissingMeasurementError(NoFeasiblePlanError):
    """Required real measurement fields were absent."""


def _jsonable(value: Any, path: str = "identity") -> Any:
    """Copy JSON data while rejecting unstable/non-serialisable values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidPlanError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise InvalidPlanError(f"{path} contains unsupported value type {type(value).__name__}")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _jsonable(value),
        # Match ``identity.digest`` used by the runtime adapter.
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidPlanError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256_text(value: Any, name: str) -> str:
    value = _nonempty_text(value, name).lower()
    if HEX_SHA256.fullmatch(value) is None:
        raise InvalidPlanError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidPlanError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidPlanError(f"{name} must be a non-negative integer")
    return value


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidPlanError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (not positive and result < 0):
        qualifier = "greater than zero" if positive else "non-negative"
        raise InvalidPlanError(f"{name} must be {qualifier}")
    return result


def _validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or not identity:
        raise InvalidPlanError("identity must be a non-empty object")
    missing = [key for key in REQUIRED_IDENTITY_KEYS if key not in identity]
    if missing:
        raise InvalidPlanError("identity missing required keys: " + ", ".join(missing))
    normalized = _jsonable(identity)
    if not isinstance(normalized, dict):  # defensive; Mapping above guarantees it
        raise InvalidPlanError("identity must be an object")
    for group in ("sglang", "model", "gpu", "config"):
        if not isinstance(normalized[group], Mapping):
            raise InvalidPlanError(f"identity.{group} must be an object")
    for group, key in (("sglang", "base_commit"), ("model", "revision"), ("gpu", "name")):
        if not normalized[group].get(key):
            raise InvalidPlanError(f"identity.{group}.{key} is required")
    return normalized


def fingerprint(identity: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a validated canonical runtime identity object."""

    normalized = _validate_identity(identity)
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def _identity_record(value: Any) -> tuple[dict[str, Any] | None, str]:
    """Resolve an identity object, a ``{identity, sha256}`` record, or digest."""

    if isinstance(value, str):
        return None, _sha256_text(value, "fingerprint")
    if not isinstance(value, Mapping):
        raise InvalidPlanError("fingerprint must be an identity object, digest, or record")
    if "identity" in value:
        identity = _validate_identity(value["identity"])
        digest = value.get("sha256", value.get("digest"))
        digest = _sha256_text(digest, "fingerprint.sha256")
        expected = fingerprint(identity)
        if digest != expected:
            raise InvalidPlanError("fingerprint sha256 does not match canonical identity")
        return identity, digest
    identity = _validate_identity(value)
    return identity, fingerprint(identity)


def validate_fingerprint(expected: Any, actual: Any) -> str:
    """Require exact digest equality, including the full identity object when present."""

    expected_identity, expected_digest = _identity_record(expected)
    actual_identity, actual_digest = _identity_record(actual)
    if expected_digest != actual_digest:
        raise StalePlanError(
            f"stale runtime fingerprint: expected {expected_digest}, got {actual_digest}"
        )
    if expected_identity is not None and actual_identity is not None and expected_identity != actual_identity:
        raise StalePlanError("stale runtime fingerprint: canonical identities differ")
    return actual_digest


def _validate_budget(budget: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(budget, Mapping):
        raise InvalidPlanError("budget must be an object")
    required = ("allowed_device_bytes", "min_kv_tokens", "max_startup_seconds")
    missing = [key for key in required if key not in budget]
    if missing:
        raise InvalidPlanError("budget missing required keys: " + ", ".join(missing))
    unknown = sorted(set(budget) - set(required))
    if unknown:
        raise InvalidPlanError("unsupported budget keys: " + ", ".join(unknown))
    allowed = _positive_int(budget["allowed_device_bytes"], "allowed_device_bytes")
    minimum = _nonnegative_int(budget["min_kv_tokens"], "min_kv_tokens")
    startup = _finite(budget["max_startup_seconds"], "max_startup_seconds")
    return {
        "allowed_device_bytes": allowed,
        "min_kv_tokens": minimum,
        "max_startup_seconds": startup,
    }


def _server_arg_reasons(server_args: Any) -> list[str]:
    reasons: list[str] = []
    if not isinstance(server_args, Mapping):
        return ["server_args is missing or not an object"]
    keys = set(server_args)
    unknown = sorted(keys - ALLOWED_SERVER_ARGS)
    missing = sorted(ALLOWED_SERVER_ARGS - keys)
    if unknown:
        reasons.append("unsupported server_args: " + ", ".join(unknown))
    if missing:
        reasons.append("missing required server_args: " + ", ".join(missing))
    if unknown or missing:
        return reasons
    try:
        fraction = _finite(server_args["mem_fraction_static"], "server_args.mem_fraction_static", positive=True)
        if fraction > 1:
            reasons.append("server_args.mem_fraction_static must be <= 1")
    except InvalidPlanError as exc:
        reasons.append(str(exc))
    for key in ("max_total_tokens", "max_running_requests", "context_length"):
        try:
            _positive_int(server_args[key], f"server_args.{key}")
        except InvalidPlanError as exc:
            reasons.append(str(exc))
    return reasons


def _candidate_reasons(
    candidate: Mapping[str, Any],
    budget: Mapping[str, Any],
    *,
    allow_synthetic: bool,
    expected_config: Mapping[str, Any] | None = None,
    expected_fingerprint: str | None = None,
) -> list[str]:
    reasons: list[str] = []
    required = (
        "id",
        "capture_sizes",
        "server_args",
        "ready",
        "runtime_peak_device_used_bytes",
        "calibration_seconds",
        "status",
        "reason",
        "evidence",
    )
    reasons.extend(f"missing {key}" for key in required if key not in candidate)
    candidate_id = candidate.get("id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise InvalidPlanError("candidate id must be a non-empty string")
    sizes = candidate.get("capture_sizes")
    if not isinstance(sizes, (list, tuple)) or not sizes:
        reasons.append("capture_sizes must be a non-empty list")
    else:
        valid_sizes = all(
            isinstance(size, int) and not isinstance(size, bool) and size > 0
            for size in sizes
        )
        if not valid_sizes:
            reasons.append("capture_sizes must contain positive integers")
        else:
            if list(sizes) != sorted(sizes):
                reasons.append("capture_sizes must be sorted ascending")
            if len(set(sizes)) != len(sizes):
                reasons.append("capture_sizes contains duplicates")
    reasons.extend(_server_arg_reasons(candidate.get("server_args")))

    if expected_config is not None:
        server_args = candidate.get("server_args")
        if isinstance(server_args, Mapping):
            for key in sorted(ALLOWED_SERVER_ARGS):
                if key not in expected_config:
                    reasons.append(f"identity.config missing {key}")
                elif server_args.get(key) != expected_config[key]:
                    reasons.append(
                        f"server_args.{key} differs from identity.config.{key}"
                    )
        if isinstance(sizes, (list, tuple)) and sizes and all(
            isinstance(size, int) and not isinstance(size, bool) and size > 0
            for size in sizes
        ):
            max_running = expected_config.get("max_running_requests")
            if isinstance(max_running, int) and not isinstance(max_running, bool):
                if max(sizes) > max_running:
                    reasons.append(
                        f"capture ceiling {max(sizes)} > configured max_running_requests {max_running}"
                    )
            else:
                reasons.append("identity.config.max_running_requests is missing or invalid")

    ready = candidate.get("ready")
    if not isinstance(ready, Mapping):
        reasons.append("ready is missing or not an object")
        ready = {}
    ready_required = (
        "device_total_bytes",
        "device_used_bytes",
        "device_free_bytes",
        "kv_tokens",
        "startup_seconds",
        "capture_seconds",
    )
    reasons.extend(f"missing ready.{key}" for key in ready_required if key not in ready)
    ready_used: int | None = None
    ready_total: int | None = None
    if all(key in ready for key in ("device_total_bytes", "device_used_bytes", "device_free_bytes")):
        try:
            total = _positive_int(ready["device_total_bytes"], "ready.device_total_bytes")
            used = _nonnegative_int(ready["device_used_bytes"], "ready.device_used_bytes")
            free = _nonnegative_int(ready["device_free_bytes"], "ready.device_free_bytes")
            ready_total, ready_used = total, used
            if used + free != total:
                reasons.append("ready device total != used + free from one sample")
            if used > budget["allowed_device_bytes"]:
                reasons.append(
                    f"ready device used {used} > budget {budget['allowed_device_bytes']}"
                )
        except InvalidPlanError as exc:
            reasons.append(str(exc))
    if "kv_tokens" in ready:
        try:
            kv = _nonnegative_int(ready["kv_tokens"], "ready.kv_tokens")
            if kv < budget["min_kv_tokens"]:
                reasons.append(f"KV capacity {kv} < required {budget['min_kv_tokens']}")
        except InvalidPlanError as exc:
            reasons.append(str(exc))
    for key in ("startup_seconds", "capture_seconds"):
        if key in ready:
            try:
                value = _finite(ready[key], f"ready.{key}")
                if key == "startup_seconds" and value > budget["max_startup_seconds"]:
                    reasons.append(
                        f"startup {value:g}s > budget {budget['max_startup_seconds']:g}s"
                    )
            except InvalidPlanError as exc:
                reasons.append(str(exc))
    if (
        "startup_seconds" in ready
        and "capture_seconds" in ready
        and isinstance(ready.get("startup_seconds"), (int, float))
        and isinstance(ready.get("capture_seconds"), (int, float))
        and math.isfinite(float(ready["startup_seconds"]))
        and math.isfinite(float(ready["capture_seconds"]))
        and float(ready["capture_seconds"]) > float(ready["startup_seconds"])
    ):
        reasons.append("ready.capture_seconds exceeds ready.startup_seconds")

    if "runtime_peak_device_used_bytes" in candidate:
        try:
            runtime_peak = _nonnegative_int(
                candidate["runtime_peak_device_used_bytes"],
                "runtime_peak_device_used_bytes",
            )
            if runtime_peak > budget["allowed_device_bytes"]:
                reasons.append(
                    f"runtime peak used {runtime_peak} > budget {budget['allowed_device_bytes']}"
                )
            if ready_used is not None and runtime_peak < ready_used:
                reasons.append("runtime peak used is below ready device used")
            if ready_total is not None and runtime_peak > ready_total:
                reasons.append("runtime peak used exceeds ready device total")
        except InvalidPlanError as exc:
            reasons.append(str(exc))
    if "calibration_seconds" in candidate:
        try:
            _finite(candidate["calibration_seconds"], "calibration_seconds", positive=True)
        except InvalidPlanError as exc:
            reasons.append(str(exc))

    status = candidate.get("status")
    if status not in VALID_STATUS:
        reasons.append(f"unsupported status {status!r}")
    elif status != "ok":
        detail = candidate.get("reason") or status
        reasons.append(f"measurement status={status}: {detail}")
    if not isinstance(candidate.get("reason"), str):
        reasons.append("reason must be a string")

    evidence = candidate.get("evidence")
    if not isinstance(evidence, Mapping):
        reasons.append("evidence is missing or not an object")
    else:
        provenance = evidence.get("provenance")
        if provenance not in VALID_PROVENANCE:
            reasons.append("evidence.provenance must be measured or synthetic_cpu_fixture")
        elif provenance == SYNTHETIC_CPU_PROVENANCE and not allow_synthetic:
            reasons.append("synthetic CPU fixture; real measured provenance required")
        actual_fingerprint = evidence.get("actual_fingerprint", evidence.get("fingerprint"))
        if expected_fingerprint is not None:
            if actual_fingerprint != expected_fingerprint:
                reasons.append("evidence actual_fingerprint does not match plan identity")
        elif not actual_fingerprint:
            reasons.append("evidence missing actual_fingerprint")
        actual_sizes = evidence.get("actual_capture_sizes")
        requested_sizes = evidence.get("requested_capture_sizes")
        if actual_sizes is None:
            reasons.append("evidence missing actual_capture_sizes")
        elif not isinstance(actual_sizes, (list, tuple)) or tuple(actual_sizes) != tuple(sizes):
            reasons.append("evidence actual_capture_sizes differ from candidate capture_sizes")
        if requested_sizes is None:
            reasons.append("evidence missing requested_capture_sizes")
        elif not isinstance(requested_sizes, (list, tuple)) or tuple(requested_sizes) != tuple(sizes):
            reasons.append("evidence requested_capture_sizes differ from candidate capture_sizes")
    return reasons


def _normalize_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise InvalidPlanError("each candidate must be an object")
    normalized = deepcopy(dict(candidate))
    if "id" not in normalized:
        raise InvalidPlanError("candidate id is required")
    capture_sizes = normalized.get("capture_sizes", ())
    normalized["capture_sizes"] = (
        list(capture_sizes) if isinstance(capture_sizes, (list, tuple)) else capture_sizes
    )
    if isinstance(normalized.get("server_args"), Mapping):
        normalized["server_args"] = dict(normalized["server_args"])
    if isinstance(normalized.get("ready"), Mapping):
        normalized["ready"] = dict(normalized["ready"])
    if isinstance(normalized.get("evidence"), Mapping):
        normalized["evidence"] = dict(normalized["evidence"])
    return normalized


def _objective(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    ready = candidate["ready"]
    return (
        float(ready["device_used_bytes"]),
        float(candidate["runtime_peak_device_used_bytes"]),
        float(ready["startup_seconds"]),
        float(candidate["calibration_seconds"]),
        -float(ready["kv_tokens"]),
    )


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    lhs, rhs = _objective(left), _objective(right)
    return all(a <= b for a, b in zip(lhs, rhs)) and any(a < b for a, b in zip(lhs, rhs))


def _pareto(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        candidate["id"]
        for candidate in candidates
        if not any(
            other["id"] != candidate["id"] and _dominates(other, candidate)
            for other in candidates
        )
    ]


def select_plan(
    identity: Mapping[str, Any],
    budget: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int = 5,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Select one measured whole-config candidate and retain all reasons."""

    normalized_identity = _validate_identity(identity)
    digest = fingerprint(normalized_identity)
    normalized_budget = _validate_budget(budget)
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not (1 <= max_candidates <= 5):
        raise InvalidPlanError("max_candidates must be between 1 and 5")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise InvalidPlanError("candidates must be a sequence")
    if len(candidates) == 0:
        raise NoFeasiblePlanError("no capture candidates were supplied")
    if len(candidates) > max_candidates:
        raise InvalidPlanError(f"at most {max_candidates} whole-config candidates are allowed")

    all_candidates: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in candidates:
        candidate = _normalize_candidate(raw)
        candidate_id = candidate["id"]
        if candidate_id in seen:
            raise InvalidPlanError(f"duplicate candidate id {candidate_id!r}")
        seen.add(candidate_id)
        all_candidates.append(candidate)
        reasons = _candidate_reasons(
            candidate,
            normalized_budget,
            allow_synthetic=allow_synthetic,
            expected_config=normalized_identity["config"],
            expected_fingerprint=digest,
        )
        if reasons:
            rejections.append({"id": candidate_id, "reasons": reasons})
        else:
            feasible.append(candidate)
    if not feasible:
        error_cls = MissingMeasurementError if rejections else NoFeasiblePlanError
        raise error_cls("no feasible capture plan; all candidates were rejected", rejections)

    pareto_ids = _pareto(feasible)
    pareto_candidates = [candidate for candidate in feasible if candidate["id"] in pareto_ids]
    selected = min(
        pareto_candidates,
        key=lambda candidate: (
            float(candidate["calibration_seconds"]),
            _objective(candidate),
            candidate["id"],
        ),
    )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "fingerprint": {"identity": normalized_identity, "sha256": digest},
        "budget": normalized_budget,
        "selected": deepcopy(selected),
        "candidates": deepcopy(all_candidates),
        "feasible_ids": [candidate["id"] for candidate in feasible],
        "pareto_ids": pareto_ids,
        "rejections": rejections,
    }


def validate_plan(
    plan: Mapping[str, Any],
    *,
    actual_fingerprint: Any = None,
    actual_budget: Mapping[str, Any] | None = None,
    require_measured: bool = True,
) -> dict[str, Any]:
    """Validate a frozen dict plan and optionally bind it to current state."""

    if not isinstance(plan, Mapping):
        raise InvalidPlanError("plan must be an object")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise InvalidPlanError("unsupported plan schema_version")
    fp = plan.get("fingerprint")
    if not isinstance(fp, Mapping):
        raise InvalidPlanError("plan.fingerprint must be an object")
    identity = _validate_identity(fp.get("identity"))
    digest = fp.get("sha256")
    if digest != fingerprint(identity):
        raise InvalidPlanError("plan fingerprint sha256 does not match identity")
    if actual_fingerprint is not None:
        validate_fingerprint(fp, actual_fingerprint)
    budget = _validate_budget(plan.get("budget", {}))
    if actual_budget is not None and budget != _validate_budget(actual_budget):
        raise StalePlanError("stale plan budget: current constraints differ")
    candidates = plan.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise InvalidPlanError("plan.candidates must be a sequence")
    if not candidates or len(candidates) > 5:
        raise InvalidPlanError("plan must contain one to five candidates")
    normalized = [_normalize_candidate(candidate) for candidate in candidates]
    ids = [candidate.get("id") for candidate in normalized]
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in ids):
        raise InvalidPlanError("plan candidate IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise InvalidPlanError("plan candidate IDs must be unique")
    feasible_ids: list[str] = []
    rejection_ids: set[str] = set()
    for candidate in normalized:
        reasons = _candidate_reasons(
            candidate,
            budget,
            allow_synthetic=not require_measured,
            expected_config=identity["config"],
            expected_fingerprint=digest,
        )
        if reasons:
            rejection_ids.add(candidate["id"])
        else:
            feasible_ids.append(candidate["id"])
    if list(plan.get("feasible_ids", ())) != feasible_ids:
        raise InvalidPlanError("plan.feasible_ids do not match candidate evidence")
    pareto_ids = list(plan.get("pareto_ids", ()))
    if not pareto_ids or not set(pareto_ids).issubset(set(feasible_ids)):
        raise InvalidPlanError("plan.pareto_ids must be a non-empty subset of feasible_ids")
    expected_pareto_ids = _pareto(
        [candidate for candidate in normalized if candidate["id"] in set(feasible_ids)]
    )
    if pareto_ids != expected_pareto_ids:
        raise InvalidPlanError("plan.pareto_ids do not match candidate objectives")
    selected = plan.get("selected")
    if not isinstance(selected, Mapping) or selected.get("id") not in set(feasible_ids):
        raise InvalidPlanError("plan.selected must be one feasible candidate")
    selected_id = selected["id"]
    selected_candidate = next(candidate for candidate in normalized if candidate["id"] == selected_id)
    if dict(selected) != selected_candidate:
        raise InvalidPlanError("plan.selected differs from its candidate record")
    if selected.get("id") != min(
        [candidate for candidate in normalized if candidate["id"] in set(pareto_ids)],
        key=lambda candidate: (
            float(candidate["calibration_seconds"]),
            _objective(candidate),
            candidate["id"],
        ),
    )["id"]:
        raise InvalidPlanError("plan.selected is not the deterministic best Pareto candidate")
    rejection_records = plan.get("rejections", ())
    if not isinstance(rejection_records, Sequence) or isinstance(rejection_records, (str, bytes)):
        raise InvalidPlanError("plan.rejections must be a sequence")
    rejection_record_ids = {record.get("id") for record in rejection_records if isinstance(record, Mapping)}
    if rejection_record_ids != rejection_ids:
        raise InvalidPlanError("plan.rejections do not match candidate evidence")
    for record in rejection_records:
        if not isinstance(record, Mapping) or not isinstance(record.get("reasons"), list) or not record["reasons"]:
            raise InvalidPlanError("each rejection must retain a non-empty reasons list")
        candidate = next(candidate for candidate in normalized if candidate["id"] == record["id"])
        expected_reasons = _candidate_reasons(
            candidate,
            budget,
            allow_synthetic=not require_measured,
            expected_config=identity["config"],
            expected_fingerprint=digest,
        )
        if list(record["reasons"]) != expected_reasons:
            raise InvalidPlanError(
                f"rejection reasons for {record['id']!r} do not match candidate evidence"
            )
    return deepcopy(dict(plan))


def load_plan(
    value: str | Mapping[str, Any],
    *,
    actual_fingerprint: Any = None,
    actual_budget: Mapping[str, Any] | None = None,
    require_measured: bool = True,
) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise InvalidPlanError("plan JSON is invalid") from exc
    return validate_plan(
        value,
        actual_fingerprint=actual_fingerprint,
        actual_budget=actual_budget,
        require_measured=require_measured,
    )


def dump_plan(plan: Mapping[str, Any], *, indent: int = 2) -> str:
    return json.dumps(validate_plan(plan), ensure_ascii=False, sort_keys=True, indent=indent)


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "MEASURED_PROVENANCE",
    "SYNTHETIC_CPU_PROVENANCE",
    "ALLOWED_SERVER_ARGS",
    "REQUIRED_IDENTITY_KEYS",
    "GraphBudgetError",
    "InvalidPlanError",
    "StalePlanError",
    "NoFeasiblePlanError",
    "MissingMeasurementError",
    "fingerprint",
    "validate_fingerprint",
    "select_plan",
    "validate_plan",
    "load_plan",
    "dump_plan",
]
