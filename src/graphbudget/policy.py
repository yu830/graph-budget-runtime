"""Frozen CPU-only graph/eager cost policy.

The public policy API is dict-first.  It performs one exact lookup on
``(batch_size, padded_size, context_length)`` and never reads sequence tensors,
launches CUDA work, or expands upstream graph eligibility.  Call
``compile_policy`` once at startup; the compiled form is immutable and the
decode-step decision is an O(1) tuple-key lookup.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any

from .plan import MEASURED_PROVENANCE


POLICY_SCHEMA_VERSION = 1
PATH_NATIVE = "native"
PATH_GRAPH = "graph"
PATH_EAGER = "eager"
VALID_PROVENANCE = frozenset({MEASURED_PROVENANCE, "synthetic_cpu_fixture"})
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
MIN_USEFUL_SAMPLES = 3
COMPILED_MARKER = "__graphbudget_compiled__"


class PolicyError(ValueError):
    """Base error for malformed policy input."""


class InvalidPolicyError(PolicyError):
    """A policy cannot be trusted as a frozen lookup table."""


class ForcedEagerError(PolicyError):
    """A test-only eager override was requested without an explicit marker."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidPolicyError(f"{name} must be a positive integer")
    return value


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidPolicyError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise InvalidPolicyError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise InvalidPolicyError(f"{name} must be a finite number")
    if (positive and number <= 0) or (not positive and number < 0):
        qualifier = "greater than zero" if positive else "non-negative"
        raise InvalidPolicyError(f"{name} must be {qualifier}")
    return number


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_HEX.fullmatch(value.lower()) is None:
        raise InvalidPolicyError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return value.lower()


def _key(entry_or_context: Mapping[str, Any]) -> tuple[int, int, int]:
    try:
        return (
            _positive_int(entry_or_context["batch_size"], "batch_size"),
            _positive_int(entry_or_context["padded_size"], "padded_size"),
            _positive_int(entry_or_context["context_length"], "context_length"),
        )
    except KeyError as exc:
        raise InvalidPolicyError(f"missing context key {exc.args[0]}") from exc


def _context(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return _key(value)
    except InvalidPolicyError:
        return None


def _entry_reasons(
    entry: Mapping[str, Any],
    capture_sizes: tuple[int, ...] | None = None,
) -> list[str]:
    reasons: list[str] = []
    required = (
        "batch_size",
        "padded_size",
        "context_length",
        "graph_ns",
        "eager_ns",
        "noise_ns",
        "samples",
        "provenance",
    )
    reasons.extend(f"missing {key}" for key in required if key not in entry)
    try:
        batch_size, padded_size, _ = _key(entry)
        if capture_sizes is not None:
            ceiling = next((size for size in capture_sizes if size >= batch_size), None)
            if ceiling is None:
                reasons.append("batch_size exceeds capture ceiling")
            elif padded_size != ceiling:
                reasons.append(
                    f"padded_size {padded_size} is not capture ceiling {ceiling} for batch_size {batch_size}"
                )
    except InvalidPolicyError as exc:
        reasons.append(str(exc))
    for name in ("graph_ns", "eager_ns"):
        if name in entry:
            try:
                _finite(entry[name], name, positive=True)
            except InvalidPolicyError as exc:
                reasons.append(str(exc))
    if "noise_ns" in entry:
        try:
            _finite(entry["noise_ns"], "noise_ns")
        except InvalidPolicyError as exc:
            reasons.append(str(exc))
    if "samples" in entry:
        try:
            _positive_int(entry["samples"], "samples")
        except InvalidPolicyError as exc:
            reasons.append(str(exc))
    provenance = entry.get("provenance")
    if not isinstance(provenance, str) or provenance not in VALID_PROVENANCE:
        reasons.append("provenance must be measured or synthetic_cpu_fixture")
    return reasons


def validate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate shape, duplicate keys, and fingerprint without enabling rows."""

    if not isinstance(policy, Mapping):
        raise InvalidPolicyError("policy must be an object")
    if policy.get("schema_version", POLICY_SCHEMA_VERSION) != POLICY_SCHEMA_VERSION:
        raise InvalidPolicyError("unsupported policy schema_version")
    digest = policy.get("fingerprint")
    digest = _sha256(digest, "policy.fingerprint")
    capture_sizes = policy.get("capture_sizes")
    if not isinstance(capture_sizes, (list, tuple)) or not capture_sizes:
        raise InvalidPolicyError("policy.capture_sizes must be a non-empty list")
    valid_sizes = all(
        isinstance(size, int) and not isinstance(size, bool) and size > 0
        for size in capture_sizes
    )
    if not valid_sizes:
        raise InvalidPolicyError("policy.capture_sizes must contain positive integers")
    if list(capture_sizes) != sorted(capture_sizes):
        raise InvalidPolicyError("policy.capture_sizes must be sorted ascending")
    if len(set(capture_sizes)) != len(capture_sizes):
        raise InvalidPolicyError("policy.capture_sizes contains duplicates")
    entries = policy.get("entries")
    if not isinstance(entries, (list, tuple)):
        raise InvalidPolicyError("policy.entries must be a list")
    enabled = policy.get("enabled", True)
    if not isinstance(enabled, bool):
        raise InvalidPolicyError("policy.enabled must be boolean")
    seen: set[tuple[int, int, int]] = set()
    normalized_entries: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise InvalidPolicyError("each policy entry must be an object")
        entry = deepcopy(dict(raw))
        reasons = _entry_reasons(entry, tuple(capture_sizes))
        strict_padding_errors = [
            reason
            for reason in reasons
            if reason.startswith("padded_size ") or reason == "batch_size exceeds capture ceiling"
        ]
        if strict_padding_errors:
            raise InvalidPolicyError("invalid policy capture bucket: " + "; ".join(strict_padding_errors))
        if reasons:
            # Retain malformed rows for diagnostics; decide_policy will return
            # native for them instead of turning runtime startup into eager.
            key = _context(entry)
            if key is not None and key in seen:
                raise InvalidPolicyError("policy has duplicate context keys")
            if key is not None:
                seen.add(key)
            normalized_entries.append(entry)
            continue
        key = _key(entry)
        if key in seen:
            raise InvalidPolicyError("policy has duplicate context keys")
        seen.add(key)
        normalized_entries.append(entry)
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "fingerprint": digest,
        "capture_sizes": list(capture_sizes),
        "entries": normalized_entries,
        "enabled": enabled,
    }


def _compiled_fallback(
    reason: str,
    *,
    fingerprint: str = "",
    capture_sizes: tuple[int, ...] = (),
    enabled: bool = False,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            COMPILED_MARKER: True,
            "schema_version": POLICY_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "capture_sizes": capture_sizes,
            "entries": MappingProxyType({}),
            "enabled": enabled,
            "status": "native_fallback",
            "fallback_reason": reason,
        }
    )


def compile_policy(
    policy: Mapping[str, Any] | None,
    actual_fingerprint: str | None,
) -> Mapping[str, Any]:
    """Validate once at startup and build an immutable tuple-key table.

    Invalid/stale profiles intentionally compile to a native-fallback object
    instead of being retried or scanned on every decode step.  A row with too
    few samples remains in the table for diagnostics but is marked for a
    native decision; one bad row does not make unrelated exact rows unusable.
    """

    if policy is None:
        return _compiled_fallback("policy_missing")
    try:
        table = validate_policy(policy)
    except InvalidPolicyError as exc:
        return _compiled_fallback(f"invalid_policy: {exc}")
    if not table["enabled"]:
        return _compiled_fallback(
            "policy_disabled",
            fingerprint=table["fingerprint"],
            capture_sizes=tuple(table["capture_sizes"]),
            enabled=False,
        )
    try:
        actual = _sha256(actual_fingerprint, "actual_fingerprint")
    except InvalidPolicyError as exc:
        return _compiled_fallback(f"profile_fingerprint_missing: {exc}")
    if actual != table["fingerprint"]:
        return _compiled_fallback(
            "profile_fingerprint_mismatch",
            fingerprint=table["fingerprint"],
            capture_sizes=tuple(table["capture_sizes"]),
            enabled=table["enabled"],
        )
    capture_sizes = tuple(table["capture_sizes"])
    entries: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    for raw in table["entries"]:
        key = _context(raw)
        if key is None:
            continue
        row = dict(raw)
        reasons = _entry_reasons(row, capture_sizes)
        if row.get("provenance") != MEASURED_PROVENANCE:
            row["__native_reason"] = "profile_not_measured"
        elif row.get("graph_ns") is None or row.get("eager_ns") is None:
            row["__native_reason"] = "profile_cost_missing"
        elif row.get("noise_ns") is None:
            row["__native_reason"] = "profile_noise_missing"
        elif row.get("samples") is None:
            row["__native_reason"] = "profile_samples_missing"
        elif reasons:
            row["__native_reason"] = "profile_entry_invalid"
        elif row.get("samples", 0) < MIN_USEFUL_SAMPLES:
            row["__native_reason"] = "profile_samples_insufficient"
        entries[key] = MappingProxyType(row)
    return MappingProxyType(
        {
            COMPILED_MARKER: True,
            "schema_version": POLICY_SCHEMA_VERSION,
            "fingerprint": table["fingerprint"],
            "capture_sizes": capture_sizes,
            "entries": MappingProxyType(entries),
            "enabled": True,
            "status": "loaded",
            "fallback_reason": "",
        }
    )


def _native(eligible: bool, reason: str, *, forced_test: bool = False) -> dict[str, Any]:
    return {
        "eligible": bool(eligible),
        "selected_path": PATH_NATIVE,
        "reason": reason,
        "forced_test": forced_test,
    }


def decide_policy(
    policy: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
    upstream_eligible: bool,
    actual_fingerprint: str | None,
    *,
    force_eager: bool = False,
    test_only_force_eager: bool = False,
) -> dict[str, Any]:
    """Return a path selection while preserving upstream safety eligibility."""

    if not isinstance(upstream_eligible, bool):
        raise PolicyError("upstream_eligible must be boolean")
    if force_eager and not test_only_force_eager:
        raise ForcedEagerError(
            "force_eager is test-only; pass test_only_force_eager=True explicitly"
        )
    if not upstream_eligible:
        return _native(False, "upstream_ineligible")
    if policy is None:
        if force_eager:
            return {
                "eligible": True,
                "selected_path": PATH_EAGER,
                "reason": "forced_eager_test",
                "forced_test": True,
            }
        return _native(True, "policy_missing")
    compiled = (
        policy
        if isinstance(policy, Mapping) and policy.get(COMPILED_MARKER) is True
        else compile_policy(policy, actual_fingerprint)
    )
    fallback_reason = compiled.get("fallback_reason", "")
    if fallback_reason == "policy_disabled" or (
        compiled.get("status") == "native_fallback" and not compiled.get("enabled", False)
        and not fallback_reason.startswith("profile_")
        and not fallback_reason.startswith("invalid_policy")
    ):
        return _native(True, "policy_disabled")
    if force_eager:
        return {
            "eligible": True,
            "selected_path": PATH_EAGER,
            "reason": "forced_eager_test",
            "forced_test": True,
        }
    if compiled.get("status") != "loaded":
        if fallback_reason.startswith("invalid_policy"):
            fallback_reason = "invalid_policy"
        elif fallback_reason.startswith("profile_fingerprint_missing"):
            fallback_reason = "profile_fingerprint_missing"
        elif fallback_reason.startswith("profile_fingerprint_mismatch"):
            fallback_reason = "profile_fingerprint_mismatch"
        return _native(True, fallback_reason or "native_fallback")
    try:
        actual = _sha256(actual_fingerprint, "actual_fingerprint")
    except InvalidPolicyError:
        return _native(True, "profile_fingerprint_missing")
    if actual != compiled["fingerprint"]:
        return _native(True, "profile_fingerprint_mismatch")
    context_key = _context(context)
    if context_key is None:
        return _native(True, "missing_or_heterogeneous_context")
    entry = compiled["entries"].get(context_key)
    if entry is None:
        return _native(True, "profile_entry_missing")
    if entry.get("__native_reason"):
        return _native(True, entry["__native_reason"])
    if entry["eager_ns"] + entry["noise_ns"] < entry["graph_ns"]:
        return {
            "eligible": True,
            "selected_path": PATH_EAGER,
            "reason": "eager_faster_beyond_noise",
            "forced_test": False,
        }
    return _native(True, "graph_not_slower_beyond_noise")


select_path = decide_policy
evaluate_policy = decide_policy


def load_policy(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise InvalidPolicyError("policy JSON is invalid") from exc
    return validate_policy(value)


def dump_policy(policy: Mapping[str, Any], *, indent: int = 2) -> str:
    return json.dumps(validate_policy(policy), ensure_ascii=False, sort_keys=True, indent=indent)


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "PATH_NATIVE",
    "PATH_GRAPH",
    "PATH_EAGER",
    "PolicyError",
    "InvalidPolicyError",
    "ForcedEagerError",
    "validate_policy",
    "compile_policy",
    "decide_policy",
    "select_path",
    "evaluate_policy",
    "load_policy",
    "dump_policy",
]
