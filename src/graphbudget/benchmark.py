"""A small, dependency-free benchmark client for a real SGLang server.

This module deliberately talks to the native SGLang ``POST /generate`` API
using only Python's standard library.  The request body always contains fixed
``input_ids`` and the runner owns the sampling fields that matter for a
repeatable validation run: ``temperature=0.0`` and ``ignore_eos=true``.  A
workload is therefore a list of *planned* requests, not a closed-loop stream
of requests that stops when the previous request finishes.

The native v0.5.9 endpoint emits server-sent events (SSE):
``data: {"output_ids": ..., "meta_info": ...}\n\n`` followed by
``data: [DONE]\n\n``.  We retain every raw event and its receive timestamp.  The
request result keeps four distinct times, relative to the benchmark start:
``planned_at_s``, ``actual_send_at_s``, ``first_token_at_s`` and
``completion_at_s``.  A stable workload request ID is kept separately from a
run-scoped client request ID and the server's ``meta_info.id``.

The cache reset option is explicit.  It calls SGLang's ``/flush_cache`` once
before a run and records the response; it only clears the server's radix/KV
prefix cache when the server accepts that request.  It does not recreate the
process, graph pool, allocator, weights, or CUDA state, so it must not be
described as a full fresh-process reset.

Metrics are intentionally conservative:

* effective output tokens/s counts output IDs only from successful requests;
* request TPOT is ``(completion - first_token) / (tokens - 1)`` when at least
  two output tokens are available;
* token ITL is returned only when every observed token event contributes one
  token.  A multi-token SSE chunk makes ITL unavailable rather than turning a
  chunk interval into a fabricated per-token interval;
* P99 is omitted as a usable statistic below ``P99_MIN_SAMPLES`` (100).
* exact validation requires every response event to carry an integer
  ``meta_info.completion_tokens``; contradictory counts, a mismatched
  ``meta_info.id``, or a finish reason other than ``{"type": "length",
  "length": max_new_tokens}`` fail the request instead of being inferred.

The module also exposes exact-token comparison and paired-round summaries.  A
paired summary reports an ``INSUFFICIENT_ROUNDS`` status below five rounds and
does not turn a small sample into an adoption claim.  The JSON workloads in
``examples/workloads`` use tiny literal IDs as safe request-shape examples;
they explicitly remain unvalidated against a particular model tokenizer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _datetime
import json
import math
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
P99_MIN_SAMPLES = 100
MIN_PAIRED_ROUNDS = 5
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_ENDPOINT = "/generate"


class BenchmarkError(RuntimeError):
    """Raised for invalid benchmark inputs or an unusable client setup."""


class BenchmarkTimeout(TimeoutError):
    """Raised when one HTTP stream exceeds its total request timeout."""


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _json_load(source: Any) -> Any:
    """Load a mapping from a path or return an already-loaded JSON value."""

    if isinstance(source, Mapping):
        return source
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise TypeError(f"expected a path or mapping, got {type(source).__name__}")


def _as_int(value: Any, field_name: str) -> int:
    # bool is an int subclass but is never a valid token/count here.
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkError(f"{field_name} must be an integer")
    return value


def _as_nonnegative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise BenchmarkError(f"{field_name} must be finite and >= 0")
    return result


@dataclass(frozen=True)
class WorkloadRequest:
    """One fixed-input request in a workload schedule."""

    request_id: str
    input_ids: tuple[int, ...]
    max_new_tokens: int
    planned_at_s: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "input_ids": list(self.input_ids),
            "max_new_tokens": self.max_new_tokens,
            "planned_at_s": self.planned_at_s,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Workload:
    """Validated benchmark workload loaded from JSON."""

    workload_id: str
    purpose: str
    requests: tuple[WorkloadRequest, ...]
    description: str = ""
    token_ids_note: str = ""
    token_ids_validated_against_model: bool = False
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "graphbudget_workload",
            "workload_id": self.workload_id,
            "purpose": self.purpose,
            "description": self.description,
            "token_ids_note": self.token_ids_note,
            "token_ids_validated_against_model": self.token_ids_validated_against_model,
            "requests": [request.to_dict() for request in self.requests],
        }


def load_workload(source: Any) -> Workload:
    """Load and validate a workload JSON path or mapping.

    The accepted request time key is ``planned_at_s``.  ``at_s`` is accepted
    as a short alias to make hand-authored schedules less error prone, but the
    canonical serialized representation always uses ``planned_at_s``.
    """

    raw = _json_load(source)
    if not isinstance(raw, Mapping):
        raise BenchmarkError("workload root must be a JSON object")
    schema_version = _as_int(raw.get("schema_version", SCHEMA_VERSION), "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise BenchmarkError(
            f"unsupported workload schema_version={schema_version}; expected {SCHEMA_VERSION}"
        )
    workload_id = str(raw.get("workload_id", raw.get("name", "workload"))).strip()
    if not workload_id:
        raise BenchmarkError("workload_id must be non-empty")
    purpose = str(raw.get("purpose", "unspecified")).strip() or "unspecified"
    requests_raw = raw.get("requests")
    if not isinstance(requests_raw, list):
        raise BenchmarkError("workload.requests must be a JSON array")

    requests: list[WorkloadRequest] = []
    seen: set[str] = set()
    last_planned = 0.0
    for index, item in enumerate(requests_raw):
        if not isinstance(item, Mapping):
            raise BenchmarkError(f"requests[{index}] must be a JSON object")
        request_id = str(item.get("request_id", item.get("id", ""))).strip()
        if not request_id:
            raise BenchmarkError(f"requests[{index}].request_id must be non-empty")
        if request_id in seen:
            raise BenchmarkError(f"duplicate workload request_id: {request_id!r}")
        seen.add(request_id)
        input_ids_raw = item.get("input_ids")
        if not isinstance(input_ids_raw, list) or not input_ids_raw:
            raise BenchmarkError(f"requests[{index}].input_ids must be a non-empty array")
        input_ids: list[int] = []
        for token_index, token in enumerate(input_ids_raw):
            token_id = _as_int(token, f"requests[{index}].input_ids[{token_index}]")
            if token_id < 0:
                raise BenchmarkError(
                    f"requests[{index}].input_ids[{token_index}] must be >= 0"
                )
            input_ids.append(token_id)
        max_new_tokens = _as_int(
            item.get("max_new_tokens", 8), f"requests[{index}].max_new_tokens"
        )
        if max_new_tokens <= 0:
            raise BenchmarkError(f"requests[{index}].max_new_tokens must be > 0")
        planned_value = item.get("planned_at_s", item.get("at_s", 0.0))
        planned_at_s = _as_nonnegative_float(
            planned_value, f"requests[{index}].planned_at_s"
        )
        # A non-monotonic schedule is valid but almost always an authoring
        # mistake.  Preserve it as authored while making the issue explicit
        # only through a validation error, rather than silently reordering it.
        if requests and planned_at_s < last_planned:
            raise BenchmarkError(
                "workload requests must be ordered by nondecreasing planned_at_s"
            )
        last_planned = planned_at_s
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise BenchmarkError(f"requests[{index}].metadata must be an object")
        requests.append(
            WorkloadRequest(
                request_id=request_id,
                input_ids=tuple(input_ids),
                max_new_tokens=max_new_tokens,
                planned_at_s=planned_at_s,
                metadata=dict(metadata),
            )
        )

    validated_raw = raw.get("token_ids_validated_against_model", False)
    if not isinstance(validated_raw, bool):
        raise BenchmarkError("token_ids_validated_against_model must be boolean")
    return Workload(
        workload_id=workload_id,
        purpose=purpose,
        requests=tuple(requests),
        description=str(raw.get("description", "")),
        token_ids_note=str(raw.get("token_ids_note", "")),
        token_ids_validated_against_model=validated_raw,
        schema_version=schema_version,
    )


@dataclass
class SSEEvent:
    """One raw SSE event and the parsed JSON payload, if any."""

    sequence: int
    received_at_s: float
    kind: str
    data: str
    raw: str
    payload: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "received_at_s": self.received_at_s,
            "kind": self.kind,
            "data": self.data,
            "raw": self.raw,
            "payload": self.payload,
        }


def _decode_sse_line(line: Any) -> str:
    if isinstance(line, bytes):
        return line.decode("utf-8", errors="replace")
    if isinstance(line, str):
        return line
    return str(line)


def _classify_sse_data(data: str) -> tuple[str, Any]:
    if data.strip() == "[DONE]":
        return "done", None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return "invalid", None
    if isinstance(payload, Mapping) and "error" in payload:
        return "error", payload
    if isinstance(payload, Mapping):
        return "response", payload
    return "invalid", payload


def parse_sse_stream(
    stream: Any,
    *,
    time_origin: Optional[float] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Iterable[SSEEvent]:
    """Yield SSE events from a file-like response without external packages.

    Raw lines are retained exactly as decoded (including line endings).  Only
    ``data:`` fields are parsed; comments and other SSE fields remain in
    ``raw``.  The native SGLang endpoint emits one JSON data field per event,
    but joining multiple data fields follows the SSE convention and keeps the
    parser useful for a fixture or a proxy that folds lines.
    """

    origin = clock() if time_origin is None else time_origin
    raw_lines: list[str] = []
    data_lines: list[str] = []
    sequence = 0

    def emit() -> Optional[SSEEvent]:
        nonlocal raw_lines, data_lines, sequence
        if not raw_lines:
            return None
        raw = "".join(raw_lines)
        data = "\n".join(data_lines)
        raw_lines = []
        data_lines = []
        if not data:
            return None
        sequence += 1
        kind, payload = _classify_sse_data(data)
        return SSEEvent(
            sequence=sequence,
            received_at_s=clock() - origin,
            kind=kind,
            data=data,
            raw=raw,
            payload=payload,
        )

    while True:
        line = stream.readline()
        if line in (b"", ""):
            event = emit()
            if event is not None:
                yield event
            break
        text = _decode_sse_line(line)
        raw_lines.append(text)
        stripped = text.rstrip("\r\n")
        if stripped == "":
            event = emit()
            if event is not None:
                yield event
        elif stripped.startswith("data:"):
            value = stripped[5:]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
        # ``event:``, ``id:``, ``retry:`` and comments are intentionally
        # ignored for parsing but remain in the raw event.


class _StreamProtocolError(ValueError):
    """A native stream cannot support exact token evidence."""


class _TokenAccumulator:
    """Reconstruct output IDs from native delta or cumulative stream chunks."""

    def __init__(self) -> None:
        self.ids: list[int] = []
        self.confidence = "reported_completion_tokens"

    @staticmethod
    def _valid_ids(value: Any) -> Optional[list[int]]:
        if not isinstance(value, list):
            return None
        result: list[int] = []
        for token in value:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                return None
            result.append(token)
        return result

    def add(self, ids_value: Any, reported_completion_tokens: Optional[int]) -> list[int]:
        ids = self._valid_ids(ids_value)
        if ids is None:
            raise _StreamProtocolError("output_ids must be a list of non-negative integers")
        before = len(self.ids)
        if reported_completion_tokens is None:
            raise _StreamProtocolError(
                "meta_info.completion_tokens is required for exact stream reconstruction"
            )
        if reported_completion_tokens < before:
            raise _StreamProtocolError(
                f"completion_tokens regressed from {before} to {reported_completion_tokens}"
            )

        # The native server normally reports cumulative completion_tokens and
        # emits a delta output_ids.  It can also emit a cumulative list when a
        # different stream configuration is used.  The count disambiguates
        # both forms and handles repeated token IDs correctly.
        if reported_completion_tokens == before + len(ids):
            self.ids.extend(ids)
            return ids
        if (
            reported_completion_tokens == len(ids)
            and len(ids) >= before
            and ids[:before] == self.ids
        ):
            self.ids = list(ids)
            return ids[before:]
        raise _StreamProtocolError(
            "output_ids and meta_info.completion_tokens are inconsistent "
            f"(previous={before}, chunk={len(ids)}, reported={reported_completion_tokens})"
        )


@dataclass
class RequestResult:
    """Observed lifecycle and stream evidence for one planned request."""

    workload_request_id: str
    client_request_id: str
    planned_at_s: float
    actual_send_at_s: Optional[float]
    first_token_at_s: Optional[float]
    completion_at_s: Optional[float]
    status: str
    server_request_id: Optional[str] = None
    http_status: Optional[int] = None
    error: Optional[str] = None
    output_ids: list[int] = field(default_factory=list)
    completion_tokens: Optional[int] = None
    token_event_counts: list[int] = field(default_factory=list)
    token_event_times_s: list[float] = field(default_factory=list)
    events: list[SSEEvent] = field(default_factory=list)
    backlog_at_send: Optional[int] = None
    client_queue_delay_s: Optional[float] = None
    finish_reason: Any = None
    token_ids_confidence: str = "not_observed"
    done_seen: bool = False
    finish_seen: bool = False
    request_metadata: dict[str, Any] = field(default_factory=dict)

    def metrics(self) -> dict[str, Any]:
        ttft_s: Optional[float] = None
        if self.actual_send_at_s is not None and self.first_token_at_s is not None:
            ttft_s = max(0.0, self.first_token_at_s - self.actual_send_at_s)
        tpot_s: Optional[float] = None
        if (
            self.status == "success"
            and self.first_token_at_s is not None
            and self.completion_at_s is not None
            and len(self.output_ids) >= 2
        ):
            tpot_s = max(0.0, self.completion_at_s - self.first_token_at_s) / (
                len(self.output_ids) - 1
            )
        itl_s: Optional[list[float]]
        if self.token_event_counts and all(count == 1 for count in self.token_event_counts):
            itl_s = [
                max(0.0, right - left)
                for left, right in zip(
                    self.token_event_times_s, self.token_event_times_s[1:]
                )
            ]
        else:
            itl_s = None
        return {
            "ttft_s": ttft_s,
            "request_tpot_s": tpot_s,
            "token_itl_s": itl_s,
            "output_tokens": len(self.output_ids),
            "successful_output_tokens": len(self.output_ids)
            if self.status == "success"
            else 0,
        }

    def to_dict(self) -> dict[str, Any]:
        metrics = self.metrics()
        return {
            "workload_request_id": self.workload_request_id,
            "client_request_id": self.client_request_id,
            "server_request_id": self.server_request_id,
            "planned_at_s": self.planned_at_s,
            "actual_send_at_s": self.actual_send_at_s,
            "first_token_at_s": self.first_token_at_s,
            "completion_at_s": self.completion_at_s,
            "status": self.status,
            "http_status": self.http_status,
            "error": self.error,
            "output_ids": list(self.output_ids),
            "completion_tokens": self.completion_tokens,
            "token_event_counts": list(self.token_event_counts),
            "token_event_times_s": list(self.token_event_times_s),
            "backlog_at_send": self.backlog_at_send,
            "client_queue_delay_s": self.client_queue_delay_s,
            "finish_reason": self.finish_reason,
            "token_ids_confidence": self.token_ids_confidence,
            "done_seen": self.done_seen,
            "finish_seen": self.finish_seen,
            "request_metadata": dict(self.request_metadata),
            "metrics": metrics,
            "events": [event.to_dict() for event in self.events],
        }


def _parse_reported_completion(meta_info: Mapping[str, Any]) -> Optional[int]:
    value = meta_info.get("completion_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _process_stream_events(
    events: list[SSEEvent],
    *,
    result: RequestResult,
    expected_client_request_id: Optional[str] = None,
    expected_max_new_tokens: Optional[int] = None,
) -> None:
    accumulator = _TokenAccumulator()
    protocol_errors: list[str] = []
    for event in events:
        if event.kind == "done":
            if result.done_seen:
                protocol_errors.append("native [DONE] marker appeared more than once")
            result.done_seen = True
            if result.completion_at_s is None:
                result.completion_at_s = event.received_at_s
            continue
        if event.kind == "error":
            result.error = str(
                event.payload.get("error") if isinstance(event.payload, Mapping) else event.data
            )
            continue
        if event.kind != "response" or not isinstance(event.payload, Mapping):
            protocol_errors.append(f"invalid SSE response event {event.sequence}")
            continue
        if result.done_seen:
            protocol_errors.append(
                f"response event {event.sequence} appeared after native [DONE] marker"
            )
        payload = event.payload
        meta = payload.get("meta_info")
        if not isinstance(meta, Mapping):
            protocol_errors.append(f"event {event.sequence} has no meta_info object")
            continue
        server_id = meta.get("id")
        if server_id is None:
            protocol_errors.append(f"event {event.sequence} has no meta_info.id")
        else:
            server_id = str(server_id)
            if result.server_request_id is None:
                result.server_request_id = server_id
            elif result.server_request_id != server_id:
                protocol_errors.append(
                    f"server request ID changed from {result.server_request_id!r} to {server_id!r}"
                )
            if (
                expected_client_request_id is not None
                and server_id != expected_client_request_id
            ):
                protocol_errors.append(
                    f"server request ID {server_id!r} did not match "
                    f"requested ID {expected_client_request_id!r}"
                )
        reported = _parse_reported_completion(meta)
        if reported is None:
            protocol_errors.append(
                f"event {event.sequence} has no valid meta_info.completion_tokens"
            )
        else:
            result.completion_tokens = reported
        if "output_ids" not in payload:
            protocol_errors.append(f"event {event.sequence} has no output_ids list")
            valid_output_ids = None
        else:
            valid_output_ids = _TokenAccumulator._valid_ids(payload.get("output_ids"))
        if "output_ids" in payload and valid_output_ids is None:
            protocol_errors.append(f"event {event.sequence} has invalid output_ids")
        delta: list[int] = []
        if valid_output_ids is not None and reported is not None:
            try:
                delta = accumulator.add(valid_output_ids, reported)
            except _StreamProtocolError as exc:
                protocol_errors.append(f"event {event.sequence}: {exc}")
        if delta:
            if result.first_token_at_s is None:
                result.first_token_at_s = event.received_at_s
            result.token_event_counts.append(len(delta))
            result.token_event_times_s.append(event.received_at_s)
        finish_reason = meta.get("finish_reason")
        if finish_reason is not None:
            if result.finish_seen:
                protocol_errors.append("more than one response event carried finish_reason")
            result.finish_seen = True
            result.finish_reason = finish_reason
            if result.completion_at_s is None:
                result.completion_at_s = event.received_at_s
    result.output_ids = list(accumulator.ids)
    if result.finish_seen:
        finish_reason = result.finish_reason
        if not isinstance(finish_reason, Mapping):
            protocol_errors.append("finish_reason must be an object with type=length")
        elif finish_reason.get("type") != "length":
            protocol_errors.append(
                f"finish_reason.type must be 'length', got {finish_reason.get('type')!r}"
            )
        else:
            finish_length = finish_reason.get("length")
            if (
                isinstance(finish_length, bool)
                or not isinstance(finish_length, int)
                or finish_length < 0
            ):
                protocol_errors.append("finish_reason.length must be a non-negative integer")
            elif (
                expected_max_new_tokens is not None
                and finish_length != expected_max_new_tokens
            ):
                protocol_errors.append(
                    f"finish_reason.length={finish_length} did not match "
                    f"requested max_new_tokens={expected_max_new_tokens}"
                )
    if expected_max_new_tokens is not None:
        if len(result.output_ids) != expected_max_new_tokens:
            protocol_errors.append(
                f"observed {len(result.output_ids)} output IDs; expected "
                f"{expected_max_new_tokens} with ignore_eos=true"
            )
        if result.completion_tokens != expected_max_new_tokens:
            protocol_errors.append(
                f"final completion_tokens={result.completion_tokens!r}; expected "
                f"{expected_max_new_tokens} with ignore_eos=true"
            )
    if result.error is None and not result.done_seen:
        protocol_errors.append("stream ended without native [DONE] marker")
    if result.error is None and not result.finish_seen:
        protocol_errors.append("stream ended without a finish_reason")
    if protocol_errors:
        result.token_ids_confidence = "protocol_error"
    else:
        result.token_ids_confidence = (
            accumulator.confidence if result.output_ids else "no_output_tokens_observed"
        )
    if protocol_errors:
        result.error = "; ".join(
            [part for part in [result.error, *protocol_errors] if part]
        )
    if result.error is None:
        result.status = "success"
    elif result.status not in {"timeout", "error"}:
        result.status = "error"


class BenchmarkClient:
    """Native SGLang HTTP client used by :func:`run_benchmark`."""

    def __init__(
        self,
        url: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise BenchmarkError("url must be non-empty")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BenchmarkError("url must include an http:// or https:// scheme and host")
        self.base_url = url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else "/" + endpoint
        if urllib.parse.urlparse(self.base_url).path.rstrip("/").endswith(
            self.endpoint.rstrip("/")
        ):
            self.generate_url = self.base_url
        else:
            self.generate_url = self.base_url + self.endpoint
        self.headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if headers:
            self.headers.update({str(key): str(value) for key, value in headers.items()})

    def _set_socket_timeout(self, response: Any, timeout_s: float) -> None:
        # urllib exposes the underlying socket through HTTPResponse.fp on the
        # CPython versions used here.  Keep this best-effort so synthetic file
        # fixtures and alternate stdlib transports remain usable.
        try:
            sock = response.fp.raw._sock
        except AttributeError:
            try:
                sock = response.fp._sock
            except AttributeError:
                sock = None
        if sock is not None:
            try:
                sock.settimeout(timeout_s)
            except (AttributeError, OSError):
                pass

    def reset_cache(self, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
        """Explicitly request SGLang's radix/KV prefix cache flush."""

        url = self.base_url
        parsed_base = urllib.parse.urlparse(url)
        if parsed_base.path.rstrip("/").endswith(self.endpoint.rstrip("/")):
            base_path = parsed_base.path.rstrip("/")
            endpoint_path = self.endpoint.rstrip("/")
            parent_path = base_path[: -len(endpoint_path)] or "/"
            parsed = urllib.parse.urlparse(url)
            url = urllib.parse.urlunparse(parsed._replace(path=parent_path, params=""))
        url = url.rstrip("/") + "/flush_cache"
        request = urllib.request.Request(url, method="POST", headers=self.headers)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = response.read(4096).decode("utf-8", errors="replace")
                return {
                    "status": "success" if 200 <= response.status < 300 else "error",
                    "http_status": response.status,
                    "body": body,
                    "elapsed_s": time.perf_counter() - started,
                    "scope": "SGLang radix/KV prefix cache only; no process/graph/allocator reset",
                }
        except Exception as exc:  # preserve the exact operation failure in output
            return {
                "status": "error",
                "http_status": getattr(exc, "code", None),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": time.perf_counter() - started,
                "scope": "SGLang radix/KV prefix cache only; no process/graph/allocator reset",
            }

    def stream_request(
        self,
        payload: Mapping[str, Any],
        *,
        timeout_s: float,
        time_origin: float,
    ) -> tuple[Optional[int], list[SSEEvent], Optional[str]]:
        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise BenchmarkError("timeout_s must be finite and > 0")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        request = urllib.request.Request(
            self.generate_url,
            data=encoded,
            method="POST",
            headers=self.headers,
        )
        deadline = time.perf_counter() + timeout_s
        events: list[SSEEvent] = []
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                http_status = getattr(response, "status", None)
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise BenchmarkTimeout(f"request exceeded timeout_s={timeout_s}")
                client = self

                class _DeadlineReader:
                    def readline(self_inner: Any) -> Any:
                        remaining_inner = deadline - time.perf_counter()
                        if remaining_inner <= 0:
                            raise BenchmarkTimeout(
                                f"request exceeded timeout_s={timeout_s}"
                            )
                        client._set_socket_timeout(response, remaining_inner)
                        return response.readline()

                # parse_sse_stream reads to EOF; the wrapper bounds every idle
                # read and the checks below bound the complete request.
                event_iter = parse_sse_stream(
                    _DeadlineReader(), time_origin=time_origin, clock=time.perf_counter
                )
                for event in event_iter:
                    events.append(event)
                    if time.perf_counter() > deadline:
                        raise BenchmarkTimeout(f"request exceeded timeout_s={timeout_s}")
                if time.perf_counter() > deadline:
                    raise BenchmarkTimeout(f"request exceeded timeout_s={timeout_s}")
                return http_status, events, None
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read(4096).decode("utf-8", errors="replace")
            except Exception:
                pass
            detail = f"HTTP {exc.code}"
            if body:
                detail += f": {body[:1000]}"
            return exc.code, events, detail
        except (socket.timeout, TimeoutError, BenchmarkTimeout) as exc:
            return None, events, f"timeout: {exc or 'request deadline exceeded'}"
        except urllib.error.URLError as exc:
            return None, events, f"URL error: {exc.reason}"
        except Exception as exc:
            return None, events, f"{type(exc).__name__}: {exc}"


def _request_id(run_id: str, workload_request_id: str) -> str:
    # Keep the run scope visible in both server logs and the saved artifact.
    return f"graphbudget/{run_id}/{workload_request_id}"


def _request_payload(spec: WorkloadRequest, client_request_id: str) -> dict[str, Any]:
    return {
        "input_ids": list(spec.input_ids),
        "sampling_params": {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_new_tokens": spec.max_new_tokens,
        },
        "stream": True,
        "rid": client_request_id,
    }


def _run_one_request(
    client: BenchmarkClient,
    spec: WorkloadRequest,
    *,
    run_id: str,
    time_origin: float,
    timeout_s: float,
    state: dict[str, int],
    state_lock: threading.Lock,
    planned_at_s: float,
) -> RequestResult:
    client_request_id = _request_id(run_id, spec.request_id)
    with state_lock:
        # Count requests that were already due but had not reached the HTTP
        # call yet.  The current request is included in this point-in-time
        # queue depth, which makes a delayed worker visible in the artifact.
        backlog = max(0, state["planned_due"] - state["actual_started"])
        state["actual_started"] += 1
    actual_send_at_s = time.perf_counter() - time_origin
    result = RequestResult(
        workload_request_id=spec.request_id,
        client_request_id=client_request_id,
        planned_at_s=planned_at_s,
        actual_send_at_s=actual_send_at_s,
        first_token_at_s=None,
        completion_at_s=None,
        status="error",
        backlog_at_send=backlog,
        client_queue_delay_s=max(0.0, actual_send_at_s - planned_at_s),
        request_metadata=dict(spec.metadata),
    )
    payload = _request_payload(spec, client_request_id)
    http_status, events, error = client.stream_request(
        payload, timeout_s=timeout_s, time_origin=time_origin
    )
    result.http_status = http_status
    result.events = events
    if error is not None:
        result.error = error
        result.status = "timeout" if error.startswith("timeout:") else "error"
    _process_stream_events(
        events,
        result=result,
        expected_client_request_id=client_request_id,
        expected_max_new_tokens=spec.max_new_tokens,
    )
    if result.server_request_id is None:
        result.error = result.error or "stream responses did not expose meta_info.id"
        if result.status != "timeout":
            result.status = "error"
    elif result.server_request_id != client_request_id:
        result.error = (
            result.error
            or f"server request ID {result.server_request_id!r} did not match scoped client ID"
        )
        if result.status != "timeout":
            result.status = "error"
    if error is not None:
        # A timeout or transport error remains the authoritative status even
        # when a partial stream contained otherwise valid output IDs.
        result.error = error
        result.status = "timeout" if error.startswith("timeout:") else "error"
    return result


def _quantile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_distribution(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    result = {
        "sample_count": len(clean),
        "p50": _quantile(clean, 0.50),
        "p95": _quantile(clean, 0.95),
        "p99": _quantile(clean, 0.99) if len(clean) >= P99_MIN_SAMPLES else None,
        "p99_status": "reported" if len(clean) >= P99_MIN_SAMPLES else "NOT_REPORTED_INSUFFICIENT_SAMPLE",
    }
    return result


def summarize_request_results(
    results: Sequence[RequestResult],
    *,
    run_duration_s: Optional[float] = None,
) -> dict[str, Any]:
    """Summarize request evidence without hiding planned failures/timeouts."""

    completed = [result for result in results if result.status == "success"]
    failed = [result for result in results if result.status == "error"]
    timed_out = [result for result in results if result.status == "timeout"]
    actual = [result for result in results if result.actual_send_at_s is not None]
    ttft_values = [
        result.metrics()["ttft_s"]
        for result in completed
        if result.metrics()["ttft_s"] is not None
    ]
    tpot_values = [
        result.metrics()["request_tpot_s"]
        for result in completed
        if result.metrics()["request_tpot_s"] is not None
    ]
    itl_values: list[float] = []
    itl_request_count = 0
    itl_unavailable_count = 0
    for result in completed:
        itl = result.metrics()["token_itl_s"]
        if itl is None:
            itl_unavailable_count += 1
        else:
            itl_request_count += 1
            itl_values.extend(itl)
    successful_output_tokens = sum(len(result.output_ids) for result in completed)
    if run_duration_s is None:
        observed_times = [
            value
            for result in results
            for value in (result.actual_send_at_s, result.completion_at_s)
            if value is not None
        ]
        run_duration_s = max(observed_times) - min(observed_times) if observed_times else 0.0
    run_duration_s = max(0.0, float(run_duration_s))
    output_tokens_per_s = (
        successful_output_tokens / run_duration_s if run_duration_s > 0 else None
    )
    return {
        "planned_requests": len(results),
        "actual_send_requests": len(actual),
        "not_started_requests": max(0, len(results) - len(actual)),
        "completed_requests": len(completed),
        "failed_requests": len(failed),
        "timed_out_requests": len(timed_out),
        "successful_output_tokens": successful_output_tokens,
        "effective_output_tokens_per_s": output_tokens_per_s,
        "makespan_s": run_duration_s,
        "max_backlog_at_send": max(
            (result.backlog_at_send or 0 for result in actual), default=0
        ),
        "max_client_queue_delay_s": max(
            (result.client_queue_delay_s or 0.0 for result in actual), default=0.0
        ),
        "ttft_s": _metric_distribution(ttft_values),
        "request_tpot_s": _metric_distribution(tpot_values),
        "token_itl_s": _metric_distribution(itl_values),
        "token_itl_request_count": itl_request_count,
        "token_itl_unavailable_request_count": itl_unavailable_count,
        "p99_min_samples": P99_MIN_SAMPLES,
    }


def run_benchmark(
    url: str,
    workload: Any,
    *,
    output: Optional[Any] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    run_id: Optional[str] = None,
    fixed_work: bool = False,
    max_workers: Optional[int] = None,
    reset_cache: bool = False,
    headers: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Run every planned request against a real native SGLang server.

    ``fixed_work=False`` is open-loop: each request is submitted at its
    ``planned_at_s`` offset even when earlier requests are still pending.
    ``fixed_work=True`` dispatches the same complete request set immediately
    (all effective planned offsets are zero) and reports its makespan.  In
    both modes every planned request receives a result record.
    """

    if timeout_s <= 0 or not math.isfinite(timeout_s):
        raise BenchmarkError("timeout_s must be finite and > 0")
    loaded = load_workload(workload) if not isinstance(workload, Workload) else workload
    client = BenchmarkClient(url, headers=headers)
    run_id = (run_id or uuid.uuid4().hex).strip()
    if not run_id:
        raise BenchmarkError("run_id must be non-empty")
    started_wall = _utc_now()
    state = {"planned_due": 0, "actual_started": 0, "max_pending": 0}
    state_lock = threading.Lock()
    cache_result: Optional[dict[str, Any]] = None
    if reset_cache:
        cache_result = client.reset_cache(timeout_s=timeout_s)

    # Cache maintenance is setup, not request service time.  Start the
    # planned schedule only after the optional reset has returned.
    started_wall = _utc_now()
    time_origin = time.perf_counter()

    specs = list(loaded.requests)
    workers = max_workers if max_workers is not None else max(1, len(specs))
    if workers <= 0:
        raise BenchmarkError("max_workers must be > 0")
    futures: list[tuple[WorkloadRequest, float, concurrent.futures.Future[RequestResult]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for spec in specs:
            effective_planned = 0.0 if fixed_work else spec.planned_at_s
            target = time_origin + effective_planned
            if not fixed_work:
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            with state_lock:
                state["planned_due"] += 1
                state["max_pending"] = max(
                    state["max_pending"],
                    state["planned_due"] - state["actual_started"],
                )
            future = executor.submit(
                _run_one_request,
                client,
                spec,
                run_id=run_id,
                time_origin=time_origin,
                timeout_s=timeout_s,
                state=state,
                state_lock=state_lock,
                planned_at_s=effective_planned,
            )
            futures.append((spec, effective_planned, future))
        results: list[RequestResult] = []
        for spec, planned_at, future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                # A worker bug must still leave a visible record for its
                # planned request instead of shrinking the denominator.
                results.append(
                    RequestResult(
                        workload_request_id=spec.request_id,
                        client_request_id=_request_id(run_id, spec.request_id),
                        planned_at_s=planned_at,
                        actual_send_at_s=None,
                        first_token_at_s=None,
                        completion_at_s=None,
                        status="error",
                        error=f"worker exception: {type(exc).__name__}: {exc}",
                        request_metadata=dict(spec.metadata),
                    )
                )
    finished_wall = _utc_now()
    run_duration_s = max(0.0, time.perf_counter() - time_origin)
    summary = summarize_request_results(results, run_duration_s=run_duration_s)
    summary["max_client_backlog"] = state["max_pending"]
    result_dict = {
        "schema_version": SCHEMA_VERSION,
        "kind": "graphbudget_benchmark_run",
        "protocol": {
            "endpoint": "native_sglang_generate_sse",
            "fixed_input_ids": True,
            "sampling": {"temperature": 0.0, "ignore_eos": True},
            "request_id_scope": "graphbudget/<run_id>/<stable workload request_id>",
            "backlog_definition": "due requests not yet marked actual at worker start; includes current request",
            "cache_reset_scope": "SGLang radix/KV prefix cache only; no process/graph/allocator reset",
            "itl_rule": "reported only when every token event contributes exactly one token",
            "p99_min_samples": P99_MIN_SAMPLES,
        },
        "run": {
            "run_id": run_id,
            "url": client.generate_url,
            "started_at_utc": started_wall,
            "finished_at_utc": finished_wall,
            "timeout_s": timeout_s,
            "schedule_mode": "fixed_work" if fixed_work else "open_loop",
            "max_workers": workers,
            "reset_cache_requested": reset_cache,
        },
        "workload": loaded.to_dict(),
        "cache_reset": cache_result,
        "requests": [result.to_dict() for result in results],
        "summary": summary,
    }
    if output is not None:
        write_json(output, result_dict)
    return result_dict


def write_json(destination: Any, value: Mapping[str, Any]) -> None:
    """Write one UTF-8, indented JSON artifact to a path or text stream."""

    if hasattr(destination, "write"):
        json.dump(value, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_run(source: Any) -> Mapping[str, Any]:
    value = _json_load(source)
    if not isinstance(value, Mapping) or value.get("kind") != "graphbudget_benchmark_run":
        raise BenchmarkError("expected a graphbudget_benchmark_run JSON object")
    return value


def _request_map(run: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    requests = run.get("requests")
    if not isinstance(requests, list):
        raise BenchmarkError("benchmark run has no requests array")
    result: dict[str, Mapping[str, Any]] = {}
    for item in requests:
        if not isinstance(item, Mapping):
            raise BenchmarkError("benchmark request result must be an object")
        identity = item.get("workload_request_id", item.get("request_id"))
        if not isinstance(identity, str) or not identity:
            raise BenchmarkError("benchmark request is missing stable workload_request_id")
        if identity in result:
            raise BenchmarkError(f"duplicate stable request identity: {identity!r}")
        result[identity] = item
    return result


def _workload_spec_map(run: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    workload = run.get("workload")
    if not isinstance(workload, Mapping):
        return {}
    requests = workload.get("requests")
    if not isinstance(requests, list):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for item in requests:
        if isinstance(item, Mapping):
            identity = item.get("request_id")
            if isinstance(identity, str):
                result[identity] = item
    return result


def compare_exact_tokens(baseline: Any, candidate: Any) -> dict[str, Any]:
    """Compare exact output token IDs by stable workload request identity."""

    base_run = _load_run(baseline)
    cand_run = _load_run(candidate)
    base = _request_map(base_run)
    cand = _request_map(cand_run)
    base_specs = _workload_spec_map(base_run)
    cand_specs = _workload_spec_map(cand_run)
    identities = sorted(set(base) | set(cand))
    matches: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for identity in identities:
        left = base.get(identity)
        right = cand.get(identity)
        if left is None or right is None:
            mismatches.append(
                {
                    "workload_request_id": identity,
                    "reason": "missing_in_baseline" if left is None else "missing_in_candidate",
                }
            )
            continue
        left_status = left.get("status")
        right_status = right.get("status")
        left_spec = base_specs.get(identity)
        right_spec = cand_specs.get(identity)
        if left_spec is not None and right_spec is not None:
            comparable_fields = ("input_ids", "max_new_tokens")
            if any(left_spec.get(field) != right_spec.get(field) for field in comparable_fields):
                mismatches.append(
                    {
                        "workload_request_id": identity,
                        "reason": "workload_request_spec_differ",
                        "baseline_spec": dict(left_spec),
                        "candidate_spec": dict(right_spec),
                    }
                )
                continue
        left_ids = left.get("output_ids")
        right_ids = right.get("output_ids")
        if left_status != "success" or right_status != "success":
            mismatches.append(
                {
                    "workload_request_id": identity,
                    "reason": "request_not_successful",
                    "baseline_status": left_status,
                    "candidate_status": right_status,
                    "baseline_error": left.get("error"),
                    "candidate_error": right.get("error"),
                }
            )
            continue
        if not isinstance(left_ids, list) or not isinstance(right_ids, list):
            mismatches.append(
                {
                    "workload_request_id": identity,
                    "reason": "missing_output_token_ids",
                }
            )
            continue
        if left.get("token_ids_confidence") == "no_output_tokens_observed" or right.get(
            "token_ids_confidence"
        ) == "no_output_tokens_observed":
            mismatches.append(
                {
                    "workload_request_id": identity,
                    "reason": "output_token_ids_not_observed",
                    "baseline_token_ids_confidence": left.get("token_ids_confidence"),
                    "candidate_token_ids_confidence": right.get("token_ids_confidence"),
                }
            )
            continue
        if left_ids != right_ids:
            mismatch_index = next(
                (
                    index
                    for index, (left_token, right_token) in enumerate(zip(left_ids, right_ids))
                    if left_token != right_token
                ),
                min(len(left_ids), len(right_ids)),
            )
            mismatches.append(
                {
                    "workload_request_id": identity,
                    "reason": "output_token_ids_differ",
                    "first_difference_index": mismatch_index,
                    "baseline_output_ids": left_ids,
                    "candidate_output_ids": right_ids,
                }
            )
        else:
            matches.append(identity)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "graphbudget_exact_token_comparison",
        "baseline_run_id": base_run.get("run", {}).get("run_id"),
        "candidate_run_id": cand_run.get("run", {}).get("run_id"),
        "request_count": len(identities),
        "matched_request_count": len(matches),
        "mismatched_request_count": len(mismatches),
        "matched_workload_request_ids": matches,
        "mismatches": mismatches,
        "equivalent": not mismatches and set(base) == set(cand),
    }


def _summary_scalar(run: Mapping[str, Any], metric: str) -> Optional[float]:
    summary = run.get("summary", {})
    if not isinstance(summary, Mapping):
        return None
    if metric in {"effective_output_tokens_per_s", "makespan_s"}:
        value = summary.get(metric)
    elif metric == "ttft_p50_s":
        value = (summary.get("ttft_s") or {}).get("p50")
    elif metric == "request_tpot_p50_s":
        value = (summary.get("request_tpot_s") or {}).get("p50")
    elif metric == "token_itl_p50_s":
        value = (summary.get("token_itl_s") or {}).get("p50")
    else:
        value = None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _normalize_pairs(pairs: Any) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise BenchmarkError("paired rounds must be an array")
    normalized: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for index, pair in enumerate(pairs):
        if isinstance(pair, Mapping):
            baseline = pair.get("baseline")
            candidate = pair.get("candidate")
        elif isinstance(pair, Sequence) and len(pair) == 2:
            baseline, candidate = pair
        else:
            raise BenchmarkError(f"paired round {index} must contain baseline and candidate")
        normalized.append((_load_run(baseline), _load_run(candidate)))
    return normalized


def _paired_metric_summary(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], metric: str
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    relative: list[float] = []
    for index, (baseline, candidate) in enumerate(pairs, start=1):
        left = _summary_scalar(baseline, metric)
        right = _summary_scalar(candidate, metric)
        row: dict[str, Any] = {"round": index, "baseline": left, "candidate": right}
        if left is not None and right is not None:
            delta = right - left
            row["delta_candidate_minus_baseline"] = delta
            if left != 0:
                row["relative_delta"] = delta / abs(left)
                relative.append(delta / abs(left))
            deltas.append(delta)
        else:
            row["status"] = "MISSING_METRIC"
        rows.append(row)
    return {
        "metric": metric,
        "rounds": rows,
        "paired_sample_count": len(deltas),
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "median_delta": _quantile(deltas, 0.5),
        "mean_relative_delta": sum(relative) / len(relative) if relative else None,
    }


def summarize_paired_rounds(
    pairs: Any,
    *,
    aa_pairs: Any = None,
    min_rounds: int = MIN_PAIRED_ROUNDS,
) -> dict[str, Any]:
    """Summarize interleaved baseline/candidate rounds and A/A noise.

    ``pairs`` is a sequence of ``{"baseline": run, "candidate": run}``
    objects, where each value is a loaded run mapping or a JSON path.  The
    function reports a non-adoptable ``INSUFFICIENT_ROUNDS`` status below five
    rounds.  ``aa_pairs`` uses the same shape and is optional; when absent the
    result explicitly says ``NOT_PROVIDED`` instead of inventing a noise band.
    """

    normalized = _normalize_pairs(pairs)
    if min_rounds < MIN_PAIRED_ROUNDS:
        raise BenchmarkError(f"min_rounds must be >= {MIN_PAIRED_ROUNDS}")
    metrics = [
        _paired_metric_summary(normalized, metric)
        for metric in (
            "effective_output_tokens_per_s",
            "makespan_s",
            "ttft_p50_s",
            "request_tpot_p50_s",
            "token_itl_p50_s",
        )
    ]
    if aa_pairs is None:
        aa_summary: dict[str, Any] = {
            "status": "NOT_PROVIDED",
            "metrics": [],
            "note": "Run explicit A/A pairs before treating a delta as beyond noise.",
        }
    else:
        aa_normalized = _normalize_pairs(aa_pairs)
        aa_metrics = []
        for metric in (
            "effective_output_tokens_per_s",
            "makespan_s",
            "ttft_p50_s",
            "request_tpot_p50_s",
            "token_itl_p50_s",
        ):
            summary = _paired_metric_summary(aa_normalized, metric)
            relative = [
                row["relative_delta"]
                for row in summary["rounds"]
                if isinstance(row.get("relative_delta"), (int, float))
            ]
            summary["absolute_relative_delta_p95"] = _quantile(
                [abs(value) for value in relative], 0.95
            )
            aa_metrics.append(summary)
        aa_summary = {
            "status": "OK" if len(aa_normalized) >= min_rounds else "INSUFFICIENT_ROUNDS",
            "round_count": len(aa_normalized),
            "metrics": aa_metrics,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "graphbudget_paired_round_summary",
        "status": "OK" if len(normalized) >= min_rounds else "INSUFFICIENT_ROUNDS",
        "minimum_rounds": min_rounds,
        "round_count": len(normalized),
        "metrics": metrics,
        "aa_noise": aa_summary,
        "adoption_claim": "NOT_SUPPORTED" if len(normalized) < min_rounds else "REQUIRES_AA_AND_TARGET_REVIEW",
    }


def _parse_headers(values: Optional[Sequence[str]]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values or []:
        if ":" not in value:
            raise BenchmarkError(f"header must use Name: value syntax: {value!r}")
        name, header_value = value.split(":", 1)
        name, header_value = name.strip(), header_value.strip()
        if not name:
            raise BenchmarkError("header name must be non-empty")
        headers[name] = header_value
    return headers


def _run_cli(args: argparse.Namespace) -> int:
    result = run_benchmark(
        args.url,
        args.workload,
        output=args.output,
        timeout_s=args.timeout,
        run_id=args.run_id,
        fixed_work=args.fixed_work,
        max_workers=args.max_workers,
        reset_cache=args.reset_cache,
        headers=_parse_headers(args.header),
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


def _compare_cli(args: argparse.Namespace) -> int:
    result = compare_exact_tokens(args.baseline, args.candidate)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["equivalent"] else 2


def _summarize_cli(args: argparse.Namespace) -> int:
    pairs: list[dict[str, Any]] = []
    for value in args.pair:
        if "," not in value:
            raise BenchmarkError("--pair expects baseline.json,candidate.json")
        baseline, candidate = value.split(",", 1)
        pairs.append({"baseline": baseline, "candidate": candidate})
    aa_pairs: Optional[list[dict[str, Any]]] = None
    if args.aa_pair:
        aa_pairs = []
        for value in args.aa_pair:
            if "," not in value:
                raise BenchmarkError("--aa-pair expects first.json,second.json")
            baseline, candidate = value.split(",", 1)
            aa_pairs.append({"baseline": baseline, "candidate": candidate})
    result = summarize_paired_rounds(pairs, aa_pairs=aa_pairs)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "OK" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a workload against native SGLang /generate")
    run.add_argument("--url", required=True, help="SGLang base URL or full /generate URL")
    run.add_argument("--workload", required=True, help="workload JSON path")
    run.add_argument("--output", required=True, help="result JSON path")
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    run.add_argument("--run-id")
    run.add_argument("--fixed-work", action="store_true")
    run.add_argument("--max-workers", type=int)
    run.add_argument("--reset-cache", action="store_true")
    run.add_argument("--header", action="append", help="HTTP header, repeatable: Name: value")
    run.set_defaults(handler=_run_cli)

    compare = subparsers.add_parser("compare", help="compare exact output token IDs")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output")
    compare.set_defaults(handler=_compare_cli)

    summary = subparsers.add_parser("summarize", help="summarize paired benchmark rounds")
    summary.add_argument(
        "--pair",
        action="append",
        required=True,
        help="baseline.json,candidate.json; repeat for interleaved rounds",
    )
    summary.add_argument(
        "--aa-pair",
        action="append",
        help="A/A first.json,second.json; repeat for noise rounds",
    )
    summary.add_argument("--output")
    summary.set_defaults(handler=_summarize_cli)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (BenchmarkError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
