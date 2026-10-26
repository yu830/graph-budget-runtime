"""Synthetic HTTP/SSE tests for the GraphBudget benchmark harness.

These tests deliberately exercise a local stdlib HTTP fixture only.  They do
not load a model, require CUDA, or claim SGLang/GPU validation; the real-server
contract is covered by the request JSON and raw SSE shapes in the fixture.
"""

from __future__ import annotations

import copy
import io
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from graphbudget.benchmark import (
    BenchmarkError,
    RequestResult,
    SSEEvent,
    _process_stream_events,
    compare_exact_tokens,
    load_workload,
    parse_sse_stream,
    run_benchmark,
    summarize_paired_rounds,
)


class _SyntheticSSEHandler(BaseHTTPRequestHandler):
    """A deterministic native-/generate-shaped fixture, not a GPU server."""

    protocol_version = "HTTP/1.1"
    requests: list[dict[str, Any]] = []
    lock = threading.Lock()

    def log_message(self, *_args: Any) -> None:  # pragma: no cover - silence fixture noise
        return

    def _json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_text(self, status: int, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/flush_cache":
            self._send_text(200, "Cache flushed.\n")
            return
        if self.path != "/generate":
            self._send_text(404, "not found")
            return
        body = self._json_body()
        with self.lock:
            self.requests.append(body)
        rid = str(body["rid"])
        if "slow" in rid:
            time.sleep(0.15)
        base = 100 + (sum(body["input_ids"]) % 10)
        multi = "multi" in rid
        target = int(body["sampling_params"]["max_new_tokens"])
        all_ids = [base + offset for offset in range(target)]
        first_count = min(2, target) if multi else 1
        events = [
            {
                "output_ids": all_ids[:first_count],
                "meta_info": {
                    "id": rid,
                    "finish_reason": None,
                    "prompt_tokens": len(body["input_ids"]),
                    "completion_tokens": first_count,
                },
            },
        ]
        for token_index in range(first_count, target):
            events.append(
                {
                    "output_ids": [all_ids[token_index]],
                    "meta_info": {
                        "id": rid,
                        "finish_reason": None,
                        "prompt_tokens": len(body["input_ids"]),
                        "completion_tokens": token_index + 1,
                    },
                }
            )
        events.append(
            {
                "output_ids": [],
                "meta_info": {
                    "id": rid,
                    "finish_reason": {"type": "length", "length": target},
                    "prompt_tokens": len(body["input_ids"]),
                    "completion_tokens": target,
                },
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in events:
            wire = "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"
            try:
                self.wfile.write(wire.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            time.sleep(0.002)
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


class BenchmarkHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _SyntheticSSEHandler.requests = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _SyntheticSSEHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def workload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workload_id": "synthetic-heldout",
            "purpose": "heldout",
            "token_ids_note": "synthetic fixture IDs; not tokenizer validation",
            "requests": [
                {
                    "request_id": "normal-00",
                    "input_ids": [1, 2],
                    "max_new_tokens": 2,
                    "planned_at_s": 0.0,
                },
                {
                    "request_id": "multi-boundary-01",
                    "input_ids": [3, 4, 5],
                    "max_new_tokens": 2,
                    "planned_at_s": 0.01,
                },
            ],
        }

    def test_sse_parser_retains_raw_events_and_done(self) -> None:
        fixture = (
            b": keep this raw comment\n"
            b"data: {\"output_ids\":[7],\n"
            b"data: \"meta_info\":{\"id\":\"r\"}}\n\n"
            b"data: [DONE]\n\n"
        )
        events = list(parse_sse_stream(io.BytesIO(fixture), time_origin=0.0))
        self.assertEqual([event.kind for event in events], ["response", "done"])
        self.assertIn(": keep this raw comment", events[0].raw)
        self.assertEqual(events[0].payload["output_ids"], [7])
        self.assertEqual(events[1].data, "[DONE]")

    def _process_raw_fixture(
        self,
        payloads: list[dict[str, Any]],
        *,
        expected_id: str = "expected",
        expected_max: int = 2,
    ) -> RequestResult:
        wire = b"".join(
            b"data: "
            + json.dumps(payload, separators=(",", ":")).encode("utf-8")
            + b"\n\n"
            for payload in payloads
        ) + b"data: [DONE]\n\n"
        events = list(parse_sse_stream(io.BytesIO(wire), time_origin=0.0))
        result = RequestResult(
            workload_request_id="fixture",
            client_request_id=expected_id,
            planned_at_s=0.0,
            actual_send_at_s=0.0,
            first_token_at_s=None,
            completion_at_s=None,
            status="error",
        )
        _process_stream_events(
            events,
            result=result,
            expected_client_request_id=expected_id,
            expected_max_new_tokens=expected_max,
        )
        return result

    @staticmethod
    def _response(
        output_ids: list[int],
        completion_tokens: Any,
        *,
        request_id: str = "expected",
        finish_reason: Any = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "id": request_id,
            "finish_reason": finish_reason,
        }
        if completion_tokens is not None:
            meta["completion_tokens"] = completion_tokens
        return {"output_ids": output_ids, "meta_info": meta}

    def test_repeated_token_ids_are_exact_for_delta_and_cumulative_chunks(self) -> None:
        final = {"type": "length", "length": 2}
        delta = self._process_raw_fixture(
            [
                self._response([7], 1),
                self._response([7], 2),
                self._response([], 2, finish_reason=final),
            ]
        )
        cumulative = self._process_raw_fixture(
            [
                self._response([7], 1),
                self._response([7, 7], 2),
                self._response([], 2, finish_reason=final),
            ]
        )
        self.assertEqual(delta.status, "success")
        self.assertEqual(cumulative.status, "success")
        self.assertEqual(delta.output_ids, [7, 7])
        self.assertEqual(cumulative.output_ids, [7, 7])

    def test_inconsistent_or_missing_completion_counts_fail_closed(self) -> None:
        contradictory = self._process_raw_fixture(
            [
                self._response([5], 1),
                self._response([6], 3),
                self._response([], 3, finish_reason={"type": "length", "length": 2}),
            ]
        )
        missing = self._process_raw_fixture(
            [
                self._response([5], None),
                self._response([6], 2),
                self._response([], 2, finish_reason={"type": "length", "length": 2}),
            ]
        )
        self.assertEqual(contradictory.status, "error")
        self.assertIn("inconsistent", contradictory.error or "")
        self.assertEqual(contradictory.token_ids_confidence, "protocol_error")
        self.assertEqual(missing.status, "error")
        self.assertIn("completion_tokens", missing.error or "")
        self.assertEqual(missing.token_ids_confidence, "protocol_error")

    def test_request_id_truncation_and_finish_reason_are_protocol_errors(self) -> None:
        wrong_id = self._process_raw_fixture(
            [
                self._response([1], 1, request_id="other"),
                self._response([2], 2, request_id="other"),
                self._response(
                    [], 2, request_id="other", finish_reason={"type": "length", "length": 2}
                ),
            ]
        )
        truncated = self._process_raw_fixture(
            [
                self._response([1], 1),
                self._response([], 1, finish_reason={"type": "length", "length": 2}),
            ]
        )
        wrong_finish = self._process_raw_fixture(
            [
                self._response([1], 1),
                self._response([2], 2),
                self._response([], 2, finish_reason={"type": "stop"}),
            ]
        )
        self.assertEqual(wrong_id.status, "error")
        self.assertIn("did not match", wrong_id.error or "")
        self.assertEqual(truncated.status, "error")
        self.assertIn("observed 1 output IDs", truncated.error or "")
        self.assertEqual(wrong_finish.status, "error")
        self.assertIn("type", wrong_finish.error or "")

    def test_multi_token_chunk_does_not_produce_token_itl(self) -> None:
        result = self._process_raw_fixture(
            [
                self._response([1, 2], 2),
                self._response([3], 3),
                self._response([], 3, finish_reason={"type": "length", "length": 3}),
            ],
            expected_max=3,
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.token_event_counts, [2, 1])
        self.assertIsNone(result.metrics()["token_itl_s"])

    def test_open_loop_sends_all_and_preserves_ids_and_metrics(self) -> None:
        result = run_benchmark(
            self.url,
            self.workload(),
            run_id="synthetic-run",
            timeout_s=2.0,
            reset_cache=True,
        )
        self.assertEqual(result["run"]["schedule_mode"], "open_loop")
        self.assertEqual(result["summary"]["planned_requests"], 2)
        self.assertEqual(result["summary"]["actual_send_requests"], 2)
        self.assertEqual(result["summary"]["completed_requests"], 2)
        self.assertEqual(result["summary"]["failed_requests"], 0)
        self.assertIsNone(result["summary"]["ttft_s"]["p99"])
        self.assertEqual(result["cache_reset"]["status"], "success")
        records = {item["workload_request_id"]: item for item in result["requests"]}
        self.assertEqual(set(records), {"normal-00", "multi-boundary-01"})
        for stable_id, record in records.items():
            self.assertEqual(record["status"], "success")
            self.assertEqual(
                record["client_request_id"], f"graphbudget/synthetic-run/{stable_id}"
            )
            self.assertEqual(record["server_request_id"], record["client_request_id"])
            self.assertEqual(record["output_ids"][-1] - record["output_ids"][0], 1)
            self.assertTrue(record["events"])
        multi = records["multi-boundary-01"]
        self.assertEqual(multi["token_event_counts"][0], 2)
        self.assertIsNone(multi["metrics"]["token_itl_s"])
        sent_payload = next(
            item for item in _SyntheticSSEHandler.requests if item["rid"].endswith("normal-00")
        )
        self.assertEqual(sent_payload["input_ids"], [1, 2])
        self.assertEqual(sent_payload["sampling_params"]["temperature"], 0.0)
        self.assertTrue(sent_payload["sampling_params"]["ignore_eos"])
        self.assertTrue(sent_payload["stream"])

    def test_exact_comparison_uses_stable_identity(self) -> None:
        baseline = run_benchmark(self.url, self.workload(), run_id="base", timeout_s=2.0)
        candidate = run_benchmark(self.url, self.workload(), run_id="candidate", timeout_s=2.0)
        equivalent = compare_exact_tokens(baseline, candidate)
        self.assertTrue(equivalent["equivalent"])
        changed = copy.deepcopy(candidate)
        changed["requests"][0]["output_ids"][0] += 1
        mismatch = compare_exact_tokens(baseline, changed)
        self.assertFalse(mismatch["equivalent"])
        self.assertEqual(mismatch["mismatches"][0]["workload_request_id"], "normal-00")

    def test_fixed_work_keeps_timeout_record_and_does_not_drop_planned_request(self) -> None:
        workload = self.workload()
        workload["requests"][0]["request_id"] = "slow-00"
        result = run_benchmark(
            self.url,
            workload,
            run_id="fixed-timeout",
            timeout_s=0.03,
            fixed_work=True,
            max_workers=1,
        )
        records = {item["workload_request_id"]: item for item in result["requests"]}
        self.assertEqual(result["run"]["schedule_mode"], "fixed_work")
        self.assertEqual(result["summary"]["planned_requests"], 2)
        self.assertEqual(result["summary"]["actual_send_requests"], 2)
        self.assertEqual(records["slow-00"]["status"], "timeout")
        self.assertEqual(records["multi-boundary-01"]["status"], "success")
        self.assertEqual(result["summary"]["timed_out_requests"], 1)
        self.assertGreaterEqual(result["summary"]["max_client_backlog"], 1)

    def test_paired_summary_requires_five_rounds_and_reports_aa_state(self) -> None:
        baseline = run_benchmark(self.url, self.workload(), run_id="summary-base", timeout_s=2.0)
        candidate = run_benchmark(self.url, self.workload(), run_id="summary-candidate", timeout_s=2.0)
        short = summarize_paired_rounds(
            [{"baseline": baseline, "candidate": candidate}]
        )
        self.assertEqual(short["status"], "INSUFFICIENT_ROUNDS")
        self.assertEqual(short["aa_noise"]["status"], "NOT_PROVIDED")
        pairs = [
            {"baseline": baseline, "candidate": candidate} for _ in range(5)
        ]
        full = summarize_paired_rounds(pairs, aa_pairs=pairs)
        self.assertEqual(full["status"], "OK")
        self.assertEqual(full["aa_noise"]["status"], "OK")

    def test_workload_validation_rejects_duplicate_or_unsorted_ids(self) -> None:
        duplicate = self.workload()
        duplicate["requests"][1]["request_id"] = duplicate["requests"][0]["request_id"]
        with self.assertRaises(BenchmarkError):
            load_workload(duplicate)
        unsorted = self.workload()
        unsorted["requests"][1]["planned_at_s"] = -1
        with self.assertRaises(BenchmarkError):
            load_workload(unsorted)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
