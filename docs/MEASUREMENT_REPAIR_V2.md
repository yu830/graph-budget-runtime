# Decode v2: controlled local measurement

This protocol is a new experiment, not a correction to the historical raw
results. The 2026-10-28 v2 comparison remains inconclusive. Its legacy
`actual_send_at_s` timestamp is taken before payload creation and the HTTP
call; it does not establish when bytes were sent. A client/server duration
comparison does not identify a historical proxy or network cause.

## Transport and phase boundaries

The benchmark supports `--transport direct` and `--transport inherit`.
Direct uses a private urllib opener with an empty proxy configuration; it does
not change system proxy settings. In automatic mode only `127.0.0.1`, `::1`
and `localhost` default to direct; other hosts retain inherited behavior.
The case runner uses the same mode for health, generation and cache flush.
Each run records only proxy-environment presence, configuration and the observed
peer category, never proxy values. A peer observation can be UNKNOWN.

The optional `--request-diagnostics` flag records client call, DNS,
connection, completed request write, response headers, SSE, token and EOF
timestamps. The server writes request-level ASGI entry, complete body,
TokenizerManager entry, scheduler dispatch return, first output and completion
events. Scheduler dispatch is an IPC send boundary, not proof of scheduler
queue admission. Both sides record the same Linux boot ID and use the host's
monotonic perf-counter clock; wall-clock values are not subtracted from it.
The hooks and request-level writes are diagnostic only and are disabled during
formal measurement. Missing stages must be reported as UNKNOWN.

## Frozen experiment

The paired runner freezes the Git tree, model, SGLang manifest, boot ID,
transport settings, workloads, arm switches and full execution order before
starting. It first runs two fresh-server B cases that are separate from every
formal B/C pair. A difference greater than 5% in the complete primary
throughput stops the schedule before A/A or B/C. If the control passes, the
runner executes A/A, five alternating B/C pairs, and a final B/B control.
Each case includes calibration, the primary workload, legacy holdout,
interactive and acceptance workloads, with exact-token checks.

The primary metric remains all successful output tokens divided by the
complete fixed-work makespan for 256 requests of 128 tokens each. Slow
requests are retained. The screening rule requires at least four of five B/C
pairs to improve, and a median improvement strictly above both 5% and the
same-boot A/A noise; B/B and drift can still require review. These thresholds
are engineering screening criteria, not a claim of statistical significance.

Run with the environment documented in `scripts/environment/README.md`, a
verified model snapshot and an existing shared GPU lock:

```bash
"$GB_PYTHON" -P -s scripts/run_paired_decode.py \
  --model-path "$MODEL_PATH" --gpu-lock "$GRAPHBUDGET_SHARED_GPU_LOCK" \
  --transport direct --output work/decode-v2-controlled
```

The runner requires a clean source tree and saves the protocol before running.
If the independent B/B control fails, preserve its data and investigate;
do not present the unstarted B/C pairs as results.
