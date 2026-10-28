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

## Observed results on 2026-10-28

The local target/proxy-sentinel tests exercised both actual HTTP endpoints.
The opt-in model diagnostic completed 145 requests, with all observed peers
classified as the target loopback port; its 43 calibration outputs matched
the prior B reference exactly. That diagnostic used the precommit working tree
(loaded GraphBudget fingerprint `dfb35c1628aa5da5c458a9d19d45db4ec1e0d63fcfeb559a0d9cd1696570c220`),
not the later frozen measurement commit. Subsequent edits recorded cache-reset
peers and omitted an unused timing call when diagnostics were disabled.
It did not reproduce the historical 35–68 s
wait. Five calibration requests spent approximately 3.18 s between the ASGI
body-complete and response-start markers, before TokenizerManager entry. This
locates those shorter waits, but does not establish their cause or the cause of
the historical waits.

Two independent B/B controls were attempted on the same WSL boot:

| Attempt | First B tokens/s | Second B tokens/s | Change | Result |
|---|---:|---:|---:|---|
| Initial | 2,038.03 | 2,002.48 | −1.74% | Gate passed |
| After CUDA recovery smoke | 2,267.09 | 2,085.28 | −8.02% | Gate failed |

After the first gate passed, the first A case completed. The second A case
failed when the scheduler reported `CUDA error: unknown error`: 144 of 256
primary requests completed and 112 failed. A subsequent bounded CUDA matrix
smoke passed; the cause of the earlier error remains UNKNOWN. The recovery
plan allowed one fresh attempt with another independent B/B, rather than
carrying a pre-error control into a post-error comparison.

In the second B/B, all 365 requests per B completed and their token sequences
matched across calibration, primary, holdout, interactive and acceptance
workloads. No 35–68 s outlier appeared: the maximum primary TTFT was about
1.02 s and 1.08 s. Median server-reported request duration rose from 1.786 s
to 1.937 s; sampled whole-device clocks and utilization also differed, but
those samples do not prove a cause. The second gate therefore stopped the run
before A/A and all five B/C pairs.

The change provides a controlled local transport and preserves complete failure
evidence, but does **not** establish an incremental B-to-C performance gain.
C remains opt-in; B remains the recommendation. No new 1.5B performance result
was run. The earlier inconclusive v2 data remains unchanged. A clean checkout
of the measured source passed 36 affected CPU checks and exposed the paired
runner; the clean-checkout B/C GPU run was not executed after the control
failed. The measured code revision is
`5d70f9af6d1bd5237d9417452d028d2943992f15`; a later documentation commit
may record these results without changing the measured code.

See the [machine-readable summary](../results/2026-10-28-measurement-repair/summary.json).
