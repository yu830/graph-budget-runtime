# Decode fast path v2: evidence and reproduction

This branch extends the published performance commit
`1f1984147608ee6c5dd8b3423e6f41877aaadeb4`. The old 72.4% result belongs to
that version's different-source comparison and is not a v2 incremental gain.
The runtime uses SGLang v0.5.9 at
`bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2`; the authoritative current patch
and complete runtime-tree fingerprints are in
[`manifest.json`](../src/graphbudget/patches/manifest.json).

## Mechanism and contract

A uses native metadata and graph policy with the small-M GEMM off. B uses the
same source and enables the existing opt-in `upstream_42145` GEMM adaptation.
C is B plus `GRAPHBUDGET_RESIDUAL_NORM=1`. For the supported residual RMSNorm
case, C fuses the FP32 conversions/add and residual write, retains PyTorch's
original `pow(2).mean()` reduction, then fuses normalization, weight multiply
and the final cast. FP32 operation boundaries are maintained and FP fusion is
disabled. The existing GEMM's FP32 fused-bias order is unchanged.

The C route is restricted to deterministic inference, BF16 contiguous 2D
inputs/weights, 1–16 rows, hidden size 896 or 1536, epsilon 1e-6, no extra
residual add, native residual and output-cast settings, and the verified
RTX 5070 Laptop SM120 device. Unsupported cases and grad-enabled calls requiring
gradients use SGLang's native route. Invalid explicit settings are errors.
The adapter remains limited to dense Qwen2/Qwen2.5, CUDA BF16, TP=PP=DP=1,
no overlap, with native graph/eager fallback. These limits do not establish
support for other GPUs, layouts, models, training, multi-GPU or production.

Related work: [SGLang #46159](https://github.com/sgl-project/sglang/pull/46159)
by Aarne Talman proposes a single fused residual RMSNorm with its own reduction.
This branch instead keeps the original PyTorch variance reduction and checks
bitwise equality to the original path. The idea of fused residual normalization
is prior work; no claim is made that it originated here. The existing small-M
GEMM geometry is adapted from [SGLang #42145](https://github.com/sgl-project/sglang/pull/42145)
by bill-h-lin, as documented in [third-party notices](../THIRD_PARTY_NOTICES.md).

## Correctness and actual paths

The frozen implementation commit is `480fd796b6bb457352b9ac4b80b1f6c4ce22a322`.
A clean independent checkout and installed Python environment passed 96 CPU
tests. The actual patched RMSNorm dispatch passed 20 GPU contract checks,
including both supported widths, row reorder, storage offsets, special values,
graph replay and gradient fallback. This tests the declared cases; it is not a
proof for arbitrary inputs or devices.

On Qwen2.5-0.5B, a clean-install GPU smoke matched B exactly for 43 calibration,
64 primary and 34 regression requests. An independent B/C smoke also matched
eight interactive requests. A separate C graph-versus-forced-eager model check
matched 43/43 requests and recorded actual eager calls. Short Nsight Systems
node traces observed 3,024 each of the new `_add` and `_finish` activities on
0.5B and 3,528 each on 1.5B. The traces prove those kernels appeared in the
captured model paths, not a speedup.

## Formal 0.5B B/C comparison

The main fixed workload has 256 requests with 128 output tokens each, 32
client workers, 16 running requests and 4,096 KV tokens. Throughput is 32,768
successful output tokens divided by the full makespan. A and B are the same
source with v1 small-M GEMM off/on; C is B plus the residual norm switch.
Each arm used a fresh server, fixed model revision
`7ae557604adf67be50417f59c2c2f167def9a775`, BF16, deterministic Triton and
capture `[1,2,4,8,12,16]`. The 16 completed cases each ran 43 calibration,
256 primary, 34 legacy regression, eight interactive and 24 new acceptance
requests. All 5,840 planned requests completed with exact token sequences;
there were no recorded failures or timeouts in the completed cases.

| Pair | B tokens/s | C tokens/s | C relative to B |
|---|---:|---:|---:|
| 1 | 399.74 | 2,260.77 | +465.6% |
| 2 | 744.37 | 701.93 | −5.7% |
| 3 | 814.47 | 2,378.53 | +192.0% |
| 4 | 2,094.89 | 430.66 | −79.4% |
| 5 | 2,066.73 | 2,448.48 | +18.5% |

The median paired change is +18.47%, but only **3/5 pairs are positive**;
the predeclared rule requires at least 4/5 and a median above
`max(5%, |A/A change|)`. It is **not met**. The median is not an accepted
incremental gain: several successful requests waited tens of seconds before
first token and dominate full makespan. All such requests remain included.

| Noise control | Primary tokens/s | Change |
|---|---:|---:|
| A/A, before interruption | 706.26 → 674.75 | −4.46% |
| B/B, same new WSL boot before pairs | 2,078.86 → 399.74 | −80.77% |
| B/B, after pairs | 2,081.54 → 2,234.43 | +7.35% |
| B/B primary, earlier boot; second case later interrupted | 1,296.39 → 825.00 | −36.36% |

The last row's second primary workload completed and matched all 256 outputs,
but its case did not finish the later workloads or write an exit receipt.
It is retained as a workload-level observation, not a completed case. The
original run was interrupted across a WSL reboot; the two completed A/A cases
and the first B were reused. The incomplete B was rerun with a new ID. All
five B/C pairs and the new B/B controls ran after reboot. The second pre-B/B
sample is also pair 1's B; those observations are not independent samples.

| Workload | B median tokens/s | C median tokens/s | B/C median TTFT p50 | B/C median client ITL p50 |
|---|---:|---:|---:|---:|
| Primary | 814.47 | 2,260.77 | 1.016 / 0.885 s | 7.28 / 6.34 ms |
| Legacy regression | 178.51 | 179.38 | 92.8 / 83.0 ms | 7.19 / 6.04 ms |
| Interactive | 55.45 | 55.95 | 31.1 / 33.2 ms | 6.66 / 5.90 ms |
| New mixed acceptance | 276.00 | 282.86 | 84.3 / 71.3 ms | 7.21 / 5.95 ms |

The median paired relative change need not equal the ratio of the two arm
medians. Historical TTFT starts at worker-side client processing (the legacy
field is named `actual_send_at_s`), before payload creation and the HTTP call.
It excludes the earlier client queue delay and does not establish when request
bytes were sent.
client ITL is the interval between received single-token events and can include
transport and buffering, not only GPU decode. Arrival-limited workloads are not
saturated capacity measurements. Interactive TTFT p95 had a
paired median change of +28.9%, with inconsistent directions across pairs;
this does not support a general latency-improvement claim. The complete
absolute and per-pair metrics are in the [sanitized summary](../results/2026-10-28-v2/summary.json).

Across the five paired 0.5B runs, median worker-ready whole-device usage was
2,242 MiB B and 2,234 MiB C; median sampled whole-device peaks were 2,742 and
2,607 MiB respectively. Sampling at 250 ms covered startup, warmup and loads,
includes desktop activity, and is not an absolute allocator peak or VRAM-saving
claim. Temperature ranged 43–54 °C, sampled power 5.24–63.52 W and SM clock
277–2,902 MHz over those full intervals.

## 1.5B control

A separate Qwen2.5-1.5B-Instruct BF16 smoke used revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, TP=1, max running 4,
2,048 KV tokens and capture `[1,2,4]`. B and C each completed the 43+64+34+8
requests exactly, with no failures or timeouts and observed graph replay.
Worker-ready whole-device usage was about 4,292 MiB B and 4,290 MiB C;
PyTorch allocated about 3,058 MiB for each. The single screening primary was
285.87 tokens/s B and 304.97 tokens/s C. One pair is not a 1.5B performance
claim and is not combined with the 0.5B result.

## Negative results and measurement limitations

A short same-source 64-request A/B control gave +128.44% and +95.49% for the
existing v1 GEMM in AB and BA order; these are diagnostics for A→B, not a v2
increment or a remeasurement of the old published 72.4%. A narrow-output GEMM
candidate passed bounded numerical checks but had mixed screening directions
(−4.47%, +12.20%) and B/B drift of −14.35%; it was not adopted.

In the formal 0.5B run, long waits of roughly 35–68 seconds occurred before the
first token in both B and C, while subsequent token-event gaps were much
shorter. Their cause remains **UNKNOWN**. Current `urllib` configuration routes
loopback via a proxy, but the historical formal proxy setting was not recorded.
One synthetic and one real B diagnostic per route did not reproduce those
long waits, so proxy causation is unproven. No long request or slow round was
removed. Hardware counters remain unavailable (`ERR_NVGPUCTRPERM`); no
occupancy or bandwidth claim is made. Ordinary non-deterministic reference N
was not run.

A newer SGLang source snapshot at
`635430d07378e2b12381353781a23b0704969909` and related PRs were inspected.
There is existing related work on both small-M GEMM and fused residual RMSNorm.
No sufficiently distinct, validated upstream slice was established, so no
separate upstream PR is proposed merely to create one.

## Reproduce

See the [v2 environment recipe](../scripts/environment/README.md#isolated-v2-profile)
for the exact tested dependency profile and clean installation. With that
profile, the pinned patched SGLang checkout and a local snapshot in place:

```bash
export PYTHONPATH="$SGLANG_SOURCE/python"
export GRAPHBUDGET_SHARED_GPU_LOCK=/absolute/path/to/a/shared-gpu.lock
export GRAPHBUDGET_MODEL_PATH="$MODEL_PATH"
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0

# GPU operator contract, including actual patched dispatch and fallback
"$GB_PYTHON" -P -s scripts/check_residual_norm.py --output work/norm-check

# Two fresh services and exact output checks; excludes the unseen acceptance set
"$GB_PYTHON" -P -s scripts/run_paired_decode.py --smoke \
  --model-path "$MODEL_PATH" --gpu-lock "$GRAPHBUDGET_SHARED_GPU_LOCK" \
  --output work/decode-smoke

# Full predeclared A/A, B/B, five interleaved B/C pairs, and final B/B
"$GB_PYTHON" -P -s scripts/run_paired_decode.py \
  --model-path "$MODEL_PATH" --gpu-lock "$GRAPHBUDGET_SHARED_GPU_LOCK" \
  --output work/decode-formal
```

For isolated local benchmarking, configure the client environment to bypass
any proxy for `127.0.0.1` and `localhost`. The formal run above did not record
its proxy environment, and this setting change is not a remeasurement.

The full runner freezes input hashes, model identity, source and sequence;
compares every completed token sequence; retains slow and failed cases; and
records throughput, TTFT, client-observed ITL, request completion, resources
and A/A/B/B controls. The 256×128 primary has a client backlog, not a production
open-arrival claim. Legacy and new burst workloads are arrival-limited; their
throughput is not saturated serving capacity. An operator-only result or a
successful smoke is not an end-to-end performance acceptance.
