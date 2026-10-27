# Decode performance experiment — 2026-10-27

The small-batch GEMM adaptation met the **predeclared primary-load rule** on
one laptop GPU. The metadata candidates did not establish a stable end-to-end
gain. Both remain opt-in experiments; this is not multi-GPU or production-load
acceptance. The original cost-policy result and its default-mode 2/43 token
mismatches remain unchanged in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Configuration and attribution

- Baseline: GraphBudget `c045569a6c295601d5124bc089202c9785e1b013` with its
  original five-file SGLang patch; candidate runtime:
  `c62c7432e6e8e0bed43ead82b4d2f42ede2ac855`, SGLang v0.5.9 base
  `bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2`, patch SHA-256
  `d14d6349aec82c82ee96b2d3ce1798ee5e319c337e26dff8dfa7b192afe410c5`.
- RTX 5070 Laptop GPU, WSL, Torch 2.9.1+cu128, Triton 3.5.1,
  Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, BF16, TP=1,
  deterministic Triton, overlap disabled, greedy output, capture sizes
  `[1,2,4,8,12,16]`, 4,096 actual KV tokens and 16 running requests.
- The selected `GRAPHBUDGET_SMALL_M_GEMM=upstream_42145` adapts the small-M
  geometry of the open, unmerged
  [SGLang #42145](https://github.com/sgl-project/sglang/pull/42145): M16,
  N64, K128 and one CTA per output tile. It reuses the pinned persistent
  Triton kernel and preserves that baseline's FP32 fused-bias semantics.
  It is not a new GEMM algorithm or an upstream acceptance claim.
- The final loaded GEMM source passed 340 checked cases across M=1–17 and
  five model-relevant shapes; 170 used an independent exact CPU oracle, and
  all 340 matched the native output bit for bit. Another 255 checks reused
  rows across M=1–17, reordered them and used random bias. These checks do not
  establish correctness for arbitrary shapes or devices.
- All measured arms used the same hash-verified model copy, task-local Triton
  cache and disabled-mmap loading. The model and dependency identities match;
  failures before these common conditions are retained as diagnosis.

## Unprofiled request comparison

Five paired fresh-server rounds ran in AB, BA, AB, BA, AB order. Each primary
round comprised 256 fixed requests with 128 output tokens each; the client
allowed 32 workers while the server admitted at most 16 running requests.
The holdout comprised six arrival bursts and 34 requests with mixed output
lengths. There were no failed or timed-out requests in the ten paired arms;
all 43 calibration, 256 primary and 34 holdout outputs in each arm matched
the fixed reference exactly.

| Primary pair | Original tokens/s | Candidate tokens/s | Paired change |
|---|---:|---:|---:|
| 1 | 1,164.15 | 2,007.24 | +72.4% |
| 2 | 1,096.61 | 1,783.15 | +62.6% |
| 3 | 1,122.35 | 2,087.16 | +86.0% |
| 4 | 1,128.60 | 1,933.98 | +71.4% |
| 5 | 1,135.00 | 2,023.29 | +78.3% |

The median of the five **paired relative changes** is **+72.4%**; all five
are positive. Before/after A/A throughput changes were +7.9% and +8.8%,
so the observed median exceeds the predeclared max(5%, A/A) threshold.
The range of original paired throughput was 1,096.61–1,164.15 tokens/s.
The A/A endpoints ranged from 1,049.50 to 1,274.96 tokens/s; the two A/A
pairs are a limited noise estimate.

| Metric | Median original | Median candidate | Median paired change |
|---|---:|---:|---:|
| Primary effective output throughput | 1,128.60 tokens/s | 2,007.24 tokens/s | +72.4% |
| Primary TTFT p50 | 2.021 s | 1.060 s | −47.1% |
| Primary TTFT p95 | 2.396 s | 1.190 s | −50.4% |
| Primary client-observed token ITL p50 | 14.96 ms | 7.56 ms | −48.6% |
| Holdout effective output throughput | 168.91 tokens/s | 178.75 tokens/s | +5.7% |
| Holdout TTFT p50 | 75.15 ms | 91.32 ms | +21.5% |
| Holdout TTFT p95 | 214.02 ms | 137.48 ms | −35.4% |
| Holdout client-observed token ITL p50 | 14.70 ms | 7.01 ms | −51.9% |

Median paired changes are not necessarily the ratio of the two displayed
medians. The primary is a fixed-workload result with client backlog, not an
open-ended production throughput claim. Each reported ITL event carried
exactly one token; it is observed at the client and can include transport and
buffering effects, not a pure GPU decode interval.

Holdout TTFT is a material limitation. Its p50 paired changes were
`+21.5%, −16.9%, +58.3%, −35.3%, +38.5%`; the median indicates a possible
regression, while the direction changes across rounds. Holdout p95 changes
also changed sign and the two A/A p95 differences reached 84.9%. A separate
single screening run had p95 TTFT 1.016 s versus 0.134 s on the original
screen; its cause was not established. These observations do not support a
general TTFT or tail-latency improvement. The arrival-limited holdout
throughput should not be interpreted as a saturation benchmark.

## Metadata candidates and retained slow case

Two balanced blocks compared the original, patched native, preallocated,
fused and prefix-only paths with the same 256-by-128 primary and holdout.
Every output matched exactly; no request failed or timed out.

| Mode | Block 1 tokens/s | Block 2 tokens/s | Changes versus that block's original |
|---|---:|---:|---:|
| Original | 1,188.57 | 1,190.19 | — |
| Patched native | 594.89 | 1,106.43 | −49.9%, −7.0% |
| Preallocated | 1,197.92 | 1,178.19 | +0.8%, −1.0% |
| Fused prefix and splits | 1,148.66 | 1,117.66 | −3.4%, −6.1% |
| Prefix-only ablation | 1,115.30 | 1,176.62 | −6.2%, −1.1% |

In block 1 patched-native request `primary-159` had a 35.73 s TTFT and
completed successfully; it remains in the throughput. Its cause is unknown.
Two blocks and this straggler do not establish a stable metadata gain or
attribute the straggler to a code path. The independent integer oracle matched
168 supported metadata cases, including padding, boundaries, reordered pool
indices and nonzero sequence storage offsets. A non-contiguous sequence case
also exposed the original gather's unsupported layout and is retained as a
failed boundary experiment.

## Diagnostic decomposition and actual paths

For the native batch-4 worker, 126 diagnostic records gave the following
median spans. CPU values are host submission offsets; CUDA Event values can
include host gaps and measurement overhead. Do not sum them as critical-path
percentages. Sampling and output processing were not isolated.

| Diagnostic span | CPU offset | Same-stream event span |
|---|---:|---:|
| Input copy to metadata start | 161.3 µs | 153.7 µs |
| Metadata preparation | 464.9 µs | 454.6 µs |
| Graph launch interval | 57.8 µs | 8,725.1 µs |
| Observed forward interval | 808.2 µs | 9,546.5 µs |
| Sampling / output | UNKNOWN | UNKNOWN |

Short Nsight Systems node traces observed 63 `cudaGraphLaunch` calls in
each compared worker. Native had no `_prefix_splits` kernel; fused had
63 while retaining the original KV gather. In the GEMM adaptation trace,
6,113 of 6,305 `matmul_kernel_persistent` activities used block size 128
and 192 used block size 256; native used block size 256 for all 6,305.
Node tracing is diagnostic and its kernel times are not the formal result.

## Resources and limitations

Across the five primary pairs, both arms had median startup snapshots of
2,242.44 MiB whole-device usage and 1,023.21 MiB PyTorch allocated, with
4,096 KV tokens. The median of sampled whole-device peaks was 2,856 MiB for
original and 2,886 MiB for candidate; samples include desktop activity and
are not absolute allocation peaks. These data do not establish VRAM savings.
Recorded temperature was 42–60 °C, power 4.79–69.2 W and SM clock
277–2,902 MHz across all samples, including idle/startup periods.

Median HTTP-ready was 24.86 s original and 56.27 s candidate; the original
SGLang tree was on ext4 and the candidate tree on a Windows mount, so source
identity scanning and startup I/O were not matched. This difference cannot be
attributed to the kernel. Median graph capture was 1.26 s and 1.81 s
respectively. The single empty-cache versus cached original starts separately
recorded HTTP-ready 78.52/62.91 s and capture 16.16/6.43 s; neither is an
independent startup-speedup claim.

Nsight Compute first failed to load its counter library from the isolated
tool path; after an ASCII-path diagnostic it returned
`ERR_NVGPUCTRPERM`. No hardware counters, occupancy or bandwidth result is
claimed. No driver or permission was changed. A WSL restart, a SIGBUS during
weight loading, a Triton cache JSON error and a DrvFS `ENOMEM` during model
identity hashing are retained separately; none is a candidate performance
result.

The compact [formal summary](../results/2026-10-27-performance/formal07-summary.json)
and [metadata comparison](../results/2026-10-27-performance/h1-paired-summary.json)
contain the derived numbers. Native trace files, full request logs, failed
runs and machine paths remain in the private evidence delivery.

## Reproduction

Use the pinned runtime overlays and apply the packaged patch as described in
the [README](../README.md). Set `GRAPHBUDGET_MODEL_PATH`,
`GRAPHBUDGET_SHARED_GPU_LOCK`, `GRAPHBUDGET_BASELINE_ROOT`,
`GRAPHBUDGET_BASELINE_SGLANG_SOURCE` and a task-local
`TRITON_CACHE_DIR`; the environment setup also requires the overlay and
candidate `SGLANG_SOURCE` variables. Then, for example:

```bash
bash scripts/environment/with_runtime.sh scripts/run_metadata_case.py \
  --mode original --rounds 1 --output work/repro/original
bash scripts/environment/with_runtime.sh scripts/run_metadata_case.py \
  --mode upstream_gemm --rounds 1 --output work/repro/upstream-gemm \
  --reference work/repro/original/correctness.json
for workload in primary-0 holdout-0; do
  bash scripts/environment/with_runtime.sh -m graphbudget.benchmark compare \
    --baseline "work/repro/original/${workload}.json" \
    --candidate "work/repro/upstream-gemm/${workload}.json" \
    --output "work/repro/${workload}-exact.json"
done
```

The original predeclared sequence, exact request comparisons, per-run hashes,
telemetry, raw outputs and interruption records are retained privately.
Re-running a single pair does not reproduce the five-pair acceptance result.
