# GraphBudget: Measured CUDA-Graph Planning and Conservative Dispatch

## Executive status

GraphBudget is an opt-in runtime adapter for dense Qwen2.5 BF16 decode on one
RTX 5070 Laptop GPU, TP=1, using SGLang v0.5.9. It adds a bounded,
whole-configuration capture planner, identity-bound startup enforcement, a
CPU-only reject-only graph/eager policy, and request-bound path evidence. It
does not add an allocator, online recapture, KV resizing, admission control, or
a custom CUDA/Triton kernel.

| Claim | Current status | Evidence boundary |
|---|---|---|
| CPU implementation and tests | PASS | Last suite: 55 tests; clean Windows install and patch replay also passed. |
| Patched-source binding | PASS | Base `bbe9c7e`; five-file patch SHA `3146afe...`; source-tree identity is checked at startup. |
| Deterministic graph/eager correctness probe | PASS | 43/43 requests and 520 output tokens matched in `work/determinism-probe-02`. |
| Management eager instrumentation | PASS | 43/43 requests, 520 output tokens, 82 timed eager steps in `work/management-eager-probe-01`. |
| Default-mode strict exact-token route | FAILED / NOT ADOPTED | The default-mode calibration retained 2/43 graph/eager token differences and 2/43 native-vs-native cadence differences. |
| Full deterministic suite | IN_PROGRESS | `work/core-deterministic/` is still in measurement; final select/profile/validation/performance artifacts are not complete. |
| Cost-policy adoption | NOT ESTABLISHED | No relaxed tolerance or profile acceptance was derived from the default-mode mismatch evidence. |
| End-to-end performance gain | NOT ESTABLISHED | No completed deterministic five-round T/C result is available yet. |

The status table is intentionally claim-specific. A passing CPU suite, graph
capture, or forced eager probe is not an end-to-end optimization result.

## Problem and mechanism

Dynamic decode batches can cross the captured-batch ceiling and fall back to
eager execution. Capturing more shapes may increase graph-pool/static-buffer
cost and reduce KV capacity. The upstream SGLang mechanisms remain the safety
authority: capture buckets, ceil padding, shared graph-pool behavior, and the
existing overflow fallback are credited to SGLang rather than claimed as new
work.

GraphBudget measures each candidate as a complete configuration in a fresh
process. The plan records capture sizes, startup/capture time, KV capacity,
same-sample device total/used/free memory, runtime peak usage, and raw evidence.
It rejects missing or synthetic evidence, stale fingerprints, mismatched
capture sets, and configurations outside the device/KV/startup constraints.
The planner keeps rejection reasons and a Pareto set; it never sums per-graph
memory as a knapsack model.

At startup the selected capture set and allow-listed server arguments are
applied before SGLang's memory sizing and graph construction. The runtime hook
runs after upstream graph eligibility and before graph/eager preparation. It
uses only CPU metadata `(batch_size, padded_size, exact_context_length)` and
can only reject an eligible graph. Missing, stale, heterogeneous, noisy, or
insufficient policy data returns to native dispatch. Actual `selected_path`
and `used_path` are recorded separately.

## Source and environment

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, BF16, tokenizer IDs frozen from
  the local snapshot.
- Runtime: Torch `2.9.1+cu128`, CUDA runtime `12.8`, Triton `3.5.1`, SGLang
  `0.5.9`, `sgl-kernel 0.3.21`, FlashInfer `0.6.3`; GPU driver `617.42`.
- Upstream base: SGLang commit
  `bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2`. The final patch binds five
  upstream files and SHA-256 `3146afe85bcd1e687dcdd3b1ba656a70687dabacfbed3301e4579205962fecec`.
- Deterministic lane: upstream deterministic inference with
  `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`, selecting the existing
  upstream Triton batch-invariant path on SM120. The first deterministic probe
  failed because DeepGEMM rejected the architecture; that failure is retained
  rather than converted into a software claim.

The v0.5.9 lane allocates KV before graph capture. GraphBudget records and
reuses that pre-capture behavior; it does not attribute the research SHA's
post-capture KV mechanism to this implementation. Environment and source
receipts are in [work/environment/ENVIRONMENT_REPORT.md](../work/environment/ENVIRONMENT_REPORT.md)
and [src/graphbudget/patches/manifest.json](../src/graphbudget/patches/manifest.json).

## Observed results and limitations

The deterministic probe-02 graph/native and forced-eager runs completed all 43
requests with identical 520-token outputs. The timed path receipts observed
80 graph steps in the native run and 48 eager steps in the forced-eager run;
these are scheduler steps, not token counts or a speedup claim. The management
probe independently exercised the eager branch with 82 timed eager steps and
43/43 completed requests.

The default-mode profile evidence does not support a cost policy: graph/eager
outputs differed for 2 of 43 requests, and repeated native-graph observations
also differed for 2 of 43. No tolerance was relaxed and no profile was accepted
from those runs. The forced-eager probe is a functional/calibration control,
not evidence that eager dispatch improves production latency.

Whole-device NVML peaks include unrelated desktop/WSL activity and vary across
runs; they are retained as sampled resource observations, not causal proof of
graph memory cost. An intermittent WSL service failure and an ENOMEM read path
were treated as host/DrvFS environment failures, not CUDA OOM diagnoses.

## Reproduction and next boundary

CPU checks:

```bash
python -m unittest discover -s tests -v
```

The bounded deterministic lane is launched through the pinned environment
wrapper, for example:

```bash
bash scripts/environment/with_runtime.sh scripts/run_core.py \
  --model-path "$MODEL_PATH" --revision "$MODEL_REVISION" \
  --backend triton --sglang-checkout "$SGLANG_SOURCE" \
  --deterministic --output work/core-deterministic --phase all
```

The current full lane remains **IN_PROGRESS**. Until its plan, profile,
validation, A/A, and paired performance artifacts are complete, there is no
adoption decision, no causal performance gain, and no claim that the optional
cost policy is useful. Local recovery evidence under `work/` is retained for
audit and is not presented as committed public raw data.

See the [design](DESIGN.md), [benchmark protocol](BENCHMARK_PROTOCOL.md),
[source index](SOURCE_INDEX.md), and [project status](STATUS.md) for the
original contracts and verification boundaries.
