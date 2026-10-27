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

## Current evidence before formal evaluation

- The same-source 64-request control used separate fresh servers. A/A changed
  +3.66%; AB and BA gave +128.44% and +95.49% for B over A respectively.
  All calibration, primary and regression token outputs matched exactly.
  These short controls establish neither a formal v2 benefit nor a new 72.4%.
- The first candidate, a narrow-output GEMM dispatch, passed its bounded
  numerical tests but produced screening changes −4.47% and +12.20%, while
  B/B changed −14.35%. It is not adopted; its patch and raw results are retained
  in the private evidence.
- The residual RMSNorm prototype and actual patched layer passed bitwise
  comparisons, bounded shapes and row reorder, special-value checks,
  real norm weights, multiple graph replays and gradient fallback. A short
  Nsight Systems graph-node trace observed 3,024 `_add` and 3,024 `_finish`
  activities in the actual model path. Profiling is diagnostic, not the
  unprofiled performance result.
- A 256-request screening series had exact tokens throughout, but a B request
  took 35.54 seconds to first token and the machine also reported a WSL service
  timeout around the series. Its cause is unknown and the slow request remains
  included. One paired screening change was +24.43%; the other is distorted by
  that B outlier. These figures are not an accepted B→C gain.
- A separately installed Python environment and an independently patched
  SGLang checkout completed a B/C model smoke: 43 calibration, 64 primary,
  34 regression and eight interactive requests per arm, no failures/timeouts,
  exact output tokens and observed graph replay. This preview used an
  uncommitted GraphBudget checkout; the clean-commit replay is recorded
  separately below.

Formal five-pair B/C performance, the new acceptance workload, clean-commit GPU
replay, and the 1.5B model are **NOT RUN** until corresponding receipts are
available. Do not infer a result from the existence of this document.

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

The full runner freezes input hashes, model identity, source and sequence;
compares every completed token sequence; retains slow and failed cases; and
records throughput, TTFT, client-observed ITL, request completion, resources
and A/A/B/B controls. The 256×128 primary has a client backlog, not a production
open-arrival claim. Legacy and new burst workloads are arrival-limited; their
throughput is not saturated serving capacity. An operator-only result or a
successful smoke is not an end-to-end performance acceptance.
