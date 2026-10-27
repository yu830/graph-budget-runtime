# 2026-10-27 fixed experiment protocol

Frozen before the unprofiled model comparison. Baseline is the original clean
GraphBudget `c045569a6c295601d5124bc089202c9785e1b013` with its original five-file
SGLang patch. The patched `native` arm is an additional wrapper-cost control.
All use the same Qwen2.5-0.5B revision, BF16, deterministic Triton fallback,
overlap disabled, capture `[1,2,4,8,12,16]`, 4096 actual KV capacity, 16 running
request limit and greedy fixed tokens. No shared environment modifications.

H1 compares native, preallocated prefix/splits, fused prefix/splits, and
prefix-only ablation. The actual upstream compact index gather stays unchanged.
Non-contiguous sequence/pool inputs are outside the original gather contract;
the retained first test failure demonstrates this instead of silently copying.
Only continuous sequence strides enter the candidate. Storage offsets and
request-pool row strides remain covered. Dummy graph rows have sequence length 1.

H2 follows the observed persistent GEMM kernels. It reuses the pinned kernel,
with opt-in BF16 M<=16 tile geometry. SGLang PR #42145 at
`8ceab1e77e0402a9cb23675f3e8733d36fe305da` is a related **open, unmerged** design.
Its M16/N64/K128, single-tile geometry is a comparison arm adapted to this
baseline's FP32-fused-bias contract, not a complete PR backport. The local
M16/N128/K64 geometry retains source K tiling. Neither is an original GEMM
invention or an upstream-accepted change. Exact numerical tests decide scope.

1. Screening: one fresh server each for original, patched native, prealloc,
   prefix-only, fused, and exact-valid small-M geometries. Each serves the 43
   original correctness requests, 64x128 decode tokens, and frozen holdout.
   Screening supports mechanism selection only, no tail/adoption claim.
2. Select the fastest exact-valid small-M geometry on screening primary
   throughput; if a tied difference is <=5%, prefer the existing upstream
   geometry adaptation. Do not tune further on holdout.
3. Formal: five original/candidate pairs, order AB, BA, AB, BA, AB. Each fresh
   server runs 256 fixed requests x128 generated tokens (32768 total tokens)
   plus six holdout arrival bursts of 3/5/9/3/9/5, with longer repeated fixed
   inputs and mixed 64/96 output lengths. Same max_workers=32 in every arm;
   client scheduling delay is retained. No failures or slow rounds are removed.
4. Original A/A runs at the beginning and end, two identical primary/holdout
   rounds per server. Record frequencies/temperature/power at 250 ms. Desktop
   GPU usage is observed, not falsely treated as zero or controlled power mode.
5. Primary metric is effective output tokens/s for the full primary load.
   Require median paired gain >max(5%, maximum absolute primary A/A delta),
   at least 4/5 positive pairs, and exact tokens for all matching requests.
   Holdout may invalidate a general adoption if it materially regresses.
   TTFT, request TPOT and true per-token ITL stay distinct; tail claims additionally
   require improvement beyond their own A/A envelope. No p-value stopping.

Nsight node traces diagnose launch shape and activity; their times are not
formal performance. CUDA Event diagnostic spans include possible host gaps and
are read only at idle; formal runs disable both profiling and those events.
Cold JIT, cached start, capture time, worker-init and HTTP-ready remain separate.
All paths have fresh real identities; no old plan/profile fingerprint is relabeled.
Only the two named mechanisms are explored. Publication and Drive upload are
handled by the coordinating chat, not this experiment runner.

Before any screening result, the original arm encountered SIGBUS in safetensors
weight loading from the existing DrvFS snapshot. `/dev/shm` had >5 GiB free,
WSL had ~5.9 GiB available including reclaimable cache, and root disk had >800
GiB free. The retained stack stops at `linear.py:693 weight_loader`; this is
not a candidate execution or performance failure. All subsequent screening and
formal arms uniformly set `--weight-loader-disable-mmap`, which is fingerprinted
by the existing loading-identity contract. The model snapshot is unchanged.

The next candidate screening encountered a Triton cache group JSON parse error;
the shared cache was left untouched. A subsequent attempt to hash the existing
DrvFS model snapshot failed with `ENOMEM` before model loading. Later arms use
the same task-local `TRITON_CACHE_DIR` and a task-local ext4 copy of the same
snapshot. Nine copied files were checked against the earlier loaded identity or
their source; all hashes matched. These are new common conditions for both arms:
earlier screening is retained as diagnosis, not paired with the new runs. The
first run begins with an empty task-local Triton cache; cached runs, capture and
HTTP readiness are recorded separately. The errors' root causes remain unknown.

The experiment scripts require explicit `GRAPHBUDGET_MODEL_PATH` and
`GRAPHBUDGET_SHARED_GPU_LOCK`; the original arm also requires
`GRAPHBUDGET_BASELINE_ROOT` and `GRAPHBUDGET_BASELINE_SGLANG_SOURCE`.
`TRITON_CACHE_DIR` must point to the same task-local cache in every arm.
The existing `scripts/environment/with_runtime.sh` supplies the runtime
overlay and source paths; retain the original per-run commands and hashes in
the private evidence instead of embedding machine-specific paths here.
