# Decode v2: focused B/C validation

This is a new prospective protocol. Earlier v1 and v2 results, including
failed controls and the A-only CUDA error, remain unchanged. B is the direct
baseline for C; no A/A run is required by this protocol.

## Model health and operator mechanism

After the historical A error, a same-boot B service completed calibration,
primary, holdout, interactive and acceptance workloads. Its 365 requests
matched an earlier B reference exactly; the primary recorded 2,032 batch-16
graph replays, and its log showed prefill, decode and no CUDA error. The
sampling backend was PyTorch. This supports that specific model execution; it
does not explain or exclude an intermittent CUDA fault.

The separate `bench_residual_norm_chain.py` diagnostic compares B's full
native residual RMSNorm chain with C's actual patched dispatch in the same
process. It uses separate captured graphs, a real 896-wide model weight,
identical inputs and weights, and checks output and residual bitwise after
changing inputs and replaying both graphs. The 1536-wide shapes use a
deterministic synthetic weight. CUDA Event spans include replay submission and
can contain host gaps; they are not a serving throughput measurement.

For the measured 896×16 case, B and C median event spans were 16.13 and
9.22 microseconds per graph replay. The median paired latency change was
−46.56%, with seven of seven blocks improving and a maximum same-B control
change of 1.29%. All six requested shapes passed their changed-input bitwise
checks. The 896×1 same-B control varied 25.58%, so not every shape had equally
stable controls. Raw samples and the exact script fingerprint are retained
with the private evidence.

## Frozen serving protocol: `bc-focused-v1`

The explicit sequence is independent B/B, then BC/CB/BC/CB/BC, then
independent B/B. The default legacy protocol remains available unchanged.
Each fresh service receives the same fixed warmup: 4×64 requests, 43
calibration requests, and a separate primary-shaped 256×128 workload. Warmup
inputs, outputs and timings are retained and validated, and warmup is outside
the timed primary. No arm gets additional warmup based on its observed speed.

The timed primary remains 256×128 with 32 client workers, 16 running requests,
4096 KV tokens and the original capture sizes. It uses same-boot, same-source,
local direct transport, no profiler and no request diagnostics. Previously
used holdout, interactive and acceptance workloads are regression sets, not
unseen inputs.

The first B/B must be within 5% or the run stops before any B/C pair. A
candidate must improve at least four of five pairs, with median improvement
strictly above 5% and all measured B/B changes and B drift. A post-control or
before-to-after drift over 5%, a consistent latency regression, missing
metrics or missing control evidence requires review and prevents automatic
adoption. These are engineering rules, not statistical-significance claims.
Every request, including slow and failed requests, stays in the evidence.

With the existing pinned environment, model snapshot and shared lock:

```bash
"$GB_PYTHON" -P -s scripts/bench_residual_norm_chain.py \
  --model-path "$MODEL_PATH" --gpu-lock "$GRAPHBUDGET_SHARED_GPU_LOCK" \
  --output work/residual-chain

"$GB_PYTHON" -P -s scripts/run_paired_decode.py \
  --protocol bc-focused-v1 --transport direct \
  --model-path "$MODEL_PATH" --gpu-lock "$GRAPHBUDGET_SHARED_GPU_LOCK" \
  --output work/bc-focused
```

The focused runner rejects other running-request, KV and transport settings,
as well as `CUDA_LAUNCH_BLOCKING`. The actual experimental protocol records
the source identity and input hashes before execution. A separate prior
exploratory B/C driver proceeded after an 8.31% B/B change and is not treated
as satisfying this protocol's 5% gate.
