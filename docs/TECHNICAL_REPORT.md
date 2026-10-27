# GraphBudget: executed core and strong-static-baseline evaluation

The original 2026-10-26 measured version ran the planning/startup/dispatch/
evidence chain on Qwen2.5-0.5B BF16 and an RTX 5070 Laptop GPU. The validated configuration uses
upstream deterministic inference with Triton batch-invariant matmul. The cost
policy found no useful eager switch; the recommended configuration is the
selected plan with native dispatch. Incremental performance over the strong
static baseline is not established. This report describes that frozen measured
version; documentation was clarified on 2026-10-27. See [current status](STATUS.md)
for subsequent repairs and their verification.

## Acceptance by claim

| Claim | Observed result |
|---|---|
| Core code and clean-source installation using the existing pinned runtime | 57 CPU tests passed, zero skipped; exact-base patch application verified |
| Model correctness | 21 validation cases, 815 requests, 40 pairwise exact-token comparisons plus the overflow control passed |
| Actual decode execution | Graph replay, batch 3 padded to 4, capture-ceiling eager fallback and test-forced eager observed |
| Frozen cost profile | Seven matched batch-size-one/context keys; zero eager wins beyond the declared noise threshold |
| End-to-end evaluation | 2,078 requests: 1,060 formal T/C, 700 boundary A/A and 318 U/E/P exploratory; zero failures/timeouts |
| Clean-source GPU replay using the existing pinned runtime | 43/43 exact outputs; loaded clean project/upstream checkouts, applied the exported plan, and used native fallback for a missing cost profile |
| Adoption | Keep native dispatch; no strong-static-baseline performance gain established |

The default non-deterministic mode failed strict token comparison for 2 of 43
calibration requests. Those records remain failures; no tolerance was widened.
The successful results below use the explicitly different deterministic setting
for every baseline and candidate. They are not compared as speedups over the
earlier default-mode runs.

## Pinned implementation and existing mechanisms

- Runtime/controller source: `9340ee0b3f6e7d6054b4062540ece46c5f087c42`; runtime content fingerprint
  `819a0ddbc7ff7e3c60e3d8cc4c8068b405d76af19dd2a846fd5fb25acc38a07b`.
- SGLang v0.5.9 base: `bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2`.
- Five-file upstream patch: `3146afe85bcd1e687dcdd3b1ba656a70687dabacfbed3301e4579205962fecec`.
- Model and tokenizer: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, BF16, TP=1.
- Actual stack: Python 3.12.13 in WSL Ubuntu 24.04.4, Torch 2.9.1+cu128,
  Triton 3.5.1, sgl-kernel 0.3.21, FlashInfer 0.6.3, driver 617.42.

This revision was selected to reuse the working cu128 stack. It allocates KV
before graph capture. The later research revision's conditional post-capture KV
sizing is not used. Capture buckets, ceil padding, graph-pool sharing and safe
overflow fallback are upstream SGLang mechanisms. No allocator, scheduler,
online recapture, KV resize or custom kernel was added.

The project applies a frozen plan at the beginning of ServerArgs normalization,
then checks actual source/model/environment identity, capture set, KV capacity
and device budget before service. A stale hard plan fails startup; missing or stale cost data preserves native
dispatch. The independent review later found malformed cost data could escape
that fallback in this measured version. The reject-only hook follows upstream eligibility
and precedes graph/eager-specific preparation. Actual replay/eager return sites
produce separate eligibility, selection and usage evidence. Production lookup
uses CPU metadata without GPU readback or synchronization.

Diagnostics buffer records and export at the scheduler's existing idle cache-
flush barrier. That operation clears cache bookkeeping and unused allocations,
preserving graph-referenced KV backing tensors. It occurs outside timed requests.
Calibration deliberately synchronizes complete forwards, including metadata,
padding and copies; those step timings are not end-to-end speedups.

## Whole-configuration selection

The predeclared budget was 7 GiB of whole-device usage, at least 1,296 KV tokens
(16 concurrent requests times the worst frozen prompt-plus-output length), and
600 seconds to HTTP readiness. All three final candidates were feasible.

| Captured batch sizes | Calibration s | Ready MiB | Sampled peak MiB | KV tokens |
|---|---:|---:|---:|---:|
| 1,2,4 | 1.958 | 2214.44 | 2327 | 4096 |
| 1,2,4,8 | 1.828 | 2216.44 | 2342 | 4096 |
| 1,2,4,8,12,16 | 1.673 | 2242.44 | 2316 | 4096 |

The selected set is `[1,2,4,8,12,16]`. Startup took 29.96–34.27 seconds and
capture took 1.51–2.18 seconds in these cached-environment runs. A cold compiler-
cache installation was not timed. Shared pools are measured as complete sets;
their memory is not modeled as a sum of individual graphs. The small and
non-monotonic resource differences do not establish a memory-saving result.

Ready usage is from CUDA total/free snapshots. The peak is the observed sampled
whole-device peak from 100-ms NVML sampling across startup, warmup and requests, including desktop activity; it is not an
absolute peak bound. PyTorch allocated/reserved/peaks are recorded separately in
the raw readiness receipts. Host RAM is not added to any VRAM number.

T is the frozen static configuration selected by this calibration within the
same bounded candidate set; it is not a global optimum. P applies that exact
configuration automatically; T and P have the same capture set, KV capacity and backend. All diagnostic calibration data
is retained for both comparisons. C adds only the frozen cost lookup to that
same plan. Three additional graph/test-eager calibration pairs are an explicit
setup cost. The seven usable keys cover homogeneous batch-one contexts of
length 32–38; other contexts retain native dispatch.

## End-to-end results

Each row reports the median of five paired `(C-T)/T` differences. Positive
throughput is better; positive latency is worse. Workload input IDs, request
counts, sampling, offered schedules, backend and resource settings were frozen.
All planned requests were sent; client scheduling delay and backlog were retained.
The measured prompts were 14–45 tokens and outputs 8–16 tokens; the configured
4,096-token context limit is not a 4K-context validation.

| Held-out load | Tokens/s | P95 TTFT | P95 request TPOT | P95 token ITL |
|---|---:|---:|---:|---:|
| Normal | -0.25% | -5.53% | -7.95% | +0.22% |
| Boundary | -2.18% | +10.52% | +3.92% | -20.18% |
| Mixed | -0.11% | +13.18% | +52.35% | +88.07% |

The five boundary-load A/A pairs showed maximum absolute throughput variation
of 16.22% and P95 ITL variation of 64.50%. A/A was not separately repeated for
normal and mixed loads, so their deltas remain descriptive. Individual token
arrival events were available for all formal paired requests; chunk intervals
were not substituted for ITL. No P99 claim is made.

The measured cost table covered 116 of 1,347 formal C decode steps (8.61%);
boundary covered 0 of 395. There were zero cost-driven graph rejections in
the measured C runs. The nominal boundary ITL reduction is smaller than the observed A/A ITL variation,
accompanies worse throughput/TTFT, and has no intended dispatch change to
support a cost-policy gain. Mixed-load tail latencies were also adverse. These
results do not meet the adoption criterion. P's configuration equals T's, so
the current value is repeatable planning and auditable runtime evidence rather
than an established improvement over static tuning.

## Failures and boundaries retained

- An isolated dependency overlay repaired missing packages while preserving
  Torch/Triton/CUDA. The exact repair and metadata are in
  `scripts/environment/`; shared installations were not modified.
- Default-mode graph/eager and native/native cadence comparisons each showed
  2/43 token differences. Full-vocabulary logits were not collected to certify
  that mode. Upstream deterministic inference removed the observed differences.
- Its default DeepGEMM branch rejected SM120. The existing upstream
  `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0` Triton path was used; no new
  kernel or upstream fix is claimed.
- Earlier signal-based observation export failed and was replaced by the
  synchronous management hook. WSL service interruptions and a file-read
  ENOMEM were retained separately from CUDA errors. A 64-KiB unbuffered read
  reproduced the complete model hash without that read failure.
- The performance run resumed after a pre-execution lease rejection. Completed
  cases were reused only after checking source fingerprint, workload, arguments
  and observation mode; their original timestamps/data were not rewritten.
- Compute Sanitizer was unavailable and was not installed. Multi-GPU, other
  models, long-context production traffic and global memory safety are not
  certified by this work.

## Reproduce and inspect

Follow [README.md](../README.md) and [upstream.lock.json](../upstream.lock.json).
Set `GRAPHBUDGET_SGLANG_SOURCE` to the patched checkout when running the CPU
suite so that all five real-source control-flow tests execute. The primary
entry is `scripts/run_core.py --deterministic --phase all`; a separate frozen
output directory creates a new experiment.

[Derived results](../results/2026-10-26/README.md) contain the frozen plan/profile,
path validation, every formal paired metric and hashes of the original raw
records. Full request events, logs and telemetry are a separate local archive
excluded from Git because they contain original machine paths. The local
receipt [PROJECT_COMPLETION.json](../PROJECT_COMPLETION.json) records the final
claim boundaries and replay entry. No public push or upstream PR was performed.
