# Project status

2026-10-27: **v0.1.1 reliability fixes validated; keep native dispatch**.

The current machine-readable result is
[reliability-fixes.json](../results/2026-10-27/reliability-fixes.json).
[Repair notes](REVIEW_FIXES.md) explain C1/C2/C3, version binding, reproduction,
the retained failed diagnostic attempt and verification limits.

| Claim | Current evidence |
|---|---|
| C1 execution support/identity | Loading boundary rejected before normalization; supported settings fingerprinted; real old-plan/dummy/TorchAO/offload refusals observed |
| C2 optional cost data | Invalid rows select native with reasons; missing/stale tables fall back; hard-plan/source failures remain hard |
| C3 service readiness | Shared bounded HTTP supervisor; real POSIX CPU lifecycle tests and public CLI GPU startup/shutdown passed |
| CPU source / installed wheel | 80 / 80 passed, zero skipped in each WSL suite |
| GPU regression | Six smoke cases, 258 timed requests plus 215 explicit warmups; five 43/43 exact comparisons and historical 43/43 comparison passed |
| Actual paths | Native eligibility/selection/usage consistency; corrupt rows observed with specific fallback reasons |
| Performance | Not rerun; no new cost calibration or gain claim; historical adoption remains rejected |

The runtime implementation is `07f58142d1fa4589c917da174f7a3fd9f34f46b8`.
The SGLang patch and inherited runtime dependencies are unchanged. The GPU job
used a staged working source tree whose module hashes exactly match that commit;
the installed-wheel checks were CPU-only.

## Frozen history

[PROJECT_COMPLETION.json](../PROJECT_COMPLETION.json) and
[2026-10-26 results](../results/2026-10-26/README.md) remain immutable v0.1.0
snapshots. Their old publication fields and runtime fingerprint are historical.
The [independent review](reviews/2026-10-26-independent-review.md) and
[fix handoff](reviews/2026-10-26-fix-handoff.md) are preserved as source documents
with whitespace-only normalization; their statements are attributed to that review.

The original 21-case validation, 43/43 clean Git GPU replay and negative
strong-static-baseline study remain in the [technical report](TECHNICAL_REPORT.md).
Default-mode 2/43 mismatches are still failures. This repair does not establish
VRAM savings, throughput gains, production-load acceptance, multi-GPU support or
Compute Sanitizer coverage. No further optimization experiment is running.
