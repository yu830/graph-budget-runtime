# Reliability fixes after the independent review

2026-10-27, GraphBudget 0.1.1. This change closes the C1/C2/C3 boundaries in the
[independent review](reviews/2026-10-26-independent-review.md). It does not add
a new optimization policy or change the historical negative performance result.

Runtime implementation: `07f58142d1fa4589c917da174f7a3fd9f34f46b8`.
Runtime module digest: `e921223e3fee4f8bd0a6fce8517ecc65c72ae5dad22691ad16100cd6255ce74e`.
New measured identity: `0b8a6f6973b62a8dcad8d685b309b2da752e796aa8fbf6f9f3d90c1ff3fbd774`.
SGLang base and the five-file upstream patch are unchanged. The new identity
has schema version 2; old hard plans must be rejected and old optional costs
must fall back. Historical identities were not rewritten.

## Changes and evidence

| Finding | Before | After |
|---|---|---|
| C1: loading/execution identity | Real support/identity functions accepted dummy weights, TorchAO and CPU offload with identical fingerprints in an external-observation CPU fixture | A bounded loading allowlist runs before upstream ServerArgs normalization, again at worker support checking, and before identity collection. Accepted normalized loading settings are bound to identity. |
| C2: malformed optional cost fields | `provenance=[]` raised TypeError; timing `10**400` raised OverflowError | Typed provenance validation and checked finite conversion retain the existing invalid-row/native decision contract. Invalid complete tables fail soft at startup; hard-plan/source/model failures remain outside that boundary. |
| C3: total startup deadline | The direct CLI waited for blocking run_server without HTTP supervision, even when worker init fit the budget but HTTP never became ready | CLI launch and experiments share the same owned-process startup supervisor and total HTTP deadline. The public CLI remains a service entry, without benchmark requests or a post-start health watchdog. |

C1 also rejects nonempty loader extras/model overrides, decrypted external
configuration, forward hooks, remote seed-loading service mode, remote/custom
loaders, alternate model implementations and other unvalidated weight-changing
features. Empty model overrides must remain JSON text, as required by upstream;
loader extras may use its supported empty-object representations. Ordinary
`auto`/`safetensors` loading and the mmap flag are fingerprinted; ports and log
destinations are excluded. This is a narrow supported boundary, not a promise
to support arbitrary future ServerArgs or external weight mutation.

C2 intentionally retains a malformed row for diagnosis while keeping unrelated
valid exact keys usable. Its table status can therefore remain `loaded` while
that row selects native with `profile_not_measured` or `profile_entry_invalid`.
Missing or stale entire profiles report `native_fallback`. The GPU fault tables
are clearly marked invalid test inputs; their numbers are not cost measurements.

For C3, the fixed upstream `/health` returns 503 while ServerStatus is Starting.
The ordinary warmup path changes it to Up only after its generation request
succeeds. Explicit upstream warmup skipping remains explicit. The supervisor
requires matching worker/endpoint run IDs and an owned process-group PID before
accepting the HTTP response, and checks elapsed time after the response. A port
preflight rejects an existing listener. `ready.json` retains the worker-init
measurement; `lifecycle.json` records the total HTTP-ready measurement.

Deadline failure returns 124, spawn failure 125, invalid readiness 126, and an
early child failure propagates its nonzero code. A child that exits 0 before
readiness maps to failure 1. Linux/WSL POSIX process groups are required; native
Windows service launch fails before spawning. Cleanup is limited to process
groups created by this launcher, including remaining workers after leader exit.

## Verification scope

- Full WSL source suite: **80 tests passed, zero skipped**, including all five
  original real-upstream source checks. The added checks cover configuration
  boundaries, actual RuntimeState profile construction, real CPU subprocesses,
  late/absent/slow HTTP readiness, early exit, foreign records, port conflict,
  owned descendants, unrelated-process preservation and successful service life.
- Isolated installed wheel: **80 tests passed, zero skipped**. All nine packaged
  runtime modules and the packaged patch match source. This reuses the pinned
  existing Python runtime; it is not a fresh-machine dependency reconstruction.
- RTX 5070 Laptop / WSL / Qwen2.5-0.5B BF16: **six successful smoke cases**, each
  with 43 fixed requests (258 total), plus 215 explicitly recorded warmup
  requests. Cases are one measured configuration, its native plan, missing
  profile, stale profile, bad provenance, and bad timing through public CLI.
- Each of the five planned/fallback cases matches the new measurement's
  **43/43 token outputs**. The new measurement also matches the frozen original
  calibration **43/43**. Actual timed calls retain native eligibility/selection/
  usage consistency; corrupted rows were exercised with their specific reasons.
- Public CLI reached HTTP-ready, served the fixed requests, and shut down via
  SIGTERM with exit 143 and no live owned server group. The late/never-ready
  fault injection is CPU-based; positive readiness is also observed on GPU.
- Real CLI negative launches rejected the old hard plan (StalePlanError in the
  worker log, propagated child exit 137) and rejected dummy/TorchAO/CPU-offload
  modes (exit 1), without reaching weight loading or worker readiness. An initial
  old-plan probe hit its explicit 90-second deadline before the stale-plan
  diagnostic; this failed diagnostic attempt is retained. A single retry with
  300 seconds, within the original 600-second cap, observed the intended refusal.

The single predeclared smoke capture set is `[1,2,4,8,12,16]`, with actual KV
capacity 4096, a 7 GiB device budget and the original 600-second startup cap.
This is a freshly measured **one-candidate regression plan**, not a repeat of
the original three-candidate search. HTTP readiness was 59.10–78.40 seconds in
these runs; those observations are not a before/after speedup comparison.

The GPU job started from the staged working tree immediately before the
implementation commit. Its original receipt records parent `a587e20`; the
retained staged diff and every runtime module hash bind it exactly to `07f5814`.
The source-binding receipt preserves this distinction. The installed-wheel
check was CPU-only; this round makes no new clean-source GPU replay claim.

## Reproduction and retained results

Use the existing environment recipe and run:

```bash
GRAPHBUDGET_SGLANG_SOURCE="$SGLANG_SOURCE" \
  bash scripts/environment/with_runtime.sh -m unittest discover -s tests -v

bash scripts/environment/with_runtime.sh scripts/verify_review_fixes.py \
  --model-path "$MODEL_PATH" --revision "$MODEL_REVISION" \
  --output work/review-smoke-new
```

The optional `--reference` accepts the original private calibration
`requests.json` for the historical exact comparison. Always use a new output
directory. The smoke script measures a new plan and never relabels an old plan
or profile with a new fingerprint. Keep native dispatch until new cost data and
a separate performance hypothesis justify another evaluation.

[Machine results](../results/2026-10-27/reliability-fixes.json),
[source tests](../results/2026-10-27/cpu-wsl-final.txt),
[installed-package tests](../results/2026-10-27/package-tests.txt) and the
[raw manifest](../results/2026-10-27/RAW_MANIFEST.json) are versioned here.
The original-byte archive is private and includes the before/after counterexamples,
failed attempts, commands, actual requests, paths, lifecycle records and telemetry.
It excludes model weights and dependency environments.

The original `PROJECT_COMPLETION.json`, `results/2026-10-26` and their fingerprints
remain unchanged. `upstream.lock.json` retains that original experiment snapshot;
the dependencies and upstream patch are still the same. The original technical
report now clarifies denominators, sampled device peaks, short input lengths and
cost coverage. No full performance rerun, new cost calibration, Compute Sanitizer,
multi-GPU or production-load acceptance is claimed. Performance remains
**NOT_ESTABLISHED / KEEP_NATIVE_DISPATCH**.
