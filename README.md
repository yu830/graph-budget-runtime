# GraphBudget

Measured, frozen CUDA Graph plans for SGLang under device-memory, KV-capacity,
and startup constraints. The first adapter targets dense Qwen2.5 BF16 decode,
one GPU, and SGLang **v0.5.9 / bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2**.

The implementation includes a bounded whole-configuration selector, startup
application before upstream sizing/capture, a conservative graph/eager hook,
and request-bound observations of the actual replay/eager calls. It neither
changes the allocator nor recaptures/resizes graphs or KV during service.

**Verification is split by claim.** See [status](docs/STATUS.md) and the
[technical report](docs/TECHNICAL_REPORT.md) for implementation, CPU checks,
model correctness, native execution paths, and performance separately.

**Original 2026-10-26 snapshot:** 57 CPU checks passed; strict model validation
and clean-source GPU replay using the existing pinned runtime passed in the
pinned deterministic mode. This snapshot describes the measured version; see
[status](docs/STATUS.md) for the current repair and verification status. Five
paired rounds per held-out load established no benefit over the strong static plan. Keep native
dispatch. The measured plan, profile and round data are in
[results/2026-10-26](results/2026-10-26/README.md).

## Install and reproduce

The CPU package has no runtime dependencies:

~~~bash
uv pip install --python /path/to/python --no-deps -e .
python -m unittest discover -s tests -v
~~~

GPU execution reuses a pinned Torch 2.9.1+cu128 / Triton 3.5.1 environment,
SGLang 0.5.9, sgl-kernel 0.3.21 and FlashInfer 0.6.3. The bounded overlay
repair is in [overlay-requirements.txt](scripts/environment/overlay-requirements.txt).
Install those exact packages into a **separate overlay**, with --no-deps;
do not run an unconstrained dependency upgrade over an existing GPU stack.
This overlay recipe presumes the inherited environments already exist; it is
not a claim that an arbitrary clean machine has the same full dependency set.

Create a dedicated upstream checkout, then apply the packaged patch:

~~~bash
git clone --branch v0.5.9 --single-branch https://github.com/sgl-project/sglang.git "$SGLANG_SOURCE"
git -C "$SGLANG_SOURCE" checkout bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2
python -m graphbudget.cli apply-patch "$SGLANG_SOURCE"
GRAPHBUDGET_SGLANG_SOURCE="$SGLANG_SOURCE" python -m unittest discover -s tests -v
~~~

The reported 57 checks include all five pinned-source control-flow checks;
those five are skipped when GRAPHBUDGET_SGLANG_SOURCE is not set.

The patch command rejects the wrong base and dirty trees. Startup verifies the
loaded patched files and the full SGLang runtime source tree. The package also
fingerprints its Python code, actual dependencies, model/tokenizer files,
GPU/driver, dtype, backend, and relevant server settings.

Set GRAPHBUDGET_ENV to the repair overlay, SGLANG_OVERLAY to the existing
0.5.9 environment, TORCH_ENV to the existing cu128 environment, and
SGLANG_SOURCE to the dedicated patched checkout. Use an immutable local
model snapshot with its exact Hub commit:

~~~bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
export MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
bash scripts/environment/with_runtime.sh scripts/run_core.py \
  --model-path "$MODEL_PATH" --revision "$MODEL_REVISION" --backend triton \
  --sglang-checkout "$SGLANG_SOURCE" --deterministic --output work/core --phase all
~~~

Phases are prepare, measure, select, profile, validate, and performance.
Use --resume with the same model/backend/output to continue; failed attempts
remain intact. Preparation freezes separate calibration and held-out workloads.
GPU cases are serialized by a project lease, which does not reserve the GPU
against unrelated applications.

The validated correctness route uses the upstream deterministic mode and its
existing Triton batch-invariant matmul. The default-mode calibration exposed
token differences under changing batch shapes; those failures are retained.
On SM120 this release's DeepGEMM deterministic path rejects the architecture,
so the explicit deterministic option selects its supported Triton path for
every baseline. These results must be kept separate from default-mode speed.

To launch a frozen plan with the supervised Linux/WSL entry point:

~~~bash
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
bash scripts/environment/with_runtime.sh -m graphbudget.cli launch \
  --output work/serve-01 --plan /path/to/plan.json --policy native \
  --startup-timeout 600 -- \
  --model-path "$MODEL_PATH" --revision "$MODEL_REVISION" \
  --dtype bfloat16 --device cuda --attention-backend triton \
  --disable-overlap-schedule --stream-output --enable-deterministic-inference --random-seed 0
~~~

The startup deadline defaults to 600 seconds and is the smaller of the supplied
`--startup-timeout` and the plan's `max_startup_seconds`. `ready.json` records
worker initialization; `lifecycle.json` records the supervised HTTP-ready
result. The pinned upstream `/health` endpoint is gated by its Starting state;
default startup includes general warmup. After startup, this supervisor does
not act as an ongoing health watchdog. `_serve` is an internal child entry.

Supply the same remaining server settings used for measurement; mismatches
fail closed. Cost mode uses --policy cost --profile /path/to/profile.json.
Missing/stale cost data preserves native dispatch; a stale plan or violated
hard memory/KV/startup constraint rejects startup. The test_eager mode
requires --test-only and is solely a functional/calibration control.

## Measurement semantics

- Candidates are entire shared-pool configurations in separate processes.
  No per-graph memory sums are used.
- Device usage (total minus free), PyTorch allocated/reserved/peaks, KV tokens,
  capture delta, and startup duration retain separate meanings. Device usage
  includes other applications; sampled peaks are not absolute allocation peaks.
- The policy can only reject an upstream-eligible graph. Unsupported or
  heterogeneous CPU context metadata uses the native path. A compiled table
  lookup adds no per-step GPU readback or synchronization.
- Diagnostic calibration times complete forwards including preparation and
  copies. It synchronizes deliberately and is excluded from clean throughput
  results. Normal observation buffers events/counters; flushing is untimed.
- T is the static choice selected by this calibration within the same bounded
  candidate set, not a global optimum. P applies the same choice automatically;
  their configuration equivalence is explicit. C uses the same plan/KV/backend
  with the optional cost policy.
- Exact greedy token comparison, actual branch observations, A/A noise, and at
  least five paired strong-baseline rounds are required for an adoption claim.
  Chunked stream arrivals do not become per-token ITL.

## Scope and attribution

The original [design](docs/DESIGN.md), [protocol](docs/BENCHMARK_PROTOCOL.md),
and [research source index](docs/SOURCE_INDEX.md) preserve the design-phase
research SHA c00e7e12b64b587da336c4350b1d2484c7dffcd6. Development uses the
compatible pinned release above. v0.5.9 allocates KV **before** graph capture;
the adapter does not claim the research revision's post-capture sizing.

SGLang already supplies graph sizes, ceil padding, graph-pool sharing and
safe overflow fallback. GraphBudget adds measured planning, fingerprinted
startup enforcement, a reject-only cost hook and reproducible evidence.
No online recapture/eviction, admission scheduler, multi-GPU support or
custom CUDA/Triton kernel is introduced.

Original project code is Apache-2.0. The SGLang patch retains upstream
Apache-2.0 attribution. Model weights are obtained from the Apache-2.0 licensed
Qwen repository and are not redistributed here.
