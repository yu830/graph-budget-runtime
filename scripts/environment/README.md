# Environment bootstrap and inspection

These helpers keep the GraphBudget task overlay separate from the inherited
Torch/Triton/SGLang environments. Replace the `/opt/...` examples with paths
in the local deployment; the public files intentionally contain no machine-
specific user paths.

The overlay repair set is exact-pinned in `overlay-requirements.txt`. Install
it only in the task overlay, without dependency resolution:

```bash
"$GRAPHBUDGET_ENV/bin/python" -m pip install \
  --no-deps --only-binary=:all: -r scripts/environment/overlay-requirements.txt
```

Run a command with project source, patched SGLang source, and the pinned
runtime paths in the intended order:

```bash
export GRAPHBUDGET_ENV=/opt/graphbudget/env
export SGLANG_OVERLAY=/opt/sglang-runtime-059/.venv
export TORCH_ENV=/opt/flagos-runtime/.venv
export SGLANG_SOURCE=/opt/graphbudget/sglang-source
bash scripts/environment/with_runtime.sh -c \
  "import importlib.util; print(importlib.util.find_spec('sglang').origin)"
```

Create a full local lock and a path-free pinned manifest without importing
Torch or creating a CUDA context:

```bash
bash scripts/environment/with_runtime.sh \
  scripts/environment/inspect_runtime.py \
  --output work/environment/runtime-lock.json \
  --pinned-output scripts/environment/runtime-pinned.json
```

The full lock contains effective metadata candidates, module resolution paths,
source identity, and exact `Requires-Dist` values. The pinned manifest is safe
to carry with public source because it omits local paths and credentials.

## Current deterministic validation metadata

`runtime-pinned.json` describes two different evidence boundaries explicitly:

- The current packaged SGLang source identity is base commit
  `bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2` with patch SHA-256
  `3146afe85bcd1e687dcdd3b1ba656a70687dabacfbed3301e4579205962fecec`.
- The earlier loaded patch fingerprint
  `efbde2ca6637b6838409c15d162a9baee86f39387e59da472cc97294a498f800` is
  retained only as historical provenance for the initial metadata/GPU
  observation. It is not relabeled as the current deterministic result.

The validated strict route uses:

```bash
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
# launch configuration also includes:
#   --enable-deterministic-inference --attention-backend triton
```

This selects the existing upstream Triton fallback because SM120 DeepGEMM is
unsupported on this route. The deterministic probe is correctness-only: native
graph versus forced eager matched exact tokens for 43/43 requests. An earlier
default-mode comparison remains failed with 2 mismatches and 41 matches out of
43 requests and is preserved as a separate result. Neither result is a throughput or replay-benefit claim.

The metadata lock itself is CPU/read-only: it requires the pinned Python
environment, performs no install, does not import Torch, and does not use a
GPU. The deterministic probe has different prerequisites: the immutable model,
the pinned runtime, an SM120 GPU, and sufficient VRAM for the configured graph
capture set. The later 2026-10-26 benchmarking results are recorded in the
[technical report](../../docs/TECHNICAL_REPORT.md); consult
[status](../../docs/STATUS.md) for subsequent repair verification.
