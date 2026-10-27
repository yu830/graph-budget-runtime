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

## Historical deterministic validation metadata (2026-10-26)

`runtime-pinned.json` describes two different evidence boundaries explicitly:

- The packaged SGLang source identity at that historical validation was base commit
  `bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2` with patch SHA-256
  `3146afe85bcd1e687dcdd3b1ba656a70687dabacfbed3301e4579205962fecec`.
- The earlier loaded patch fingerprint
  `efbde2ca6637b6838409c15d162a9baee86f39387e59da472cc97294a498f800` is
  retained only as historical provenance for the initial metadata/GPU
  observation. It is not relabeled as the later result. For the current checkout,
  use `src/graphbudget/patches/manifest.json` rather than this historical snapshot.

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

## Isolated v2 profile

The v2 candidate uses a single isolated Python 3.12.13 environment rather than
requiring the historical overlay paths above. The observed exact versions are in
[requirements-v2.txt](requirements-v2.txt). They describe the tested trimmed
runtime, installed with `--no-deps`; optional SGLang backends and extras are
not asserted to be complete. Source identity is checked separately against the
current packaged patch and SGLang runtime-tree manifest.

The following commands use the same package indexes, versions and non-editable
GraphBudget installation used in the isolated smoke. Supply local absolute
paths and use a model snapshot whose directory is its exact revision:

```bash
export GB_REPO="$(pwd)"
export GB_ENV=/absolute/path/to/new-graphbudget-env
export GB_PYTHON="$GB_ENV/bin/python"
export SGLANG_SOURCE=/absolute/path/to/new-sglang-checkout
export MODEL_PATH=/absolute/path/to/model/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
export GRAPHBUDGET_SHARED_GPU_LOCK=/absolute/path/to/shared-gpu.lock

uv --no-config venv --python 3.12.13 --no-python-downloads "$GB_ENV"
uv --no-config pip install --python "$GB_PYTHON" --no-deps --only-binary :all: \
  --link-mode copy --index-strategy unsafe-best-match \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --requirement "$GB_REPO/scripts/environment/requirements-v2.txt"
uv --no-config pip install --python "$GB_PYTHON" --no-deps --no-build-isolation "$GB_REPO"

git clone --branch v0.5.9 --single-branch https://github.com/sgl-project/sglang.git "$SGLANG_SOURCE"
git -C "$SGLANG_SOURCE" checkout bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2
"$GB_PYTHON" -m graphbudget.cli apply-patch "$SGLANG_SOURCE"

# CPU tests use the checkout only for its scripts and tests; GraphBudget itself
# is loaded from the installed environment.
cd /tmp
env -u SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM \
  -u GRAPHBUDGET_SMALL_M_GEMM -u GRAPHBUDGET_RESIDUAL_NORM \
  PYTHONPATH="$GB_REPO:$SGLANG_SOURCE/python" \
  GRAPHBUDGET_SGLANG_SOURCE="$SGLANG_SOURCE" PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  "$GB_PYTHON" -P -s -m unittest discover -s "$GB_REPO/tests" -v
cd "$GB_REPO"
export PYTHONPATH="$SGLANG_SOURCE/python"
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
```

Run the [v2 smoke and paired comparison](../../docs/DECODE_FAST_PATH_V2.md#reproduce)
with that environment. The GPU lock must be shared by cooperating experiments;
it cannot reserve GPU use by unrelated desktop applications. Model weights are
not included in this repository. This recipe pins versions, not wheel hashes;
a later unavailable or replaced upstream wheel can prevent exact recreation.
