#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: sglang_server.sh control|graph PORT RUN_TAG}"
port="${2:?usage: sglang_server.sh control|graph PORT RUN_TAG}"
run_tag="${3:?usage: sglang_server.sh control|graph PORT RUN_TAG}"

if [[ "${mode}" != "control" && "${mode}" != "graph" ]]; then
  echo "unsupported mode: ${mode}" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "${script_dir}/../.." && pwd)"
overlay="${GRAPHBUDGET_ENV:?set GRAPHBUDGET_ENV to the task-specific overlay}"
sglang_overlay="${SGLANG_OVERLAY:?set SGLANG_OVERLAY to the SGLang overlay}"
base="${TORCH_ENV:?set TORCH_ENV to the pinned Torch environment}"
model="${MODEL_PATH:?set MODEL_PATH to the immutable model snapshot}"
model_revision="${MODEL_REVISION:?set MODEL_REVISION to the immutable model SHA}"
log_path="${root}/work/environment/${mode}_${run_tag}.log"

mkdir -p "${root}/work/environment"
exec > >(tee "${log_path}") 2>&1

echo "[graphbudget] mode=${mode} port=${port} run_tag=${run_tag}"
echo "[graphbudget] log=${log_path}"
echo "[graphbudget] model=${model}"
echo "[graphbudget] model_revision=${model_revision}"
echo "[graphbudget] overlay=${overlay}"
echo "[graphbudget] python=${overlay}/bin/python"
echo "[graphbudget] torch/triton inherited from=${base}"

common_args=(
  --model-path "${model}"
  --tokenizer-path "${model}"
  --revision "${model_revision}"
  --host 127.0.0.1
  --port "${port}"
  --attention-backend triton
  --decode-attention-backend triton
  --skip-server-warmup
  --mem-fraction-static 0.55
  --max-running-requests 1
  --max-total-tokens 2048
  --context-length 2048
  --tensor-parallel-size 1
  --dtype bfloat16
  --random-seed 0
  --sampling-backend pytorch
  --disable-overlap-schedule
  --log-level info
  --log-requests
  --log-requests-level 3
)

if [[ "${mode}" == "control" ]]; then
  mode_args=(--disable-cuda-graph)
else
  mode_args=(--cuda-graph-bs 1)
fi

export PYTHONPATH="${overlay}/lib/python3.12/site-packages:${sglang_overlay}/lib/python3.12/site-packages:${base}/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="${base}/lib/python3.12/site-packages/torch/lib:${base}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
export PYTHONUNBUFFERED=1

exec "${overlay}/bin/python" -m sglang.launch_server "${common_args[@]}" "${mode_args[@]}"
