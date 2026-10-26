#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

overlay="${GRAPHBUDGET_ENV:?set GRAPHBUDGET_ENV to the task-specific overlay}"
sglang_overlay="${SGLANG_OVERLAY:?set SGLANG_OVERLAY to the SGLang overlay}"
torch_env="${TORCH_ENV:?set TORCH_ENV to the pinned Torch environment}"
sglang_source="${SGLANG_SOURCE:?set SGLANG_SOURCE to the patched SGLang source checkout}"

export PYTHONPATH="${repo_root}/src:${sglang_source}/python:${overlay}/lib/python3.12/site-packages:${sglang_overlay}/lib/python3.12/site-packages:${torch_env}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${torch_env}/lib/python3.12/site-packages/torch/lib:${torch_env}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

exec "${overlay}/bin/python" "$@"
