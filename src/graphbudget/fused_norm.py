"""Opt-in residual RMSNorm retaining PyTorch's variance reduction."""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _add(X, R, SUM, RESOUT, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    x = tl.load(X + row * N + col, mask=mask, other=0).to(tl.float32)
    r = tl.load(R + row * N + col, mask=mask, other=0).to(tl.float32)
    value = x + r
    tl.store(SUM + row * N + col, value, mask=mask)
    tl.store(RESOUT + row * N + col, value, mask=mask)


@triton.jit
def _finish(SUM, VAR, WEIGHT, OUT, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    value = tl.load(SUM + row * N + col, mask=mask, other=0)
    variance = tl.load(VAR + row)
    weight = tl.load(WEIGHT + col, mask=mask, other=0).to(tl.float32)
    inv = tl.rsqrt(variance + EPS)
    normalized = value * inv
    result = normalized * weight
    tl.store(OUT + row * N + col, result, mask=mask)


def residual_rms_norm(x, residual, weight, eps):
    """Keep the native square/mean reduction and round only at native boundaries."""
    rows, n = x.shape
    total = torch.empty_like(x, dtype=torch.float32)
    residual_out = torch.empty_like(x)
    output = torch.empty_like(x)
    block = triton.next_power_of_2(n)
    _add[(rows,)](x, residual, total, residual_out, n, block, enable_fp_fusion=False)
    variance = total.pow(2).mean(dim=-1, keepdim=True)
    _finish[(rows,)](total, variance, weight, output, n, eps, block, enable_fp_fusion=False)
    return output, residual_out


@lru_cache(maxsize=None)
def _verified_device(device):
    props = torch.cuda.get_device_properties(device)
    return (
        (props.major, props.minor) == (12, 0)
        and props.name == "NVIDIA GeForce RTX 5070 Laptop GPU"
    )


def supports_residual_rms_norm(layer, x, residual, post_residual_addition):
    """Return false for every configuration outside the measured fast path."""
    if (
        residual is None
        or post_residual_addition is not None
        or not layer.has_weight
        or layer.cast_x_before_out_mul
        or layer.fp32_residual
        or layer.override_orig_dtype is not None
        or layer.variance_size_override is not None
        or layer.variance_epsilon != 1e-6
        or x.ndim != 2
        or not 1 <= x.shape[0] <= 16
        or x.shape[1] not in (896, 1536)
        or x.shape[1] != layer.hidden_size
        or x.dtype != torch.bfloat16
        or residual.dtype != x.dtype
        or layer.weight.dtype != x.dtype
        or x.shape != residual.shape
        or layer.weight.shape != (x.shape[1],)
        or not x.is_cuda
        or residual.device != x.device
        or layer.weight.device != x.device
        or not x.is_contiguous()
        or not residual.is_contiguous()
        or not layer.weight.is_contiguous()
        or (
            torch.is_grad_enabled()
            and (x.requires_grad or residual.requires_grad or layer.weight.requires_grad)
        )
    ):
        return False
    return _verified_device(x.device)
