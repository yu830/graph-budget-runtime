"""Opt-in dense deterministic decode metadata preparation, SGLang v0.5.9.

Only prefix/split preparation changes. The upstream compact index gather runs
after this kernel in the same stream; it never reads another CTA's unfinished
prefix. No metadata values are cached across steps.
"""
import os
from collections import Counter
import torch
import triton
import triton.language as tl

MODES = ('native', 'prealloc', 'fused', 'prefix_only')
COUNTS = Counter()

@triton.jit
def _prefix_splits(Seq, Indptr, Splits, N: tl.constexpr, TILE: tl.constexpr,
                   SEQ_STRIDE: tl.constexpr, BLOCK: tl.constexpr, WRITE_SPLIT: tl.constexpr):
    i = tl.arange(0, BLOCK)
    lengths = tl.load(Seq + i * SEQ_STRIDE, i < N, other=0).to(tl.int64)
    prefix = tl.cumsum(lengths, 0)
    tl.store(Indptr + i + 1, prefix, i < N)
    tl.store(Indptr, 0)
    if WRITE_SPLIT:
        tl.store(Splits + i, (lengths + TILE - 1) // TILE, i < N)

def configure(backend, max_bs):
    mode = os.environ.get('GRAPHBUDGET_METADATA_MODE', 'native')
    if mode not in MODES:
        raise ValueError(f'Invalid GRAPHBUDGET_METADATA_MODE: {mode}')
    backend.graphbudget_metadata_mode = mode
    backend.graphbudget_metadata_tmp = (
        torch.empty(max_bs, dtype=torch.int32, device=backend.device)
        if mode == 'prealloc' else None)

def supported(backend, bs, seq, indptr, splits):
    # The existing max-context invariant bounds all active and padded lengths.
    # Bound the prefix sum before launching, without per-step GPU readback.
    return (0 < bs <= 16 and backend.enable_deterministic
            and isinstance(backend.split_tile_size, int) and backend.split_tile_size > 0
            and (backend.sliding_window_size is None or backend.sliding_window_size <= 0)
            and bs * backend.max_context_len < 2**31
            and backend.max_context_len + backend.split_tile_size - 1 < 2**31
            and seq.dtype == indptr.dtype == splits.dtype == torch.int32
            and seq.is_cuda and seq.device == indptr.device == splits.device
            and seq.ndim == indptr.ndim == splits.ndim == 1
            and seq.numel() >= bs and indptr.numel() >= bs+1 and splits.numel() >= bs
            and seq.stride(0) == indptr.stride(0) == splits.stride(0) == 1)

def prepare(backend, bs, seq, indptr, splits):
    """Return 0 for native, 1 for prefix-only, 2 for prefix plus splits."""
    mode = backend.graphbudget_metadata_mode
    if mode == 'native' or not supported(backend, bs, seq, indptr, splits):
        COUNTS[(mode, 'native', bs)] += 1
        return 0
    seq = seq[:bs]
    if mode == 'prealloc':
        torch.cumsum(seq, dim=0, dtype=torch.int32, out=indptr[1:bs+1])
        torch.add(seq, backend.split_tile_size-1, out=backend.graphbudget_metadata_tmp[:bs])
        torch.div(backend.graphbudget_metadata_tmp[:bs], backend.split_tile_size,
                  rounding_mode='floor', out=splits[:bs])
        COUNTS[(mode, 'preallocated_prefix_splits', bs)] += 1
        return 2
    _prefix_splits[(1,)](seq, indptr, splits, bs, backend.split_tile_size,
                         seq.stride(0), triton.next_power_of_2(bs), mode == 'fused', num_warps=4)
    COUNTS[(mode, 'fused_prefix_splits' if mode == 'fused' else 'prefix_only', bs)] += 1
    return 2 if mode == 'fused' else 1
