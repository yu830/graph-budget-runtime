# Third-party notices

The SGLang v0.5.9 patch in this repository modifies
`python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`. The pinned
SGLang source identifies that file as adapted from
[Thinking Machines Lab's batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops),
whose MIT license and copyright notice are reproduced below. SGLang is
distributed under [Apache-2.0](https://github.com/sgl-project/sglang/blob/bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2/LICENSE).

The small-M tile geometry was adapted from
[SGLang PR #42145](https://github.com/sgl-project/sglang/pull/42145) by
bill-h-lin. This project's implementation retains the pinned kernel's FP32
fused-bias semantics and is not a verbatim backport or an upstream merge.

Related prior work on residual normalization is [SGLang PR #46159](https://github.com/sgl-project/sglang/pull/46159)
by Aarne Talman. It proposes a fully fused kernel with its own reduction;
GraphBudget's separate implementation retains the PyTorch variance reduction
and tests exact equality to the pinned native path. This is not an upstream
merge or a claim that fused residual normalization is new.

## Thinking Machines Lab: MIT License

MIT License

Copyright 2025 Thinking Machines Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
