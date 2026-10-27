# GraphBudget

**Deterministic Decode Optimization for SGLang** · [简体中文](README.zh-CN.md)

GraphBudget investigates low-latency, reproducible LLM decoding at the SGLang runtime boundary. Its demonstrated optimization is an opt-in BF16 small-M GEMM path, accompanied by explicit execution-path evidence, exact-output checks, and paired serving measurements.

| Measured outcome | Validation | Environment |
|:--|:--|:--|
| **+72.4%** median paired effective output throughput | **5** interleaved A/B pairs; **340** GEMM and **255** reorder checks | SGLang v0.5.9 · Qwen2.5-0.5B BF16 · deterministic Triton · one GPU |

**Scope:** This number compares a pinned native deterministic Triton path with the same path plus the small-M adaptation, on one fixed serving workload. It is not a speedup claim over every SGLang backend, model, or deployment.

![GraphBudget: SGLang decode architecture](docs/figures/decode_architecture.svg)

## The optimization

Deterministic decoding at small batch sizes exposes GEMM calls with few output rows. In the pinned SGLang runtime, the opt-in path checks the supported BF16 shape and stride conditions before selecting a more suitable persistent Triton GEMM tile. Unsupported cases retain the native route.

| Aspect | Native deterministic path | Opt-in small-M adaptation |
|:--|:--|:--|
| GEMM tile (M / N / K) | 128 / 128 / 64 | 16 / 64 / 128 |
| Warps | 8 | 4 |
| Execution contract | Original persistent Triton, FP32 fused bias | Same arithmetic contract and native fallback |
| Selection | Runtime's original dispatch | Explicit switch with dtype, shape and stride guards |

The candidate tile geometry follows [bill-h-lin's SGLang PR #42145](https://github.com/sgl-project/sglang/pull/42145), adapted to the SGLang v0.5.9 persistent Triton path and its fused-bias behavior. The project contribution is the pinned-runtime adaptation, profiling and reproducible validation; the upstream tile design is credited to its author.

## Performance evidence

![Five paired serving throughput comparisons](docs/figures/paired_throughput.svg)

Five fresh-server pairs ran in AB / BA / AB / BA / AB order, with **256 fixed requests × 128 output tokens per arm**. The median of the five *paired relative gains* is **+72.4%**. Two surrounding A/A checks recorded up to **8.8%** change, a limited noise observation rather than a confidence interval.

The original/candidate serving outputs matched exactly in the tested runs. Nsight Systems independently confirmed a different persistent GEMM launch geometry; the measured throughput is taken from unprofiled request runs, not profiler timings.

[Technical report (English, PDF)](docs/reports/GraphBudget_Technical_Report_EN.pdf) · [技术报告（中文 PDF）](docs/reports/GraphBudget_Technical_Report_ZH.pdf) · [Original paired results](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/docs/PERFORMANCE_RESULTS_20261027.md)

## Method and reproducibility

![Experimental workflow](docs/figures/validation_workflow.svg)

The reproducible implementation is preserved in the [fixed performance revision](https://github.com/yu830/graph-budget-runtime/tree/1f1984147608ee6c5dd8b3423e6f41877aaadeb4). It includes an opt-in SGLang patch, a source-locked environment, CPU and GPU correctness checks, diagnostic traces, request-level A/B tests, and machine-readable statistics.

| Artifact | Read or reproduce |
|:--|:--|
| SGLang integration patch | [Pinned patch](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/src/graphbudget/patches/sglang-v0.5.9.patch) |
| Benchmark summary | [Five paired observations (JSON)](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/results/2026-10-27-performance/formal07-summary.json) |
| Reproduction steps and pinned dependencies | [Performance branch README](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/README.md) |
| Full measurements and ablations | [Performance results](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/docs/PERFORMANCE_RESULTS_20261027.md) |
| Human-readable technical analysis | [Chinese walkthrough](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/docs/PERFORMANCE_WALKTHROUGH_ZH.md) |
| Report source | [English LaTeX](docs/reports/GraphBudget_Technical_Report_EN.tex) · [Chinese LaTeX](docs/reports/GraphBudget_Technical_Report_ZH.tex) |

Build either report with XeLaTeX from `docs/reports/`:

```bash
xelatex GraphBudget_Technical_Report_EN.tex
xelatex GraphBudget_Technical_Report_ZH.tex
```

The PDF and SVG figures are included as vector assets. The figures describe actual integration boundaries; CUDA Graph replay uses captured launches rather than re-evaluating the GEMM guard per output token.

## Limits and adjacent findings

The measured result is scoped to one pinned model and GPU, deterministic BF16 inference, and the stated serving workload. Separate experiments found no stable incremental end-to-end benefit from the earlier CUDA Graph cost-selection policy or from decode-metadata preallocation/fusion. Holdout TTFT results were mixed; GPU occupancy and bandwidth counters were unavailable. Those observations are documented in the [technical report](docs/reports/GraphBudget_Technical_Report_EN.pdf) and [original measurement record](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/docs/PERFORMANCE_RESULTS_20261027.md).

## Provenance and license

Original project materials are licensed under [Apache-2.0](LICENSE). SGLang and other referenced work retain their own licensing and attribution, including the applicable [third-party notices](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/THIRD_PARTY_NOTICES.md). The small-M geometry is attributed to SGLang PR #42145, not claimed as an independently invented algorithm.
