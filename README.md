# GraphBudget — Deterministic Decode Optimization for SGLang

**SGLang 确定性解码优化与验证。** 本项目已完成运行时接入和限定场景的性能验证；性能代码位于独立分支，`main` 用作项目入口并保留早期设计材料。

On an RTX 5070 Laptop, the small-M GEMM adaptation improved effective output throughput by a **median 72.4% across five paired runs** for the workload below. This is a result against the matched, unadapted deterministic Triton path in SGLang v0.5.9—not a universal speedup over current SGLang backends.

## Start here / 成果入口

| 入口 | 内容 |
|---|---|
| [性能实现与复现入口](https://github.com/yu830/graph-budget-runtime/tree/1f1984147608ee6c5dd8b3423e6f41877aaadeb4) | 固定交付版本；按该版本 README 配置并复现 |
| [性能报告、配对结果与限制](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/docs/PERFORMANCE_RESULTS_20261027.md) | 主负载、holdout、A/A 和负结果 |
| [SGLang 集成补丁](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/src/graphbudget/patches/sglang-v0.5.9.patch) | 实际源码变更、显式开关和原生回退 |
| [正式实验汇总](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/results/2026-10-27-performance/formal07-summary.json) | 五组配对的机器可读统计 |
| [技术讲解](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/docs/PERFORMANCE_WALKTHROUGH_ZH.md) | 机制、归属与实验边界 |

## Measured result / 已验证结果

- **设备与软件**：RTX 5070 Laptop，WSL，SGLang v0.5.9 固定源码，Torch 2.9.1+cu128，Triton 3.5.1。
- **模型与执行条件**：Qwen2.5-0.5B-Instruct BF16，TP=1，确定性 Triton，overlap disabled，最多 16 个运行请求；graph capture sizes 为 `[1,2,4,8,12,16]`。
- **负载与基线**：每个正式实验臂 256 个固定请求，每请求 128 个输出 token；五组交错 A/B，比较同配置、未启用该 GEMM 适配的原生确定性路径。
- **吞吐**：五个配对相对变化的中位数为 **+72.4%**；前后两组 A/A 的最大主负载变化为 **8.8%**。A/A 是有限的噪声观测，不是置信区间。
- **正确性与执行证据**：所测请求输出逐 token 一致；最终源码有 340 项 GEMM 数值检查和 255 项跨 batch／重排检查。Nsight Systems 记录确认了实际 GEMM 启动配置变化。

These measurements establish neither general tail-latency improvements nor multi-GPU, production-workload, or cross-model performance. The candidate remains opt-in.

## Mechanism and attribution / 实现与归属

小 M 路径参照 [bill-h-lin 的 SGLang #42145](https://github.com/sgl-project/sglang/pull/42145) 中的 **M16/N64/K128、4 warps、每个输出 tile 一个 CTA** 方案，适配到本项目固定版本的 Triton 路径，并保留原 **FP32 fused-bias** 语义。

本项目工作包括热点定位、源码适配、元数据方案消融、数值与 batch／重排验证，以及请求级端到端评估。**不将上游 tile 思路声明为独立发明的 GEMM 算法，也不把本地验证称为上游接受或合并。** 第三方代码与思路保留原有归属和许可。

## Negative results and boundaries / 负结果与适用边界

- 早期 graph/eager 成本分派没有建立稳定的增量收益；继续使用原生分派。**72.4% 来自上述 GEMM 适配，不归因于图预算规划器。**
- Decode 元数据预分配和融合未建立稳定端到端收益；不作为性能成果，保持默认关闭。
- Holdout TTFT 的方向不一致，p50 配对变化中位数为 +21.5%；不声明所有负载的首 token 或尾延迟改善。
- 没有已确立的 VRAM 节省、KV 扩容或冷启动加速。启动对照存在源码文件系统差异，不能归因于 kernel。
- Nsight Compute 返回 `ERR_NVGPUCTRPERM`；没有 occupancy、warp-stall 或带宽计数器结论。
- 旧默认非确定性模式的 2/43 token 不一致仍保留；不能用确定性模式的成功覆盖该历史失败。

原始请求、trace 和失败记录继续私有保存；公开仓库提供脱敏后的源码、汇总、复现入口和限制，不上传模型权重、个人路径或大型私有日志。

## Version anchors / 版本绑定

| 角色 | 固定提交 |
|---|---|
| 原始工程基线 | `c045569a6c295601d5124bc089202c9785e1b013` |
| 性能实现 | `c62c7432e6e8e0bed43ead82b4d2f42ede2ac855` |
| 性能文档交付 | `b8394da59fb2573c3732c6567d7977774fd922a0` |
| 公开归属与许可补充 | `1f1984147608ee6c5dd8b3423e6f41877aaadeb4` |

实现分支：`codex/decode-metadata-perf-20261027`。本首页不会把后续未经验证的实验自动并入上述性能声明。

## Historical design / 历史设计

本项目起点是 CUDA Graph 显存预算与捕获集合规划。相关实现和负结果保留在性能版本的 [原始技术报告](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/docs/TECHNICAL_REPORT.md) 中。

[2026-10-26 设计阶段 README](https://github.com/yu830/graph-budget-runtime/blob/ce5a41cc069302dd14f671cce2473e040b9aa430/README.md) 及 `main` 中既有设计文件是历史材料，不代表整个项目当前仍未实现。后续 Decode Fast Path 工作独立验证，不覆盖历史结论。

## License

本仓库原创材料见 [Apache-2.0](LICENSE)。SGLang、Triton 及引用的上游修改遵守各自原有许可和归属；本仓库不重新许可第三方材料。SGLang 的 batch-invariant 文件还源自 Thinking Machines Lab 的 MIT 许可实现，完整版权与许可见[第三方许可与归属](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/THIRD_PARTY_NOTICES.md)。项目实现与文档整理使用过 AI 辅助。
