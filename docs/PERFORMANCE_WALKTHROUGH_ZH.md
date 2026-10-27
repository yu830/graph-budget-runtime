# GraphBudget 性能实验讲解提纲

本轮主负载已完成配对测量；结果仅适用于报告中的设备、模型与负载，不代替本人对代码与原理的掌握。

## 输入、输出与实际路径

固定底座为 SGLang v0.5.9，测试模型为 Qwen2.5-0.5B BF16。普通 decode 的 graph replay 前，attention backend 根据当前 batch 的 `seq_lens` 和请求池索引准备 `kv_indptr`、`num_kv_splits`，再由原有 gather 构造紧凑的 KV 索引。`kv_indptr` 是长度为 batch 加一的前缀偏移，split 数控制后续 attention 归约；padding 行也参与准备。

## 候选一：元数据准备

- 原生路径分别执行前缀和、输出写入、split 计算，再执行 KV 索引 gather。简单对照直接写预分配输出；融合候选用单 CTA Triton kernel 同时生成前缀偏移和确定性 split，之后仍在同一 stream 调用原 gather。
- 实现入口是 [`metadata.py`](../src/graphbudget/metadata.py)，接入点随 [SGLang patch](../src/graphbudget/patches/sglang-v0.5.9.patch) 应用到固定上游版本。候选只覆盖代码守卫明确允许的普通 dense、确定性、小 batch 路径，其余情况返回原生路径；每一步均重新读取会变化的元数据。
- 整数结果必须与独立 Python 参考逐元素一致。单独的入口计时包含 Python 提交与同步边界，不能当成纯 kernel 时间，更不能直接当成完整请求收益。

## 候选二：小 batch 矩阵乘法分块

现有确定性矩阵乘法接收 `A[M,K]`、`B[K,N]`，输出 `C[M,N]`，并保留原内核的 FP32 accumulator 内 bias 语义。候选复用既有 persistent Triton kernel，在明确的 BF16 小 `M`、形状与步长条件下，将行分块由 128 调为 16，并调整 warp 数；其余形状继续使用原配置。改变分块是否影响数值须由独立参照、逐值比较和跨 batch 检查决定，不能只从源码推断。

相关 [SGLang PR #42145](https://github.com/sgl-project/sglang/pull/42145) 仍在审，采用另一组小 `M` 几何；本实验将其几何作为对照并保留固定底座的 bias 语义，不将该思路称为自主发明或已被上游合并。

## 基线、证据和失效条件

强基线是原始 `c045569` GraphBudget 及其原生 SGLang patch，使用相同模型、确定性模式、Triton backend、捕获集合、KV 容量、采样与负载；它已经使用 CUDA Graph 和经过选择的静态集合。另有新补丁的 native、预分配、融合与消融对照。旧成本查表实验未证实收益，不应拿较弱的禁图路径替代这个基线。

实际 worker 的 Nsight Systems 节点 trace 观察到 63 次 graph launch；其中 `matmul_kernel_persistent` 有 6,305 条 kernel 活动记录。节点追踪有开销，累计 kernel 时间不等于完整 decode 的关键路径占比；正式收益必须来自无 profiler 的配对运行和 A/A 噪声对照。

在固定的 256 请求、每请求 128 输出 token 主负载下，选中 GEMM 配置的五组吞吐配对提升中位为 72.4%，前后 A/A 最大变化为 8.8%，所有输出精确一致。持出负载 TTFT 变化方向不一致，首 token 中位的配对变化为 +21.5%，不据此声称普遍降低延迟。元数据预分配与融合在两组配对中未建立稳定端到端收益。细节见 [性能结果](PERFORMANCE_RESULTS_20261027.md)。

以下情况不能外推：未命中候选守卫的形状、非连续请求索引、SWA/spec decode、其他精度与设备、不同模型和生产负载。必须分别说明整数契约、真实调用、固定请求输出、局部时延和端到端结果；任何一项未测或失败都保留原样。
