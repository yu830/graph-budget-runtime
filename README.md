# GraphBudget

**显存预算约束下的 LLM CUDA Graph 执行规划**

**Status: design complete; implementation and GPU validation pending.**

在给定显存、最低 KV 容量和启动预算下，选择 decode CUDA Graph 捕获集合，并在上游安全条件内评估 graph/eager 执行路径的成本。当前为设计文档仓库，没有可运行的 Runtime，也没有本项目性能结果。

## 为什么做

动态 batch 超过捕获 ceiling 时可能退回 eager，延迟和在途请求数会相互影响。扩大覆盖又可能占用共享图池、静态 buffer 和启动时间，挤占 KV 容量。[SGLang #33483](https://github.com/sgl-project/sglang/issues/33483) 提供了特定设备上的现象；是否在目标设备重现、是否存在优于手调配置的增量，均需实验回答。

## 设计中的模块

1. **捕获集合测量与选择**：最多 4–6 套候选，独立进程测整套配置，依据显存、KV 和启动约束以及独立校准负载选出 frozen plan。共享 pool 成本不按单图相加。
2. **可关闭的成本策略**：保留上游 eligibility，只允许进一步拒绝 graph；缺少 profile、环境不匹配或收益未超过噪声时回到原生策略。路径选择在 graph/eager 专用准备之前完成。
3. **实际执行与请求证据**：记录 loaded module、真实 replay/fallback、batch/padding 和请求事件，评估 TTFT、真实逐 token ITL、吞吐以及资源代价。

capture sizes、ceil padding、graph pool sharing 和部分配置的 post-capture KV sizing 已属于上游。GraphBudget 尚未实现这些设计中的增量；改变一个 cap 不构成项目完成。

## 文档入口

| 文档 | 内容 |
|---|---|
| [设计](docs/DESIGN.md) | 范围、机制、不变量、成立与退出条件 |
| [实验协议](docs/BENCHMARK_PROTOCOL.md) | 同预算强基线、负载、正确性、统计和测量边界 |
| [源码索引](docs/SOURCE_INDEX.md) | 固定研究版本的可定位源码证据 |
| [相关工作](docs/RELATED_WORK.md) | 上游机制、已有提案与结果归属 |
| [当前状态](docs/STATUS.md) | 已完成文档与未运行项目的分别记录 |
| [下一阶段](docs/NEXT_STEPS.md) | 实施前成立检查与停止条件 |
| [研究版本锁](upstream.lock.json) | 研究 SHA；未验证的运行环境字段保留 null |

## 后续验证

首个目标是 RTX 5070 Laptop 8GB / WSL Ubuntu、Qwen2.5-0.5B-Instruct BF16、TP=1、普通 decode，attention backend 待实机确认。此处不是已通过的兼容性声明。

主基线是**同显存、同 KV/启动约束和同调优预算的人工静态配置**。同时比较上游默认、禁用 graph、自动 plan + 原分派、同 plan + 新分派。校准与测试分开，先做 A/A 噪声检查，再做配对实验。全部 eligible 区域 graph 都更快时停用成本策略；仅调 cap 即达到同样效果时降级为调优研究。

研究输入固定为 SGLang [`c00e7e12b64b587da336c4350b1d2484c7dffcd6`](https://github.com/sgl-project/sglang/tree/c00e7e12b64b587da336c4350b1d2484c7dffcd6)。实施时另行选择开发版本，保留此研究快照。

## 已验证结果

2026-10-26 完成设计整理及原始来源/固定源码读取。Runtime 实现 **NOT_STARTED**，GPU 验证 **NOT_RUN**，性能收益 **UNKNOWN**。第三方 issue 的数字只支持其原始环境，不是本项目结果。

## 许可

本仓库新原创文档采用 [Apache-2.0](LICENSE)。第三方源码和文章通过链接引用，权利仍归其原作者；本仓库不重新许可第三方材料。
