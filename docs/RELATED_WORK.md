# Related work and attribution

来源与状态复核日期：2026-10-26。性能数字属于各作者的设备、模型和负载；GraphBudget 未复现，未将其作为自身收益。

| 原始来源 | 已读内容与状态 | 对本项目的约束 |
|---|---|---|
| [SGLang #33483](https://github.com/sgl-project/sglang/issues/33483) | Open；L40 / Qwen2.5-0.5B 的 graph ceiling 与 admission 失配报告 | 说明研究问题；不预设 Laptop 必然出现相同退化 |
| [SGLang #33900](https://github.com/sgl-project/sglang/pull/33900) | Open，未合并；超出最大捕获 shape 的一次性告警 | 告警不是新的执行策略 |
| [SGLang #37898](https://github.com/sgl-project/sglang/pull/37898) | 已合并；32GB 档默认范围与 prefill 设置调整 | 上游已有配置改进，不能重新认领 |
| [SGLang #40346](https://github.com/sgl-project/sglang/pull/40346) | Open，未合并；32GB SM120 默认 cap 提议 | 静态上限是必须竞争的已有路线 |
| [SGLang #46132](https://github.com/sgl-project/sglang/pull/46132) | Open，未合并；35–60 GiB 默认 cap 提议 | 不能只比较未经调优默认值 |
| [SGLang #38628](https://github.com/sgl-project/sglang/issues/38628) | Open；GB300 TP=4 的 tiny_gemm 局部更快、端到端更慢报告 | kernel/step 测量不代替请求层验证；不声称在 8GB 复现 |
| [vLLM #63399](https://github.com/vllm-project/vllm/issues/63399) 与 [作者 proposal](https://github.com/lktr-wq/flashinfer-workspace-study/blob/main/PROPOSAL.md) | Issue Open；工作区缩容、原生 KV 预算接入及端到端取舍 | 394→32 MiB、KV +21.27% 和相关性能结果归 lktr-wq；不是 GraphBudget 成果 |

workspace study 的范围包括 SM89、FA2、固定模型/后端/服务范围及冻结的 source；其 proposal 还披露首请求变慢、质量任务未全部正确及原型集成限制。GraphBudget 不复制其 32 MiB 上界，首版保留 backend 合法 workspace 配置。

## 上游已有机制

固定研究版本已包含 capture size 配置、ceil padding、超过范围的 eager fallback、共享 graph pool、graph memory 统计和有条件的 post-capture KV sizing。源码入口与具体边界见 [SOURCE_INDEX.md](SOURCE_INDEX.md)。

本项目计划增加可重复的预算/配置决策、最小可关闭成本分派及实际路径证据。是否值得实现仍取决于与同预算静态强基线的比较。检索未发现等价实现不构成首次发明声明。

## 语义和模型资料

- [PyTorch CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)：固定地址、动态行为限制与共享 pool 的 replay 顺序/非并发条件；本次 stable 入口转至 2.14 文档。
- [NVIDIA CUDA Graph memory issues](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/memory-issues.html)：allocator、pool/stream 隔离、静态张量和 reserved memory 边界。
- [Qwen2.5-0.5B-Instruct 官方配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json)：24 层、2 KV heads、896/14=64 head dimension，BF16 K/V 的理论大小为 `2 * 24 * 2 * 64 * 2 = 12,288 bytes/token`；不包含权重、workspace、pool、临时与 allocator 开销。

实施前需再次核对版本、状态与支持域。以上文档读取没有执行 proposal 或 issue 内的命令。
