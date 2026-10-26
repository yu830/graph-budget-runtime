# GraphBudget：显存预算约束下的 LLM CUDA Graph 执行规划

> Historical design input. Implementation was explicitly authorized on
> 2026-10-26. See STATUS.md and TECHNICAL_REPORT.md for executed results;
> the design-phase restrictions below do not govern the active implementation.

设计日期：2026-10-25；文档建仓及来源复核：2026-10-26。状态：**设计完成；Runtime 尚未实现；目标 GPU 尚未验证；性能收益未知。**

仓库：`yu830/graph-budget-runtime`。工程底座：SGLang。初版只支持单 GPU、TP=1、dense 小模型、普通 decode；不实现通用推理引擎。

## 1. 一句话定义与真实问题

在给定的 GPU 显存预算和最低 KV 容量下，依据真实捕获成本与执行测量选择 decode CUDA Graph 捕获集合，并在现有安全分派条件内决定执行路径，验证它相对上游默认和同预算人工调优配置的收益与代价。

真实痛点是：动态 Batch 可能跨过 graph capture ceiling，导致执行从 graph replay 退回 eager；服务时间变长又可能增加系统内的并发请求，放大尾延迟。增大所有捕获 shape 则可能增加图池、静态 buffer 和启动成本，挤占 KV 或其它运行余量。

这不是承诺所有模型都会出现断崖，也不是提出新的 CUDA Graph 算法。项目目标是一个能被反证的 Runtime 工程优化：哪一组图、为什么选它、占多少真实内存、执行时到底走什么路径、对请求指标有什么影响。

## 2. 证据与选择理由

### 2.1 直接的问题来源

[SGLang #33483](https://github.com/sgl-project/sglang/issues/33483) 由作者在 L40、Qwen2.5-0.5B 上报告：运行请求上限和 CUDA Graph 上限独立推导，跨越捕获上限后出现持续的执行退化。作者同时比较了增大 graph ceiling 与限制 admission，显示两者的尾延迟代价不同。原文的多倍延迟是**作者设备和负载的结果**。

当前研究基线为 SGLang：

`c00e7e12b64b587da336c4350b1d2484c7dffcd6`

当前实现对低显存设备仍有较小的默认 decode graph cap；但已有不少相邻工作：[#33900](https://github.com/sgl-project/sglang/pull/33900) warning、[#37898](https://github.com/sgl-project/sglang/pull/37898) 已合并的默认范围调整、[#40346](https://github.com/sgl-project/sglang/pull/40346) 与 [#46132](https://github.com/sgl-project/sglang/pull/46132) 的其它显存档位调整。2026-10-26 复核：#37898 已合并；#33900、#40346、#46132 仍 Open。实施时再次刷新状态。不能把这些已知建议算成自己的发现。

### 2.2 为什么不直接做 Workspace 缩容

[vLLM #63399](https://github.com/vllm-project/vllm/issues/63399) 已有他人公开的 [完整 study 与 proposal](https://github.com/lktr-wq/flashinfer-workspace-study/blob/main/PROPOSAL.md)，包括支持条件、范围上界、原生 KV 预算接入和 fallback。其 394→32 MiB、KV +21.27% 和性能结果不属于本项目。

当前 vLLM 也已有 WorkspaceManager、共享资源、graph memory profiling 和 KV budgeting。直接改环境变量或复制该原型，工程增量不充分。Workspace 在 GraphBudget 首版中保持现有 backend 的合法配置，作为要测量的背景资源，不自创未经证明安全的 32 MiB 上限。

### 2.3 为什么必须做端到端验证

[SGLang #38628](https://github.com/sgl-project/sglang/issues/38628) 报告 tiny_gemm 独立更快但多 GPU 推理更慢；原问题需要 GB300 TP=4 和巨型模型，不能在 8GB Laptop 声称复现。它在此只说明：局部 Kernel / graph step 成本是机制证据，最终采用还需要服务请求层面的实验。

## 3. 已有机制与自主增量的严格边界

| 已有机制，不作为项目创新 | 本项目计划增加的部分 |
|---|---|
| 配置 capture batch sizes、ceil padding、超范围 eager fallback | 把实测的整套配置成本、最低 KV 容量、允许的启动预算连成可重复的选择过程 |
| CUDA Graph pool sharing、常见 capture 顺序、graph memory 统计 | 同进程边界下的整集合测量与配置可行性证据；不把单图内存相加 |
| SGLang 部分配置已有 post-capture KV sizing | 在既有适用条件内集成；不另造 KV allocator，不把该能力归为自主成果 |
| 上游 can_run_graph 的安全条件 | 在 eligible=True 后增加可关闭的、依据测量作出的成本选择；只能进一步拒绝 graph，不能放宽安全条件 |
| 上游 Serving benchmark | 生成可复现的路径级证据和公平的端到端比较，保留损失与失败 |

**成立条件：** 最终至少有真实 Runtime 源码变更、正确性和实际路径证据，并在同预算强基线面前提供有用的性能/资源取舍。若只剩一个选配置的脚本，或只是 cap=8 改成 cap=32，则收口为调优研究或局部贡献，不能包装成完成的强主项目。

## 4. 首版机制

### 4.1 Bounded capture-plan selection

输入：固定模型、GPU/软件栈、attention backend、上下文与 batch 范围、GPU 使用预算、最低 KV token capacity、允许的启动时间，以及一份与最终测试分离的校准负载。

候选集合从上游默认、少量人工合理调优的 cap/bucket 配置出发，例如当前设备能运行的 8/16/32/64 范围；**这些只是候选，不是硬编码最优值**。先保留最多 4–6 套候选配置，禁止无界搜索。

每套候选在独立进程中，使用相同模型、依赖、KV/内存规则和预热条件测量整个捕获集合。现有 capture 前测点位于 eager/backend warmup 与 shared-output 准备之后，因此 capture phase delta 不包含更早分配的资源；还须记录整个进程 ready 总量与运行期占用。记录：

- 捕获前后和捕获期间观察到的 GPU 使用情况；PyTorch allocated / reserved 与设备空闲内存分别记录，不相加。只有 PyTorch 统计或设备采样时，准确写“PyTorch 峰值”或“采样观察峰值”，不能称全部 CUDA 分配的绝对峰值。
- 实际 KV blocks / tokens、graph 覆盖范围、启动与 capture 时间。
- eager 与 graph replay 的代表性 decode step 成本；包含 padding 与必要的准备/拷贝边界。
- capture 或执行失败、不可支持的 shape、OOM 和回退原因。

选择可行配置的 Pareto 集合；在预先给定的内存/KV/启动约束下以校准负载目标选一套。选择器的输出是带版本与环境身份的 plan，不是跨设备通用最优解。相同候选与调优预算也提供给静态基线；不能让本项目看最终测试集再选优。

### 4.2 Frozen cost policy in the Runtime

第一版不在线改图、不在请求服务中重建 KV，也不修改 admission scheduler。原因是这些会引入额外正确性状态和干扰，使首版难以定位收益。

从实际执行路径读取上下文特征：实际 batch size、可用的 padded graph size、模型/backend/dtype 和必要的 context-length 分档。只有在校准支持域内，且上游已经判定 graph 可执行时，成本策略才可以选择 graph 或 eager。

核心规则：

1. 上游 `can_run_graph` / 其它 eligibility 判断保留为安全权威。
2. 在任何 graph padding / replay 专用准备和 eager 准备之前确定路径，避免选 eager 后保留错误的 graph 状态。
3. 收益未超过测量噪声、profile 缺失或环境身份失效时，返回原生策略；不得为了展示命中率强行选另一条路径。
4. 超过捕获范围仍走既有安全 fallback。项目不声称消除了所有 overload，也不拒绝请求来伪造吞吐。
5. policy 仅是一张可审查的有界决策表；不训练 Agent、线上 RL 或新模型。

额外两条接口不变量：安全 eligibility 与实际 selected/used path 分开记录；策略拒绝 graph 后，返回的执行标识和统计不能仍误报 graph 已使用。决策只用已有主机元数据或预先维护的信息，不在每步新增 GPU `.item()` / 同步读回来查表；缺少可靠特征时回原生路径。相同一步的目标 graph size 只决定一次，观测与实际执行必须一致。

当前 SGLang 的 bucket 已较密，**eligible 时 eager 更快的区域可能根本不存在**。若实测没有成本翻转，就禁用这部分策略，保留负面结果；不能先宣称 cost dispatch 有收益。

### 4.3 启动闭环与资源约束

预算约束的概念形式：

`weights + KV + workspace + graph/static buffers + transient headroom <= allowed device budget`

实现不能把这些可能重叠的 allocator 统计直接相加。预算判定以一致边界的实际测量为准，分项仅用于解释。

捕获集合涉及共享 pool，多个图的内存成本不一定可加；某些额外小图甚至不增加 pool 峰值。不使用“逐图 size 相加”的背包模型。地址在 graph replay 期间必须稳定；不能释放后重新分配同名 KV 张量却保留原图。

优先复用上游的 KV sizing 和 memory guard。当前 post-capture KV sizing 有 CUDA、环境开关、模型类型和 prefill graph 覆盖等条件；**关闭 prefill graph 后，不能假定仍走后置 KV 分配**。若首版实际走 pre-capture 分支，则保留现有 eager activation headroom，并完整记录路径。

选择计划后，最终启动还要校验实际配置是否满足预算。若预算不满足，启动阶段失败关闭并用预先定义的较小计划重启；最多两次有依据的候选调整。初版不在服务流量中 resize / recapture。未知共享池生命周期条件不靠一次测试通过来跳过：先沿用上游已审查路径，新增共享行为须另行证明。

## 5. 源码落点

研究到的固定版本入口（实施前确认符号与路径仍然存在）：

- `python/sglang/srt/model_executor/model_runner.py`：`ModelRunner._forward_raw` 的 graph eligible → 分派点。
- `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`：capture、eligibility、replay 及相关 padding 行为。
- `python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py`：decode graph 整集合捕获及前后显存计量。
- `python/sglang/srt/model_executor/model_runner_components/kv_pool_runtime.py`：现有 KV sizing / 分配顺序。
- `python/sglang/srt/arg_groups/memory_hook.py` 与 `python/sglang/srt/arg_groups/overrides.py`：分别核对默认 graph 范围和 post-capture KV sizing 适用条件，不凭旧行号修改。

最小 patchset 应把 plan/policy 与原执行器用一个可关闭开关连接；不开启时行为和接口保持原样。不要顺便重构 scheduler、通信、模型层或第三方 CUDA Kernel。

**内核贡献不是首版的硬性装饰。** 首版修改 CUDA Graph 与 Runtime 核心执行路径即可体现系统深度；只有 profiling 证明某个数据准备/packing kernel 是瓶颈时才另加 CUDA/Triton 优化。不能把计划中的 Kernel 写成已实现。

## 6. 本机与实验边界

- RTX 5070 Laptop 8GB，WSL Ubuntu；以实际检查的驱动、功耗、时钟、温度、CUDA/PyTorch/FlashInfer 版本为准。
- 首个模型 `Qwen/Qwen2.5-0.5B-Instruct`，BF16，TP=1，显式选择本机可用的 attention backend。软件支持和最小启动必须实测；不能因为同属 Blackwell 就套用 SM100 内核。
- 条件允许再加入 1.5B 作为 held-out model；不是首版的强制下载项。
- 不在 8GB 上复现 GB300 四卡 tiny_gemm 结果；不购买云 GPU。
- GPU 性能实验串行、接电、稳定功耗；模型编译/JIT 与正式计时分开。NVTX/profiler 只用于诊断，清洁运行用于最终性能。

0.5B 的 KV 容量可由官方 config 算得每 token 12,288 bytes（BF16、24 层、2 KV heads、head_dim=64），但这不含模型、图池、workspace、allocator 和临时 buffer；不能用它代替实际总 VRAM 测量。

## 7. 基线与指标

| 基线/消融 | 用途 |
|---|---|
| U：固定版本上游默认 | 描述现实入口；不能只击败它就称机制成功 |
| E：关闭 CUDA Graph | 检查执行机制，不作为唯一主基线 |
| T：同预算人工调优的静态 cap/buckets | 主要竞争基线；与候选相同校准数据和搜索预算，测试前冻结 |
| P：自动 plan + 原生分派 | 隔离配置选择的作用 |
| C：同一 plan + 成本分派 | 隔离新增 Runtime policy 的作用 |

另可给出逐负载最佳静态配置作为事后 oracle 上界，明确它看过测试集，不冒充可部署基线。

至少比较三类负载：低并发正常 decode；靠近捕获边界的稳定/突发到达；不同 prompt/output 长度造成的 batch 波动。先在本机测可稳定支持的 offered load，再冻结速率；不复制 L40 的 31 req/s。

开放到达率实验应记录每个请求的计划发送、实际发送和完成时间，以及积压/拒绝/超时；不能通过客户端闭环把服务变慢时减少请求数隐藏掉。固定工作总量的 makespan 实验另外报告。

核心指标：

- TTFT、真正的逐 token ITL、请求级 TPOT、P50/P95，样本足够才给 P99；不能用 `batch/throughput` 推导值冒充逐 token 实测。
- completed / failed / timed-out requests、有效输出 tokens/s、makespan。
- graph eligible、真正 replay、各 fallback 原因、实际/填充 batch、padding ratio。
- 实际 KV capacity、GPU allocated/reserved、设备 free-memory 的边界测量、capture time 和 startup time。

先做 A/A 噪声检查，再至少 5 个配对交错回合；校准与测试分开，多 seed。预注册目标可以设为目标负载下 P95 ITL 降低至少 10% 或有效吞吐提高至少 5%，正常负载退化不超过 5%，且均超过 A/A 噪声；**这是是否值得继续的目标，不是已有结果。** TTFT、KV 或启动时间变差必须同时披露。

## 8. 正确性与安全验收

- 同一固定输入与采样条件下，验证输出 token 数、请求归属、终止行为，必要时比较 logits 容差；不能只看自然语言相似。
- 覆盖 captured size、前后边界、padding、有意 fallback，以及 disabled policy；不能用测试直接调用新函数代替实际 loaded module / dispatch 证据。
- 完整记录 source commit、应用 patch、安装模块绝对路径、图或 policy 的实际命中和执行状态。
- 成本策略不能放宽 graph 安全谓词，不能让不支持的 seq/layout/backend 误入图。
- CPU 单测、配置验证、graph capture 成功、真实模型执行通过分别报告。
- Compute Sanitizer 若无法在 WSL 初始化，保留错误并标注该部分未测；不能写“全部内存安全已证明”。

## 9. 开发成立检查点与一天边界

**D0 文档阶段：** 创建设计文档、源码索引、实验协议和公开文档仓库。实现与 GPU 验证尚未开始；实际启动工程实现后才升级为“开发中”。不自动进入下面的实现。

未来授权实施时，以一个有效工程日为目标：

1. M0（60–90 分钟上限）：最小模型启动、真实 graph 路径与少量配置的内存边界；检查初步差异，确认当前上游是否已有等价实现。这是成立信号筛选，不包含第7节完整的多负载、多seed和至少5个配对回合；如果安装占用了预算、剩余观测不足，结论为“未定，需补验证”，不能把未发现收益当成已经证明没有收益。
2. M1：冻结一个可验证的增量，接入有界选择器和最小 Runtime policy；必要 CPU 准备可并行，GPU 串行。
3. M2：独立正确性、真实路径、A/A 和强基线配对实验；形成采用/不采用结论。
4. M3：干净 checkout 重放、技术报告、仓库状态与证据更新。

退出或降级条件：

- 当前栈在 45 分钟的有界安装/诊断后仍不能启动：交付环境阻碍，不在同一天无界换依赖。
- 只靠一个常规静态参数就达到相同效果，且没有可验证的资源/成本决策增量：降级为调优案例，不称完成强主项目。
- 图/静态显存成本在可行范围内几乎无差异：拒绝“节省图内存”的说法，重新评估题目，不能强设不现实预算制造正结果。
- 所有可用区域 graph 都更快：停用成本分派，保留证据；若剩余 plan 只等于手调配置，停止继续包装。
- 确定性正确性失败、graph/KV 地址生命周期无法闭合、必须改多卡才能成立：停止主路径，报告具体阻碍。

若选题需要调整，应显式记录新范围与原因，不能沿用本项目名称替换已声明的成果。

## 10. 文档阶段的交付与后续入口

本仓库包含设计、[实验协议](BENCHMARK_PROTOCOL.md)、[相关工作](RELATED_WORK.md)、[源码索引](SOURCE_INDEX.md)、[实际状态](STATUS.md)和 [upstream lock](../upstream.lock.json)。没有可运行的推理服务、算法实现或测量结果。

进入实施前按 [下一阶段检查点](NEXT_STEPS.md)重新确认开发版本、环境和成立信号；本次设计审查不替代运行验证。

## 11. 原理来源

- [PyTorch CUDA Graph 语义与内存池](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)：固定地址、动态行为限制、共享池约束。
- [NVIDIA CUDA Graph memory issues](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/memory-issues.html)：共享池和内存占用的常见误判。
- [Qwen2.5-0.5B 官方配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json)：KV 算术来源。

本文件是可执行的设计与实验协议，不是已经通过的性能报告。
