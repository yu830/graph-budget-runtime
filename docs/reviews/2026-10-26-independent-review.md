# GraphBudget 首个核心版本：独立审核

审核日期：2026-10-26（Asia/Taipei）
固定提交：[a587e20b4a3164f9c8331a25c3e855bd08168240](https://github.com/yu830/graph-budget-runtime/tree/a587e20b4a3164f9c8331a25c3e855bd08168240)
上游：SGLang v0.5.9，bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2
[原审核入口](https://drive.google.com/file/d/1WA-8Vv6cZWfmS0wZSI0f58UP0NxFAk-m/view) · [原私有证据目录](https://drive.google.com/drive/folders/1qoFuk-AXM9gbIh3N_KhpV9eh2OzZeU8C)

## 1. 结论与建议

**核心链路已经真实实现，提交的实验结果可以复算；“性能收益未确立”判断成立。作为可复用、承诺严格配置失效处理的 Runtime 工具，当前仍有三个需要修复的边界问题，不能直接给予无条件的完整验收。**

本次发现不推翻已记录的默认参数实验。它们限制的是启动器在配置变化、坏成本表和直接服务启动情况下的可靠性。建议在现有分支完成最小修复，追加新版本验收；保留全部既有负结果。

| 审核维度 | 判断 | 含义 |
|---|---|---|
| 核心实现 | PARTIAL | 正常限定配置的完整链路实现；C1/C2/C3 尚未闭合 |
| CPU 安装、补丁与已有测试 | PASS | 隔离安装包、固定上游补丁、57 项 CPU 测试，零跳过 |
| 模型正确性证据 | PASS，限定已提交模式 | deterministic + Triton batch-invariant matmul 的原始输出可独立复算；默认模式仍有失败 |
| 真实执行路径证据 | PASS，限定软件调用观察 | 调用位置、请求归属、计数及输出一致；不等于全局 CUDA 内存安全证明 |
| 端到端性能收益 | UNKNOWN / NOT_ESTABLISHED | 本轮采纳门槛 FAIL；不能宣称加速，也不能外推所有设备绝无收益 |
| 可复现性 | PARTIAL | CPU 与原始数据复算已独立执行；作者的干净源码 GPU replay 记录成立，审核者未重跑 GPU |

审核环境没有可用 NVIDIA GPU。这里的独立验证包括源码检查、CPU 执行及原始数据复算，不把这些工作称为审核者独立 GPU 复现。

## 2. 实际完成的独立验证

- 从固定 GitHub 提交取得全部 51 个文件，逐一核对 Git blob SHA。
- 核对 main 到提交的三个提交及文件差异；从实际运行提交 9340ee0 到最终提交 a587e20 的变化未改动 src/graphbudget 运行实现。
- Drive 的 11 个报告/结果镜像与固定 GitHub 文件逐字节一致；入口和发布回执是额外文件。原目录共 14 个文件，包括原始 ZIP。
- 下载原始 ZIP，SHA-256 为 14cf06626a7dfebecc563e9020b83123fa51458ebf6f01628ca6ef3248a665c4；包内 ORIGINAL_MANIFEST 的 1,352 个文件全部匹配，公开 RAW_MANIFEST 的 955 个索引项全部匹配。
- 克隆上游固定提交，调用项目原有 apply-patch，确认补丁及五个修改文件的 hash。
- 在单独 Python 环境中离线安装项目包，确认补丁文件随包安装；运行全部 57 项 CPU 测试，退出码 0、零跳过。第一次执行因审核沙箱禁用本机回环 socket 而受阻，原日志已保留；这是环境阻断，后续在允许回环服务的环境中完整重跑。
- 独立复算脚本不导入项目的验证或统计函数。直接解析原始 SSE 字符串，重建 token ID、请求 ID、结束条件和时间记录，核对发布结果。
- 另用未修改的项目函数执行 CPU 最小反例，证实 C1 和 C2。外部 GPU、依赖和模型观察在这些反例中是明确标识的 stub。

完整复算与 CPU 日志在本包 checks/；原私有 GPU 数据不重复打包。

## 3. 必须修复的三个问题

### C1 / P1：改变实际执行方式的参数没有进入指纹，也未被支持范围检查拒绝

位置：

- [identity.py](https://github.com/yu830/graph-budget-runtime/blob/a587e20b4a3164f9c8331a25c3e855bd08168240/src/graphbudget/identity.py)：CONFIG_KEYS 与 collect，19–24、81–113 行。
- [runtime.py](https://github.com/yu830/graph-budget-runtime/blob/a587e20b4a3164f9c8331a25c3e855bd08168240/src/graphbudget/runtime.py)：check_supported 与计划/成本指纹检查，46–62、76–100 行。

最小反例调用真实 check_supported 和 identity.collect，仅对外部观察作 CPU stub。以下各项单独改变，都被接受，而且得到完全相同的 identity 和 fingerprint：

| 参数 | 原值 | 新值 |
|---|---|---|
| load_format | auto | dummy |
| torchao_config | 空字符串 | int8wo |
| cpu_offload_gb | 0 | 1 |

固定上游的模型加载器、TorchAO 应用和 offloader 确实使用这些参数。其中 [DummyModelLoader](https://github.com/sgl-project/sglang/blob/bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2/python/sglang/srt/model_loader/loader.py) 生成随机权重，因此仅对指定模型文件作 hash 不能证明本次执行加载了那些权重。

**影响：**执行机制改变后，旧计划/成本数据仍可能通过身份检查。**这不表示本次实验实际使用了 dummy loader 或量化权重。** 本次默认参数、命令和数据仍须按各自记录判断。

**最小修复：**v1 明确拒绝未支持的 dummy/custom loading、TorchAO 和 CPU offload；将仍支持的、会改变加载与执行的规范化设置纳入身份。检查同一加载边界的模型 override、额外 loader 配置，建立有限的支持范围，避免把无关端口和日志路径全部塞进指纹。

**验收：**合法默认设置继续通过；上述改动必须在模型加载前被拒绝，或使身份变化并拒绝旧硬计划。旧成本表按软失败规则回退 native。原历史证据的指纹不得改写。

### C2 / P2：损坏的可选成本表会让启动崩溃，而不是回退 native

位置：[policy.py](https://github.com/yu830/graph-budget-runtime/blob/a587e20b4a3164f9c8331a25c3e855bd08168240/src/graphbudget/policy.py) 51–58、131–132 行，以及 runtime.py 93–109 行。

对发布的有效 profile 设置一个坏字段：

| 输入 | compile_policy | RuntimeState 构造 |
|---|---|---|
| 原有效 profile | loaded | loaded |
| entry.provenance = [] | TypeError | TypeError |
| entry.graph_ns = 10**400 | OverflowError | OverflowError |

它们是合法 JSON，但不符合成本表字段要求；当前异常越过软失败处理。已在未修改的源码上重复证实。

**最小修复：**集合成员判断前检查 provenance 类型；将数值转换溢出统一转换为明确的 InvalidPolicyError 或现有无效行状态。用少量表驱动用例覆盖同类字段即可。不要使用包住整个启动过程的 except Exception 来掩盖硬预算或模型错误。

**验收：**坏成本表产生有原因记录的 native fallback；有效表行为不变；失效硬计划继续拒绝启动。

### C3 / P2：直接启动检查 worker 初始化时间，却没有执行同一 HTTP-ready 截止时间

位置：

- runtime.py 143–175：模型 worker 初始化完成时检查 elapsed time。
- [cli.py](https://github.com/yu830/graph-budget-runtime/blob/a587e20b4a3164f9c8331a25c3e855bd08168240/src/graphbudget/cli.py) 68–72：直接调用阻塞的 run_server。
- [experiment.py](https://github.com/yu830/graph-budget-runtime/blob/a587e20b4a3164f9c8331a25c3e855bd08168240/src/graphbudget/experiment.py) 117–137：只有受控实验路径等待 /health 并检查 HTTP-ready 总时长。

计划中的 startup 指标定义为到 HTTP readiness 的时间。worker 初始化之后仍有 HTTP/lifespan/warmup 阶段。README 给出的直接 cli launch 路径没有监督这段额外时间。

**最小修复：**复用现有生命周期管理，提供统一、有截止时间的受监督启动路径；或明确区分 worker-init 与 HTTP-ready 两个指标，并将保证后者的正式入口限定为受监督入口。优先使 README 的推荐入口真正满足承诺。

**验收：**CPU 生命周期测试覆盖“worker 很快完成、HTTP 延迟或永不 ready”“按时 ready”“子进程提前退出”，确认准确的状态、退出码和本次进程清理。不要清理不属于本次启动的进程。

这是源码可证实的入口合同差异，没有执行 GPU 超时故障注入。本次 run_case 已有 HTTP 检查，既有启动数据不因此失效。

## 4. 已可认可的工程实现

限定配置下，以下机制有源码及测试支持：

1. 计划在 ServerArgs 默认值归一化、资源分配和 graph capture 前生效，不是捕获后修改字段。
2. 选择器按完整共享池配置测量，不累加可能共享内存的单图开销。
3. 策略在上游 eligibility 后、graph/eager 各自准备前执行；只能拒绝已允许的 graph。
4. used 由真正 replay 调用和 eager forward 返回位置记录，能够与 selected 区分。
5. 请求归属和 warmup/正式请求分离得到验证；校准同步不混入正式性能轮次。
6. 正常策略使用已有 CPU 元数据，没有额外逐步 GPU readback 或 synchronization。只有编译后的表查找是 O(1)，整个 context 提取仍与 batch 长度有关。
7. KV 在所选 SGLang v0.5.9 中先于 capture 分配。项目未增加 allocator、在线 recapture、KV resize、admission scheduler 或自定义 CUDA/Triton kernel。

显存约束应表述为“完整候选的实测预算筛选与启动时资源检查”。100 ms NVML 采样和 ready 快照不构成整个服务运行期的严格显存配额或绝对峰值上界。

## 5. 正确性与真实路径：数据复算通过的精确范围

已重建并核对：

| 范围 | 结果 |
|---|---|
| 模型验证 | 21 个场景、815 次请求 |
| U/E/T/P/C 同负载两两比较 | 40 组全部精确 token 一致 |
| 捕获上限控制 | overflow control 与参照精确一致 |
| 验证期间 timed decode | 1,839 步；每条请求均对应 output_tokens−1 个 decode 步 |
| 干净源码 replay | 43/43 精确一致，520 output tokens，68 timed decode 步 |
| 成本校准 | 7 个表项的中位值、噪声、来源 hash 可重构，0 个 eager win |

所验证设置是 Qwen2.5-0.5B-Instruct BF16、TP=1、RTX 5070 Laptop、固定 SGLang/Torch/Triton 环境、deterministic + 上游 Triton batch-invariant matmul。默认模式的 cal-007、cal-014 两个请求仍保留 2/43 不一致，没有被覆盖为成功。

固定输入使用 ignore_eos 和指定输出长度，不能自动扩展为真实 EOS early-stop 的完整覆盖。没有全词表 logits 分析或 Compute Sanitizer；没有多卡、其他模型、长上下文生产流量的验证。

## 6. 性能结论与项目当前价值

### 6.1 请求总数需要拆开说明

| 性能阶段组成 | Case 数 | 请求数 |
|---|---:|---:|
| 正式 T/C 配对，三类负载各五轮 | 30 | 1,060 |
| A/A 边界负载，五轮 | 10 | 700 |
| U/E/P 单轮探索 | 9 | 318 |
| 合计 | 49 | 2,078 |

2,078 是性能阶段合计，并非全部来自 T/C 正式配对。

### 6.2 发布的数值可以独立复算，但不能支持加速

以下均为五轮配对 (C−T)/T 的中位数；吞吐正数较好，延迟正数较差。

| 负载 | Tokens/s | P95 TTFT | P95 TPOT | P95 token ITL |
|---|---:|---:|---:|---:|
| Normal | −0.25% | −5.53% | −7.95% | +0.22% |
| Boundary | −2.18% | +10.52% | +3.92% | −20.18% |
| Mixed | −0.11% | +13.18% | +52.35% | +88.07% |

边界 A/A 的最大绝对变化为吞吐 16.22%、TTFT 95.54%、TPOT 37.70%、ITL 64.50%。这些只是已观察的对照波动，不是置信区间，也不能充当 normal/mixed 的 A/A 数据。

原始正式请求确有单 token SSE 到达事件，因此可计算客户端观察的 token ITL；它不是 kernel latency。单轮仅约 1.6–2.9 秒，prompt 14–45 tokens、输出 8–16 tokens；context_length=4096 是配置上限，不是测试了 4K 上下文。

### 6.3 策略实际上几乎没有覆盖目标运行区间

| 正式 C 负载 | decode 步数 | 命中实测成本表 | 命中率 | 因成本拒绝 graph |
|---|---:|---:|---:|---:|
| Normal | 666 | 110 | 16.52% | 0 |
| Boundary | 395 | 0 | 0% | 0 |
| Mixed | 286 | 6 | 2.10% | 0 |
| 合计 | 1,347 | 116 | 8.61% | 0 |

7 个成本 key 均是 batch=1、context 32–38。边界负载没有一次命中成本表，全部正式运行没有一次成本驱动的 eager 切换。因而边界 ITL 的名义下降没有“策略改变实际执行”的因果依据。

P 的配置与 T 相同。T 是同一有界候选集内单次校准选出的静态配置，公平性比 graph-off 对照更强，但不能称为统计稳定的全局最优静态配置。

7 GiB 预算下三个候选仅约 2.2–2.3 GiB，全部可行、KV 都是 4,096 tokens，没有显示资源紧张场景的改善。现在可以证明预算规划和验证机制运行，不能证明降低 VRAM、扩大 KV 或提高吞吐。

**判断：**当前价值是一个具备真实 Runtime 接入与证据闭环的工程原型，以及一次保留负结果的性能研究。它还没有完成原先“解决有收益的资源/执行瓶颈”的优化目标。优先关闭可靠性缺口；只有找到会改变真实路径或资源配置的可测假设，再投入性能实验。无需先购买算力或无限增加重复轮次。

## 7. 文档与完成状态应如何更新

保留原冻结 PROJECT_COMPLETION.json 和 2026-10-26 原始结果。追加独立审核记录及新版本状态，不把旧指纹替换成新指纹。

建议新的状态措辞：

> 固定配置下核心执行与验证链路已实现，原始结果独立复算一致；三个启动/配置边界问题待修复；性能收益未确立，维持 native dispatch。

同时修正：

- “2/43 matches” → “2 of 43 requests mismatched (41 matched)”。
- scripts/environment/README.md 的 “Full benchmarking remains pending” 为早期状态，应注明历史并链接本次结果。
- “conservative peak” → “observed sampled whole-device peak”。
- “clean installation” → “clean-source installation and replay using the pinned existing runtime”。
- benchmark.py 顶部 examples 的说明更新为实际 frozen tokenizer workloads。
- 补充请求总数分母、成本命中率、候选搜索范围和短负载限制。

## 8. 下一步执行及验收边界

配套 GRAPHBUDGET_CODEX_FIX_HANDOFF_20261026.md 给出最小修复任务。建议顺序：

1. C1：支持范围和身份匹配。
2. C2：可选成本表的输入验证及 native fallback。
3. C3：统一服务 readiness 截止时间及生命周期。
4. 文档纠正、保留历史证据、追加新版本记录。
5. CPU 原有套件与必要新回归；本机只做受影响启动路径的最小 GPU smoke 与精确输出对照。

代码或身份结构变化后，旧 plan/profile 应依原规则失效。新版本需要新的实测身份和对应记录；不可复制旧性能数字并改指纹。没有改变优化策略和发现新收益机制前，不要求重新跑整套性能实验。

本轮没有修改用户项目、GitHub 分支、main、PR、原始完成票据或简历，也没有向第三方发送证据。审核产物是对原交付的补充，不是替换。
