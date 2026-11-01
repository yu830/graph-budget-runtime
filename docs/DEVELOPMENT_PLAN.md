# GraphBudget｜下一轮技术研发与证据补强计划

> **Plan ID**: GB-IMPROVEMENT-20261101-v0.1.0  
> **日期**: 2026-11-01  
> **状态**: 设计文档已准备；本轮未执行新的 GPU 测试、Kernel 修改或性能验收。  
> **工程权威**: Git commit 与原始实验证据；本文件是下一轮开发规范，不覆盖历史记录或本机受控任务状态。  
> **责任分工**: Human 决定重大方向、成本及最终接受；ChatGPT 负责技术方案和独立审查；Codex 是后续代码、实验及受控任务状态的执行者/writer。

## 1. 目标与明确边界

GraphBudget 的下一轮目标不是提高宣传中的加速百分比，而是把已有 **SGLang deterministic decode / Small-M BF16 GEMM** 工程成果提升为可独立复核、可合理归因、具备明确性能边界的 AI Infra 项目；之后仅在真实 Profiling 和上游查重支持时，增加**一项**具有独立技术解释的优化。

最终应能够回答四个问题：

1. 哪个实际 Kernel/Runtime 路径构成瓶颈？时间消耗在哪里，为什么既有实现不适合该形状？
2. 相同源码、相同数值语义、相同服务配置，仅改变优化开关，端到端改善是否仍然存在？
3. 在哪些模型、Batch、请求到达模式下成立；在哪里退化，代价是什么？
4. 新增机制与 SGLang 已有方案有何实质区别，为什么值得保留或提交上游？

**硬约束**：仅使用现有 RTX 5070 Laptop 8GB 与可用本地资源；不租 GPU、不购买 API、不开多 GPU 或不受支持的生产性能声明；不清除旧失败样本、不修改冻结原始结果、不破坏已有 dirty worktree。独立 GPU 测试串行使用现有共享锁。

## 2. 证据基线与版本身份

| 组件 | 已核验身份 | 当前结论 |
| --- | --- | --- |
| GitHub 默认分支 | `main`，2026-11-01 规划前 HEAD `8a35bb98e6147f2e15e109dcc1dc40798b5dce50` | 公开索引及报告，不是可直接运行的完整性能工作树 |
| 原始 Graph/CUDA Graph 预算策略 | `codex/core-v1`，`c045569a6c295601d5124bc089202c9785e1b013` | 在同预算强静态配置前未建立增量收益，保持原生分派 |
| 已发表的 Small-M GEMM 实验 | `codex/decode-metadata-perf-20261027`，`1f1984147608ee6c5dd8b3423e6f41877aaadeb4` | 五组固定负载配对吞吐增益中位数 **+72.4%**；历史、限域观察 |
| v2 residual RMSNorm fast path | `codex/decode-fastpath-v2`，`6cd092bf0cbfaa8c63e2e6ce6eb0bedee6cd86a3` | 局部算子信号积极；正式服务 B/B 稳定性门槛失败，**未采用增量吞吐结论** |
| 受控任务 | 本机 `.agent-workspace/tasks/T-GRAPHBUDGET-DECODE.json`，审查时 revision 23 | `closed / partial`；不得通过编辑文档或手写 JSON 假装重开 |

核心历史事实：

- 2026-10-27：原始路径与 Small-M 路径的五组不同源码版本对比，均为正向（+62.6% 至 +86.0%）；两次 A/A 观测约 +7.9%、+8.8%。该结果支持所测配置上的收益，**不等于**同源码开关级的因果隔离，也没有独立跨设备复现。
- 主负载 256 requests × 128 output tokens、Qwen2.5-0.5B-Instruct、BF16、确定性 Triton、最多 16 running requests；Holdout 吞吐配对中位数约 +5.7%，而 Holdout TTFT p50 配对中位数 **+21.5%（变差）**，方向并不一致。
- 2026-10-28：v2 首轮 B/C 仅 3/5 正向，同代码 B/B 曾达 −80.77%；后续聚焦冻结协议的 B/B 为 +8.69%，超过预定 ±5% 门槛，因此五组正式 B/C **NOT_RUN**；此前探索性 B/C +14.07% 不满足该门槛，不可当作正式采用证据。
- v2 同进程 896×16 Residual RMSNorm 链路测得事件间隔 16.13→9.22 μs（约 −46.6%），只能说明局部机制有效信号，不能推导服务吞吐改善；896×1 同代码噪声仍有 25.58%。
- 上游 **[SGLang #42145](https://github.com/sgl-project/sglang/pull/42145)** 已提出 Small-M M16/N64/K128 几何（审查时 Open）。**[SGLang #46159](https://github.com/sgl-project/sglang/pull/46159)** 已于 2026-10-28 合并 residual add + RMSNorm 融合。必须保留作者归属并重新核对后续差异。

公开证据： [v1 原始性能记录](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/docs/PERFORMANCE_RESULTS_20261027.md) · [v1 JSON](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/results/2026-10-27-performance/formal07-summary.json) · [v2 复测与负结果](https://github.com/yu830/graph-budget-runtime/blob/6cd092bf0cbfaa8c63e2e6ce6eb0bedee6cd86a3/docs/DECODE_FAST_PATH_FOCUSED_V2.md)。

## 3. 研发路线选择（按证据闸门推进）

| 路线 | 价值与代价 | 决定 |
| --- | --- | --- |
| **R1：修复测量可辨识性，重做同源码 GEMM A/B** | 可直接检验当前最有价值的 +72.4% 归因；开发/复测规模可控 | **优先执行** |
| R2：继续打磨 v2 RMSNorm 融合 | 局部延迟改善明确，但同代码服务噪声过大；上游 #46159 已合并，独立性不足 | **冻结，不自动继续** |
| R3：重启 CUDA Graph/Eager 预算路由 | 原始数据没有发现稳定成本翻转，静态强基线已解释结果 | **不重启，除非出现新的瓶颈证据** |
| R4：在可靠基线上新增一个 Profiling 驱动的优化机制 | 有机会产生更强独立贡献，但风险、创新性和收益未知 | **待 R1 判定后选题** |

如果 R1 无法在本地获得可信的服务测量，不能把其工作量无限扩大为“修好所有 WSL 问题”。保留现有有限结果，转为可复现的局部 Kernel/执行路径案例，明确服务级结论尚未建立。

## 4. 运行时与代码变更边界

现有关键链路：请求进入 SGLang Scheduler → ModelRunner → deterministic backend → CUDA Graph capture/replay 或原生安全 fallback → persistent Triton GEMM → logits / token 输出。

**计划的最小新实验接口**：

- 定义三种实验臂：**N** = 同源码、关闭 Small-M；**S** = 同源码、开启 Small-M；**F** = S 加 v2 RMSNorm（冻结备选）。比较 N/S 时除单个优化开关及其必要重新捕获之外，其他源码、模型、依赖、数据、GPU 预算与后端保持一致。
- GEMM 路由只允许显式 opt-in 和已验证的 dtype/shape/stride，当前支持域为 BF16、`0 < M <= 16`、`K % 64 == 0`、`N % 128 == 0` 与已验证 Stride；不支持时原生回退。优先复用当前 patch，而不是重写核心算法。
- CUDA Graph replay 执行的是捕获时的 Kernel 路径；**不要声称每个 token 可以通过环境变量动态选择 Kernel**。N/S 必须各自使用与其开关一致的捕获/进程。
- 继续遵守现有 fused-bias、FP32 累加顺序、数值确定性和合法 fallback。正确性覆盖 **实际加载的 SGLang Patch / captured execution**，不能仅用直接调用自测 Kernel 代替。
- 不增改 scheduler admission、KV pool 生命周期、通用 allocator、跨模型的训练路径；性能实验要从业务请求到输出形成闭环。
- 新工作放在独立、干净的开发分支与工作目录；保留 `main` 的既有公开成果、`codex/core-v1` 和 v1/v2 冻结分支。受控任务是否新建继任条目由 Codex 按现有 helper/writer 规则决定；不直接修改历史 revision 23。

## 5. 分阶段执行与验收

### G0 — 接续、来源对齐与测试资格（Codex）

**输入**：上述三个冻结 commit、v2 原始报告、现有任务 revision 23、当前 SGLang Patch 与本机 GPU 锁。

**动作**：读取当前 Git worktree/dirty 状态、HEAD、可执行环境、模型修订、脚本与私有证据 manifest；禁止覆盖 dirty R1 工作树。检查实验软件全链路是否在可复用的 ext4 路径，记录实际加载 module、图捕获配置与数值 contract。发现上游变更时先写比较说明，**不可静默升级**。

**输出/验收**：一个最小、可回放的环境与源码身份记录；明确 N/S 是否可以由同一源码构建产生，且优化开关独立。若不能，先提交最小可逆的隔离补丁，不进行性能对比。状态为 `READY_FOR_CONTROL` 或 `BLOCKED`，不是性能 PASS。

### G1 — 先解答“环境波动能否区分优化收益”

按同 boot、同 commit、同模型、相同预热、同 transport 与固定负载执行独立 **S/S 控制**。保持现有共享 GPU 锁和实验任务独占；记录每臂发送偏差、完整 makespan、完成/失败/超时请求、SSE 重组 token、TTFT 和 ITL，以及 GPU SM 时钟、功耗、温度、服务阶段与日志异常。

针对已观察的问题分层诊断：客户端/网络发送 → HTTP 接收 → TokenizerManager 入口 → Scheduler 队列 → GPU graph replay → 输出。历史曾在收齐请求体后、TokenizerManager 入口前出现约 3.18s 等待；CUDA unknown error 原因仍未知。**SM 时钟差异仅是相关观测，不提前认定为因果根因**。

沿用 v2 已冻结的 **独立同代码变化绝对值 ≤5%** 作为本轮资格门槛（工程阈值，不是统计显著性）。一次失败先提出一项有日志支持的可逆控制/诊断修复，再进行一次完整控制；没有可验证的新解释则停止服务级性能验收。不要延长测试、删慢样本或放宽门槛直到通过。

**验收**：PASS = 独立控制满足预声明门槛、成功请求与失败记录完整、没有未解释的 CUDA 错误；否则 `BLOCKED_BY_MEASUREMENT_ENVIRONMENT`。若持续失败，允许整理局部 GPU 证据，但禁止声称新的 E2E 吞吐加速。

### G2 — 同源码 GEMM 因果验证与跨负载复现

只有 G1 PASS 才开始新的、预注册的 N/S 服务对比：

1. 在同一源码版本仅切换 Small-M 开关，每臂独立启动及 capture；相同模型、后端、图集合、KV capacity、预热、请求序列和功耗条件。按 **NS / SN / NS / SN / NS** 交错完成至少五组配对；固定负载仍为 256×128，不可用临时改变负载找成绩。
2. 对每个实验臂逐请求验证输出 token 和边界行为；对 GEMM 支持域做精确数值、M=1/4/8/16、M=17 fallback、不同合法 Stride 及重排检查。记录 Kernel 路径的真实命中与 capture replay；诊断可用 Nsight Systems，但正式吞吐来自无 profiler 的运行。
3. 在不看新主负载结果选参的前提下，另测低并发、混合输出长度/突发到达与 Holdout；有余量再尝试 **Qwen2.5-1.5B 的已知可运行低并发配置**，如不能保证 8GB 运行则明确 `NOT_RUN`。性能数据按工作负载分别报告，不能把跨场景样本合并成一个通用加速比。
4. 与固定版本的 **SGLang 默认推理配置**另外做现实参照；它未必具有相同的严格确定性数值契约，因此需要分别说明性能和正确性，不把它混作 N/S 的因果基线。
5. 保留每个请求与失败、Holdout TTFT p50/p95、ITL、makespan、显存/KV/启动及 capture 代价。Nsight Compute 权限不可用时标记 `NOT_RUN`，不臆造 occupancy/带宽数据。

**采用规则（需在数据前冻结）**：五组配对至少 4/5 正向，吞吐中位数增益严格超过 `max(5%, 前后同源码控制绝对变化)`；所有声明支持域的数值/Token 验证 PASS，且没有超出预定门槛的关键 TTFT / tail-latency / 错误率退化。若收益仅在固定饱和负载上成立，应按该范围接受，不升级为广义 serving 性能改善。五组样本不自动构成显著性检验。

### G3 — 新增独立技术价值：先定位再选机制

**不预设必须继续做 RMSNorm，也不为了补充“创新点”而新增 Kernel。**

先用真实 decode Trace 对 Kernel 耗时、调用次数、shape 分布、graph 节点/调度开销做瓶颈分解，并用 Amdahl 上界判断：某阶段占比为 `p`，即使该阶段加速 `s` 倍，端到端最多改善 `1 / ((1-p) + p/s)` 倍。任何主张必须复核该阶段是否处于关键路径，不能把重叠时长直接相加。

**可能选题，非已决定的实现**：
- 形状相关的确定性 GEMM 路由/Tile 选择，证明优于单一 M16 配置而不破坏 bitwise contract、capture 路径与 fallback；与 #42145 明确分离原创增量。
- 若 Trace 证实另外一个 GPU 数据准备或 Kernel path 在关键路径上具有真实开销，设计一次有对照的融合/访存/launch 优化，附负载和资源代价。
- 若新 SGLang 版本已有等价实现，则优先做差异审查或可复现 regression 贡献，不包装重复技术为新发明。

**选题门槛**：上游源码/PR 查重、具体输入触发点、独立强基线、理论收益上界、最小反例、数值契约、可在 8GB GPU 实际执行。选一个机制与一个主要消融；无法区分现有方案时停止。进入上游 PR 另需 Human 审核、可维护性和仓库政策检查。

### G4 — 交付、复核与公开声明

交付物以修改实际代码和复现实验为核心，而非报告数量：
- 可定位 Git commit / SGLang patch / 精确环境锁及单入口复现命令；干净 checkout 验证。
- 机器可读的 N/S 或 S/F 控制与配对结果、请求级输出和独立统计复算；公开只放已脱敏的摘要及可复现指令，原始私有 Trace 保留在受保护存储。
- 对于每项假设给出 `ADOPT`、`REJECT` 或 `NEEDS VERIFICATION`，明确证据范围、失败/反例及决定。更新当前综合成果与技术报告，不覆盖 10-27/10-28 冻结报告。
- 更新中英文 README、STATUS、相关工作及（必要时）LaTeX 技术报告源；只写可被 commit 和原始运行数据支持的事实。最终由 ChatGPT 独立回读远端与关键产物、Human 接受。

## 6. 停止与改向条件

| 观察 | 处理 |
| --- | --- |
| 同源码 S/S 在一次针对性修复后仍不能通过已冻结控制门槛 | 停止 E2E 性能比较；保留测量失败，转为局部证据或收尾 |
| N/S 不再有稳定收益，或低于同等成本/更强原生路径 | 不继续宣传 +72.4% 为当前独立效果；更正当前综合结论与 README |
| 主负载正向，但 Holdout/关键 TTFT 或资源代价明显恶化 | 限定负载，保持 opt-in；拒绝通用加速声明 |
| 上游已有等价机制，或新的 Kernel 时间上界不足以影响请求级性能 | 不新开发同质化 Kernel，寻求其他已证实热点或关闭该方向 |
| 正确性/地址生命周期/graph replay 契约无法证明 | 停止性能采用；先处理确定性与安全，不放宽测试 |
| 需要付费 GPU、改变核心产品方向或重做整个项目 | 返回 Human 决策，不能视为已有授权 |

## 7. 给 Codex 的首个工程交接

**当前只授权本次文档设计与 GitHub 同步；本文件本身不代表已运行下面的步骤。** 后续启动时：

1. 读取本文件版本及 GitHub 远端提交、[v1 实验报告](https://github.com/yu830/graph-budget-runtime/blob/1f1984147608ee6c5dd8b3423e6f41877aaadeb4/docs/PERFORMANCE_RESULTS_20261027.md)、[v2 focused 结果](https://github.com/yu830/graph-budget-runtime/blob/6cd092bf0cbfaa8c63e2e6ce6eb0bedee6cd86a3/docs/DECODE_FAST_PATH_FOCUSED_V2.md)，并回读本机受控任务及 writer/revision，不复活旧 closed task 或触碰 dirty worktree。
2. 在独立开发目录建立新任务/分支与最小 G0 计划；若任务 ID 或 writer 与旧状态冲突，以实际 helper 返回为准，不手改受控 JSON。
3. 先交付 `G0` 的源码身份、N/S 可切换性、控制资格方案以及最小风险实现差异。获得该项有效证据后，才进入 G1，不提前声称任何新的 GPU 结果。
4. 每一验收单元回传：`plan id/version`、源码/环境哈希、实际执行命令、输出位置、正负结果、原始证据、阻塞/偏离与下一步。ChatGPT 复核后再决定是否推进下一依赖阶段。

**成功标准**不是“重新跑出 72.4%”，而是：可解释的真实机制、能证伪的公平对比、受控波动、准确的性能边界，以及至少一个可以独立讲清楚的 Kernel/Runtime 工程改进。
