# Benchmark protocol

状态：协议已制定，实验未运行。规范版本：2026-10-26。主设计见 [DESIGN.md](DESIGN.md)。

## 实验冻结与公平预算

每次实验先固定 source commit、patch、依赖版本、模型 revision、后端、dtype、功耗条件、capture 配置、KV 规则、启动预算及 workload seed。当前未知项保留 null，不把研究快照当可运行环境锁。

候选限制为 4–6 套；每个候选独立进程测量整个捕获集合。人工静态 T 与自动计划 P 使用相同显存、最低 KV、启动和调优预算、相同校准负载；最终测试集不参与选择。计划与策略在测试前冻结。

capture phase delta、进程 ready 总量、运行期占用分别记录；PyTorch allocated/reserved 与设备空闲内存不相加。graph pool 不按逐图大小求和。失败、OOM、fallback 均进入记录。

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


## 退出条件与报告

成立筛选、45 分钟环境诊断上限、60–90 分钟 M0 上限和降级条件见 [DESIGN.md 第 9 节](DESIGN.md#9-开发成立检查点与一天边界)。M0 是筛选，不是完整性能验收。安装耗尽预算时报告未定。

确定性错误、地址生命周期无法闭合时停止主路径；所有 eligible 区域 graph 都更快时关闭成本策略并报告负结果。仅提高静态 cap 达到相同效果、或 pool 成本无可辨别差异时降级或调整题目。

最终报告同时呈现收益、退化、失败数、噪声、原始请求事件、路径统计和资源代价。至少 5 个配对交错回合并先做 A/A；若样本不支持 P99 或显著结论，应明确不报告或标为不确定。没有运行记录时不得创建结果 JSON 或通过徽章。
