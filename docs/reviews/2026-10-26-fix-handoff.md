# GraphBudget：独立审核后的最小修复任务

请完整阅读，然后给出简短执行顺序并直接开始。此任务授权的是隔离、可逆的代码修复与验证，不需要在 M0 或每个子步骤后重新等待确认。保持现有项目范围，不重新做选题调查。

## 1. 角色与目标

你负责将 GraphBudget 的首个核心版本从“限定路径已实现、三处可靠性边界未闭合”推进到“已修复、关键回归已验证、历史证据保留”的下一版。

停止点是可审核的代码改动、验证记录和明确分项状态；本任务不授权合并 main、创建上游 PR、购买算力或公开原始日志。现有明确授权允许的普通开发分支同步可以沿用；没有则完成本地交付，不因发布问题阻塞修复。

顺序：先收紧执行配置与指纹，再修坏 profile 的回退，随后统一 readiness 截止时间，最后最小回归和文档同步。

## 2. 起始事实

- 仓库：https://github.com/yu830/graph-budget-runtime
- 交付分支：codex/core-v1。
- 固定审核提交：a587e20b4a3164f9c8331a25c3e855bd08168240。
- 运行实现提交：9340ee0b3f6e7d6054b4062540ece46c5f087c42。
- SGLang 基线：v0.5.9 / bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2。
- 原 main：ce5a41cc069302dd14f671cce2473e040b9aa430；先读远端核实，不假定其没有后续变化。
- 当前结论：核心普通路径实现；无已建立性能收益；推荐 native dispatch。
- 审核者已离线安装项目包、应用固定上游补丁并独立重跑 57 项 CPU 测试，全部通过、零跳过。
- 审核者独立复算 1,352 个原始文件和最终 SSE 输出。21 场景/815 请求/40 对照、clean 43/43 及所有报告指标一致；没有独立重跑 GPU。
- 当前三项发现并不推翻既有正常配置的实验数据。

没有外部 PR 合并或维护者 Review 前置条件。不要等待其他仓库。

## 3. 先读这些材料

1. 当前 AGENTS.md、git status、git log，以及 main/审核提交到当前分支的 diff；保护未提交工作。
2. 附件 GRAPHBUDGET_INDEPENDENT_REVIEW_20261026.md；若有审核包，读 checks/code/CODE_AUDIT.md 和反例脚本。
3. README.md、docs/STATUS.md、docs/TECHNICAL_REPORT.md。
4. src/graphbudget/identity.py、runtime.py、policy.py。
5. cli.py、experiment.py 和 scripts/run_core.py 的启动/恢复入口。
6. src/graphbudget/patches/manifest.json、sglang-v0.5.9.patch 以及固定上游相关源文件。
7. tests/ 与 results/2026-10-26 中的 plan/profile/summary；GPU 原始证据保留私有。
8. upstream.lock.json、scripts/environment/README.md 和现有隔离 runtime 配置。

若路径不存在，先在现有项目工作区用 rg 查找；不要要求我复制大量命令或重建已经可用的 GPU 环境。

## 4. 必须保留的行为与成果

1. 计划在归一化/内存池/capture 前生效，硬计划错误不允许静默退回无预算默认配置。
2. 策略仅能拒绝上游 eligible graph；不得扩张 eligibility 或把 overflow fallback 说成新的优化。
3. 真实 used 来自实际 replay/eager 调用返回位置；保留 request/run/source 绑定。
4. 生产 hot path 不增加 GPU .item()、D2H、同步或逐 token 文件写入。
5. 完整共享池配置测量，ready/torch allocated/reserved/采样峰值/KV 分开记账。
6. 原默认模式 2/43 不一致、零成本驱动 eager、无性能增益，以及全部旧日志与结果保留。
7. 修改只在任务专用分支或 worktree，SGLang 使用专用 checkout。现有 Skills、其他 PR、简历和共享 GPU 安装不在此次范围。
8. 不为满足“完成”而增加新框架、dashboard、泛用管理平台或第三个 Skill。

## 5. 修复 C1：执行配置的支持范围与身份

已确认的反例：

- load_format: auto → dummy
- torchao_config: 空字符串 → int8wo
- cpu_offload_gb: 0 → 1

当前真实 check_supported 接受以上变化，identity.collect 却生成相同完整身份。上游确实分别改变权重加载、量化实现与 CPU offload；dummy loader 生成随机权重。

请按当前 v1 的窄范围修：

- 对不支持的 dummy/custom loading、TorchAO 和 CPU offload 在模型加载前明确拒绝。
- 对允许的执行相关配置，使用规范化值进入身份匹配；把相同加载边界的 model override、额外 loader 设置纳入有界审查。
- 优先明确 allowlist/不支持条件，不承诺支持未经验证的新量化或卸载方式。
- 不要盲目 hash 全部 ServerArgs；端口、日志目的地等与性能契约无关的项不应造成无意义失效。
- 保持合法默认路径。旧硬计划失配必须拒绝，旧成本 profile 失配回退 native。
- 若新增身份字段或改变实现 fingerprint，生成新版本的实测记录；不能手改旧票据或把旧 profile 换个 fingerprint 继续说是重新校准。

回归必须证明：默认设置通过；上述三种改动被明确拒绝或使旧计划失效；相同语义的规范化值一致；未支持执行模式不会借 identity hash 通过。

## 6. 修复 C2：坏成本表必须按软失败规则回退

已确认的最小输入：

- 已发布 profile 的某 entry.provenance 改为 [] → TypeError。
- entry.graph_ns 改为 JSON 整数 10**400 → OverflowError。

它们在 compile_policy 和真实 RuntimeState 构造中都越过当前 fallback；原有效 profile 为 loaded 正对照。

实现要求：

- membership 前先检查 provenance 类型。
- timing 数值转换与有限性错误统一为 InvalidPolicyError，或按现有设计标记无效行。
- 对 graph_ns/eager_ns/noise_ns、provenance 等同类字段采用有限的表驱动回归，包含 bool/null/数组/过大整数等有意义错误域。
- 有效行策略不变。坏可选数据必须给出可理解原因并沿用 native；硬计划、source drift、模型错误不能被一起吞掉。
- 不用 except Exception 包住整个启动来隐藏缺陷。

## 7. 修复 C3：统一 HTTP readiness 语义与截止时间

现状：

- RuntimeState.ready 在 worker_initialized 时检查 elapsed time。
- 直接 cli launch 阻塞于 run_server，没有后续 HTTP-ready deadline。
- experiment.run_case 已等待 /health 并检查 HTTP-ready 总时长。
- 计划、报告及推荐直接启动命令使用的是 HTTP-ready 意义。

决策：

| 条件 | 采取的实现 |
|---|---|
| 现有生命周期管理可安全复用 | 抽出小型共用监督逻辑，推荐的 direct launch 与实验使用同一 readiness 定义及总deadline |
| 直接复用 run_case 会夹带 benchmark/清理等副作用 | 提取必要的启动、ready等待、退出和清理部分；不要把 benchmark 当生产服务入口 |
| 固定上游 /health 实际只代表 liveness | 明确检查上游真正可服务的 readiness 条件；记录与旧指标的差异，不悄悄重命名历史数据 |
| 为改入口必须进行大规模架构重写 | 先提交最小可监督入口并收紧文档；清楚标记尚未保证的旧入口，不虚报全部修复 |

CPU 生命周期测试至少覆盖：

- worker 已完成、HTTP 迟到或始终未 ready，deadline 到达后状态/退出码准确。
- 正常按时 ready。
- 子进程提前退出或启动失败。
- 只清理本次拥有的进程组；无孤儿进程和无关进程误杀。
- 若启动已成功进入服务，不因单次瞬态健康探测失败杀掉服务；启动截止时间与运行期健康管理分开。

这次不引入通用服务编排框架，也不把启动资源检查宣传为运行期绝对显存配额。

## 8. 文档与最小验证

文档修改：

- 性能阶段 2,078 请求拆成 T/C 1,060、A/A 700、U/E/P 探索 318。
- 成本表实际覆盖 116/1,347 decode steps（8.61%）；boundary 为 0/395；成本拒绝 graph 共 0。
- 静态 T 限定为同一有界候选集内本轮校准选择，不称全局最优。
- “conservative peak” 改为 observed sampled whole-device peak。
- “clean installation” 明确为干净源码加既有 pinned runtime，不伪称全新机器完整依赖重建。
- scripts/environment/README.md 中 2/43 matches 改为 2 mismatches / 41 matches；更新过时 benchmarking pending 状态。
- benchmark.py 顶部 workload 描述同步实际 frozen tokenizer 输入。
- 保留原 PROJECT_COMPLETION.json 与原结果为冻结历史；追加新的修复/审核状态并从 STATUS 链接，避免多份“当前”状态互相矛盾。

验证顺序：

1. 先复现审核反例，保存失败；再修复并确认回归行为。
2. 跑完整已有 CPU 套件，设置 GRAPHBUDGET_SGLANG_SOURCE，使五项真实上游源码检查执行。新增测试只覆盖真实缺口，不堆数量。
3. 检查可安装包、补丁 hash、源码身份、计划失效行为和启动生命周期。
4. 本机 RTX 5070/现有 runtime 可用时，针对受影响启动路径做最小 GPU smoke：实际正确模型加载、native plan、缺/坏 profile fallback、readiness，固定请求逐 token 比较。
5. 若因 fingerprint 变化旧计划失效，以新源码重新产生必要的实测计划/记录；可以为 smoke 用明确标注的一候选小型校准，不把它伪装成完整三候选重选或新性能验收。
6. 没有改变策略且没有新收益假设时，不要求重跑整套 2,078 请求性能实验，不租算力。
7. GPU 受阻时完成所有独立可做的代码/CPU/文档工作，GPU 项标 BLOCKED/NOT_RUN，并写清下一条准确命令。

## 9. 失败处理与停止条件

- 发现上游/API/分支已变化：先核对实际 diff；不覆盖别人工作。可在原固定 SHA 的新 worktree 完成修复。
- 某路径需要未授权云成本、修改共享安装、force push、修改第三方 PR 或公开私有原始证据：该动作暂停；其他工作继续。
- 同一环境阻塞最多三次有依据的尝试，保留具体错误并转向不依赖它的工作，不循环重装环境。
- 修复后没有性能收益是允许的结果，不调整统计口径来制造成功。
- 不把新增 CPU 测试通过称为新的 GPU 或端到端验收通过。

## 10. 交付与自检

交付：

- 小而可审的代码 diff、提交 SHA、当前分支。
- C1/C2/C3 各自修复机制和回归证据。
- 原失败→修复后行为的对照记录。
- 更新文档，保留旧的性能负结果和旧版本身份。
- 新版本分项状态：代码、CPU、真实模型、实际路径、readiness、性能结论分别报告。
- 仍未验证的项目与最短下一条命令。

最终自检：

- [ ] 未支持的实际执行模式无法复用旧硬计划。
- [ ] 合法默认执行配置仍通过。
- [ ] 坏可选 profile 返回 native 且说明原因，硬预算/身份错误仍拒绝启动。
- [ ] 推荐服务启动入口执行准确的 readiness 截止时间并只清理本次进程。
- [ ] 所有原有 CPU 测试与必要新回归实际运行；跳过与环境阻塞单列。
- [ ] GPU 验证没有执行就明确 NOT_RUN；执行了则保留真实模型输出与运行身份。
- [ ] 旧记录未改写，未声称获得加速、VRAM 降低或新的内核优化。
- [ ] 未改动简历、其他 PR、Skills、共享环境或 main。
