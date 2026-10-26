# 下一阶段入口

当前只完成 D0 文档。此列表不自动启动实现、安装、下载模型或 GPU 实验。

1. 刷新上游，选择并记录开发 commit；保留研究 commit，不能把旧观察移到新版本。
2. 复核 GPU 是否空闲、磁盘空间与现有 WSL 栈，在项目隔离环境中进行有界启动检查。
3. M0 用 60–90 分钟核对真实 loaded module、graph replay/fallback、配置整集合内存与初步差异。环境诊断最长 45 分钟；观测不足记未定。
4. 有成立信号后冻结最小增量；按照 [设计](DESIGN.md)接入可关闭 policy，保持原 eligibility，最多两次有依据的启动计划调整。
5. 按 [协议](BENCHMARK_PROTOCOL.md)做正确性、A/A、同预算静态强基线和消融；没有成本翻转就关闭成本策略。
6. 独立检查真实执行证据，干净 checkout 重放，再更新 [状态](STATUS.md)。

不扩展多卡、在线 recapture、admission scheduler 或独立 KV allocator。profiling 未显示需要时，不附加 CUDA/Triton kernel 工作。
