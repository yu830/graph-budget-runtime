# Source index

读取日期：2026-10-26。所有以下链接固定到 SGLang 研究 commit `c00e7e12b64b587da336c4350b1d2484c7dffcd6`；未使用当前 main 替代研究版本。Blob SHA 同时记录于 [upstream.lock.json](../upstream.lock.json)。

## 1. model_runner.py

[固定源码入口](https://github.com/sgl-project/sglang/blob/c00e7e12b64b587da336c4350b1d2484c7dffcd6/python/sglang/srt/model_executor/model_runner.py#L1971)

`ModelRunner._forward_raw` 先计算 can_run_graph，再执行 decode runner；回退执行 eager preparation。新策略应在专用准备前决定路径，并将 eligibility 与 used path 分开。

## 2. decode_cuda_graph_runner.py

[固定源码入口](https://github.com/sgl-project/sglang/blob/c00e7e12b64b587da336c4350b1d2484c7dffcd6/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L844)

`can_run_graph` 同时检查 batch ceiling、布局/模式/后端等条件；不能把它替换成单一 batch 判断。capture 路径已有大 shape 优先及共享资源安排。

## 3. cuda_graph_setup.py

[固定源码入口](https://github.com/sgl-project/sglang/blob/c00e7e12b64b587da336c4350b1d2484c7dffcd6/python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py#L703)

decode 捕获在 before/after available-memory 测点内创建 runner；测点之前已有 eager warmup 和共享输出准备，delta 不覆盖所有启动资源。

## 4. kv_pool_runtime.py

[固定源码入口](https://github.com/sgl-project/sglang/blob/c00e7e12b64b587da336c4350b1d2484c7dffcd6/python/sglang/srt/model_executor/model_runner_components/kv_pool_runtime.py#L34)

`is_post_capture_kv_active` 结合计划谓词和 CUDA 平台；`compute_post_capture_kv_resize` 复用现有 backing/finalize 路径。

## 5. memory_hook.py

[固定源码入口](https://github.com/sgl-project/sglang/blob/c00e7e12b64b587da336c4350b1d2484c7dffcd6/python/sglang/srt/arg_groups/memory_hook.py#L89)

固定版本在 <20 GiB 档默认 decode max_bs=8；20–35 GiB 与 35–60 GiB 使用不同档位。只提高默认值不能视为新机制。

## 6. overrides.py

[固定源码入口](https://github.com/sgl-project/sglang/blob/c00e7e12b64b587da336c4350b1d2484c7dffcd6/python/sglang/srt/arg_groups/overrides.py#L1791)

`post_capture_kv_sizing_planned` 检查环境开关、CUDA、DCP、MLA、allocator 和 prefill 覆盖等条件；普通非 decode-only 配置关闭 prefill graph 会使其失效。

## 阅读结论的边界

这里只记录源代码中存在的机制和拟接入位置；没有 import、启动、捕获、replay 或 benchmark 证据。实施时须重新核对调用者、backend 和 pool 生命周期，不能沿旧行号直接修改新版本。

模型 config 的 BF16 KV 算术来自[官方配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json)。2026-10-26 读取文件 SHA-256 为 `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`；这不是模型权重锁，未下载权重。
