# brief-get-goal-by-utov-v1

## 任务目标

用 `clark-utov-build` 解决
`/libs/arm64-v8a/libEncryptor.so`。

默认runner在
`runner`

目标不是写一段“看起来合理”的解释，而是让下一个短上下文 LLM 通过 utov 和真实
runner 自己拿到闭合成果：

1. 还原出可运行的 Python 算法；
2. 用真实 runner output 和 Python output 对拍；
3. 用 utov 证明 sink / provenance 或 boundary / parity；
4. 留下一份简短报告，说明公式、证据和验证命令。

## 红线

在你自己闭合候选公式之前，不要读取这些历史答案文件：

 

你可以读取这些：

 
- `libs/arm64-v8a/libEncryptor.so`
- `libs/arm64-v8a/trace.txt`
- `runner/**`
- `clark-utov-build/**`
- 你自己为本次尝试新建的工作文件

不要在以下条件没满足时宣布“算法闭合”：

1. output sink 已确认；
2. 候选公式能连到 output-producing path，或者有明确 boundary 解释；
3. 真实 runner 的 held-out parity 通过。

不要把不同 `runner.rerun(...)` 的 observation 静默混在一起。如果某个值每次执行都可能变，就必须让 observation 和同一次执行的 `rr.output` 配对。

utov 已经有的 primitive，要优先使用。必须手搓时，要在报告里写清楚为什么 utov 递不到这一步。

不要预设目标是纯 `input -> output`。依赖关系必须由证据证明。

## 边界

只闭合 runner 返回的可见 output。

不要求证明每一段内部 native 计算，除非它是解释可见 output 必需的路径。

不要修改 `clark-utov-build` 引擎源码。

不要为了通过 parity 降低阈值。

不要把某个局部窗口的 symbolic/local closure 当成最终算法闭合。

你可以在新 work 目录，或 `work-chatgpt5.4/` 里，用新名字创建脚本和报告。

## 必须尽量使用的 utov 能力

能用就用，不能用就说明原因：

- `engine.runner_client.UnidbgTextTraceReader`
- `engine.runner_client.SubprocessRunnerAdapter`
- `engine.runner_client.ObservePoint`
- `engine.runner_client.mem_snapshots_from_rerun`
- `engine.oracle_sink.validate_sink`
- `engine.oracle_provenance.trace_provenance`
- `engine.oracle_provenance.BoundaryEdge`
- `engine.setup_symex.ParityVector`
- `engine.setup_symex.check_parity_vectors`
- `engine.static_tools.run_tool`
- `engine.manifest.build_manifest`
- `engine.authority_projection.project_authority`
- `python3 -m engine.cli trace-info`
- `python3 -m engine.cli pipeline --estimate-only`

## 建议路径

1. 先读原始 brief 和 clark-utov 文档。

   建议从这些开始：

   ```text
   clark-utov-build/README.md
   clark-utov-build/agent-usage-guide.md
   clark-utov-build/contracts/runner_interface.md
   brief_tc2_mem_input_classify_then_parity.md
   ```

2. 先检查 target 和 runner。

   用 utov 的 trace / runner 工具采这些基础信息：

   - trace 指令数量；
   - entry / exit PC；
   - runner 是否可用；
   - observe 能力是否满足当前需要。

3. 先确认 sink，再谈公式。

   用 `validate_sink`。如果 sink 没确认，不要继续宣布公式闭合。

4. 从 confirmed sink 追 provenance。

   用 `trace_provenance`。如果 provenance 停在合理的外部调用或边界调用，要把这个 boundary 明确记录下来。

5. 采候选公式需要的运行时证据。

   用 runner observe points。每条 observation 要和同一次执行的 `rr.output` 绑定。

6. 写 Python 候选。

   Python 函数必须在给定它声明需要的证据值后稳定复现输出。如果它依赖外部执行状态，要把这个状态作为显式参数或显式说明写进报告。

7. 做真实对拍。

   从真实 runner 执行构造 `ParityVector`，调用 `check_parity_vectors`。用独立 held-out executions。若输出因为外部/session 状态重复，就继续采，直到达到要求的 distinct observed outputs 数量。

8. 交付最终包。

   最少要有：

   - Python 公式文件；
   - runner-vs-Python parity JSON；
   - utov primitive evidence JSON；
   - 简短最终报告；
   - 一个 verifier 脚本，能检查证据包。

## 验收标准

完成必须同时满足：

1. Python 还原存在，并且能运行；
2. 真实 runner parity 通过，verdict 来自 utov `check_parity_vectors`；
3. parity report 至少有 100 个 independent distinct observed outputs；
4. sink evidence 是 `SINK_CONFIRMED`；
5. provenance 或 boundary evidence 被明确记录；
6. 最终报告说明可见 output 依赖 plaintext input、外部状态，还是两者都有；
7. 有一个 verifier 命令退出码为 `0`，并检查关键 evidence fields。

## 什么时候停下

只有遇到下面情况才停：

1. utov 缺少某个能力，而这个能力能明显减少大量手工 runner 工作；
2. runner observation 做不到 same-execution；
3. sink 无法确认；
4. parity 无法收集到足够 independent distinct observed outputs；
5. 需要的外部模型无法从证据里合理解释。

停下时，不要写“还需要更多分析”这种空话。要写成具体 utov / runner 需求：缺什么能力、为什么卡住、下一步应该补什么。

