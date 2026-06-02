# example/ — 样例目录（不属于引擎）

这里放两类样例，引擎本身一概不依赖它们。两个子目录分工不同：

## `runner-sha256/` — 如何实现一个满足 `contracts/` 协议的 runner

一个**参考 runner 实现**：用 Java + unidbg 把一个 SHA-256 标的
（`libs/arm64-v8a/libsha256.so`，我们自建的 OLLVM 风格样本，过 NIST FIPS 180-2
全部向量）跑起来，并通过 stdio NDJSON 实现
[`../contracts/runner_interface.md`](../contracts/runner_interface.md) 的
`metadata()` / `get_trace()` / `rerun()`。

它回答的问题是：**「我要怎么写一个引擎能驱动的 runner？」** 看这里。
详见 [`runner-sha256/README.md`](runner-sha256/README.md)（含构建、运行、
以及刻意保留的 C5 一致性反例）。

## `task-libEncryptor/` — 一个完整解题样例

一个**端到端解题样例**：针对看雪 `libEncryptor.so`，给齐解题所需的全部料——
任务 brief、分析标的、真实 trace、候选模型脚本：

- `brief-get-goal-by-utov-v1.md` — 目标说明（brief）
- `libs/arm64-v8a/libEncryptor.so` — 分析标的
- `libs/arm64-v8a/trace.txt` — 真实 unidbg 文本 trace
- `rand_model_candidate.py` — 候选模型脚本

它回答的问题是：**「拿到一个目标，一次完整解题长什么样？」** 看这里。

---

English: `runner-sha256/` shows how to implement a `contracts/`-compliant runner
(Java + unidbg, SHA-256 target); `task-libEncryptor/` is one complete worked
example (brief + target + trace + candidate). Neither is part of the engine.
