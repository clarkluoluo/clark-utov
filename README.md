# clark-utov

> **clark-utov** 是 agent 的朋友 / 工具 / 长程任务和猜想链的账本助手。

> 本文档（中文）为主版本 · English mirror: [README.en.md](README.en.md)

## 方法论卡片（clark-utov）

| # | 原则 | the reference target 证据 |
|---|------|-----------------|
| 1 | **锁死入口出口** | I/O 93/93 ≠ digest；`0x32350c` 非出口 → R1 @ `0xb7bb0` |
| 2 | **记好账本** | `hook_src_valid:626` invalidated；R1/R2 只认 gate JSON |
| 3 | **错了回退** | `state_reconciliation.md` 后再 R1 |
| 4 | **差分定位错点** | random_007 utov diff → SignRunner 非输入 |
| 5 | **重新出发** | R2 换 VMP 观测层；枚举尽 → suspend 等 dev |

首个目标部分归档：`work-tc3-samples/work/legacy/reference-target_partial_archive/` · [RELEASE_NOTES.md](RELEASE_NOTES.md) **v0.1.0-partial**

> **这是一个对 agent 友善的工具。** 每个分析步骤都暴露为带类型的 JSON-RPC
> 接口（见 `contracts/agent_protocol.md`），每次 promotion 都可审计
> （`interventions` 账本），每个 batch 都可回滚（`discard_batch`），每条
> finding 都带 agent 反驳所需的依据（`anchors_seen`、`evidence_score`、
> `reference_impl`、`io_test`）。引擎是为被 LLM agent 驱动而设计的，
> 不是给人盯着看的。

> **而且它让上下文/记忆没那么强大的 agent 也具备长程工作能力。** 过程数据外置、
> agent 上下文只装决策要素，配可审计账本 + gated 流水线 + by-situation 能力地图，
> 让一个上下文有限的 agent 也能跨多轮持续推进一项长程任务（比如本 build 的 VMP
> 还原），而不必把整条链塞进上下文。utov 用确定性机制补足 agent 的上下文短板。

> **关键诊断（决定优先级）**:窄上下文 agent 跑长程的**主限是「飘」——装不下几十步的全局地图/
> 历史/已闭合状态,而不是判断不准**。判断可拆成**有界 checkpoint（证据摆眼前、一次判一件）+ parity 闸
> 兜**(错判当场抓);记忆装不下却是「窄」的硬限。所以 utov 最高杠杆 = **把长程记忆外置到极致**
> (账本 `cvd_ledger` + 全局 stage 地图投影):让 agent **从不 holds 全程**,每步只看 utov 投影出的
> 当前全貌、判一件有界事、跑 utov 机械。**账本/投影对窄 agent 比对大 agent 重要得多**——大的还能硬记
> 一阵,窄的一飘就废。

基于 trace 的自动算法还原引擎。针对被 VMP / OLLVM 保护的 ARM64 Android native
库：给定一个 JNI 入口已能在 unidbg（或等价物）中稳定调用的目标，本系统消费
其规范化指令 trace，跑确定性流水线（S1–S5），并在一个已验证的假设账本里
积累"这个算法是什么、它怎么工作"的可追溯结论。

**当前状态**：P0 + P1 端到端可跑。
- S1–S5 各 stage、conformance 门禁、runner 契约、假设账本、脚本/agent 两驱动
  模式、蓝军审查、规则提炼 + 准入测试 全部实现。
- **跨 stage 反馈回路**接通：S1.5 把指纹命中的指令 idx 写到 session；S4 自动
  把它们当额外 sink。libsha256.so 上 kept_nodes 从 4 → 245 验证了回路工作。
- S3 是 concrete 数据流图（Triton 符号化在 P1.5 roadmap）。S6（LLM 假设回路）
  完整 wire；DeepSeek API 实调通过 `--mode aggressive` 开关（frugal 模式永不调）。

---

## 版本推进（version progression）

> 详细 changelog 见 [`RELEASE_NOTES.md`](RELEASE_NOTES.md)；前瞻路线见 `dev_doc/PLAN.md` §21。

- **v0.1.0-partial** — 首个目标 the reference target（SM3）端到端跑通：trace → S1–S5 → I/O → Triton → R1 hook。
- **框架泛化（v0.2–v0.5）** — profile 层（声明式判定语义）、现场经验扩展、Task 目标管理机制。
- **Level-1（set-up symex Tier-1）** — 不透明符号还原脚手架：四契约 + 前/后向双模式 + 执行型
  driver（`drive`）+ **多向量 cross-run parity 闸**（防 1/1 同义反复假 exact）。
- **Level-2（utov 自拥 concolic）✅ tag `hypotask-level2-complete`** — 不再让 agent 手搓 symex runner：
  Triton 覆盖大头 + 可扩展**逃生口**（没建模指令→精确 BLOCK，手填语义注入符号态 + 缓存），
  trace-guided concolic（执行序段 + 分支跟 trace 不发散 + 满符号 seed）。
- **Level-3 — 开发中（分支 `level3`），已落地的几个大功能：**
  - **final-sink-first 工具链** — 输出是「定长头 ‖ 拷贝活缓冲」这类最终构造时，锚在最终 sink 那条链上，不被拉进无关内部链（import-map / libc 边界边合成 / real-gold distinct 下限采集）。
  - **mem-write sink 还原** — 输出是 STORE 而非寄存器时按符号字节自检 + parity；定不下来 → 结构化终态，绝不静默回退。
  - **extern 可执行模型注册表** — extern 符号解析成可运行、带证据的参考模型并按族排序，verifier 不再手搓 PRNG/libc。

---

## 攻 VMP 目标：由轻到重（phase 顺序）

面对一个还不知道算法的 VMP 目标，**先做便宜的取证，全量 trace 留到最后**。这个顺序
就是 sign5 方法论（roadmap §8.12/§9.4），并固化成 `engine.vmp_phase_api`
（`VmpPhaseApi`）里的强制顺序接口，跨长程不衰减：

| 阶段 | 动作 | 成本 |
|---|---|---|
| **1 · `phase_1_io_observe`** | calltrace 找 crypto 入口 + hook I/O，只取 I/O **形态** | 轻 |
| **2 · `phase_2_materialization_trace`** | hook 输出物化序列（`strb`）——公式结构在这里显形 | 轻 |
| **3 · `phase_3_provenance`** | 追数据流：watch first-write + 5-way provenance → producer 链。*这里没有"猜算法"动作，只有"追数据流"。* | 轻 |
| **4 · `phase_4_formula_induction`** | 你的判断：从 2–3 看到的归纳**一条**公式（不是撒候选） | — |
| **5 · `phase_5_parity`** | 整链逐字节对拍 | 轻 |
| **`phase_heavy_vmtrace`** | 全量指令级 trace——**升级档，不是开局动作** | **重** |

每个阶段在前序没记 verdict 前拒绝启动，所以顺序是结构强制的，不是劝告。驱动循环见
[`AGENT-WORKFLOW.md`](AGENT-WORKFLOW.md) §2 *Step 0* runbook。

### 使用 vmtrace（`phase_heavy_vmtrace`）

> ⚠️ **不推荐上来直接全量 vmtrace。** 面对新 VMP 目标它很少是对的开局——phase 1–3
> 能更便宜地闭掉多数情况。全量 trace 是"轻路径真的走不通"时的升级档。

所以 vmtrace 挂在一道刻意的闸后面，两种方式之一满足：

- **自主**——`EscalationProof` 引用 phase 1–3 里记了 `COULD_NOT_CLOSE` 的结论
  （轻路径确实撞了墙），或
- **交互**——agent 必须清掉一层**警告问答**：*"已尝试 phase_1-3 均未闭合，确认升级到
  vmtrace 吗？"*（若轻阶段被跳过，则是更重的 *"尚未尝试 phase_1-3 就要上 vmtrace，
  确定？"*）。人/driver 答 `yes` 会被记录留痕。

无论哪条，agent 都要先填一份 **`VmtraceBudget`**——预估 `runtime_s` + `disk_mb`——
让成本成为有意识、可留痕的决定，警告问答也会把它显示出来。这一切都**不硬挡**（底层
原语始终可达）：它把便宜路径做成默认，把昂贵那条做成可见、可审的选择。

---

## 1. 仓库内容

```
clark-utov/
├── README.md / README.en.md       ← 双语 README（中文为主）
├── NOTICE                         ← 第三方代码归属（MIT）
├── contracts/                     ← 公开接口规范
│   ├── runner_interface.md          v2（PLAN §17 三方法接口 + 一致性自检）
│   └── java/                        Java 版契约（接口 + DTO）
├── engine/                        ← Python 引擎（consumer）
│   ├── pyproject.toml
│   ├── engine/                      核心包
│   │   ├── stages/                    S1–S6 流水线
│   │   ├── core.py                    驱动无关 facade
│   │   ├── orchestrators/             script_mode (run_full_pipeline) / agent_mode (JSON-RPC stdio)
│   │   ├── rules/                     规则提炼/准入/registry/降级（完整实现）
│   │   ├── verifier.py                3 个具体策略
│   │   ├── conformance.py             C1-C4 门禁
│   │   ├── llm_client.py              DeepSeek + MiMo 双后端
│   │   ├── runner_client.py           trace 解析器 + RunnerAdapter 实现
│   │   ├── hyp_tree.py                N 叉带回溯账本 CRUD
│   │   ├── fold.py                    行 / 块感知折叠
│   │   ├── dataflow.py                regflow / producer / semop 原语
│   │   ├── static_tools.py            白名单 subprocess（radare2 / readelf / ...）
│   │   ├── discipline.py              LLM 反漂移提醒注入
│   │   ├── blue_team.py               蓝军审查（P2）
│   │   ├── store.py                   workdir + SQLite 布局
│   │   ├── data/fingerprints.py       97 条 crypto/hash 常数指纹
│   │   └── profile/                   v0.3.0 — 可声明判定语义
│   │                                    (base 机制 vs domain 语义 两层；
│   │                                     新领域 = 加 JSON，不改源码)
│   ├── profiles/                      已落地的 profile JSON
│   │   ├── base.json                    5 个机制 probe (M1/M3/CP/VP/WFW)
│   │   ├── vmp_algorithm_extraction.json
│   │   ├── key_extraction.json
│   │   └── weird_target_x.json
│   ├── tests/
│   ├── dry_run.py                   File 模式演示（吃静态看雪 trace）
│   └── dry_run_live.py              Live 模式演示（Python 驱动 Java runner）
├── example/                       ← 样例（不属于引擎，见 example/README.md）
│   ├── runner-sha256/               ← 参考 runner：如何实现 contracts/ 协议（SHA-256 标的）
│   │   ├── libs/arm64-v8a/libsha256.so  我们的 SHA-256 OLLVM 风格样本（ground truth）
│   │   ├── java/                        Java 胶水类
│   │   └── runner/                      ← Maven 项目：基于 unidbg 的测试 runner
│   │       ├── pom.xml                    拉 unidbg-android 0.9.9
│   │       └── src/main/java/...          Sha256TestRunner + Main(serve|demo)
│   └── task-libEncryptor/           ← 一个完整解题样例（brief + 标的 + trace + 候选）
└── testTarget/                    ← 仅剩第三方样本（已 gitignore）
    └── vmp/                          看雪 libEncryptor.so + trace
```

公开仓库屏蔽（`.gitignore`）：`dev_doc/`（内部规划）、`.env`（API keys）、
`tmp/`（构建临时）、`testTarget/vmp/`（第三方样本）。

---

## 2. 架构原则（红线）

继承自 `dev_doc/PLAN.md`，代码结构强制执行。**跨模式、跨 PR 周期都不变**。

1. **verifier 是唯一真理来源**。任何来源（插件 / LLM / Triton）的结论
   都是**未验证**状态，必须 verifier 用真实 trace 数据裁决后才能依赖。无例外。
2. **findings ≠ hypotheses**。findings 已验证、是地基；hypotheses 未验证、
   进树可回溯。两张表两个 SQLite 文件，物理分离。
3. **假设产出即验证**，不积压到流水线终点。假设树在同一回路里失败即回溯。
4. **假设树是 N 叉带回溯**。DFS，同层按 LLM 给的可信度排序。
5. **LLM 只看洗干净的小数据**，永远不喂原始 trace。确定性 stage 做重活，
   LLM 做模式识别。
6. **禁止时间轴二分**。缩小到有效逻辑用反向数据流切片（S4），不用区间二分。
7. **VMP 和 OLLVM 处理方式不同**，不要在同 stage 混用策略。
8. **核心 / 驱动严格分层**。`engine/engine/core.py` 是 `orchestrators/` 唯一能
   import 的入口。脚本 / agent 两模式共用同一核心实现，不允许各自实现。
9. **反调试归 runner 管**。引擎的起点是"unidbg 已能稳定调用目标的那一刻"。
10. **語意層開放、機制底線守住**。v0.3.0 把證據檔位、節點狀態、閉環判據、
    scope 標註、cause→action 路由全部抽到 **profile 層**
    (`engine/profiles/*.json` + `engine.profile.*`)。接新領域（密鑰提取、
    冷門怪異目標）= 改 profile JSON，不動引擎源碼。但機制底線（M1
    observation≠closure / M3 虛假塊偵測 / `constant_provenance` 框架 /
    observation 必須封頂 / observation→producer 追蹤）住在 **base profile**，
    由三道獨立鎖（load-time registry 拒絕 / runtime gate force-include /
    雙側 lint）守住，子 profile 動不了。完整 spec：`dev_doc/PLAN.md` §19。

> **紅線劃邊界，線路給路——兩個都要給。** 紅線是*硬擋*，說什麼**不准**；線路是
> *默認路*，那條由輕到重的 phase 路徑，說接下來該**做什麼**。只給紅線、不給路，
> agent 會在邊界上繞來繞去找縫——把預算花在找繞法、而不是幹活（the reference case §8.11
> 的陷阱；我們已親眼看到一個新 agent 對著「只有紅線」的 brief 正是這樣空轉）。所以
> **每條紅線都要配一個線路階段**:「禁止 X」永遠要帶上「改做 Y」。只給邊界不給路不是
> 嚴格,是逼 agent 在「形式偷換」和「硬撐」之間二選一。一份做對了的 brief 範例:
> [`sample_brief_aes_vmp.md`](sample_brief_aes_vmp.md)。

---

## 3. 上手

### 3.1 引擎依赖

**只要 Python 3.11+**。引擎本身就这一个依赖。

```bash
python3 -m pip install --user clark_utov_engine-0.1.0-py3-none-any.whl
utov doctor       # 自检环境
```

Python 包列表在 `engine/pyproject.toml`，pip 自动拉。

引擎消费任何能 NDJSON 协议（见 [`contracts/agent_protocol.md`](contracts/agent_protocol.md)）的 runner 子进程，**不在意 runner 是什么语言写的、用什么仿真器**。实测过 sample runner 用 **unidbg-android 0.9.9**（兼容矩阵见 [DEPENDENCIES.md](DEPENDENCIES.md)）。

Python 依赖：
```bash
cd engine
python3 -m pip install --user click python-dotenv jsonschema openai tqdm pyarrow
```

可选 dev 依赖：
```bash
python3 -m pip install --user ruff pytest
```

### 3.2 跑流水线

引擎要一个 **runner 命令字符串** —— shell 命令，能 spawn 你的 runner、NDJSON 跑在 stdio。

```bash
# 例：你自己的 runner
RUNNER='your-runner-cmd serve /path/to/target.so'
# 或者用我们附带的 Java sample（见 example/runner-sha256/README.md）
RUNNER="$(pwd)/bin/run-runner.sh serve $(pwd)/example/runner-sha256/libs/arm64-v8a/libsha256.so"

# (a) 预估成本，不烧 LLM
utov pipeline --runner-cmd "$RUNNER" --input 616263 --estimate-only

# (b) 省钱模式（默认）：S1..S5，零 LLM 调用
utov pipeline --runner-cmd "$RUNNER" --input 616263 --mode frugal

# (c) 激进 + 预算上限
utov pipeline --runner-cmd "$RUNNER" --input 616263 --mode aggressive \
    --budget-usd 0.50 --budget-tokens 1000000 --budget-seconds 300
```

> **注意**：`example/runner-sha256/runner/` 是个**示例 runner**（演示怎么实现 contract，用 Java + unidbg）—— **不属于引擎**。要跑它需要 Java/Maven/NDK/unidbg，详见 [`example/runner-sha256/README.md`](example/runner-sha256/README.md)。引擎本身一概不依赖这些。

期望输出（节选）：
```
work dir:  /.../work/libsha256.so/<input_hash>/runs/<run_id>
pipeline summary:
  {"stage":"s1","blocks":1005,...}
  {"stage":"s1b","fingerprint_hits":16,"hypotheses_seeded":16,...}
  {"stage":"s2","unique_blocks":35,...}
  {"stage":"s3","nodes":7844,...}
  {"stage":"s4","sinks":[...],"kept_nodes":4,...}
  {"stage":"s5","annotations":4,...}

algo_signature hypotheses (16):
  hyp#1 conf=0.65 subj=SHA256.h0 fp=SHA256.h0 hits=5
  hyp#2 conf=0.65 subj=SHA256.h1 fp=SHA256.h1 hits=4
  ...
```

### 3.4 File 模式（没 runner，只有静态 trace）

```bash
# 用预生成 trace 跑 S1..S5（看雪 VMP 样本示例）
python3 -m engine.cli pipeline-file \
    --trace ../testTarget/vmp/trace.txt \
    --target-name libEncryptor.so \
    --entry 0x40007d88 \
    --exit  0x40007ed8 \
    --output-len 32
```

File 模式下 conformance C1/C2/C3 自动 SKIP（没 rerun），仅跑 C4。verifier
里依赖 rerun 的策略降级；交付物自动带"未确认缺口"标记。

### 3.5 读交付物（`utov emit`）

`utov pipeline-file` / `utov pipeline` 跑到 `algorithm_identified` 就停 ——
这是**标签**（`algorithm: SHA-256, evidence_score: 1.0, anchors_seen:
12/12`）。要把这个标签变成可粘贴可读的**还原产物**（IV 常量、被指纹覆盖
到的 K 表、σ/Σ idiom 的 PC + 寄存器分配、trace 观察到的循环次数），跑：

```bash
utov emit <run_dir>                          # 打到 stdout
utov emit <run_dir> --output pseudocode.md   # 写到文件
utov emit <run_dir> --format markdown        # 围 fenced markdown
```

`preprocess_batch` 一旦 promote 出 `algorithm_identified` 也会顺手把
`<run_dir>/pseudocode.md` 落盘 —— agent 拿到 run dir 就直接读，免一步
CLI 调用。

支持的算法在 `engine/engine/data/algorithm_pseudocode.py:ALGORITHM_SPECS`
里：当前覆盖 SHA-256 / SHA-512；SHA-1 / MD5 / SM3 / SM4 / AES round /
HMAC 的模板欢迎 PR。

---

## 4. 流水线工作机制

| 阶段 | 做什么 | 输出 |
|---|---|---|
| **C1-C4 conformance 门禁** | 分析前先把 runner 调 5 次验确定性、翻 3 字节验输入敏感性、取 1 观测点、验 trace 起止 PC。任一失败拒绝启动。 | `conformance_report.json` |
| **S1 segment** | 走一遍 trace。按控制流跳转（`b`/`bl`/`br`/`ret`/`b.cond`/`cbz`/...）或 PC 不连续切基本块。 | `s1.jsonl`（每块一行）|
| **S1.5 fingerprint** | 把每条 `regs_write` 值对 95 条 crypto 常数（MD5/SHA-1/SHA-256/SHA-512/SM3/SM4/AES/CRC/HMAC/...）+ 2 条 NEON SIMD 模式扫一遍。命中作 `confidence=0.65–0.85` 的 hyp。**仍须 verifier 才能升级**为 finding。 | `s1b.jsonl` + 账本行 |
| **S2 dedupe + fold** | 块按 PC 序列 hash 去重；同 hash 连续 ≥10 次的 run 折叠为"首块 + 哨兵 + 末块"（PLAN §12.4）。 | `s2_blocks.jsonl` + `s2_executions.jsonl` |
| **S3 dataflow graph** | 每条指令的 `regs_read` 链回最近的 producer。我们有 concrete 寄存器值，切片不需要符号化。Triton 在 P1.5。 | `s3_dfg.jsonl` |
| **S4 backward slice** | 从指定 sink（默认：最后一条指令的写）在 DFG 上 BFS 倒查；只保留 sink 的祖先。 | `s4_slice.jsonl` |
| **S5 simplify** | 轻量：识别 zero idiom、`mov #imm` 立即数，4 行窗口对 DiANa InsSub 模式反向匹配。深度符号化简在 P1.5（需 Triton）。 | `s5_simplified.jsonl` |
| **S6 hypothesis loop** | 已 wire（`engine/stages/s6_hypothesis.py`）。LLM 调用前过纪律重注入；每候选立即过 verifier；pass → finding，fail → 回溯。 | hypotheses DB |

---

## 5. Runner 契约（写你自己的 runner）

任何讲 `contracts/runner_interface.md §3` NDJSON-over-stdio 协议、过得了
conformance 门禁的子进程都是合格 runner。引擎不在意它什么语言写的。

**必需方法**（Live 模式）：

```
get_trace(input, start, end) → JSONL 或 unidbg-text trace 文件路径
rerun(input, observe_points) → {output, observations}
metadata()                   → {target_name, arch, entry_pc, exit_pc, ...}
```

**conformance 门禁**（必须全 PASS）：

```
C1 DETERMINISM        同输入跑 5 次产出逐位一致
C2 INPUT_SENSITIVITY  3 次单字节翻转中 ≥2 次输出不同
C3 OBSERVE_POINT      在 entry_pc 取观测点能拿到非空寄存器
C4 TRACE_INTEGRITY    trace 起止 PC 与 metadata 锚点匹配
```

**File 模式**只有静态 trace、没 rerun 时用（例如我们打包的看雪 libEncryptor
样本）。只跑 C4；verifier 标 `verifier_degraded=True`，交付物自动带告警。

完整参考实现见 `example/runner-sha256/runner/`：Java + `unidbg-android`，Python 侧通过
`SubprocessRunnerAdapter` 驱动。

---

## 6. 给其他 agent / 调用方

### 6.1 当 CLI 用（大多数 agent）

```bash
# 快速看一个 trace 文件
python3 -m engine.cli trace-info <path>

# 完整流水线驱动 runner
python3 -m engine.cli pipeline --runner-cmd '<你的 runner 启动命令>' --input <hex>

# 直接查 findings / hypotheses 表（对应 get_findings / get_hypotheses 两个 RPC 的 CLI 镜像）
python3 -m engine.cli findings <run-dir> --source plugin --kind algorithm_identified --json
python3 -m engine.cli hyps <run-dir> --status passed --kind algo_signature --json

# 翻一条 verdict + 记一条 intervention（对应 override_verdict 的 CLI 镜像）
python3 -m engine.cli override <run-dir> <hyp_id> fail --reason "agent 不认这条"

# 打印推荐的「由轻到重」VMP phase 路线（静态，无参数）
python3 -m engine.cli phases

# 或者吃一个预生成 trace
python3 -m engine.cli pipeline-file --trace <path> --target-name <name> \
    --entry <0x..> --exit <0x..>
```

`agent-serve` 还把「由轻到重」VMP phase 路线暴露成可调用的 RPC 方法。顺序强制
（每个 phase 在前序用 `phase_record` 记了 verdict 前拒绝启动）；`phase_heavy_vmtrace`
是升级档（需要 `EscalationProof` **或**一次确认，外加带 `runtime_s`+`disk_mb` 的
`VmtraceBudget`）。这里**没有**「枚举标准密码」的方法——唯一的 crypto 来源动作是
`phase_3` provenance。

| 方法 | 参数 | 返回 |
|---|---|---|
| `phase_state` | — | 序列 + 轨迹 + 是否已闭合 |
| `phase_1_io_observe` | `entry_pc` | instrument spec |
| `phase_2_materialization_trace` | `output_base`, `output_len` | instrument spec |
| `phase_3_watch_producer` | `addr`, `value_name` | watch-first-write spec |
| `phase_3_classify` | 带 `producer_dataflow` / `rerun_observations` 的 value 记录 | 5-way provenance 裁决 |
| `phase_4_formula_induction` | `expression`, `derived_from` | parity intent |
| `phase_5_parity` | `expression`, `inputs_min` | parity intent |
| `phase_record` | `phase`, `status`（`ran`/`closed`/`could_not_close`）, `could_not_close_reason?` | 记 verdict（下一个 phase 进入的前提）|
| `phase_heavy_vmtrace_prompt` | `budget?` | 给人看的确认问答 |
| `phase_heavy_vmtrace` | `anchor`, `budget`（`runtime_s`+`disk_mb`）, `proof` **或** `confirmation` | 全量 trace instrument spec |

读 / 写两侧的 RPC（`get_findings` / `get_hypotheses` / `override_verdict`）也都有
上面 §6.1 列出的 CLI 镜像。完整 wire 规范见
[`contracts/agent_protocol.md`](contracts/agent_protocol.md)。

### 6.2 当库用（Python agent）

```python
from pathlib import Path
from engine.runner_client import SubprocessRunnerAdapter
from engine.core import open_live

runner = SubprocessRunnerAdapter(
    cmd=["java", "-jar", str(JAR_PATH), "serve", str(SO_PATH)],
    cwd=JAR_PATH.parent,
)
try:
    core = open_live(
        work_root=Path("./work"),
        runner=runner,
        input_bytes=b"abc",
        new_run=True,
    )
    summaries = core.run_pipeline()        # → 各阶段总结
    for s in summaries:
        print(s)

    # 访问假设账本
    for h in core.get_hypotheses(kind="algo_signature"):
        print(h.id, h.subject, h.confidence, h.payload)
finally:
    runner.shutdown()
```

### 6.3 输出落盘

一切持久化到 `work/<target>/<input_hash>/runs/<run_id>/`：

```
work/<target>/<input_hash>/
├── runs/<run_id>/
│   ├── meta.json                  ← 驱动模式 / run id / target 信息
│   ├── conformance_report.json    ← C1-C4 裁决
│   ├── stage_state.json           ← 已完成 stage + code_version
│   ├── stage_outputs/
│   │   ├── s1.jsonl               ← 基本块
│   │   ├── s1b.jsonl              ← 指纹命中
│   │   ├── s2_blocks.jsonl        ← 唯一块
│   │   ├── s2_executions.jsonl    ← 块执行流（含哨兵）
│   │   ├── s3_dfg.jsonl           ← 数据流图
│   │   ├── s4_slice.jsonl         ← 切片后留下的指令
│   │   └── s5_simplified.jsonl    ← 注释后的 slice
│   ├── findings.sqlite            ← 已验证事实（含 hyp_payloads blob 表）
│   ├── hypotheses.sqlite          ← 账本 6 表 WAL（D-027）：
│   │                                hyp_payloads / claim_templates / hypotheses
│   │                                hyp_anchors / hyp_tags / hyp_dependencies
│   ├── archived/                  ← abandoned 子树归档，不在 hot path
│   ├── anomalies/                 ← verifier 无法裁决的真异常（待人审）
│   ├── session.json               ← 跨 stage 反馈上下文
│   └── notes/                     ← 蓝军备注、人工标注
└── latest -> runs/<run_id>        ← 续跑用 symlink
```

---

## 7. 测试这个项目

### 7.1 Lint + 单测

```bash
cd engine
python3 -m ruff check .                    # 风格 + bug 类
python3 -m pytest -v                       # 3 个 trace-reader 测试
find engine -name '*.py' | xargs -n1 python3 -m py_compile   # 语法检查
```

### 7.2 标准参照运行

```bash
# (a) File 模式跑看雪 VMP 样本（ground truth：SHA-512 形态）
python3 dry_run.py

# (b) Live 模式跑 libsha256.so（ground truth：SHA-256 NIST 向量）
python3 dry_run_live.py
```

两者都把 conformance 报告写到 `/tmp/`，成功 exit 0。

### 7.3 libsha256.so 上的健康跑表征

- C1-C4 conformance 门禁：**全 PASS**（Live 模式）
- S1 blocks：**约 1,000**
- S1.5 fingerprints：**16 hits — SHA256.h0..h7 + SHA256.K[0..7]**
  （不是 SHA-512；如果看到 SHA-512，说明你是在跑 `testTarget/vmp/libEncryptor.so`，
  不是 `example/runner-sha256/libs/arm64-v8a/libsha256.so`）
- S2：**约 35 唯一块**
- S3：**约 7,800 数据流节点**
- S5：K 表加载行至少有一些 `mov_immediate` 注释

---

## 8. 已知限制 / Roadmap

| 问题 | 状态 | 跟踪 |
|---|---|---|
| S3 是 concrete 版，没 Triton 符号化 | P1.5 | IMPL_PLAN §3 |
| S5 化简不处理嵌套表达式树 | P1.5 | IMPL_PLAN §4 |
| S5 的 InsSub 4 窗口模式只匹配原生 arm64 序列；源码级 OLLVM 宏展开可能不触发 | P1.5 | — |
| S6 LLM 回路已 wire；通过 `--mode aggressive` 开（frugal 永不烧 key） | 设计如此 | — |
| `--estimate-only` 不落盘 S1..S5（用临时 `_estimate` 工作目录）；正式跑会重新跑一遍 S1..S5 | 设计如此 | — |
| `pyarrow` 未启用 —— stage 输出现在是 JSONL | 已跟踪 | — |

---

## 9. 第三方归属

本项目包含从第三方 MIT 项目移植的代码。边界见 `NOTICE`：

- 97 条 crypto 指纹表、fold 算法、regflow/producer/semop 辅助原语 移植自
  [icloudza/algokiller-plugin](https://github.com/icloudza/algokiller-plugin)
  （MIT, cloudza 2026），特别是其 Sprint 1-6 插件扩展部分。上游
  `match`/`context`/`daemon` 引擎路径（[@lidongyooo](https://github.com/lidongyooo)
  另行持有版权）未移植。

`testTarget/vmp/libEncryptor.so` 是看雪论坛 thread 291195 的第三方样本，
仅本地用于分析，已 gitignore，不进公开仓库。

---

## 10. License

引擎代码 MIT 协议（见 `LICENSE`）。

`example/runner-sha256/libs/arm64-v8a/libsha256.so` 是本项目原创，同样 MIT。
看雪 VMP 样本**不属于**公开仓库。
