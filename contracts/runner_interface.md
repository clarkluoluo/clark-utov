# Runner 接口契约 v1

> 本文是 **clark-utov 系统** 对 **runner 实现者**的接口要求。
> Runner 不归本仓库实现 —— 它由系统使用者自行用 unidbg / 其他 emulator 搭好。
> 系统从这份契约定义的边界开始接管。

---

## 0. 范围声明（红线）

**runner 必须自行解决的事**（系统**不管**）：
- 反调试 / 反 Frida / root 检测绕过
- unidbg 初始化、SVC patch、syscall 模拟
- JNI 动态注册 (`RegisterNatives`) 的解析
- 加固 / 脱壳 / 解密 init
- 对 .so 本身的加载和 relocation

**runner 必须提供给系统的东西**：
1. 规范化的 trace（§2 schema）
2. 可反复调用的 `rerun(input) -> ...` 接口（§3）
3. 元信息（§4）

满足这三项 = runner 合格。**系统不关心 runner 内部用什么语言、什么 emulator、跑在哪。**

---

## 1. 通信方式

最简：**文件 + 子进程**。
- runner 把 trace 写到指定路径
- 系统通过子进程 / 标准 IO 调 runner 的 `rerun`

不限定具体 RPC 形式。`engine/runner_client.py` 提供 adapter 类，常见实现：
- `FileTraceAdapter`：trace 一次性预生成在磁盘，rerun 走 `subprocess.run(["./runner", "--input", hex])`
- `JsonRpcAdapter`：runner 起进程，stdio 走 JSON-RPC（后续按需要加）

---

## 2. Trace 格式

### 2.1 标准格式：JSONL（推荐）

一行一条已执行指令。**只记本指令真正读写的寄存器和内存**，不存全套寄存器快照（详见 §2.3 体积控制）。

```json
{
  "idx": 0,
  "pc": "0x40007d88",
  "bytes": "ff8301d1",
  "mnemonic": "sub sp, sp, #0x60",
  "regs_read":  {"sp": "0xbffff700"},
  "regs_write": {"sp": "0xbffff6a0"},
  "mem": []
}
```

字段语义：
| 字段 | 类型 | 含义 |
|---|---|---|
| `idx` | int | 指令序号，从 0 递增 |
| `pc` | hex string | 指令地址 |
| `bytes` | hex string | 4 字节原始指令编码 |
| `mnemonic` | string | 反汇编文本（仅辅助，系统不依赖它做语义） |
| `regs_read` | obj | 本指令**读**的寄存器及其值（执行前） |
| `regs_write` | obj | 本指令**写**的寄存器及其值（执行后） |
| `mem` | list | 本指令的内存访问 |

`mem` 元素：
```json
{"rw": "r", "addr": "0xbffff708", "val": "0x0", "size": 8}
```

### 2.2 兼容格式：unidbg 文本 trace

系统也接受 unidbg 默认文本输出：
```
[09:39:47 005][libEncryptor.so 0x07d88] [ff8301d1] 0x40007d88: "sub sp, sp, #0x60" sp=0xbffff700 => sp=0xbffff6a0
```
解析器在 `engine/runner_client.py::UnidbgTextTraceReader` 实现。新 runner 优先用 §2.1 JSONL。

### 2.3 体积控制（必守）

- **只 trace 算法区间**，不从 JNI 入口全程记录。runner 提供 start/end 锚点（PC 地址或符号）。
- **寄存器只记本条指令真正读写的**。一条 `add x0, x1, x2` 只记 `x1, x2 → x0`，不记 x3..x30/sp/pc。**违反这条会让 GB 级 trace 失控**。
- **内存只记实际访存**。

参考：testTarget/vmp/trace.txt 5MB / 41K 行就是符合这些约束的范例。

### 2.4 必须可定位算法区间

trace 起止由 runner 锚定。系统不负责"从 JNI 入口跑到目标函数"。看雪样本就是好例子：trace 从 `0x7d88`（.mytext 入口）开始到 `0x7ed8`（ret）结束，正好一次算法调用。

---

## 3. Runner 三方法接口（PLAN §17）

合格 runner 实现以下三个方法。模式分为 **Live 模式**（三个全实现，主流程能力完整）和 **File 模式**（仅有静态 trace 文件，verifier 降级）。

### 3.1 `get_trace(input, start, end) -> trace_stream`

- 入参: `input: bytes`、`start: int(PC)`、`end: int(PC)`
- 出参: §2 schema 的 trace 流（JSONL 或文件路径）
- 语义: 用 `input` 跑一次目标，trace 截断到 `[start, end]` PC 区间
- **File 模式可省略**：使用已 dump 好的固定 trace 文件，但仅能分析对应 input

### 3.2 `rerun(input, observe_points=[]) -> RerunResult`

- 入参:
  - `input: bytes`
  - `observe_points: list[ObservePoint]` （可空，空时只取终点 output）
- `ObservePoint`: `{pc: hex, when: "before"|"after", capture: ["regs"|"mem"], regs: [...]?, mem: [{addr, size}]?}`
- **线上请求每个 observe point 出参须带全字段**（历史 bug：只发了 `{pc, when, regs}`，
  丢了 `capture`/`mem` → runner 无从 capture mem → snapshot 端到端恒空）。请求 JSON 形如:

  ```json
  {"pc": "0x70ec4", "when": "BEFORE",
   "capture": ["REGS", "MEM"],
   "regs": ["x0", "x19"],
   "mem": [{"addr": "0x...", "size": 8}]}
  ```

  `capture` 选择 runner 记录哪些种类；`mem` 列 `(addr,size)` 范围（addr 用 hex）。
  缺 `capture`/`mem` 即破坏 same-execution mem-snapshot 路（B/C3 conformance round-trip 闸守此约）。
- **线上 `when` 与 `capture` 均为 UPPERCASE**（对应 Java enum `When{BEFORE,AFTER}` /
  `Capture{REGS,MEM}`，runner 用 `valueOf()` 严格解析）。**engine 负责在 wire 边界大写化**，
  caller 用小写惯例（`"before"` / `("regs","mem")`）。两者须对称——历史 bug2：`when` 已大写但
  `capture` 漏了同样处理，小写 `["mem"]` 令 `Capture.valueOf("mem")` 抛错 → `isolated once exit 1`。
  注意 `regs` 是寄存器**名**（自由 `List<String>`，如 `"x19"`），**不是 enum、不大写**。
- 出参: `RerunResult{output: bytes, observations: list[ObservedState], truncated: bool=false, truncated_detail?: {...}}`
- 语义: 用 `input` 跑一次，在每个观测点 dump 真实状态，最后返回 algorithm output
- **File 模式不支持** → verifier 自动降级为"中间态对照单输入 + 终点 I/O 不验证"

#### 撞记录 cap 必置 `truncated`（对称由构造保证，不留静默降级）

- runner 的 reg-relative / 宽域 concrete 采集有记录上限（如
  `X25_REGREL_CONCRETE_WRITE_MAX=8192`）。一旦某次 rerun 触达任一记录 cap、
  导致**后续匹配的 read/write 没能继续记录**（账本不完整）：
  - **必须**在 `RerunResult` 置 `truncated=true`（这是布尔语义，**不是** cap 数值本身——
    cap 值留在 runner 侧；engine 不硬编码具体上限）。
  - 宜附 `truncated_detail`（自由 dict，仅供日志/导出），例如
    `{"cap": "X25_REGREL_CONCRETE_WRITE_MAX", "limit": 8192, "kind": "write", "dropped": <n>}`。
- 这是**硬契约**：silently 丢记录却仍报 `truncated=false` = 违约。下游会把
  `truncated=false` 的 observations 当**完整、干净 provenance** 消费，被静默截断的账本最危险
  （`feedback_construct_symmetry_not_caller_obligation`）。
- engine 消费侧（`mem_snapshots_from_rerun` / `recapture.observations_to_snapshots`）
  读到 `truncated=true` 会**顶 terminal WARN（不静默）**，并把派生出的每个
  `MemSnapshot` 标 `truncated=true`，让 provenance/validate_sink 知道这份**不可当
  complete/clean provenance 用**。对称由此构造保证：caller 无需「记得检查」义务。
- 即便 PC-条件单点 watch（§3.2.1 point-watch）尚未上线，本条也**先补**——
  现在宽域采集撞 cap 是静默的，必须先消除静默截断。

#### 3.2.1 PC-条件 reg-relative 单点 watch（point-watch，精确档，免宽域噪+撞 cap）

> 背景：要采的常是「执行到某 PC 那一刻、按寄存器现值算出的一个地址处的单点
> read/write」（如 `0x70ec4` 那刻的 `[x19+0x38]` 8B）。现有宽域 `REGREL_CONCRETE`
> 只认预算好的固定范围，覆盖这种目标只能猜一段 stack 宽域硬抓 → **噪**（目标只占少数）
> + **撞 cap**（宽域 write 打满 `*_WRITE_MAX` → 后续静默丢记录）。point-watch 是中间这个
> 精确档：在目标 PC 那刻按现值算址、只采那一个点。

引擎侧已派出该 directive（`engine.watch_first_write.request_point_watch` /
`recapture_target.derive_recapture_directive(..., point_watch_pc=...)`）。**runner 须支持**
解析并执行它。directive 的 JSON 形态（`WatchFirstWriteSpec.to_dict()`）：

```json
{
  "kind":        "point_watch",
  "watch_kind":  "read",          // read | write —— 采集方向
  "pc":          462020,
  "pc_hex":      "0x70ec4",
  "addressing":  "reg_relative",
  "base_reg":    "x19",
  "offset":      56,              // 0x38
  "addr_expr":   "[x19 + 0x38]",
  "width_bytes": 8,
  "addr":        0,               // diagnostic-only（deriving run 上观测到的址，runner 不信它）
  "value_name":  "...",
  "reason":      "..."
}
```

**runner 语义契约**：

1. **arm 在 `pc`**：在 PC == `pc` 的指令处下条件断点/观测点。
2. **触发时按现值算址**：到达该指令那一刻，取 `base_reg` 的**当前**值 + `offset`
   得目标地址（**不用** directive 里的 `addr`——它跨 run 不稳，仅诊断用）。
3. **采单点**：在该地址处采 `width_bytes` 字节，方向由 `watch_kind` 定（`read`=读那刻
   该地址的内容；`write`=记该指令写入该地址的字节）。
4. **只采那一个点**：不展开成区间扫描 → 不会引入无关 read/write 噪、物理上**不会**打满
   宽域记录 cap。
5. 结果以 §3.2 `RerunResult.observations`（或 §3.7 `MemSnapshot`）回填，与其他观测并列。

与 `kind: "watch_first_write"`（first-write、无 PC gate、catch 首写）区分清楚：
point-watch 是 **PC-gated 定点单采**，是宽域 `REGREL_CONCRETE` 的精确替代。runner 收到
`kind=point_watch` 时按本节执行；收到 `kind=watch_first_write` 走原 first-write 语义。

**信息不全时引擎不派 point-watch**：当 arm PC 未知，引擎**不会** fabricate 一个 PC，而是
退回宽域 reg-relative 范围采集，并在 directive 上**显式标** `capture_mode:
"reg_relative_range"` + `capture_risk`（噪 + 可能撞 cap、非干净 provenance）。runner/下游
见到该标记须知此账本可能不完整、不可当干净 provenance（与 §3.2 `truncated` 同精神）。

##### point-watch 走 `rerun` observe-point 的 wire 形态（JSON-RPC 路）

上面的 `WatchFirstWriteSpec` directive 是**离线 directive 形态**；当 point-watch 经
**`rerun` 的 `observe_points`** 路下发时（JSON-RPC live 模式），它作为 observe point 上的
一个**新增 `mem_regrel` 字段**承载——与 §3.2 的 concrete `mem`（`[{addr,size}]`，址预先算好）
**并列、互斥**。concrete `mem` 表达不了「按 `pc` 那刻寄存器现值算址」，`mem_regrel` 补这一档。
请求里每个 observe point 形如：

```json
{"pc": "0x70ec4", "when": "BEFORE",
 "capture": ["mem"],
 "regs": [],
 "mem": [],
 "mem_regrel": [
   {"base_reg": "x19", "offset": 56, "width": 8, "pc": "0x70ec4", "kind": "read"}
 ]}
```

- `mem_regrel` **可选**：缺省（不发该键 / 空）= 纯 concrete observe point，wire 形态与历史
  逐字节一致（零回归）。runner 见到它即按 §3.2.1 语义执行：arm 在该项的 `pc`、到达时取
  `base_reg` 现值 + `offset` 算址、采 `width` 字节、方向由 `kind`（`read`|`write`）定，只采那一个点。
- 每项字段：`base_reg`（指针寄存器名）、`offset`（字节偏移，可负，十进制）、`width`（字节数）、
  `pc`（arm/访问指令 PC，hex）、`kind`（`read`|`write`）。**址永远由 runner 按现值算**——
  请求里**不带** concrete addr（与 directive 的 `addr` 仅诊断同理）。
- 一个 observe point 上 `mem`（concrete）与 `mem_regrel`（reg-relative）**互斥**：engine 侧
  `MEMREGION_WATCH` 调用按 `base_reg` 有无分流，同时给两形态会**报错拒绝**（不静默二选一）。

### 3.3 `metadata() -> TargetMeta`

- 出参: §4 描述的结构（arch / 入口 PC / 输入规格 / 输出读取方式）
- 启动时调一次，缓存

### 3.4 契约保证（验收条件，不止函数形状）

| # | 保证 | 何处验证 |
|---|---|---|
| 1 | **确定性可复现**：同 input 多次调用结果逐位一致 | conformance.C1 |
| 2 | **fake 已稳定**：时间/随机/反调试/设备指纹由环境侧固定 | runner 实现者自承 + C2 input sensitivity 反向佐证 |
| 3 | **观测点可达且准确**：任意指定 PC 可 dump 真实寄存器/内存 | conformance.C3 |
| 4 | **重跑可承受**：单次耗时合理或支持缓存 | conformance 计时（warn 阈值 > 1s/次） |
| 5 | **观测中立**：装上 observer 不改变被观测的 I/O（详见 §3.6） | conformance.C6 |
| 6 | **撞 cap 不静默截断**：采集触达记录上限时 `RerunResult.truncated=true`（见 §3.2 撞 cap 段） | engine 消费侧 WARN + 标脏；runner 实现者自承 |

### 3.5 性能预期

- 单次 rerun 期望 < 1 秒（unidbg 典型 SHA256 类算法约 10-100 ms）
- verifier 一轮验证可能调几十到上千次，慢于 1 秒/次会拖垮系统

### 3.6 观测域与 observer-set（多消费者共存）

> 背景：guard 采证需要装一组 init observer（CodeHook/WriteHook），但同一 runner
> 上跑的 color oracle 不需要它们。observer 默认全开、又没分域时，guard 的 write/
> unmapped handler 会越域裁决 color 路径上的写（实测：`WRITE_UNMAPPED @0x61924 →
> NULL`），把本来能过的 oracle 弄挂。这不是「某个 if 写反」，是**分域缺失 +
> 观测有副作用**。runner 侧按本节实现，使「两个消费者要相反观测状态」有多种解法，
> 不必各自手工绕。

**规则 1 · observer 必须 band-scoped（治本）**
任何 observer 安装时**必须声明自己拥有的 PC / 内存区间**，并且对**区间外的地址
一律 pass-through**——不读可以、但**绝不裁决**（不返回 NULL/false 改变 fault 结果、
不调 `mem_map`/`mem_write` 改内存）。参考正例：`installUtovInitObservers` 的 CodeHook
用 `hook_add_new(..., libBase+0x322000, libBase+0x328800, null)` 限定 band；
`installStagingHeapProvenanceWatch` / `installTemplateHeapWriteWatch` 的 WriteHook
既在注册期传 `[lo,hi]`、hook 内又 `if (address<lo||address>hi) return;`，且纯只读
（只 `reg_read`/`readMemSafe`/记录）——这三个都是合规范本。

**按 hook 类型分清风险（重要，避免误判）：**

| hook 类型 | 中立性风险 | 说明 |
|---|---|---|
| `CodeHook` (HOOK_CODE) | 低 | 回 void，只观测；band-scoped 即安全 |
| `WriteHook` (HOOK_MEM_WRITE) | **低（天生中立）** | 回 void，且**只在已映射写入上触发**，物理上无法产生 `WRITE_UNMAPPED`/无法改写入成败。band-scoped + 只读即可 |
| **`EventMemHook` (HOOK_MEM_*_UNMAPPED / PROT)** | **高 ← 真正的雷** | handler **回 boolean 决定 map-continue(true) vs fault(NULL/false)**。全域注册时会替**别的消费者**的未映射访问裁决出 fault —— color oracle `WRITE_UNMAPPED @0x61924 → NULL` 即此类 |
| 任何调用 `mem_map`/`mem_write`/`reg_write` 的 hook | **高** | 直接改被观测对象状态，已非 observer |

**结论**：纯 `WriteHook`/`CodeHook` 不是 neutrality 的嫌疑；排查越域问题**先看
`EventMemHook`（unmapped 事件 handler）和任何真改内存/寄存器的 hook**。这两类必须
band-scoped，且对域外地址**返回「未处理」让原路径自决**，不得短路成 fault。

**规则 2 · 具名 observer-set（给用户选项）**
runner **可**暴露具名 observer 集合，运行时按配置选一套，而不是全局 on/off：

```yaml
# metadata 可选字段
observer_sets:
  clean:          []                       # 默认：不装任何 init observer
  guard_evidence: [init_codehook, staging_provenance_watch, ...]
default_observer_set: clean                # 默认走 clean，而非「全开」
```

- 选择方式：runner 启动属性（如 `-Dutov.init.observer_set=guard_evidence`）。
- **默认必须是 `clean`**——「后来 hook 默认全开」正是 color 回归的直接原因。
- 这样 guard run 选 `guard_evidence`、oracle run 选 `clean`，是**声明式选择**，不是
  手工两次掛的隐式技巧。band 不相交时（规则 1 成立），两套甚至能在同一 run 共存。

**规则 3 · 中立性可验证**
任一 observer-set 对某条路径是否中立，用 `conformance.C6`
（`check_observer_neutrality` / `check_observer_set_matrix`）验：同一 input 在 `clean`
与该 set 下各跑一次，I/O 分歧或一侧 fault 即判 set 越域。引擎不替 runner 切旗标——
runner 提供两套配置好的 adapter，引擎只跑两边比结果。

### 3.7 snapshot 观测（必须以规范 shape 放进 trace 流）

> 背景：最终输出常常**不在 mem-trace 的 read/write 步骤里**，而是 runner 在某个
> hook 处对一块缓冲做的内存快照（如把寄存器指向的 framed 输出 dump 下来）。引擎的
> oracle sink-validator 把 snapshot 当**一等扫描来源**；若 runner 不交 snapshot，
> validator 只能扫 writes/reads，可能对「其实落在 snapshot 里的输出」误报
> `OUTPUT_NOT_OBSERVABLE`（实测踩过）。

**契约**：runner/adapter **必须**把 snapshot 观测以下述 utov 规范 shape 放进 trace
流，与指令记录并列（引擎会把它们收成 snapshot 列表喂给 validator）：

```
MemSnapshot:
  addr:   int      # 这块快照的具体起始地址（concrete）
  data:   bytes    # 该地址处观测到的字节
  label:  str = "" # 可选标签，如 "output_buffer"
  source: str = "snapshot"   # 只读观测标记
```

- **不绑任何 runner 专属格式**：本 repo 只定义这个规范 shape + validator 消费它；
  把 runner 专属捕获（寄存器指向的 dump、memcpy 目的地、`x25` 之类）转成 `MemSnapshot`
  是 **adapter glue 的职责**，引擎核心不写任何 runner 专属解析器。
- **缺失即诚实标注**：trace 不含 snapshot 时，validator 的
  `OUTPUT_NOT_OBSERVABLE` 会注明「snapshot 来源未提供」，不可被读成「任何来源都不可
  观测」。要消除该歧义，adapter 就得按本条把输出区快照交上来。

**何处验证**：`engine.oracle_sink.validate_sink(..., snapshots=[MemSnapshot,...])`；
verdict 的 `scanned_sources` 字段列出本次实际扫了哪些来源（writes/reads/snapshots）。

---

### 3.8 运行时能力声明（`TargetMeta.capabilities`）

> 背景：能力按**声明**门控时，会出现「能力其实可用、只是 runner 漏声明」→ 被静默
> 丢弃的退化（spec MEM-cap，the reference case F0 C3：MEM observe 实际成功确认了 sink，却因
> runner 没声明 `mem_capture` 被静默跳过 round-trip）。conformance 现已改为**按
> 探针（实际 round-trip）判定可用性，不纯按声明**——但「干净的声明态」仍是常态目标，
> 故 runner 必须把它真正支持的运行时能力声明出来。

**契约**：runner 把它支持的运行时能力名声明在 `TargetMeta.capabilities`（或
adapter 的静态 `CAPABILITIES`，后者被 `TargetMeta.capabilities` 覆盖，见
`block_cause.oracle_from_adapter`），元组形式。已知能力名：

| 能力名 | 含义 | conformance round-trip 闸 |
|---|---|---|
| `observe_point` | 任意 PC 可 dump 真实寄存器（§3.4 #3） | C3 / C7 |
| `mem_capture` | observe_point 可附带请求**内存区间**并把字节回填（§3.2 `ObservePoint.mem` → `RerunResult.observations[].mem`） | C3-mem / C7 mem round-trip |

- **C7 对每个声明的能力 round-trip 断言**：声明了但 round-trip 回不来 = LOUD FAIL
  （「声明了没接线」类，Bug1）。
- **未声明但实际可用 = LOUD `capability_undeclared_but_working`**：conformance 不会
  静默丢弃一个能用的能力——它会照常跑 round-trip 并打一条 inconsistency 记录（针对
  `mem_capture`），提示 runner 补声明把状态收敛回干净的声明态。**runner 端正确做法是
  声明它**，使该 inconsistency 不再出现。
- **真正不可用（未声明且 round-trip 不通）= 带 WARN 降级**：conformance 标
  `mem_skip_reason=probe_unavailable` + WARN，绝不静默空过。
- **File 模式**：rerun 不可用 → 基于 rerun 的检查 SKIP 并标 `verifier_degraded` /
  `mem_skip_reason=file_mode`（带标，非静默）。

**何处验证**：`conformance.C3`（含 mem round-trip）+ `conformance.C7`
（`CAPABILITY_PROBES` 注册表逐能力 round-trip）。

---

## 4. 元信息

runner 启动前提供给系统：

```yaml
target_name: libEncryptor.so
arch: arm64           # 目前只支持 arm64-v8a
algo_entry_pc: 0x40007d88   # trace 起始锚点
algo_exit_pc:  0x40007ed8   # trace 结束锚点
input_spec:
  kind: bytes
  length: 16        # 固定/可变；可变则给 max
output_spec:
  kind: bytes
  length: 32
```

可选：
- `capabilities`：本 runner 支持的运行时能力名元组（见 §3.8），如
  `("observe_point", "mem_capture")`。声明了 conformance 才把对应 round-trip 当
  断言；未声明但实际可用会被 LOUD `capability_undeclared_but_working` 点名（提示补
  声明），**不再静默丢弃**。
- `algo_symbol`：如果 .so 没 strip，给函数符号名（仅辅助调试）
- `known_io_pairs`: 已知的输入→输出对（终点验证基准；P4 真实样本最有用）
- `emulator_name`: 你的 runner 用的仿真器名字（`"unidbg"`/`"qiling"`/`"frida"`/...）
- `emulator_version`: 仿真器版本（如 `"0.9.9"`）

引擎用 `emulator_name`/`emulator_version` 做：
- 写入 `meta.json` 留痕
- 标到 interventions 审计（同一 finding 在不同 emulator 下重现的对比）
- §14 规则提炼的 `applicability_tags` 候选（"此规则仅在 unidbg-0.9.x 下验证过"）

不填也行，引擎降级到"unknown"。

---

## 5. Runner 实现检查清单

满足这些 = 合格 runner：

- [ ] 能稳定加载目标 .so（反调试已处理）
- [ ] 能定位算法函数入口（地址或符号）
- [ ] 能用任意 input 调它并取得 output
- [ ] 同 input 多次调用输出一致
- [ ] 能产出 §2 schema 的 trace
- [ ] trace 只覆盖算法区间，遵守 §2.3 体积约束
- [ ] 实现 §3 三方法（File 模式至少 §3.3 metadata + §3.1 静态 trace）
- [ ] 元信息按 §4 填好
- [ ] **通过 §6 一致性自检（强制门禁）**

满足之后再来对接系统。**不满足的 runner 系统拒绝吃**。

---

## 6. 一致性自检（强制前置门禁，PLAN §17）

实现完接口后 **必须先过 conformance test** 才能进入主分析流程。系统通过 `python -m engine.conformance --target ...` 运行。任何一项失败 → 报告写到 `work/<target>/<input_hash>/conformance_report.json`，**主流程拒绝启动**。

### 6.1 五项检查

| ID | 名称 | 内容 | Live | File |
|---|---|---|---|---|
| **C1** | DETERMINISM | 同 input 调 `rerun` 5 次输出逐位一致 | ✓ | skip |
| **C2** | INPUT_SENSITIVITY | 翻转 input 中 3 个随机 byte，至少 2 个产生不同输出（防 stub） | ✓ | skip |
| **C3** | OBSERVE_POINT | 在 `metadata.algo_entry_pc` 取观测点，确认能 dump 寄存器 | ✓ | skip |
| **C4** | TRACE_INTEGRITY | `get_trace` 返回的 trace 起点 PC = `start`，终点 PC = `end` 或 ret | ✓ | ✓ |
| **C5** | CROSS_CALL_INDEPENDENCE | 基线 `get_trace` 与"几次 `rerun` 后再 `get_trace`"行数比；第二次 < 50% baseline → FAIL | ✓ | skip |
| **C6** | OBSERVER_NEUTRALITY | 同 input 在 `clean` 与 instrumented observer-set 下各跑一次；I/O 分歧或一侧 fault → FAIL（§3.6 规则 3）| ✓（需两套 regime）| skip |

C5 强制 §3.2 第 4 条"无副作用：相邻调用互不影响"。常见 fail 原因：观测 CodeHook 装上没 detach，污染下一次 `get_trace`。

C6 强制 §3.4 保证 #5"观测中立"（§3.6）。常见 fail 原因：observer 没 band-scoped，write/unmapped handler 越域裁决了别的消费者的路径（color oracle `WRITE_UNMAPPED` 即此）。**C6 需调用方提供 `clean` 与 instrumented 两套 adapter，故不进强制单 adapter 门禁，是 opt-in 检查。**

### 6.2 模式语义

- **Live 模式**（三方法全实现）：C1-C4 全跑，全过才放行；verifier 全功能开启
- **File 模式**（仅有静态 trace 文件）：仅跑 C4，过即放行但 `verifier_degraded = True`；最终交付物自动注明"无 rerun 验证"

### 6.3 当前样本状态

| 样本 | 模式 | 自检状态 |
|---|---|---|
| `example/runner-sha256/libs/arm64-v8a/libsha256.so` | Live（待 runner 实现） | 未跑 |
| `testTarget/vmp/libEncryptor.so` | File（仅 trace.txt） | 未跑，预期 C4 PASS（起 0x7d88 / 终 0x7ed8），verifier 降级

---

## 6. 已支持的样本举例

| 样本 | trace 格式 | 入口 | runner 实现状态 |
|---|---|---|---|
| `example/runner-sha256/libs/arm64-v8a/libsha256.so` | 待 runner 提供 | `Java_com_clark_utov_test_Sha256_hash` | 待实现（系统使用者负责） |
| `testTarget/vmp/libEncryptor.so` | 已有 unidbg 文本 trace (`trace.txt`) | 0x7d88 / 0x7ed8 | 半成品：trace 有，rerun 需补 |

---

## 8. JniLiveRunner 参考基类 (v3, 0527 BUG_REPORT-7)

> 这一节定义 Android JNI native-method 目标的 Live-mode runner 应该如何实现，
> 把 TC1 (`Sha256TestRunner`) 和 TC2 (`LibEncryptorTestRunner`) 重复 ~120 行
> 的 unidbg 脚手架抽出来落进 `contracts/java/com/clarkutov/runner/contract/JniLiveRunner.java`。

### 8.1 谁该 extends `JniLiveRunner`

实现以下三件事**都**成立的 runner：

1. 目标是 **Android JNI native method**（不是裸 C 函数 / 不是 Frida / 不是 qiling）
2. 用 **unidbg** 加载和模拟（其他 emulator 自己写 contract）
3. **Live 模式**（不只是 File-mode trace 回放）

不满足这三条 → 直接 implement `Runner` 接口，按 §1-§7 写。

### 8.2 子类必须填的 5 个方法

```java
public class MyTargetRunner extends JniLiveRunner {
    @Override protected String targetName()   { return "mylib.so"; }
    @Override protected long   algoEntryPc()  { return 0x40001234L; }
    @Override protected long   algoExitPc()   { return 0x40005678L; }
    @Override protected int    outputLength() { return 32; }
    @Override protected File   soFile()       { return resolveSoFile(); }
}
```

可选 override：
- `dummyJClass()` — 默认 `"com/clarkutov/Target"`。如果目标用真实 Java 类
  调度（如 TC1 的 `com/clark/utov/test/Sha256`），override。
- `callArgs(EmuContext, byte[])` — 默认 ABI
  `(JNIEnv*, jobject thiz, jbyteArray msg, jint len)`。其他 ABI（buffer-out、
  jstring 输入、`RegisterNatives` 路径）按需 override。
- `resolveOutput(EmuContext, Number)` — 默认从 `jbyteArray` 返回值取 bytes。
  buffer-out 型 algorithm 改成读 emulator 内存。
- `invoke(EmuContext, byte[])` — 想完全换调用路径（如
  `vm.callStaticJniMethodObject`）就 override 这个，跳过默认
  `module.callFunction(...)`。
- `inputLength()` / `algoSymbol()` / `emulatorName()` / `emulatorVersion()` —
  metadata 字段，不影响功能。

### 8.3 基类负责什么 (子类不要再写这些)

- **EmuContext per call** — `AndroidEmulator + VM + module.load` 在每次
  `rerun` / `getTrace` 里独立构造、独立 close。C5 cross-call independence
  靠这个保证 —— 子类**不要**缓存 EmuContext。
- **`defaultGpRegs()`** —— x0..x30 + sp + pc，conformance C3 fallback。
- **`regNameToUcId()`** —— 字符串到 `Arm64Const` 的映射。
- **`captureState()`** —— `ObservePoint → ObservedState` 全部逻辑。
- **ObservePoint → CodeHook 安装循环** —— `Backend.hook_add_new` + `BEFORE/AFTER`
  PC 调整 + 不需要 detach 因 EmuContext close 时整个 emulator 释放。
- **`getTrace()` traceCode redirect 模板** —— 创建 temp 文件 + `traceHook.setRedirect()` +
  finally stopTrace + 返回 `FileTraceStream`。
- **rebase 检测** —— EmuContext 构造时校验 `algoEntryPc()` 在
  `[module.base, module.base+module.size)` 范围内，超出抛带原始/期望地址的
  IllegalStateException。

### 8.4 Live 模式失败的四种最常见原因（按概率排序）

1. **反调试 / 反 Frida** —— .so 自检检测到 emulator 立即 abort。基类的
   `EmuContext` 构造**不会**自动 patch；子类需要在 override 里加 patch
   (override `EmuContext` 构造或者继承一个 `PatchedEmuContext`)。
2. **JNI vtable 错配** —— 目标用 `RegisterNatives` 注册了名字，但默认 ABI
   不对。override `callArgs()` / `resolveOutput()` 或干脆 override
   `invoke()` 走 `vm.callStaticJniMethodObject`。
3. **Output shape 错配** —— algorithm 写到 caller-supplied buffer 而不是返回
   jbyteArray。override `resolveOutput()` 改读 emulator 内存。
4. **.so rebase** —— `algoEntryPc()` 是用一个 base 录的，unidbg 加载到了
   另一个 base。EmuContext 抛 IllegalStateException，把消息里的实际
   `module.base` 抄到代码里重算。

### 8.5 边界（基类**不**覆盖什么）

| 目标 shape | 怎么办 |
|---|---|
| Android JNI native method on stripped .so（raw-PC 调用） | `JniLiveRunner`（本基类）默认 ABI |
| Android JNI native method with `RegisterNatives` 符号 | 同上 + override `invoke()` 走 `callStaticJniMethodObject` |
| 裸 C 函数（无 JNI / 无 DalvikVM） | 不在本基类范围；将来 `CFunctionLiveRunner` |
| qiling (Python) | 不同 SDK，将来 Python 侧 `QilingLiveRunner` |
| Frida (attach to live device/process) | 不同范式，将来 `FridaAgentRunner` |
| 纯 unicorn（无 Android 层） | 将来 `UnicornLiveRunner` |

未来 sibling 基类共享 EmuContext + observe + traceCode 的部分（可能再抽更上层的
parent），但 §8 只定义 Android JNI 路径。

### 8.6 验证清单

写完一个 `extends JniLiveRunner` 的子类后：

- [ ] `mvn -DskipTests compile` 通过（构建期/依赖正确）
- [ ] `java -jar runner.jar demo` 不抛 Exception 跑完
- [ ] `utov pipeline --runner-cmd "java -jar runner.jar serve <so> <trace>" --input ...`
      conformance gate C1-C5 全 PASS
- [ ] `findings_total` > 100、`algorithm_identified` 有内容
- [ ] 没有"first rerun ok, second rerun returns 0 bytes"（C5 fail 的典型症状 ——
      表明你违反了"不要缓存 EmuContext"）

---

## 9. 版本

v3, 2026-05-27。

**v3 (2026-05-27, BUG_REPORT-7)**:
- 新增 §8 `JniLiveRunner` 参考基类规范 + 实现位置
  (`contracts/java/com/clarkutov/runner/contract/JniLiveRunner.java`)
- 把 TC1/TC2 重复 ~120 行的 unidbg 脚手架抽进基类，子类 ~12 行就够

**v2 (2026-05-26)**:
- 接口从单方法 `rerun` 升级为三方法 `get_trace` / `rerun` / `metadata`（PLAN §17）
- 区分 Live 模式 vs File 模式
- 增加 §6 一致性自检为强制门禁

**v1 (2026-05-26)**:
- 初始版本，单方法 `rerun(input)->bytes` + 静态 trace + YAML 元信息
