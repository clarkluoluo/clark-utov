# clark-utov

> **clark-utov** is an agent's friend, tool, and ledger assistant for long-running
> tasks and hypothesis chains.

> Primary doc is Chinese: [README.md](README.md) — this is the English mirror.

## Methodology card (clark-utov)

| # | Principle | the reference target evidence |
|---|-----------|---------------------|
| 1 | **Pin entry and exit** | I/O 93/93 ≠ digest; `0x32350c` is not the exit → R1 @ `0xb7bb0` |
| 2 | **Keep an auditable ledger** | `hook_src_valid:626` invalidated; R1/R2 trust gate JSON only |
| 3 | **Roll back on error** | After `state_reconciliation.md`, then R1 |
| 4 | **Diff to locate the fault** | random_007 utov diff → SignRunner is not the input |
| 5 | **Start fresh** | R2 switches VMP observation layer; when enumeration is exhausted → suspend pending dev |

First target partial archive: `work-tc3-samples/work/legacy/reference-target_partial_archive/` · [RELEASE_NOTES.md](RELEASE_NOTES.md) **v0.1.0-partial**

> **An agent-friendly tool.** Every analysis step is exposed as a typed
> JSON-RPC method (`contracts/agent_protocol.md`), every promotion is
> auditable (`interventions` ledger), every batch is reversible
> (`discard_batch`), and every finding carries the inputs the agent needs
> to challenge it (`anchors_seen`, `evidence_score`, `reference_impl`,
> `io_test`). The engine is built to be driven by an LLM agent, not
> just stared at by a human.

> **And it gives a context-limited agent long-horizon working capacity.** Process
> data is externalized, the agent's context carries decision elements only, and an
> auditable ledger + gated pipeline + a by-situation capability map let an agent
> whose context/memory isn't that strong still push a long task across many rounds —
> like this build's VMP recovery — without holding the whole chain in context. utov
> uses deterministic mechanisms to make up for the agent's context limits.

> **Key diagnosis (sets the priority).** The primary limit of a narrow-context agent
> on long-horizon work is *drift* — it cannot hold the dozens-of-steps global map /
> history / closed-state — **not** judgment accuracy. Judgment decomposes into bounded
> checkpoints (evidence in front of it, one decision at a time) behind parity gates,
> which catch wrong calls; but not-enough-memory is the intrinsic limit of "narrow".
> So utov's highest-leverage move is **externalizing long-range memory to the extreme**
> (the `cvd_ledger` + a projected global stage map): the agent **never holds the whole
> chain** — each step it reads only utov's current projection, judges one bounded thing,
> runs utov's mechanics. Ledger/projection matter **far more** for a narrow agent than a
> large one — the large one can brute-force-remember for a while; the narrow one drifts and dies.

Trace-based automated recovery of algorithms hidden inside VMP/OLLVM-protected
ARM64 Android native libraries. Given a target whose JNI entry can already be
called from `unidbg` (or equivalent), the engine consumes a regularized
instruction trace, runs a deterministic pipeline (S1–S5), and accumulates a
verified hypothesis ledger about what the algorithm is and how it works.

**Status**: P0 + P1 are functional end-to-end.
- S1–S5 stages, conformance gate, runner contract, hypothesis ledger, both
  script_mode and agent_mode orchestrators, blue-team review, rule promotion
  & admission test all implemented.
- Cross-stage **feedback context** is wired: S1.5 publishes fingerprint anchor
  indices into the session; S4 picks them up as additional sinks (kept_nodes
  on libsha256.so went from 4 → 245 once the loop closed).
- S3 ships as a concrete dataflow graph (Triton symbolic execution on the P1.5
  roadmap). S6 (LLM hypothesis loop) is fully wired; the actual DeepSeek call
  has not been burned against your API key by default.

---

## Version progression

> Full changelog: [`RELEASE_NOTES.md`](RELEASE_NOTES.md); forward roadmap:
> `dev_doc/PLAN.md` §21.

- **v0.1.0-partial** — first target, the reference target (SM3), end-to-end: trace → S1–S5 →
  I/O → Triton → R1 hook.
- **Framework generalisation (v0.2–v0.5)** — the profile layer (declarative verdict
  semantics), field-experience extensions, the Task target-management mechanism.
- **Level-1 (set-up symex, Tier-1)** — the opaque-symbol-recovery scaffold: four
  contracts + forward/backward dual mode + an executing `drive` + the **multi-vector
  cross-run parity gate** (a tautological 1/1 can no longer pass as EXACT).
- **Level-2 (utov's own concolic) ✅ tag `hypotask-level2-complete`** — the agent no
  longer hand-writes the symex runner: a Triton bulk decoder + an extensible **escape
  hatch** (an un-modeled instruction is a precise BLOCK; hand-filled semantics are
  injected into the symbolic state and cached), with trace-guided concolic
  (execution-order segment + branches taken from the trace + full symbolic seed).
- **Level-3 — in progress (branch `level3`); the big features already landed:**
  - **Final-sink-first toolkit** — when the output is a small final construction (a fixed
    header ‖ a copy from a live buffer), anchor on the final-sink chain instead of a
    true-but-irrelevant internal one (import-map / libc boundary-edge synthesis / real-gold
    distinct-output floor).
  - **Mem-write sink recovery** — when the window output is a STORE, not a
    register, self-check + parity over the symbolic bytes; can't pin it → a structured
    terminal, never a silent fallback.
  - **Extern executable-model registry** — resolve an extern symbol to a runnable,
    evidence-tagged reference model and rank candidate families — so the verifier never
    hand-writes per-target PRNG/libc code.

---

## Approaching a VMP target: light-to-heavy (phase order)

On a VMP target whose algorithm you don't yet know, work the cheap evidence
first and reach for a full trace last. This order is the sign5 methodology
(roadmap §8.12/§9.4) and is encoded as a forced-order interface in
`engine.vmp_phase_api` (`VmpPhaseApi`) so it survives across long runs:

| Phase | Move | Cost |
|---|---|---|
| **1 · `phase_1_io_observe`** | calltrace to the crypto entry + hook I/O — capture the I/O **shape** only | light |
| **2 · `phase_2_materialization_trace`** | hook the output write sequence (the `strb`s) — where the formula structure shows | light |
| **3 · `phase_3_provenance`** | follow the data flow: watch first-write + 5-way provenance → producer chain. *No "guess the algorithm" move exists here — only "trace the flow".* | light |
| **4 · `phase_4_formula_induction`** | your judgement: induce **one** formula from what 2–3 showed (not a candidate spray) | — |
| **5 · `phase_5_parity`** | full-chain bytewise parity against the oracle | light |
| **`phase_heavy_vmtrace`** | full instruction-level trace — **the escalation, not the opening move** | **heavy** |

Each phase refuses to start until its predecessor recorded a verdict, so the
order is structural, not advisory. See the
[`AGENT-WORKFLOW.md`](AGENT-WORKFLOW.md) §2 *Step 0* runbook for the driving loop.

### Using vmtrace (`phase_heavy_vmtrace`)

> ⚠️ **We do not recommend going straight to a full vmtrace.** On a fresh VMP
> target it is rarely the right opening move — phases 1–3 close most cases far
> more cheaply. A full trace is the *escalation* for when the light path is
> genuinely exhausted.

So vmtrace sits behind a deliberate gate, satisfied one of two ways:

- **autonomously** — an `EscalationProof` citing the phase 1–3 outcomes that
  recorded `COULD_NOT_CLOSE` (the light path provably hit a wall), or
- **interactively** — a **warning prompt** the agent must clear: *"已尝试
  phase_1-3 均未闭合，确认升级到 vmtrace 吗？"* (or, if the light phases were
  skipped, the louder *"尚未尝试 phase_1-3 就要上 vmtrace，确定？"*). A human/driver
  `yes` is recorded for audit.

Either way the agent must also commit a **`VmtraceBudget`** up front — estimated
`runtime_s` + `disk_mb` — so the cost is a conscious, recorded decision and the
warning prompt can show it. None of this *blocks* a determined caller (raw
primitives are always reachable); it makes the cheap path the default and the
expensive one a visible, audited choice.

---

## 1. What this repo contains

```
clark-utov/
├── README.md                      ← you're here
├── NOTICE                         ← third-party attribution (MIT)
├── contracts/                     ← public interface specs
│   ├── runner_interface.md          v2 (PLAN §17 contract + conformance test)
│   └── java/                        canonical Java version of the contract
├── engine/                        ← the Python system (consumer)
│   ├── pyproject.toml
│   ├── engine/                      core package
│   │   ├── stages/                    S1–S6 pipeline
│   │   ├── core.py                    driver-agnostic facade
│   │   ├── orchestrators/             script_mode (run_full_pipeline) / agent_mode (JSON-RPC stdio)
│   │   ├── rules/                     rule promotion / admission / registry / telemetry
│   │   ├── verifier.py                3 concrete strategies
│   │   ├── conformance.py             C1-C4 gate
│   │   ├── llm_client.py              DeepSeek + MiMo backends
│   │   ├── runner_client.py           trace readers + RunnerAdapter implementations
│   │   ├── hyp_tree.py                N-ary backtracking ledger
│   │   ├── fold.py                    line / block-aware fold
│   │   ├── dataflow.py                regflow / producer / semop primitives
│   │   ├── static_tools.py            whitelisted subprocess (radare2 / readelf / ...)
│   │   ├── discipline.py              anti-drift LLM reminder injection
│   │   ├── blue_team.py               adversarial finding review (P2)
│   │   ├── store.py                   workdir + SQLite layout
│   │   ├── data/fingerprints.py       97 crypto/hash constant fingerprints
│   │   └── profile/                   v0.3.0 — declarable judgment semantics
│   │                                    (base mechanism vs domain semantics;
│   │                                     adding a new domain = adding a JSON file)
│   ├── profiles/                      shipped profile JSON files
│   │   ├── base.json                    5 mechanism probes (M1/M3/CP/VP/WFW)
│   │   ├── vmp_algorithm_extraction.json
│   │   ├── key_extraction.json
│   │   └── weird_target_x.json
│   ├── tests/
│   ├── dry_run.py                   File-mode demo (uses static kanxue trace)
│   └── dry_run_live.py              Live-mode demo (drives the Java runner)
├── example/                       ← samples (not part of the engine, see example/README.md)
│   ├── runner-sha256/               ← reference runner: how to implement contracts/ (SHA-256 target)
│   │   ├── libs/arm64-v8a/libsha256.so  our SHA-256 OLLVM-style sample (ground truth)
│   │   ├── java/                        Java glue class
│   │   └── runner/                      ← Maven project: unidbg-based test runner
│   │       ├── pom.xml                    pulls unidbg-android 0.9.9
│   │       └── src/main/java/...          Sha256TestRunner + Main(serve|demo)
│   └── task-libEncryptor/           ← a complete worked sample (brief + target + trace + candidates)
└── testTarget/                    ← third-party sample only (gitignored)
    └── vmp/                          kanxue libEncryptor.so + trace
```

Excluded from the public repo (`.gitignore`): `dev_doc/` (internal plans),
`.env` (API keys), `tmp/` (build scratch), `testTarget/vmp/` (third-party sample).

---

## 2. Architectural principles (red lines)

These are inherited from `dev_doc/PLAN.md` and enforced in code structure.
They survive across modes and across PR cycles.

1. **Verifier is the only source of truth.** Plugin, LLM, and Triton outputs
   are *always* unverified until the verifier rules on them with real trace
   data. No exceptions.
2. **Findings ≠ hypotheses.** Findings have been verified; hypotheses haven't.
   They live in separate tables and never bleed across.
3. **Hypotheses are verified immediately**, not batched until end-of-pipeline.
   The hypothesis tree backtracks on failure within the same loop.
4. **Hypothesis tree is N-ary with backtracking.** DFS, siblings ordered by
   LLM-supplied confidence.
5. **LLM only sees clean small data.** Never raw trace. The deterministic
   stages do the heavy lifting; the LLM does pattern recognition.
6. **No time-axis binary search.** Reduce to relevant logic via backward
   data-flow slicing (S4), not range bisection.
7. **VMP and OLLVM are handled differently.** Don't mix strategies in the
   same stage.
8. **Core / driver split.** `engine/engine/core.py` is the only thing
   `orchestrators/` may import. The two driver modes (script, agent) share
   one core implementation — never reimplement.
9. **Anti-debug is the runner's problem.** The engine starts where unidbg
   can already call the target stably.
10. **Open the semantic layer, hold the mechanism baseline.** v0.3.0
    extracts evidence classes, node states, closure gates, scope rules
    and cause→action routing into a declarable **profile** layer
    (`engine/profiles/*.json` + `engine.profile.*`). Adding a new
    target domain — key extraction, weird custom protections —
    becomes a profile-JSON edit, not an engine source change. But
    the mechanism baseline (M1 observation≠closure, M3 false-block
    detection, `constant_provenance` framework, observation-must-cap,
    observation→producer trace) lives in **base profile** locked by
    three independent doors (load-time registry rejection / runtime
    gate force-include / dual-side lint) and cannot be disabled,
    overridden, or removed by any subprofile. Full spec:
    `dev_doc/PLAN.md` §19.

> **Red lines draw the boundary; a route gives the road — ship both.** A red
> line is a *hard refusal*: it says what is **not** allowed. A route is a
> *default road*: the light-to-heavy phase path that says what to **do next**.
> A red line on its own leaves an agent circling the boundary hunting for a
> seam — it spends its budget looking for bypasses instead of working (the
> reference-case trap, roadmap §8.11; we have watched a fresh agent do exactly this
> against a red-line-only brief). So **pair every red line with a route stage**:
> "forbidden X" should always arrive with "instead, do Y". Boundary without a
> road is not strictness — it is a brief that forces the agent to choose
> between a formal spoof and grinding. A brief that does this well:
> [`sample_brief_aes_vmp.md`](sample_brief_aes_vmp.md).

---

## 3. Quick start

### 3.1 Engine prerequisites

**Just Python 3.11+.** That's it for the engine itself.

```bash
python3 -m pip install --user clark_utov_engine-0.1.0-py3-none-any.whl
# or for dev:  python3 -m pip install -e engine/
utov doctor       # verify environment
```

Python dependencies are declared in `engine/pyproject.toml` and pulled by
pip automatically.

The engine consumes any runner subprocess that speaks the NDJSON wire
protocol in [`contracts/agent_protocol.md`](contracts/agent_protocol.md).
**It does not care what language your runner is in or which emulator it
uses.** Tested compatibility against the sample runner using **unidbg-android
0.9.9** (see [DEPENDENCIES.md](DEPENDENCIES.md) for the matrix).

### 3.2 Run the pipeline

You need a **runner command** — a shell command string that spawns your
runner. The engine spawns it and talks NDJSON over its stdio.

```bash
# Pass whatever spawns your runner. Examples:
RUNNER='your-runner-cmd serve /path/to/target.so'         # your own runner
# or, using the bundled Java sample (see example/runner-sha256/README.md):
RUNNER="$(pwd)/bin/run-runner.sh serve $(pwd)/example/runner-sha256/libs/arm64-v8a/libsha256.so"

# (a) Estimate cost first (no LLM, $0)
utov pipeline --runner-cmd "$RUNNER" --input 616263 --estimate-only

# (b) Frugal mode: no LLM, $0 — runs S1..S5
utov pipeline --runner-cmd "$RUNNER" --input 616263 --mode frugal

# (c) Aggressive with budget cap
utov pipeline --runner-cmd "$RUNNER" --input 616263 --mode aggressive \
    --budget-usd 0.50 --budget-tokens 1000000 --budget-seconds 300
```

> **Note on the bundled sample**: `example/runner-sha256/runner/` is an example showing
> how to implement a runner using Java + unidbg. **Not part of the engine.**
> If you use it, see [`example/runner-sha256/README.md`](example/runner-sha256/README.md) for
> Java/Maven/NDK/unidbg prerequisites — those are sample-fixture-specific,
> not engine-required.

Expected output (abbreviated, frugal mode on libsha256.so post-0526):
```
work dir:  /.../work/libsha256.so/<input_hash>/runs/<run_id>
pipeline summary:
  {"stage":"s1","blocks":1005,...}
  {"stage":"s1b","fingerprint_hits":16,"hypotheses_seeded":16,...}
  {"stage":"s2","unique_blocks":35,...}
  {"stage":"s3","nodes":7844,...}
  {"stage":"s4","sinks":[...],"kept_nodes":245,...}
  {"stage":"s5","annotations":245,"inssub_matches":0,...}
  {"stage":"s1b-verify","checked":16,"passed":16,"promoted":16}     ← layer-0 plugin pass
  {"stage":"s5-verify","checked":46,"passed":46,"promoted":46}      ← layer-0 reg-reg-reg binop (BR-4 §1)
  {"stage":"s5-verify-unary","checked":34,"passed":34,"promoted":34}      ← layer-0 unary (0526 C5)
  {"stage":"s5-verify-imm","checked":20,"passed":19,"promoted":19}        ← layer-0 reg-imm binop (0526 C5)
  {"stage":"s5-verify-ext","checked":18,"passed":18,"promoted":18}        ← layer-0 shifted/extended binop (0526 C5)
  {"stage":"s5-verify-bfx","checked":0,...}                                ← layer-0 ubfx/sbfx (0526 C5.6)
  {"stage":"s5-verify-ch","checked":0,...}                                 ← Ch idiom (0526 C3 + BR-8 #2 and/bic/orr variant)
  {"stage":"s5-verify-maj","checked":0,...}                                ← Maj idiom (BR-8 #2)
  {"stage":"s5-verify-triton","checked":58,"passed":56,"promoted":56}     ← Triton symex (0526 C1)
  {"stage":"s5-fold-sigma","matched":1,"promoted":1,"linked_members":3}    ← layer-1 σ/Σ fold (0526 C4 + BR-8 #1 DFG-grouped Phase 3)
  {"stage":"s5-algorithm-fit","matched_algorithms":["SHA-256"],...}        ← layer-2 algorithm fit (0526 E1)
  {"stage":"s5-anchor-rescan","sigma_promoted":0,"refit":...}              ← BR-8 #3 anchor-gap self-rescan

mode:                frugal
findings_promoted:   191

algo_signature hypotheses (16):
  hyp#1 conf=0.65 subj=SHA256.h0 fp=SHA256.h0 source=plugin
  ...
```

The s1b-verify / s5-verify-* / s5-fold-sigma / s5-algorithm-fit /
s5-anchor-rescan stages are the deterministic batch — they run in both
`--mode frugal` and `--mode aggressive` and produce `source` field values
`plugin / s5_deterministic / s5_triton / s5_fold_idiom / s5_algorithm_fit`
in `findings.sqlite`. Pull the per-source breakdown with
`utov status <run_dir> --by-source`.

### 3.4 File mode (no runner — static trace only)

```bash
# Run the full File-mode pipeline (S1..S5 + layer-0/1/2 verify chain)
# on a pre-generated trace (kanxue VMP sample example)
python3 -m engine.cli pipeline-file \
    --trace ../testTarget/vmp/trace.txt \
    --target-name libEncryptor.so \
    --entry 0x40007d88 \
    --exit  0x40007ed8 \
    --output-len 32
```

`pipeline-file` runs the same orchestration as Live mode's `pipeline` minus
the LLM/S6 loop: S1..S5 followed by `s1b-verify`, `s5-verify`,
`s5-verify-{unary,imm,ext,bfx,ch,maj,triton}`, `s5-fold-sigma`,
`s5-algorithm-fit`, and `s5-anchor-rescan`. Findings are written to
`findings.sqlite` and an `algorithm_identified` hypothesis is emitted
whenever the anchor set fits a known template (see §7.4 for expected
output on the bundled VMP sample).

In File mode, conformance C1/C2/C3 SKIP (no `rerun` available) and any
verifier strategy that needs rerun degrades. Only C4 (trace integrity)
runs. Algorithm-fit's `io_test` field self-documents the skipped IO
equivalence test.

For a self-contained reference implementation (e.g. when scripting around
the CLI), see `engine/examples/file_mode_full.py`.

### 3.5 Interpreting the deliverable (`utov emit`)

`utov pipeline-file` / `utov pipeline` stop at `algorithm_identified`,
which is a **label** (`algorithm: SHA-256, evidence_score: 1.0,
anchors_seen: 12/12`). To turn that label into a paste-and-read
**reconstruction** (IV constants, K table when fingerprinted, σ/Σ idiom
PCs + register assignments, observed loop counts), run:

```bash
utov emit <run_dir>                          # print to stdout
utov emit <run_dir> --output pseudocode.md   # write to file
utov emit <run_dir> --format markdown        # fenced markdown
```

`preprocess_batch` also drops `<run_dir>/pseudocode.md` automatically
whenever it promotes an `algorithm_identified` finding — agents that
already have the run path get the artefact for free.

Supported algorithms are listed in
`engine/engine/data/algorithm_pseudocode.py:ALGORITHM_SPECS`
(SHA-256 / SHA-512 today; PRs welcome for SHA-1 / MD5 / SM3 / SM4 /
AES round-form / HMAC).

---

## 4. How the pipeline works (each stage in one paragraph)

| Stage | What it does | Output |
|---|---|---|
| **C1-C4 conformance gate** | Before any analysis, the runner is poked five times for determinism, three byte-flips for input sensitivity, one observation point, and trace start/end PC integrity. If any check fails, the pipeline refuses to start. | `conformance_report.json` |
| **S1 segment** | Walk the trace once. Split into basic blocks at every control-flow transfer (`b`, `bl`, `br`, `ret`, `b.cond`, `cbz`, ...) or PC discontinuity. | `s1.jsonl` (one row per block) |
| **S1.5 fingerprint** | Scan every `regs_write` value against 95 crypto constants (MD5/SHA-1/SHA-256/SHA-512/SM3/SM4/AES/CRC/HMAC/...) and every disasm against 2 NEON SIMD patterns. Each hit becomes a `confidence=0.65–0.85` hypothesis. **Verifier still has the final say** before promotion. | `s1b.jsonl` + ledger rows |
| **S2 dedupe + fold** | Hash each block by its PC sequence; collapse identical executions; then fold runs of `≥ 10` consecutive identical-hash blocks into `first + sentinel + last` (PLAN §12.4). | `s2_blocks.jsonl` + `s2_executions.jsonl` |
| **S3 dataflow graph** | For each instruction, link each `regs_read` to its most recent producer instruction. Concrete DFG is the default; Triton symbolic execution opt-in via `--symex triton`, writes `s3_symex.jsonl` alongside. | `s3_dfg.jsonl` (+ `s3_symex.jsonl` with Triton) |
| **S4 backward slice** | From designated sinks (default: last instruction's writes), BFS backward through the DFG; keep only ancestors of the sink. | `s4_slice.jsonl` |
| **S5 simplify** | Lightweight: zero-idiom + `mov #imm` recognition + 4-line MBA reverse-match against DiANa's InsSub patterns. | `s5_simplified.jsonl` |
| **s1b-verify / s5-verify / s5-verify-{unary,imm,ext,bfx,ch,maj} / s5-verify-triton** | Layer-0 deterministic batch (0526 C5 / C5.6 / C3 / C1 + BR-8 #2). Each pass scans the trace for one ARM instruction shape (plugin fingerprints, reg-reg-reg binops, reg-imm binops, unary, shifted/extended-register, bit-field-extract, Ch idiom, Maj idiom, full-instruction Triton symex), feeds the verifier, and promotes PASS to `findings.sqlite` with the matching `source` tag. Dedupe by PC. Ch matcher covers both canonical eor/and/eor and the and/bic/orr variant (BR-8 #2). Triton uses a net-coverage filter so it doesn't double-count layer-0 PCs. | rows in `findings.sqlite` (kind=handler_semantic / algo_signature) |
| **s5-fold-sigma** | Layer-1 SHA-2 σ/Σ fold (0526 C4 + BR-8 #1). Three phases: (1) contiguous 3-insn window, (2) per-BB ILP-tolerant scan (≤8 unrelated insns between components), (3) DFG-grouped 3-tuple scan keyed by input register — catches ILP gaps that straddle BB boundaries and `dst==input` final-write patterns Phase 1/2 reject. Matches (kind, amount) frozenset against 8 SHA-256/SHA-512 templates; verifies the algebra; promotes one `kind=fold_idiom` finding bound to its 3 layer-0 members via `finding_groups`. | rows in `findings.sqlite` (kind=fold_idiom) + `finding_groups` |
| **s5-algorithm-fit** | Layer-2 structural anchor-set fit (0526 E1). Aggregates fold_idiom + plugin findings; checks against `SHA-256` / `SHA-512` / `AES` template (anchor set size ≥ template min trips a fit); promotes one `kind=algorithm_identified` finding with confidence scaling from anchor coverage. IO-equivalence is run when the runner exposes `rerun(bytes)` (BR-7 §C); skipped in File mode. | rows in `findings.sqlite` (kind=algorithm_identified) |
| **s5-anchor-rescan** | BR-8 #3 anchor-gap self-rescan. When an `algorithm_identified` finding still has missing anchors (e.g. only 11/12), re-runs σ/Σ + Ch + Maj — which dedupe internally — and calls `recompute_algorithm_fits` to refresh the existing payload's `anchors_seen` / `evidence_score`. Idempotent; no-op when nothing is missing. | rows updated in `findings.sqlite` (kind=algorithm_identified) |
| **S6 hypothesis loop** (aggressive only) | LLM-backed proposer for stuck points the layer-0/1/2 passes don't cover. Wraps LLM calls with the discipline reminder; each candidate goes straight to verifier; pass → finding, fail → backtrack. Default LLM backend is `none` (0526 D2 — no API call); pass `--llm-backend deepseek` or `--llm-backend mimo` to engage. | hypotheses DB |

---

## 5. Runner contract (writing your own runner)

Any subprocess that speaks the NDJSON-over-stdio protocol described in
`contracts/runner_interface.md §3` and passes the conformance gate is a
valid runner. The engine doesn't care what language it's written in.

**Required methods** (Live mode):

```
get_trace(input, start, end) → JSONL or unidbg-text trace file path
rerun(input, observe_points) → {output, observations}
metadata()                   → {target_name, arch, entry_pc, exit_pc, ...}
```

**Conformance gates** (must all PASS):

```
C1 DETERMINISM        5 reruns of same input produce bit-identical output
C2 INPUT_SENSITIVITY  3 byte-flipped inputs ≥ 2 produce different output
C3 OBSERVE_POINT      Observation at entry_pc returns non-empty regs
C4 TRACE_INTEGRITY    Trace start/end PC match metadata anchors
```

**File mode** is for samples where only a static trace exists and no rerun
is possible (e.g. the kanxue libEncryptor sample we ship). Only C4 runs;
the verifier flags `verifier_degraded=True` and the deliverable carries
that caveat through.

See `example/runner-sha256/runner/` for a complete reference implementation in Java
that uses `unidbg-android` and is driven via `SubprocessRunnerAdapter` from
the Python side.

---

## 6. Driving the engine from another agent

**Task-oriented usage walkthrough**: [`AGENT-WORKFLOW.md`](AGENT-WORKFLOW.md).
That document covers _which interface to call, in what order, and what
to read out of each result_; this section is a quick CLI / library
reference.

### 6.0 Interface categories at a glance

| Category | Examples | When |
|---|---|---|
| **Batch preprocessing** (`$0`, deterministic) | `utov pipeline --mode frugal` · `preprocess_batch({passes?})` RPC · individual `verify_*` RPCs | Start of session. One call runs the layer-0/1/2 chain, tags every promoted finding with a `batch_id`, returns next-step hints. |
| **Exposure / read-only query** (`$0`) | `utov status --json --by-source/--by-kind` · `utov findings` · `utov hyps` · `get_findings` · `get_hypotheses` · `stuck_statistics` · `read_trace_window` · `static_tool` | After batch — agent reads totals + per-axis breakdowns, decides whether to discard / pursue LLM. `utov findings` / `utov hyps` are CLI mirrors of `get_findings` / `get_hypotheses`. |
| **Decision / write** (audited) | `override_verdict` · `utov override` · `batch_override_verdict` · `discard_batch` · `inject_finding` · `force_status` · `add_tag` · `rerun_from_stage` · `list_interventions` | Agent disagrees with the verifier, wants to retract a batch's pass, or marks something the verifier missed. Every write writes an `interventions` audit row. `utov override` is the CLI mirror of `override_verdict`. |
| **Light-to-heavy phase route** (VMP, forced order) | `utov phases` · `phase_state` · `phase_1_io_observe` · `phase_2_materialization_trace` · `phase_3_watch_producer` · `phase_3_classify` · `phase_4_formula_induction` · `phase_5_parity` · `phase_record` · `phase_heavy_vmtrace_prompt` · `phase_heavy_vmtrace` | VMP target, algorithm unknown — work the cheap evidence first. Order enforced (`phase_record` gates the next phase); vmtrace is escalation-gated. No "enumerate ciphers" method exists. `utov phases` prints the route. |
| **Cross-run / ops** | `utov compare a b` · `utov audit` · `utov resume` · `utov doctor` · `checkpoint` / `pause` / `is_safe_to_interrupt` | Run-management: before/after experiments, lifecycle, host health checks. |
| **S6 LLM loop** (opt-in spend) | `s6_find_stuck_points` · `s6_propose_and_verify` · `s6_auto_loop` · `--mode aggressive --llm-backend deepseek/mimo` | Only for shapes the deterministic chain doesn't model. Default `--llm-backend none` = no LLM, no spend. |

Full wire spec: [`contracts/agent_protocol.md`](contracts/agent_protocol.md)
§3.2 has the method-by-method table; §4 covers the inverse channel
(`llm_request`) where the engine asks the agent to answer.

### 6.1 As a CLI (most agents)

```bash
# Get a quick read on any trace file
python3 -m engine.cli trace-info <path>

# Run the full pipeline against a runner (one-shot batch — frugal mode is $0)
python3 -m engine.cli pipeline --runner-cmd '<your runner spawn cmd>' --input <hex>

# Read what just landed
python3 -m engine.cli status <run-dir> --json --by-source --by-kind

# Query the findings / hypotheses tables directly (CLI mirrors of the RPC reads)
python3 -m engine.cli findings <run-dir> --source plugin --kind algorithm_identified --json
python3 -m engine.cli hyps <run-dir> --status passed --kind algo_signature --json

# Flip a verdict + log an intervention (CLI mirror of override_verdict)
python3 -m engine.cli override <run-dir> <hyp_id> fail --reason "agent rejects this"

# Print the recommended light-to-heavy VMP phase route (static, no args)
python3 -m engine.cli phases

# Diff two runs (e.g. before / after a code change)
python3 -m engine.cli compare <run-a> <run-b>

# Or run against a pre-existing static trace (file mode, no rerun)
python3 -m engine.cli pipeline-file --trace <path> --target-name <name> \
    --entry <0x..> --exit <0x..>
```

### 6.2 As a library (Python agents)

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
    summaries = core.run_pipeline()        # → list of stage summaries
    for s in summaries:
        print(s)

    # Access the hypothesis ledger
    for h in core.get_hypotheses(kind="algo_signature"):
        print(h.id, h.subject, h.confidence, h.payload)
finally:
    runner.shutdown()
```

### 6.3 Outputs

Everything is persisted under `work/<target>/<input_hash>/runs/<run_id>/`:

```
work/<target>/<input_hash>/
├── runs/<run_id>/
│   ├── meta.json                  ← driver mode, run id, target info
│   ├── conformance_report.json    ← C1-C4 verdicts
│   ├── stage_state.json           ← which stages are done at which code_version
│   ├── stage_outputs/
│   │   ├── s1.jsonl               ← basic blocks
│   │   ├── s1b.jsonl              ← fingerprint hits
│   │   ├── s2_blocks.jsonl        ← unique blocks
│   │   ├── s2_executions.jsonl    ← block execution stream w/ sentinels
│   │   ├── s3_dfg.jsonl           ← data-flow graph
│   │   ├── s4_slice.jsonl         ← surviving slice
│   │   └── s5_simplified.jsonl    ← annotated slice
│   ├── findings.sqlite            ← verified facts (+ hyp_payloads blob store)
│   ├── hypotheses.sqlite          ← 6-table WAL ledger (D-027):
│   │                                hyp_payloads / claim_templates / hypotheses
│   │                                hyp_anchors / hyp_tags / hyp_dependencies
│   ├── archived/                  ← abandoned-subtree archives, off hot path
│   ├── anomalies/                 ← verifier-irreducible cases (human review)
│   ├── session.json               ← cross-stage feedback bag
│   └── notes/                     ← blue-team notes, manual annotations
└── latest -> runs/<run_id>        ← symlink for resume
```

### 6.4 Envelope siblings (per-call wrapper output)

Every JSON-RPC response carries a `methodology` sibling next to
`result` — runtime anti-drift payload from the discipline wrapper
(`engine.discipline_wrapper`). Inside `methodology`, gates attach
their structured outputs as further siblings so the agent doesn't
have to re-derive the check by hand. Full wire spec in
[`contracts/agent_protocol.md`](contracts/agent_protocol.md) §3.3
and §3.5.

| Sibling | Source | Env toggle | What it carries |
|---|---|---|---|
| `footer` / `card` / `prompts` / `alerts` / `intercepted` | base wrapper | `UTOV_METHODOLOGY=off` | per-call footer, periodic discipline card, reverse-question prompts, runtime alerts, refusal flag + reason |
| `m1_audit` | `engine.m1_success_audit` | `UTOV_M1_AUDIT=off` | A/B/C grading of `target_success` / `archival_allowed` claims; B auto-downgrades to `strong_partial`, C refuses |
| `m3_bypass` | `engine.m3_bypass_block` | `UTOV_M3_BYPASS=off` | block flipped to `suspected_bypass_block` after ≥N distinct observation methods all reported variability=0; follow-up observations on the block refused before dispatch |
| `value_provenance` | `engine.value_provenance` | `UTOV_VALUE_PROVENANCE=off` | per-value `{source, provenance, final_class}` tags; hook/dump/io/snapshot capped at evidence_class B until closed-form recompute verified |
| `watch_suggestions` | `engine.watch_first_write` | `UTOV_WATCH_FIRST_WRITE=off` | auto `watch_first_write(addr)` spec on observed values with a concrete landing address (capture the producing PC + source bytes) |
| `length_chain` | `engine.length_chain_check` | `UTOV_LENGTH_CHAIN=off` | per-edge breakdown of any `length_chain: [...]` you sent; flags `unexplained_edges[]` when adjacent lengths have no explainable relation |
| `phase_discovery` *(hidden by default when router is wired)* | `engine.phase_discovery` | `UTOV_PHASE_DISCOVERY=off`; `UTOV_PHASE_DEBUG=1` to restore | value records whose producing computation lives outside the loaded trace window; emits `PhaseBoundary` (pc_range / region / anchor) for the producing phase |
| `phase_instrument_suggestions` *(hidden by default when router is wired)* | `engine.phase_instrument` | `UTOV_PHASE_INSTRUMENT=off`; `UTOV_PHASE_DEBUG=1` to restore | auto-suggested `PhaseInstrumentSpec` (anchor + granularity) for each discovered boundary |
| `block_cause` | `engine.block_cause` | `UTOV_BLOCK_CAUSE=off` | L1 routing conclusion for each unresolved node — classifier picks one of `collection_gap` / `recognition_gap` / `strategy_gap` / `true_boundary`, then auto-routes to `auto_collect` / `register_backlog` (writes `<run_dir>/capability_backlog.jsonl`) / `escalate_l2` / `escalate_l3` / `escalate_user`. Class-3 (true boundary) is the only path that reaches the user, and only with clark-prepared decision elements. |
| `constant_provenance` | `engine.constant_provenance` | `UTOV_CONSTANT_PROVENANCE=off` | 5-way deterministic verdict per value (`hardcoded_fixed` / `appkey_fixed_function` / `session_level_derived` / `per_input_variable` / `undetermined`), driven by two probes — rerun variability across 4 axes + producer dataflow. Auto-sets evidence_class ceiling, scope, and recommended_action. Dataflow trumps reruns on the entropy-locked blindspot. M3 (per-input) + M1 dimension-variability check unified. |
| `profile` *(v0.3.0)* | `engine.profile` | omitted when wrapper constructed without a profile | `{name, chain}` — advertises the v0.3.0 profile the engine is running under (e.g. `vmp_algorithm_extraction`, `key_extraction`). `chain` is the resolved inheritance chain from base to leaf. Agents that branch on domain read this; legacy agents ignore the key. |

`UTOV_METHODOLOGY=off` suppresses the wrapper entirely (envelopes
degrade to the plain `{"id":…,"result":…}` shape). Individual gate
toggles let you debug-isolate one gate without flipping the larger
switch. When the block-cause router is wired,
`phase_discovery` / `phase_instrument_suggestions` are L1
intermediates and stay hidden unless `UTOV_PHASE_DEBUG=1`.

---

## 7. For testing this project

### 7.1 Lint + unit tests

```bash
cd engine
python3 -m ruff check .                    # style + bug-class checks
python3 -m pytest -v                       # 3 trace-reader tests
find engine -name '*.py' | xargs -n1 python3 -m py_compile   # syntax check
```

### 7.2 Standing reference runs

```bash
# (a) File mode on the kanxue VMP sample (ground truth: was SHA-512-shaped)
python3 dry_run.py

# (b) Live mode on libsha256.so (ground truth: SHA-256 NIST vectors)
python3 dry_run_live.py
```

Both write a conformance report to `/tmp/` and exit 0 on success.

### 7.3 What a healthy run looks like on libsha256.so

- C1-C4 conformance gate: **all PASS** (Live mode)
- S1 blocks: **~1,000**
- S1.5 fingerprints: **16 hits — SHA256.h0..h7 + SHA256.K[0..7]**
  (NOT SHA-512; if you see SHA-512 you're tracing `testTarget/vmp/libEncryptor.so`
  instead of `example/runner-sha256/libs/arm64-v8a/libsha256.so`)
- S2: **~35 unique blocks**
- S3: **~7,800 dataflow nodes**
- S5: at least some `mov_immediate` annotations on the K-table loads

### 7.4 What a healthy File-mode run looks like on libEncryptor.so

Running `utov pipeline-file` against the bundled kanxue VMP sample
(`example/libs/arm64-v8a/trace.txt`, 41,416 instructions) should produce:

- C1-C3 conformance gate: **SKIP** (`verifier_degraded=True`), C4 PASS
- S1 blocks: **~4,668**
- S1.5 fingerprints: **10 hits — SHA512.h0..h7 and the two SHA-512 σ
  constants**
- S5 verify chain: **~955 promoted findings (fresh venv, no Triton)** across
  `s1b-verify` (10), `s5-verify*` (~890 handler-semantic),
  `s5-fold-sigma` (22 σ/Σ folds after BR-8 #1 Phase 3), `s5-algorithm-fit` (2:
  SHA-512 + AES), plus `s5-anchor-rescan` / `s5-mode-ledger` /
  `s5-primitive-timeline` / `s5-static-artifacts`.
- `algorithm_identified` payload (SHA-512):
  - `algorithm: "SHA-512"`
  - `anchors_seen`: **12/12** SHA-512 anchors (σ₀, σ₁, Σ₀, Σ₁, h0..h7).
    σ₁ now matched after BR-8 #1's DFG-grouped Phase 3 lifts `dst==input`
    + ILP-gap constraints.
  - `evidence_score = 1.0`
  - `io_test: "skipped (NullRunner — file mode)"`
- `algorithm_identified` payload (AES):
  - `confidence: 0.85` (BR-7 §B override — Te0 constants are zero-FP)
  - `anchors_seen`: 2/7 AES.Te0[0..3] hits
  - `io_test: "skipped (no IO vector — keyed algorithm)"`

`utov status <run> --json --by-source --by-kind` is the agent-friendly way
to inspect the same numbers.

---

## 8. Known limitations / roadmap

| Issue | Status | Tracked |
|---|---|---|
| S3 is concrete-only; no Triton symbolic execution | P1.5 | IMPL_PLAN §3 |
| S5 simplification doesn't handle nested expression trees | P1.5 | IMPL_PLAN §4 |
| S5 InsSub 4-window patterns only match raw arm64 sequences (source-level OLLVM macros may not trip them) | P1.5 | — |
| S6 LLM loop wired; opt-in via `--mode aggressive` (frugal mode never calls LLM) | by design | — |
| `--estimate-only` does not persist S1..S5 outputs (uses a throwaway work-dir suffixed `_estimate`); re-run without `--estimate-only` re-does S1..S5 | by design | — |
| `pyarrow` not used yet — stage outputs are JSONL | tracked | — |

---

## 9. Citations & attribution

This project includes work derived from third-party MIT-licensed sources.
See `NOTICE` for the boundaries:

- The 97-entry crypto fingerprint catalog, fold algorithm, regflow/producer/
  semop helpers were ported from
  [icloudza/algokiller-plugin](https://github.com/icloudza/algokiller-plugin)
  (MIT, cloudza 2026), specifically its Sprint 1-6 plugin extensions. The
  upstream `match`/`context`/`daemon` engine paths (separately attributable
  to [@lidongyooo](https://github.com/lidongyooo)) were not vendored.

`testTarget/vmp/libEncryptor.so` is a third-party sample from kanxue forum
thread 291195, included locally for analysis only and gitignored from the
public tree.

---

## 10. License

The engine code is MIT-licensed (see `LICENSE`).

`example/runner-sha256/libs/arm64-v8a/libsha256.so` is original to this project and
also covered by the MIT license. The kanxue VMP sample is **not** part of
the public repo.
