# Agent usage guide — what to call when (progressive), and where the boundary is

> The single agent-facing guide. **By-situation index**: given where you are, which
> capability to reach for, what to read out, and what utov will **not** do for you.
> For exact params/returns of each method, defer to [`AGENT-WORKFLOW.md`](AGENT-WORKFLOW.md)
> §1/§2 — this page routes, it does not re-spec.

**Release notes live here.** Small releases prepend a bullet to *Recent releases*
below (newest first); a major version also gets its own `WHATS_NEW-vX.md`. Every
build that adds or changes an agent-visible capability updates this page.

---

## Recent releases (newest first)

- **one-call recovery entry — `engine.cvd_recovery.run_recovery(...)`** wires a whole
  recovery CVD run for you: pure recovery registry (no `default_registry` sink noise),
  safe decision defaults with the GENUINE `mem_input_symbolize_vs_back` judgment still
  escalated to `pending_judgments`, anchor reconcile onto this run's base, and an actively
  persisted stamped `cvd_gap_map.json` with no sqlite. Prefer it over hand-assembling
  `recovery_registry` + `run_cvd_collect_to_json` (that hand-wiring is what got 3 configs
  wrong and drowned the gap map in noise).

- **recovery is one CVD run, and its output is stamped JSON — never the sqlite ledger**
  (`engine.cvd_recovery.recovery_registry`, `engine.cvd.run_cvd(..., collect_extensions=True)`,
  `engine.cvd.run_cvd_collect_to_json`, `engine.cvd.export_gap_map`,
  `engine.setup_symex.check_emit_self_consistency`).
  - **Recovery registers INTO CVD**: a `recover_window` candidate per window (from
    `dispatch_coverage` / `cohort_diff`), one heavy Verifier running the whole
    `setup_symex.drive` per window, a TerminalClassifier for opaque / seed-invariant windows.
    `collect_extensions=True` lists the WHOLE gap map in one run (confirmed +
    extension_requests + pending_judgments) instead of stopping at the first gap; default
    (`False`) is byte-for-byte unchanged.
  - **Consumer output = OUT-layer stamped JSON.** `run_cvd_collect_to_json(items, expected,
    work_root=, ts=)` runs the collect CVD and ACTIVELY writes `cvd_gap_map.json` with the
    `<!-- utov-export -->` header (one durable, traceable artifact — not a stream you lose).
    Read THIS; do NOT `open_ledger` / touch the sqlite ledger (utov's internal collect
    layer). No ledger passed → no `.sqlite`/`-wal`/`-shm` dead ledger. Use
    `engine.export_stamp.load_stamped_json` to read one back (`header=None` ⇒ a hand-written
    file, not a utov export).
  - **G4 emit self-check**: the runner should surface `trace_self_check`
    (`{seed_values, sink_value, sink_mask}`) so the recovered F is evaluated on its own
    trace's concrete seed and must equal the trace's window-exit sink BEFORE emit — a
    mismatch BLOCKs (catches a symex that silently emits a value disagreeing with its own trace).

- **seed-independence gate + cohort-diff localization — don't symbolize a constant,
  and find where the seed actually drives the computation**
  (`engine.setup_symex.check_seed_independence`, `engine.cohort_diff.localize_input_dependence`).
  A handler whose symbolized seed takes the SAME value in every cohort vector recovers a
  constant F that trivially passes parity — a false EXACT (the F0 case). The subject is the
  SEED (the recovery variable you symbolize), NOT hard-wired "input": F may be `F(input)`,
  `F(nonce)`, `F(input,nonce)`. Two complementary primitives:
  - `check_seed_independence(seed_values, min_vectors=)` — PRE-symex gate. `seed_values`
    maps each seed (a reg, or `"mem@0x…"`) to its entry value per cohort vector. BLOCK when
    EVERY seed is constant across the cohort (the window isn't seed-driven — setup, or the
    cohort never varied the seed); OK otherwise, surfacing constant seeds (those are concrete
    backing, not symbolic). `INSUFFICIENT` (surfaced, not silent) below `min_vectors`. This
    is the upstream companion to `check_parity_vectors` (which already counts independence by
    distinct `input_key` = the seed-assignment fingerprint, the same "independent evidence"
    notion CVD uses — it does not degrade to exec_id).
  - `localize_input_dependence(cohort_traces, input_keys=, ignore_regs=, ignore_addrs=)` —
    the full-trace extension: which windows vary with the seed (= recover) vs stay constant
    (= carry as a constant). Aligns by PC (a control-flow divergence is itself a seed-
    dependent branch, not a crash), diffs **register AND memory** write values (opaque values
    enter through stores — a register-only diff misses them), and `ignore_regs`/`ignore_addrs`
    control out a coupling axis (per-run nonce/time). Returns an `InputDependenceMap` with the
    seed's first entry idx and per-window verdicts. Crucially, a cohort that genuinely varies
    yet shows NO observable state change returns verdict `opaque` (real dependence hidden in
    staging → needs symex to pierce it), never a silent "no dependence".
- **concolic shadow — un-symbolized values follow the trace's ground truth, not 0**
  (`engine.setup_symex_runner`). Registers had a concrete shadow on their symbolic
  variables; the *rest of the state* did not — so a value Triton never saw seeded (a load
  off intra-handler state / a constant table, upstream thread state) was its uninitialised
  **0**, and a downstream `mul`/`eor` collapsed the whole transform to 0 (`emitted_F="0"`,
  parity 0/N — even after the window was correct). The Level-2 runner now keeps two
  complementary, taint-guarded shadows so the symbolic skeleton stays the input-dependent
  computation and *everything else follows the trace*:
  - **register reconciliation (primary, works on any register-trace)**: AFTER processing
    each step, every written register that Triton kept NON-symbolic gets its trace value
    (`regs_write[reg]`). This does **not** depend on a populated `mem[]` — the F0 reality
    is a register-trace whose `ldr` steps carry no `MemOp`, so the loaded value lives only
    in `regs_write`. Surfaced as `shadowed_reg_writes`.
  - **memory shadow (complementary, when `mem[]` is populated)**: BEFORE processing, every
    non-symbolized memory READ gets `MemOp.val`. Surfaced as `shadowed_mem_reads`.
  A register/memory cell Triton kept SYMBOLIC is input-tainted and is left untouched (F
  stays a function of the input); `isRegisterSymbolized` / `isMemorySymbolized` guard that
  and the multi-vector parity gate backstops over-concretization. The shadow is
  **self-adaptive across trace forms** (sparse-mem/full-reg, full-mem, mixed) with zero
  case-specific knowledge; a load with NEITHER a `mem[]` value nor a `regs_write` entry has
  no trace source and is counted as `unshadowed_steps` (surfaced, not a silent emit "0").
  This auto-answers the
  per-slot "symbolize vs back" question (only a genuine external-input ambiguity still
  pauses at `mem_input_symbolize_vs_back`). The backing flow is internalized too:
  `CaseConfig.concrete_backing`'s base values + pointed region bytes flow through `drive`
  and are upfront-seeded (`concrete_regs` + `concrete_mem`), so you don't inject a per-case
  decoder/hook in your pipeline. (§3)
- **dispatch preflight coverage map — see all the work before running**
  (`engine.dispatch_coverage`). A VMP dispatch loop calls ~6 handler TYPES many
  times; solving invocation-by-invocation is whack-a-mole. `preflight_dispatch_coverage(
  trace, invocations=, reg_file=, decode_probe=)` classifies the call sequence
  (you supply the `HandlerInvocation`s — `type_id` from your dispatch decode +
  idx window) and computes ONCE per type the full I/O signature: `reg_live_in` /
  `mem_live_in` (the input gaps, via the live-in derivation), `unmodeled_opcodes`
  (via `decode_probe` — `triton_decode_probe()` by default), and `outputs` (the
  live-out state carrier). The whole gap list is visible **up front** — solve each
  type to EXACT, compose along `sequence`, N invocations covered by ~6 solves.
  `CoverageMap.to_stamped_markdown(...)` projects it with the `utov-export` header. (§3)
- **auto-seed memory arm — external memory inputs** (`engine.setup_symex`). Register
  live-in only pins register inputs; a value entering through `ldr` (a carrier byte /
  table entry) was left un-symbolized → the symbolic chain never started → the window
  exit collapsed to a concrete 0 (the handler11 `symbolic=0` trap).
  `derive_window_mem_live_in(items, window=, window_is_idx=)` derives the window's
  external memory inputs (loaded bytes with no in-window writer, with `base_regs` for
  re-pin context) from byte-granular `mem_deps`. An un-pinned one makes `drive`
  **pause at the named checkpoint `mem_input_symbolize_vs_back`** (a DrivePause, not a
  silent 0) — symbolize-vs-back is your judgment: a new input that **arrived**
  (symbolize) vs a state carrier / table base that **passes through** (back). Resolve
  via `decisions={"mem_input_symbolize_vs_back": {addr: {"symbolize": value} | "back"}}`
  (or `on_checkpoint`), or pre-decide with `CaseConfig.symbolic_mem = ((addr, size,
  shadow), …)` / `concrete_backing`. The Level-2 `TritonStepDecoder` symbolizes the
  region (shadow on the variable) so it propagates. (§3)
- **drive auto-seeds per-handler symbolic inputs** (`engine.setup_symex`). Stop
  hand-configuring `symbolic_regs` per handler/window — the run-once-look-once trap
  (forget one handler → `sym_regs_n=0`, its input never propagates). Leave
  `CaseConfig.symbolic_regs=None` and `drive` derives the window's **live-in** (regs
  read inside the window with no producer inside it = the seed + the threaded state
  carrier) via dataflow, excludes concretely-**backed** pointer bases (the C2 split),
  and seeds those. A non-empty `symbolic_regs` you pass is still honoured verbatim
  (explicit override). Surfaced under `per_step` `seed_entry_state.auto_seed`
  (`live_in` / `backed_excluded` / `dropped_not_in_reg_file` / `empty`); a degenerate
  window or regs absent from `reg_file` raise an advisory `note` instead of silently
  seeding nothing. Helper `derive_window_symbolic_regs(items, window=..., reg_file=...,
  window_is_idx=...)` is also callable standalone. (§3)
- **level2 — utov's own Level-2 concolic symex** (`engine.setup_symex_runner`). Stop
  hand-writing the symbolic runner. A Triton **bulk decoder** covers the middle; an
  **escape hatch** covers the long tail — an opcode Triton can't model is a precise
  BLOCK (`DriveResult.unmodeled`), never force-concretized; you hand-fill that one
  instruction as an S-expression (`(bvmul x0 x1)`, `(bv v size)`, extract/zx/sx/concat)
  and utov compiles it to a Triton AST and **injects it into the live symbolic state**,
  caches it in a `SemanticsTable`, and continues. **Trace-guided concolic**:
  `run_window(window_kind="idx")` segments by execution order; branches are taken from
  the trace, not evaluated by Triton (no divergence). **Full symbolic seed**: all
  `symbolic_regs` symbolized, concrete shadow on the variable; `unseeded_regs` surfaces
  any gap. Conforms to `drive`'s `triton_runner` protocol. (§3)
- **Multi-vector parity gate** (`check_parity_vectors`). `phase_5_parity`'s `inputs_min`
  is a floor on **independent cross-run vectors** — a tautological `1/1` is now BLOCK,
  not EXACT. (§3)
- **Set-up symex, Tier-1** (`engine.setup_symex`). The opaque-symbol-recovery scaffold:
  four target-agnostic contracts (boundary-via-provenance, entry-completeness,
  symbol-preserving hybrid, mem[] backing) + the forward/backward dual-mode switch +
  `drive` (runs the 8-step plan, never bypasses the backing gate, surfaces the two
  judgments as checkpoints) + a hard parity gate on emit.
- **CVD — the Candidate-Verification Driver** (`engine.cvd`). The self-routing layer
  above the primitives: you PLACE (state + oracle + tools), the train ROUTES (enumerate
  → ROI-order → verify → always a defined next move); returns at a boundary
  (`SUCCESS`/`TERMINAL`/`EXTENSION_REQUEST`/`BUDGET_*`). Credibility from evidence,
  never a confidence number. (§2)
- **Oracle sink-validator** (`engine.oracle_sink.validate_sink`) and **oracle
  provenance** (`engine.oracle_provenance.trace_provenance`): prove the sink before
  slicing; backtrace the producer chain and classify how the output is produced. (§2)
- **Memory-routed dataflow**: S3 byte-granular `mem_deps`, S4 follows them + auto-detects
  the output buffer — a cipher byte reaching the outparam through memory is now traceable
  back to its value source.
- **Forward taint + scrub-capture**, **observation readers** (`engine.obs_readers`),
  and the **light-to-heavy phase API** with gated `phase_heavy_vmtrace`. (§1)

---

## 0. Mental model (one line)

> **You are the judge; utov is the deterministic batch + gate + ledger.** You
> supply target config (addresses / window / exec-identity / backing source) +
> reasoning + the one induced formula. utov returns structured verdicts behind
> gates you do not bypass. Run to a **terminal** (CLOSED, or a clean BLOCK with a
> precise reason), read the verdict off the return value — never eyeball it.

The main line is a **light→heavy forced order**. There is deliberately **no
"guess the algorithm" move** — the only crypto-source move is `phase_3`
provenance (follow the data flow).

---

## 1. Progressive main line (VMP / cipher targets)

Run the phase route in order; `phase_record` gates each next phase. Exact
params/returns: `AGENT-WORKFLOW.md` §2 Step 0. `utov phases` prints the route.

| Where you are | Call (RPC / CLI) | Read out → next |
|---|---|---|
| Just got the target; only know I/O shape | `phase_1_io_observe(entry_pc)` | I/O shape only — can't see VM internals. Light start, **not** a full vmtrace. |
| Know the output; need how it's written | `phase_2_materialization_trace(output_base, output_len)` | the output write sequence (the `strb`s) = prefix/formula source. Prove the sink first → §2 `validate_sink`. |
| Have the materialization point; who wrote these bytes? | `phase_3_watch_producer(addr, value_name)` → `phase_3_classify(records)` | producer chain + 5-way source verdict (`CONTINUOUS_BUFFER`/`STREAMING`/`OPAQUE_CALLEE`/`NEEDS_OBSERVATION`). `NEEDS_OBSERVATION` → re-capture per next-watch. **No algorithm guessing here.** |
| Provenance closed; induce the transform | `phase_4_formula_induction(expression, derived_from)` | **your judgment** — induce **one** formula, not a candidate spray. For opaque VMP handlers, drive set-up symex with the **Level-2 runner** (`build_level2_runner`) instead of hand-writing one — see §3. |
| Have a formula; prove it | `phase_5_parity(expression, inputs_min)` | full-chain bytewise parity = oracle. `inputs_min` is a floor on **independent cross-run vectors** (see §3). |
| Light phases hit a wall (recorded `could_not_close`) | `phase_heavy_vmtrace_prompt(budget)` → `phase_heavy_vmtrace(anchor, budget, proof|confirmation)` | **gated** escalation; needs an `EscalationProof` citing the wall + a `VmtraceBudget(runtime_s, disk_mb)`. Then feed the trace to Step 1. |

Already have a trace (re-analysis / deliberate heavy capture)? Skip to the batch:

- **Step 1 — one-shot batch (`$0`, deterministic):** `utov pipeline --mode frugal`
  or `preprocess_batch({passes?})` → `{batch_id, totals, next_step_hints}`.
- **Step 2 — one-look status:** `utov status <run> --json` (totals + per-axis +
  conformance) — the first thing to call after a batch.

---

## 2. Cross-cutting capabilities (reach for any time)

| Situation | Call | Note |
|---|---|---|
| Prove a sink before slicing on it | `validate_sink(items, expected, candidate_base=, snapshots=)` (`engine.oracle_sink`) | `SINK_CONFIRMED` / `WRONG_SINK` (redirects) / `OUTPUT_NOT_OBSERVABLE`. Wired as the s4 pre-gate. |
| Backtrace "who produced these bytes" | `trace_provenance(items, expected, sink_base=, snapshots=)` (`engine.oracle_provenance`) | classifies production; resolves an indirect `blr xN` target; precise next-watch list. |
| Grade a value/constant's source | `phase_3_classify` / `constant_provenance.classify_value` / `value_provenance.tag_value` | 5-way category + ceiling. utov grades evidence — don't hand-label. |
| Stuck — why, and is heavy worth it? | `stuck_statistics({max_points?})`; `get_findings({kind,stage,...})` | `by_mnemonic`/`by_verifiable_shape`/`by_pc_cluster`. Decide `--mode aggressive` from this, not a hunch. |
| Self-routing above the primitives | `run_cvd(items, expected, *, snapshots=, artifacts=, policy=, registry=)` (`engine.cvd`) | you PLACE (state + oracle + tools), the driver ROUTES; returns at a boundary (`SUCCESS`/`TERMINAL`/`EXTENSION_REQUEST`/`BUDGET_*`). Credibility from evidence, never a confidence number. |
| Record / expose / decide | `get_findings`/`get_hypotheses`/`utov status`; writes: `override_verdict`/`inject_finding`/`discard_batch`/`rerun_from_stage` | every write creates an `interventions` audit row. Don't hand-write report files. |
| Don't redo closed work | ledger `should_skip` / `is_closed`; `utov status` | exec-identity bucketed; a closed subject is closed. |

---

## 3. Set-up symex: parity is multi-vector, and don't hand-write the runner

`phase_5_parity`'s **`inputs_min` is a floor on the number of INDEPENDENT
cross-run vectors**, not a single match count. A `1/1` parity — verifying a
transform against the very trace it was derived from — is a **tautology** and is
**BLOCK, not EXACT**. A per-handler / per-window devirt (backward-alias mode)
must clear **≥N independent cross-run vectors**, each checked against its own
execution's observed output (determinism: no cross-run mixing).

Primitive: `engine.setup_symex.check_parity_vectors` → `ParityVectorReport
{independent_pass, verdict=EXACT|BLOCK, determinism_ok}`; `setup_symex.drive`
wires it as the authoritative per-window gate (the scalar `parity_min` match floor
still applies on top). Tune the floor with `UTOV_SETUP_SYMEX_PARITY_VECTORS`.

**Don't hand-write the symex runner — use Level-2.** `build_level2_runner(table=,
decoder=, gold=)` (`engine.setup_symex_runner`) is utov's own concolic runner: a
Triton bulk decoder + an **escape hatch**. An instruction it can't model is **not**
force-concretized — it's a precise BLOCK surfaced as `DriveResult.unmodeled`
(`UnmodeledInstruction`: "insn `<opcode>` (`<mnemonic>`) @ idx Y not modeled — supply
its symbolic semantics"). You hand-fill that one instruction as an S-expression
(`written_reg = (bvmul x0 x1)`, plus `(bv v size)`/extract/zx/sx/concat); utov
compiles it to a Triton AST, **injects it into the live symbolic state**, caches it in
a persistent `SemanticsTable`, and continues — and the fill still clears the parity gate.
Pass it to `drive` as the `triton_runner` (signature unchanged); supply a `gold`
callable for parity (the oracle is target-specific — the engine never fabricates it).
For a handler/window segment, pass `window_kind="idx"` with a trace-index range (so a
recurring pc / branch side-path can't pull in the wrong occurrence), seed the **whole**
input set in `entry["symbolic_regs"]` (seeding one register leaves the exit concrete),
and pass the trace's concrete inputs in `entry["concrete_regs"]`. Branches are taken
from the trace order, not evaluated by Triton; `unseeded_regs` surfaces any seed gap.

---

## 4. Boundary — what utov does / does not do / what you must supply

Three zones (authoritative: `utov-arch-index.md` §④ "边界"):

| Zone | Contents | Owner |
|---|---|---|
| General mechanisms (contract-enforced) | primitives (compute) · ledger (record) · projection + stamp (export) · **gates** (backing / multi-vector parity / determinism) · drive orchestration | **utov**, target-agnostic |
| Reasoning + target config | addresses / window / exec-identity / backing source · checkpoint decisions · `triton_runner` (L1) | **you supply** |
| Case-specific data | concrete addresses / numbers / run identity | config / ledger — **never** primitive code |

**Test:** "does it still hold for a different target?" yes → utov general
mechanism; only this case → config/ledger, not the engine.

**utov does NOT:**
- Put an LLM in the engine as a trusted stage. The engine stays
  deterministic / gated / reproducible; an LLM is only a **parity-gated candidate
  generator** (fires after the deterministic path is exhausted, trusted only if its
  output clears parity). All judgment is handed back to you (phase_4, checkpoints).
- Judge for you, or let you bypass a gate. A gate bug → fix the engine or
  BLOCK+report; never route around it with a side metric.
- Solve the case for you — utov is the automation + audit surface; **you are the
  decision maker**.

**You MUST supply:** target addresses / window / exec-identity, the backing source
(same-execution snapshot), the one induced formula, checkpoint judgments, and the
`triton_runner` (L1 — utov does not yet ship its own Triton).

### Runner / wire boundary — **this repo ships the CONTRACT, not the runner**

The live runner (unidbg / Java emulator harness, the line-protocol server) lives in a
**separate repo** (e.g. a `runner/...` tree in its own repo). When a capability needs runner-side
work — a new wire command (`base_reg+offset` watch, point-watch), a new observe-point
field (MEM capture, reg-relative addressing), a record-cap `truncated` signal — **this
repo only updates `contracts/runner_interface.md` + the engine's request/response/
verification side.** The runner-side parse/serialize/capture is implemented in the
runner repo, by whoever owns it.

- **Do NOT** edit runner code from this repo. Engine fixes must land in
  this repo's **source** (`engine/engine/**`); runner fixes must land in the runner repo.
- A symptom that "the contract declares X but X never arrives over the wire" is usually
  a **split fix**: engine side (serialize the request / consume the response / WARN on
  silent degrade) here, runner side (parse + honour it) there. Land both, or it stays
  dead end-to-end. Conformance should round-trip-assert each declared capability so a
  "declared but not wired" gap fails loudly instead of silently returning empty.

---

## Where to read more
- `AGENT-WORKFLOW.md` §1 interface categories · §2 runbook + phase method table (params/returns) · §6 driving loop.
- `README.md` (中文,主) / `README.en.md` (English) — driving the engine, CLI + RPC tables.
- `contracts/agent_protocol.md` — the wire spec.
