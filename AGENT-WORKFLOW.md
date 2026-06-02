# AGENT-WORKFLOW

> Driving clark-utov from an external agent: which interface to call,
> in what order, and what to read out of each result.
>
> Companion to [`contracts/agent_protocol.md`](contracts/agent_protocol.md)
> (the wire spec). This file is task-oriented; the protocol file is reference.

---

## 0. Mental model

**clark-utov is an automation + ledger tool an agent uses to make
decisions.** The agent is the judge; the engine is the batch and audit
surface.

```
       ┌────────────────────────────────────────┐
       │              agent                     │  ← decision maker
       │   - which algorithm is this?           │
       │   - is the verifier's verdict right?   │
       │   - is more LLM spend worth it?        │
       └──────────────┬─────────────────────────┘
                      │ uses
                      v
       ┌────────────────────────────────────────┐
       │  clark-utov                            │
       │                                         │
       │  batch:    plugin / handler-semantic    │
       │            (binop/unary/imm/ext/bfx)    │
       │            / Ch / Triton / σ-Σ fold /   │
       │            algorithm template fit       │
       │                                         │
       │  exposure: status / stuck_statistics /  │
       │            get_findings / get_hypotheses│
       │                                         │
       │  decisions:override / batch_override /  │
       │            inject_finding / discard_    │
       │            batch / rerun_from           │
       │                                         │
       │  audit:    findings.sqlite +            │
       │            interventions table          │
       └────────────────────────────────────────┘
```

A typical agent session is **one preprocess pass + a handful of read
queries + at most a few writes**. Most decisions never need an LLM.

---

## 1. Interface categories

> These are the **RPC + CLI** surface (batch / ledger / phase route). The
> recovery / opaque-VMP-handler line — CVD, `run_recovery`, set-up symex,
> the Level-2 runner, cohort localization, the oracle pre-gates — is a separate
> **import-level** surface (`import engine.X`); its params/returns are in **§6**.

### A. Batch preprocessing (deterministic, `$0`, one call)

| Interface | Surface | What it gives the agent |
|---|---|---|
| `utov pipeline --mode frugal` | CLI | Runs S1..S5 + the full layer-0/1/2 pass chain. Most agents should start here. |
| `preprocess_batch({passes?: [str]})` | JSON-RPC | One-call wrapper around the same chain, with selectable subset (e.g. skip `triton` if bindings missing). Returns `{batch_id, ran, results, totals, next_step_hints}`. Every finding promoted by this call is tagged with `batch_id`. |
| individual `verify_handler_*` / `verify_sigma_idioms` / `verify_algorithm_templates` | JSON-RPC | Run a single pass. Use when an agent wants to step manually instead of taking the batch all at once. |

`preprocess_batch` is the agent-friendly entry point — one round-trip,
one batch_id, plus a hints list summarizing what the trace looks like.

### B. Exposure (read-only, `$0`)

| Interface | Surface | What it gives the agent |
|---|---|---|
| `utov status <run> --json` / `--by-source` / `--by-stage` / `--by-kind` | CLI | One JSON object: totals + per-axis breakdowns + conformance + hypothesis status counts. **The first thing an agent should call after a batch.** |
| `get_findings({source?, stage?, kind?, subject_like?, limit?})` | JSON-RPC | Filtered findings table query (symmetric to `get_hypotheses`). |
| `utov findings <run> [--source --stage --kind --subject-like --limit --json]` | CLI | CLI mirror of `get_findings` — same filtered findings query from the shell. |
| `get_hypotheses({status?, kind?, source?, tag?, anchor_*?, limit?})` | JSON-RPC | Filtered hypotheses query. |
| `utov hyps <run> [--status --kind --source --limit --json]` | CLI | CLI mirror of `get_hypotheses` — filtered hypotheses query from the shell. |
| `stuck_statistics({max_points?, cluster_gap?})` | JSON-RPC | `{total, by_mnemonic, by_verifiable_shape, by_pc_cluster}` — turns the flat stuck-point list into something readable in one glance. Use to decide whether `--mode aggressive` is worth the spend. |
| `read_trace_window({idx_from, idx_to})` | JSON-RPC | Look at the raw trace around a PC. |
| `static_tool({tool, args})` | JSON-RPC | Whitelisted static-analysis CLI bridge (strings/nm/objdump/radare2/...). |

### C. Decision (writes; every write creates an `interventions` audit row)

| Interface | Surface | When to call |
|---|---|---|
| `override_verdict({hyp_id, new_verdict, reason})` | JSON-RPC | Flip one hyp's verdict (`pass`/`fail`/`inconclusive`). |
| `utov override <run> <hyp_id> <pass\|fail\|inconclusive> --reason "..." [--actor cli]` | CLI | CLI mirror of `override_verdict` — flip a verdict + log an intervention from the shell. |
| `batch_override_verdict({hyp_ids: [int], new_verdict, reason})` | JSON-RPC | Same, but for many at once. Returns `{total, ok, errors}`. |
| `discard_batch({batch_id, sources?, kinds?, reason?})` | JSON-RPC | Bulk-`override_verdict("fail")` the hypotheses whose finding rows carry the given `batch_id`, optionally filtered by `source` or `kind`. The append-only ledger is preserved; nothing is deleted. |
| `inject_finding({kind, subject, payload, reason, ...})` | JSON-RPC | Add a finding the verifier didn't produce (`source` tagged `agent_override` / `manual_inject`). |
| `force_status({hyp_id, new_status, reason})` | JSON-RPC | Lower-level status flip when override_verdict semantics don't fit. |
| `add_tag({hyp_id, axis, value, reason})` | JSON-RPC | Multi-axis tag on a hyp. |
| `rerun_from_stage({stage, reason})` | JSON-RPC | Cascade-invalidate from a stage; downstream hyps marked `abandoned`, downstream findings deleted. |
| `list_interventions({limit?, action?, actor?})` | JSON-RPC | Audit query. |
| `s6_propose_and_verify` / `s6_auto_loop` / `s6_find_stuck_points` | JSON-RPC | LLM hypothesis loop. Engine routes `llm_request` back to the agent (see `agent_protocol.md` §4). Default `--llm-backend none` means the engine produces no findings here unless the agent answers each `llm_request`. |

### E. Light-to-heavy phase route (VMP analysis, forced order)

The light-to-heavy phase path (§2 Step 0) is callable as RPC methods —
each phase refuses entry until its predecessor recorded a verdict via
`phase_record`; `phase_heavy_vmtrace` is gated (needs an `EscalationProof`
**or** a confirmation, plus a `VmtraceBudget` with `runtime_s`+`disk_mb`).
There is deliberately **no** "enumerate standard crypto" method — the only
crypto-source move is `phase_3` provenance. Full method table in §2 Step 0
and the driving loop in §6.

| Interface | Surface | When to call |
|---|---|---|
| `utov phases` | CLI | Print the recommended light-to-heavy phase route (`phase_1_io_observe` → `phase_5_parity`, `phase_heavy_vmtrace` as a gated escalation). Static discoverability; no args. |
| `phase_state` / `phase_1_io_observe` / `phase_2_materialization_trace` / `phase_3_watch_producer` / `phase_3_classify` / `phase_4_formula_induction` / `phase_5_parity` / `phase_record` / `phase_heavy_vmtrace_prompt` / `phase_heavy_vmtrace` | JSON-RPC | The VMP phase API as callable tools (params + returns in §2 Step 0). Run in order; `phase_record` gates the next phase; vmtrace is escalation-gated. |

### D. Cross-run / ops

| Interface | Surface | When to call |
|---|---|---|
| `utov compare <run-a> <run-b> [--json]` | CLI | Diff two runs' findings — by source / kind / stage delta + new/removed subjects. Use for patch-before-after experiments. |
| `utov audit <run>` | CLI | Print recent intervention log. |
| `utov rerun-from <run> <stage>` | CLI | Cascade-invalidate from `<stage>`; downstream stages re-run on `utov resume`. |
| `utov resume <run>` | CLI | Continue a paused / partial run. |
| `utov doctor [--sample-dir DIR]` | CLI | Host environment check (Triton bindings, Python deps, optional sample fixtures). |
| `checkpoint` / `pause` / `resume` / `is_safe_to_interrupt` / `shutdown` | JSON-RPC | Lifecycle. |

---

## 2. First-contact runbook

Wrap your agent loop around these five steps. Each step has a concrete
yield from the previous result that decides the next branch.

### Step 0 — Light-to-heavy first (VMP targets)

Nothing here *blocks* you from capturing a full vmtrace and jumping
straight to Step 1 — you can't be physically stopped, and a heavy trace
is sometimes the right call. But on a VMP target where you don't yet know
the algorithm, the cheap path closes most cases without paying for a full
trace, and **a full trace should be the escalation you reach for once the
cheap path can't close — not the default first move** (the reference case
"half-hour full vmtrace" detour came from skipping this; roadmap §8.12).

The path is encoded in `engine.vmp_phase_api` (`VmpPhaseApi`) so you don't
re-derive it per target. Run it in order; each phase is deterministic
except the formula step:

| Phase | Cheap move (deterministic) | Primitive |
|---|---|---|
| `phase_1_io_observe` | calltrace to the crypto entry + hook I/O — capture the I/O **shape** only, not VMP internals | `phase_instrument` (func entry, coarse) |
| `phase_2_materialization_trace` | hook the output write sequence (the `strb`s) — this is where the prefix/formula structure shows up | `phase_instrument` (output region first-write) |
| `phase_3_provenance` | follow the data flow: watch first write + 5-way constant-provenance classify → producer chain. **There is no "guess the algorithm" move here — only "trace the flow".** | `watch_first_write` + `constant_provenance` |
| `phase_4_formula_induction` | *your* judgement: induce **one** formula from what phases 2–3 showed (not a candidate spray) | — |
| `phase_5_parity` | full-chain bytewise parity against the oracle | conformance |
| `phase_heavy_vmtrace` | full instruction-level trace — **the escalation**. Capture this, then feed it to Step 1 (`utov pipeline`). | `phase_instrument` (`GRAN_FULL`) |

Escalating to `phase_heavy_vmtrace` asks for an `EscalationProof` that
cites which light phase hit a wall and why (e.g. "phase_3: producer PC
outside observable range"). That's not a hoop — it's the one line that
tells the next agent *why* the cheap path didn't close, so they don't
re-walk it. If you genuinely need the full trace, name the wall and go.

**Driving it.** These are real agent-serve RPC methods — call them, don't
re-invent the path in your own scripts:

| Method | Params | Returns |
|---|---|---|
| `phase_state` | — | sequence + trail + whether closed |
| `phase_1_io_observe` | `entry_pc` | instrument spec |
| `phase_2_materialization_trace` | `output_base`, `output_len` | instrument spec |
| `phase_3_watch_producer` | `addr`, `value_name` | watch-first-write spec |
| `phase_3_classify` | value record(s) w/ `producer_dataflow` / `rerun_observations` | 5-way provenance verdicts |
| `phase_4_formula_induction` | `expression`, `derived_from` | parity intent |
| `phase_5_parity` | `expression`, `inputs_min` | parity intent |
| `phase_record` | `phase`, `status` (`ran`/`closed`/`could_not_close`), `could_not_close_reason?` | records the verdict (required before the next phase can enter) |
| `phase_heavy_vmtrace_prompt` | `budget?` | the confirmation question to show a human |
| `phase_heavy_vmtrace` | `anchor`, `budget` (`runtime_s`+`disk_mb`), `proof` **or** `confirmation` | full-trace instrument spec |

There is no method to enumerate / guess standard ciphers — if SM4/AES/HMAC/CTR
don't match, that is a `phase_3` finding (`could_not_close`: non-standard F),
not a cue to spray more candidates. The next move is `phase_heavy_vmtrace`
(trace F's internals) or deeper provenance, never a wider guess. `utov phases`
prints this route.

> Already have a trace in hand (re-analysis, a target you've traced
> before, or a deliberate heavy capture)? Skip to Step 1 — it consumes
> whatever trace you give it.

### Step 1 — One-shot batch (`$0`, deterministic)

```bash
utov pipeline \
    --runner-cmd '<your runner cmd>' \
    --input <hex> \
    --mode frugal \
    --new-run \
    --work-root ./work
```

Runs S1..S5 + plugin/handler/Ch/Triton layer-0 + σ-Σ fold + algorithm-fit.
Produces 10+ `s5-verify-*` and `s5-fold-sigma` / `s5-algorithm-fit` stage
rows. Every promoted finding lands in `findings.sqlite` tagged with
`source` (and, when called via `preprocess_batch`, with `batch_id`).

Or, when you want selectable passes / programmatic control, do the same
through agent-mode:

```python
# JSON-RPC over stdio (utov agent-serve <args>)
{"id": 1, "method": "preprocess_batch",
 "params": {"passes": ["plugin", "binop", "unary", "imm",
                       "ext", "bfx", "ch", "sigma", "algorithm"]}}
# omit "triton" if host doesn't have Triton bindings;
# omit "sigma"/"algorithm" if you want to compose them yourself.
```

Result:

```json
{
  "batch_id": "2c19d0df7827",
  "ran":      ["plugin", "binop", "..."],
  "results":  {"plugin": {...}, "binop": {...}, ...},
  "totals": {
    "promoted":           1278,
    "by_source":          {"plugin": 10, "s5_deterministic": 891, ...},
    "by_kind":            {"handler_semantic": 1268, "fold_idiom": 8, "algorithm_identified": 1, ...},
    "matched_algorithms": ["SHA-512"]
  },
  "next_step_hints": [
    "algorithm_identified: SHA-512. Confidence + anchors_seen are in the finding's payload; follow up with IO-equivalence …",
    "triton model_mismatch high …"
  ]
}
```

### Step 2 — One-look status (`$0`)

```bash
utov status ./work/.../<run>/ --json
```

Agent reads the JSON and routes on three lines:

| Field | Decision |
|---|---|
| `findings_by_kind.algorithm_identified ≥ 1` | algorithm tentatively known — go to step 3a |
| `findings_by_kind.fold_idiom ≥ 2` (but no algorithm_identified) | round-function family looks plausible but anchors below the algorithm-fit threshold — go to step 3b |
| `findings_by_kind.algo_signature ≥ 1` (but no idiom/fit) | plugin fingerprint hit only — go to step 3c |
| all of the above 0 | unknown trace — go to step 3d |

### Step 3 — Branch on what step 2 told you

**3a. Algorithm identified**

```python
{"method": "get_findings", "params": {"kind": "algorithm_identified"}}
```

Inspect the finding's payload: `algorithm`, `anchors_seen`,
`evidence_score`, `io_test`. If `evidence_score ≥ 0.75`, treat the label
as solid; otherwise re-read the missing anchors and decide whether more
trace coverage is needed. **The agent stops here for most jobs** —
algorithm identification is the deliverable.

**3b. Fold idioms only**

```python
{"method": "get_findings", "params": {"kind": "fold_idiom"}}
```

Each fold finding's payload lists `components` (PCs + amounts). Map the
PCs back via `read_trace_window` to see which loop iterations they
cover. If only round-function PCs are present and message-schedule σ/σ
PCs aren't, your trace doesn't cover the whole algorithm.

**3c. Plugin fingerprints only**

The trace hit cryptographic constants but no round-function code. Two
explanations: (1) trace covers init / IV setup only, not the loop; (2)
the algorithm uses the constants but our σ-Σ / Ch idiom matchers miss
its compiler's particular emission. Inspect `stuck_statistics.by_pc_cluster`
to see whether the trace clusters where you expected.

**3d. Unknown trace**

```python
{"method": "stuck_statistics"}
```

Read `by_verifiable_shape`:

- non-zero `memory_load` / `memory_store` / `fp_neon` → verifier model
  can't reach these; LLM won't help either (it can't compute memory).
- non-zero `other` → the only bucket worth feeding to an LLM. If
  `other > a few hundred`, step 5 (aggressive) becomes worth its spend.

### Step 4 — Selectively discard untrustworthy passes (optional)

Each pass is independently auditable. If you want to keep, say,
layer-0 + σ-Σ but reject Triton's net-coverage finding because its
trace context looked iffy:

```python
{"method": "discard_batch",
 "params": {"batch_id": "2c19d0df7827",
            "sources":  ["s5_triton"],
            "reason":   "agent rejects Triton net-coverage on this run"}}
```

The hypotheses behind those findings get `override_verdict("fail")`;
the finding rows stay (append-only ledger) but their underlying hyps
are now `failed`. Every individual flip writes an `interventions` row,
so the discard is fully auditable.

### Step 5 — Aggressive LLM (optional, has spend)

Only reach for this when step 3d says `other ≥ ~hundreds` and you have
budget. Use `--resume` so the deterministic chain doesn't re-run.

```bash
utov pipeline --runner-cmd '...' --input ... \
              --mode aggressive \
              --llm-backend deepseek \
              --budget-usd 0.5 \
              --s6-concurrency 4 \
              --resume
```

After completion, LLM-derived findings appear with `source = s6_llm` so
you can break them out from the deterministic majority via
`get_findings({source: "s6_llm"})` or `utov status --by-source`.

---

## 2a. Common closure path + per-step pitfalls (recovery / opaque-VMP line)

Step 0–5 above is **discovery-ordered** (cheap forensics first). Once you have a sink to
recover, the **closure path** is the canonical order below. Each step names the helper to
reach for AND the pitfall that wastes the most time when skipped or mis-wired. This is a
default route, not a red line — but don't free-climb past a step.

| # | Step | Reach for | Pitfall (the time-sink) |
|---|---|---|---|
| 1 | **Confirm the sink** | `oracle_sink.validate_sink` (§6.1) | Wrong sink ⇒ everything downstream goes nowhere. Confirm BEFORE slicing. Don't treat the top-level program output as a window-local sink. |
| 2 | **Provenance to the producer** | `trace_provenance` → `phase_3_classify` | `OPAQUE_CALLEE`/boundary ⇒ you need a libc boundary edge (step 4) or observation (step 3), not a guess. `NEEDS_OBSERVATION` is not "stuck" — it's step 3. |
| 3 | **Same-execution observation** | `run_recapture_loop`; (planned: `suggest_observations`/`run_plan`, spec #4) | G1: output + every snapshot must be ONE rerun — never accumulate snapshots across reruns. Un-observed gap ⇒ recapture directive, not a dispatch fallback. |
| 4 | **Boundary-explicit candidate** | `libc_boundary.synthesize_boundary_edge`; `final_materialization`; (planned: `extern_model.resolve_extern_model`, spec #1) | Don't keep widening the recover window hoping it closes. An extern PRNG/time/memcpy is a *boundary*, not an opaque blob — model it / synthesize the edge. |
| 5 | **Enough distinct vectors** | `real_gold.collect_real_gold` (drives reruns to a distinct floor) | A tautological `1/1` or `observed distinct < min` is `UNCLOSABLE` — that means **fix the cohort** (output-diverse seeds), not F. Run `check_seed_independence` BEFORE symex (don't symbolize a constant). |
| 6 | **Parity** | `check_parity_vectors` (bytewise/multi-vector) + `check_emit_self_consistency` (G4, pre-emit) | Self-check BLOCK = recovered F disagrees with its own trace seed ⇒ F is wrong, fix it before parity. Don't assume input-only closure is required — boundary/observed state is legitimate. |

**Common mis-wirings (caught by the planned parity-input lint, spec #3):** `seed_values`
shape ≠ declared inputs; `sink` not sourced from this window's exit; `observed` = top-level
output instead of window-local sink; `sink_mask` width ≠ register width.

---

## 3. Cheat sheet

```
┌─────────────────────────────────────────────────────────────────┐
│  Cold start                                                      │
│                                                                  │
│  VMP target, algorithm unknown? Light-to-heavy first (Step 0):   │
│    phase_1 io_observe → phase_2 materialization → phase_3        │
│    provenance → phase_4 induce → phase_5 parity.                 │
│    Full trace = phase_heavy, reach for it once light can't close.│
│                                                                  │
│  Have a trace already (or escalated)? Run the pipeline:          │
│  utov pipeline --runner-cmd '...' --input ... --mode frugal      │
│  └──> 1278 findings, batch_id, algorithm_identified, hints       │
│                                                                  │
│  utov status <run> --json                                        │
│  └──> findings_total, by_source, by_kind                         │
└────┬─────────────────────────────────────────────────────────────┘
     │
     ├─── algorithm_identified ──> get_findings(kind='algorithm_identified')
     │                              read payload.confidence + anchors_seen
     │                              if conf ≥ 0.75 → done
     │
     ├─── fold_idiom only ───────> get_findings(kind='fold_idiom')
     │                              re-check trace coverage vs algorithm entry
     │
     ├─── plugin only ───────────> stuck_statistics → by_pc_cluster
     │                              verify trace bounds
     │
     └─── nothing ───────────────> stuck_statistics
                                    look at by_verifiable_shape.other
                                    if other > ~hundreds:
                                       --mode aggressive --llm-backend ...
                                       --resume
                                    else:
                                       unsupported trace; stop or rerun_from
```

---

## 3a. Framework gates (anti-drift primitives)

Every JSON-RPC response carries a `methodology` sibling next to
`result`. Inside that sibling the wrapper attaches **structured
gate outputs** so the agent doesn't need to re-derive these checks
by hand. Treat each one as a hard signal — the wrapper has already
either downgraded the claim in-place or intercepted the call.

| Envelope field | Gate | What it means |
|---|---|---|
| `m1_audit` | M1 success audit | `evidence_class` ∈ {A,B,C} + `action` ∈ {allow, downgrade, reject}. On `downgrade`, the wrapper rewrote `target_success`/`archival_allowed` to false in your params before dispatch — the archival landed as `strong_partial`, not full. On `reject`, dispatch never ran. |
| `m3_bypass` | M3 bypass-block detector | `suspected_bypass=true` ⇒ this candidate block failed M3 variability under ≥N distinct observation methods. Stop swapping observation methods on this block; reroute the hypothesis upstream or to a parallel path. Follow-up observation attempts on the same block will be refused before dispatch. |
| `value_provenance` | Value-source state machine | Each tagged value carries `provenance ∈ {observed, closed_form, hybrid, unknown}` + `final_class`. A value from `hook`/`dump`/`io`/`snapshot` is capped at evidence_class B until a closed-form recompute is verified. `observation_parity=True` alone never lifts the ceiling. |
| `watch_suggestions` | Watch-first-write | Observed value at a concrete `landing_address` ⇒ wrapper attaches a `WatchFirstWriteSpec`. Runner side should install a memory watchpoint at the address, resume, and capture the producing PC + source bytes. Disable via `UTOV_WATCH_FIRST_WRITE_AUTO_TRIGGER=off` for advisory-only mode. |
| `length_chain` | Length-chain consistency | Per-edge breakdown of any `length_chain: [...]` you sent. `ok=false` + `unexplained_edges[]` ⇒ adjacent lengths have no explainable relation (equal / integer-multiple / hex 2:1 / strict base64 4:3 / explicit ratio/delta). Likely wrong intermediate representation chosen. |

Each gate has an independent env kill-switch — `UTOV_M1_AUDIT`,
`UTOV_M3_BYPASS`, `UTOV_VALUE_PROVENANCE`, `UTOV_WATCH_FIRST_WRITE`,
`UTOV_LENGTH_CHAIN` — all default on. The wider
`UTOV_METHODOLOGY=off` disables every gate at once.

To use them, structure the relevant params surface so the wrapper
can find the data:

```json
{
  "report": {
    "target_success": true,
    "success_dependencies": ["prefix", "body_len", "key"],
    "samples":              [...],
    "pass_rate":            1.0,
    "scope":                "cross_session",
    "closure_paths":        [{"name": "cfbc", "digest": "..."}, ...],
    "values": [
      {"value_name": "vmp_key", "source": "hook",
       "landing_address": 4275245056, "evidence_class": "A"}
    ],
    "length_chain": [
      {"name": "input",      "length": 32},
      {"name": "decoded",    "length": 21},
      {"name": "compressed", "length": 16}
    ]
  }
}
```

For M3 variability checks, set `block_id` + `observation_method` on
the params so cross-call evidence accumulates against the right block.

---

## 4. Conventions that bite

- **Append-only ledger.** `findings.sqlite` rows aren't deleted by the
  decision surface. `discard_batch` flips the underlying hypothesis to
  `failed` + writes an audit row, but the finding row remains. Consumers
  who want "live" findings should JOIN through `origin_hyp_id`'s status,
  or filter on `hypotheses.status = 'passed'`.
- **`source` is the agent's main filter.** Promoter paths populate
  `findings.source` with: `plugin`, `s5_deterministic`, `s5_triton`,
  `s5_fold_idiom`, `s5_algorithm_fit`, `s6_llm`, `agent_override`,
  `manual_inject`. Use `--by-source` or `get_findings(source=...)`
  rather than guessing from `subject`.
- **`batch_id` is best-effort.** Set only when a finding lands inside a
  `preprocess_batch` call. Findings created outside (e.g. by manually
  running a single pass, or by `inject_finding`) have `batch_id = NULL`.
  `discard_batch` only acts on tagged rows.
- **`--llm-backend` default is `none`**. `utov pipeline --mode aggressive`
  without an explicit `--llm-backend` prints a deprecation warning and
  produces no LLM findings. Pass `--llm-backend deepseek` or `mimo` (or
  set `LLM_BACKEND` env) to engage a backend.
- **File-mode degrades cleanly.** Most verify passes don't need the
  runner once the trace is loaded; the algorithm-fit step's `io_test`
  field documents whether the IO-equivalence test ran or was skipped.
- **Triton net-coverage** skips PCs the deterministic passes already
  promoted (no double-counting). `triton.model_mismatch` counts PCs
  Triton couldn't model in fresh-context mode (status flags / memory
  state); those are silently skipped, not failed.

---

## 6. Python-API surface: CVD / recovery / set-up symex / oracle

§1–§5 cover the **RPC + CLI** surface (the batch/ledger world). The
recovery / opaque-VMP-handler line is a separate **import-level** surface —
you `import engine.X` and call it from your driver, not over the wire. This
is the section the `agent-usage-guide.md` "Recent releases" bullets and its §2/§3
point at for exact params/returns. Run a target to a **terminal** and read the
verdict off the return value; never eyeball it.

### 6.1 Oracle pre-gates (prove the sink before you slice)

| Call (`engine.oracle_*`) | Signature (key args) | Returns / read-out |
|---|---|---|
| `oracle_sink.validate_sink` | `(items, expected_output, *, candidate_idxs=None, candidate_base=None, snapshots=None)` | `SinkValidation`: `SINK_CONFIRMED` / `WRONG_SINK` (redirects) / `OUTPUT_NOT_OBSERVABLE`. A same-base snapshot that matches 32/32 **is** `SINK_CONFIRMED` — do not slice until confirmed. |
| `oracle_provenance.trace_provenance` | `(items, expected_output, *, sink_base, snapshots=None, max_steps=100000, max_breadth=None, assess_observability=False, boundary_edge=None)` | `ProvenanceResult`: producer chain + source verdict (`CONTINUOUS_BUFFER`/`STREAMING`/`OPAQUE_CALLEE`/`NEEDS_OBSERVATION`/`BOUNDARY_EDGE`), indirect `blr` target resolution, next-watch list. Pass `boundary_edge=` when the producer writes outside the bundled trace (e.g. a libc `memcpy`). |

### 6.2 CVD — the self-routing driver (`engine.cvd`)

| Call | Signature (key args) | Returns |
|---|---|---|
| `run_cvd` | `(items, expected, *, snapshots=None, window=None, budget=None, registry=None, has_runner=True, submissions=None, policy=None, artifacts=None, obs_scope=0, collect_extensions=False)` | `CvdResult`. You PLACE (state + oracle + tools), the driver ROUTES; returns at a boundary (`SUCCESS`/`TERMINAL`/`EXTENSION_REQUEST`/`BUDGET_*`). `collect_extensions=True` lists the WHOLE gap map in one run instead of stopping at the first gap. |
| `run_cvd_collect_to_json` | `(items, expected, *, work_root, ts, exec_identity=None, filename="cvd_gap_map.json", cohort_load=None, **run_kwargs)` | `(CvdResult, Path)` — runs the collect CVD and ACTIVELY writes the stamped `cvd_gap_map.json` (`<!-- utov-export -->` header). Read THIS; do **not** `open_ledger` / touch the sqlite. |
| `export_gap_map` | `(result, path=None, *, ts, exec_identity=None, source=…, from_entries=())` | stamped-JSON text — project an in-memory `CvdResult` to the OUT-layer artifact. |

Read a stamped export back with `engine.export_stamp.load_stamped_json(text)
-> (header_or_None, payload)`; `header=None` ⇒ a hand-written file, not a utov export.

### 6.3 Recovery — one CVD run (`engine.cvd_recovery`)

`run_recovery(...)` wires a whole recovery CVD run for you — **prefer it over
hand-assembling** `recovery_registry` + `run_cvd_collect_to_json` (the
hand-wiring is what got 3 configs wrong and drowned the gap map in noise).

| Call | Key args | Returns |
|---|---|---|
| `run_recovery` | `(items, *, base_config, triton_runner, expected, work_root, ts, coverage=None, dependence=None, decisions=None, cohort_traces=(), input_keys=None, pointer_chain=None, budget=None, exec_identity=None, snapshots=None, recapture_adapter=None, output_observe_pc=None, …)` | `(CvdResult, Path)` — pure recovery registry (no `default_registry` sink noise), safe decision defaults with the GENUINE `mem_input_symbolize_vs_back` judgment still escalated to `pending_judgments`, anchor reconcile, stamped `cvd_gap_map.json`, no sqlite. |
| `recovery_registry` | `(*, base_config, triton_runner, coverage=None, dependence=None, ledger=None, decisions=None, window_kind="idx", cohort_traces=(), pointer_chain=None, budget=None, …)` | a CVD `Registry` (the three recovery roles only) — for when you must drive `run_cvd` yourself. |

**mem-write output (Issue 7).** When the window's output is a STORE (bytes into a
buffer), not a register, pass `mem_sink={"sink_form":"mem","sink_idx":…,"sink_addr":…,
"sink_size":…}` to `run_recovery`/`recovery_registry` (addr/size derived from the trace
mem op / S3 `mem_deps` when omitted — `derive_mem_sink_interval`). The runner then reads
the symbolic bytes `[sink_addr, sink_addr+sink_size)` instead of `x8`; self-check + parity
compare bytewise. Can't pin the EA / can't read the bytes ⇒ the structured terminal
`TERMINAL_MEM_SINK_UNPLACEABLE` (carries `MEM_SINK_UNPLACEABLE_NEEDED`), never a silent
register/constant fallback; an input-invariant store routes to the existing
seed-independence / `UNCLOSABLE` exclusion. `sink_form="reg"` (default/omitted) is
byte-for-byte the old x8 path.

### 6.4 Set-up symex (`engine.setup_symex`) + Level-2 runner (`engine.setup_symex_runner`)

Don't hand-write the symbolic runner; don't hand-configure per-window seeds.

| Call | Key args | Returns / when |
|---|---|---|
| `drive` | `(*, trace, case_config, triton_runner, ledger=None, decisions=None, on_checkpoint=None, cfg=None, ts=None, pointer_chain=None, cohort_traces=None, cohort_keys=None, mem_sink=None)` | `DriveResult \| DrivePause`. Runs the 8-step plan, never bypasses the backing gate, surfaces the two judgments as checkpoints (`mem_input_symbolize_vs_back`). Pauses (not silent 0) on an un-pinned external mem input. `mem_sink` (Issue 7) routes the output to a STORE's symbolic bytes instead of `x8`; unreadable ⇒ `DriveResult.mem_sink_unreadable`. |
| `build_level2_runner` | `(*, table=None, decoder=None, gold=None)` | a `triton_runner` callable for `drive`. Triton bulk decoder + escape hatch: an unmodeled opcode is a precise `DriveResult.unmodeled` BLOCK (you hand-fill one S-expr), **never** force-concretized. Supply `gold` for parity (oracle is target-specific). |
| `check_seed_independence` | `(seed_values, *, min_vectors=2)` | `SeedIndependenceReport` — PRE-symex gate. BLOCK when EVERY seed is constant across the cohort (don't symbolize a constant); `INSUFFICIENT` below `min_vectors`. |
| `check_parity_vectors` | `(vectors, *, window, min_vectors, trace_exec_id=None)` | `ParityVectorReport {independent_pass, verdict=EXACT\|UNCLOSABLE\|BLOCK, determinism_ok}`. A tautological `1/1` is BLOCK, not EXACT. Floor = independent cross-run vectors (`UTOV_SETUP_SYMEX_PARITY_VECTORS`). |
| `check_emit_self_consistency` | `(*, expr_source, inputs, seed_values, trace_sink, sink_mask=None, sink_form="reg")` | `EmitSelfCheckReport` — G4: the recovered F evaluated on its own trace's seed must equal the trace's window-exit sink BEFORE emit; mismatch BLOCKs. |
| `derive_window_symbolic_regs` | `(items, *, window, reg_file=None, window_is_idx=False)` | `(regs, info)` — a window's live-in = the seed + threaded state carrier. `drive` calls this when `symbolic_regs=None`. |
| `derive_window_mem_live_in` | `(items, *, window, window_is_idx=False)` | `(MemLiveIn…, info)` — external memory inputs (loaded bytes with no in-window writer). |

### 6.5 Localization (`engine.cohort_diff`, `engine.dispatch_coverage`)

| Call | Key args | Returns |
|---|---|---|
| `cohort_diff.localize_input_dependence` | `(cohort_traces, *, input_keys=None, ignore_regs=(), ignore_addrs=(), min_observability=0.05)` | `InputDependenceMap` — which windows vary with the seed (recover) vs stay constant (carry). Diffs register AND memory writes; a genuinely-varying cohort with no observable change returns `opaque` (needs symex), never a silent "no dependence". |
| `dispatch_coverage.preflight_dispatch_coverage` | `(trace, *, invocations, reg_file=None, decode_probe=None)` | `CoverageMap` — per handler-type I/O signature (`reg_live_in`/`mem_live_in`/`unmodeled_opcodes`/`outputs`) computed ONCE; N invocations covered by ~6 solves. `to_stamped_markdown(...)` projects it. |

### 6.6 Final-sink-first toolkit (import map · libc boundary · final-materialization · real-gold)

When the runner-visible output is a **small final construction** (a fixed header
‖ a copy from a live buffer), the shortest explanation is that construction + its
source — not the upstream composite chain that may also be in the trace. This
toolkit hands you the manual "final sink → boundary call → external-call map →
same-run watch → parity-ready vectors" queue so you don't get pulled into a
true-but-irrelevant internal chain. Order: detect final materialization → map the
boundary call → synthesize its edge → backtrace provenance → collect real gold.

| Call | Signature (key args) | Returns / read-out |
|---|---|---|
| `final_materialization.detect_final_materialization` | `(items, *, sink_base, output, snapshots=None, annotated_calls=None, header_len_hint=None)` | `FinalMaterialization {verdict ∈ FINAL_COPY/FINAL_HEADER_COPY/FINAL_BYTEMAP/NO_FINAL_MATERIALIZATION, header_bytes, source_region, copy_call, next_move}`. **ROUTES, does not CLOSE** — a detected copy is never promoted to `ClosureLevel.ORACLE`; the recovered F still clears `closure_classification`. `next_move`: `recover_source_provenance` / `watch_source_buffer` (+`SOURCE_UNOBSERVABLE`) / `fall_through_composite`. |
| `import_map.build_import_map` | `(binary_path=None, *, static_artifacts=None, plt_map=None, got_map=None, timeout=30.0)` | `ImportMap` — resolves PLT/import stubs to symbol names via `static_tools` (objdump/readelf), or from an explicit `plt_map`/`got_map`. No binary ⇒ `binary_available=False` (never fabricates a name). |
| `import_map.annotate_calls` | `(trace, import_map)` | per-call `{symbol, resolved_from, external_state, state_kind}` (reuses the indirect-`blr` resolver). Unknown ⇒ `unknown@<addr>`; known symbol with no summary ⇒ `external_unknown` (not assumed pure). |
| `import_map.extern_summary` | `(symbol)` | `ExternSummary {name, abi_args, introduces_external_state, state_kind, effect} \| None`. Table keyed by symbol name (time/srand/rand/random/memcpy/memmove/memset/strlen/strcpy); TC2's concrete addresses live in the map/config, never the table. A flow through an `introduces_external_state=True` call must be surfaced (couples to `check_seed_independence`), not reported as a pure-input transform. |
| `libc_boundary.synthesize_boundary_edge` | `(trace, call_site, sink_region, import_map, *, summary=None, boundary_edge=None)` | `BoundaryEdge \| BoundaryEdgeUnresolved`. For `memcpy`/`memmove` (sink⊆dst ⇒ COPY edge to `src+offset`) and `memset` (CONST edge), built from the ABI (`extern_summary.abi_args`). Explicit `boundary_edge=` honored verbatim. Unknown call / `n` symbolic / `dst` unresolved ⇒ `BOUNDARY_EDGE_UNRESOLVED {symbol, missing}` — **never a silently-wrong edge**. Feed the result straight into `oracle_provenance.trace_provenance(..., boundary_edge=)`. |
| `real_gold.collect_real_gold` | `(runner_adapter, observe_points, seeds, *, loop_input, predict, window, distinct_output_floor=None, max_reruns=200, exec_identity=None, config=None, env=None)` | `RealGoldReport {vectors, observed_distinct, reruns_spent, floor_met, verdict_hint}`. Drives runner reruns, stamps exec-id + captured seed, emits `ParityVector`s, and **collects until N DISTINCT observed outputs** before judging via `check_parity_vectors`. Below the floor after the budget ⇒ `INSUFFICIENT_VARIANCE` (not a false `UNCLOSABLE`/`EXACT`). Floor defaults from `UTOV_SETUP_SYMEX_PARITY_VECTORS`. `seeds` are `SeedSpec` (reg / `mem@addr` / external-state). |

### 6.7 Extern-model registry · planner · advisory metadata (this wave)

Eight additive, **general-first** capabilities. Each is opt-in: unused ⇒ the
existing path is byte-for-byte unchanged. Adding a model / rule / invariant /
relation is one registry entry, never a verifier-source edit.

| Call | Signature (key args) | Returns / read-out |
|---|---|---|
| `extern_model.resolve_extern_model` | `(symbol, *, runtime_tags={})` | `ModelSpec \| ModelUnavailable`. One door for both kinds: PRNG/time (`rand`/`srand`/`time`) → **stateful** model with `eval_sequence(seed, count, *, project="raw\|low8\|...")`; `memcpy`/`memset`/`memmove` → **mem_effect** model wrapping `libc_boundary.synthesize_boundary_edge` as a provider. Adding a model = one `MODEL_REGISTRY` entry. Unknown / ambiguous-no-tag ⇒ `EXTERN_MODEL_UNAVAILABLE {reason, family_hints}` — never a silent pick. `evidence_level` (`reference\|observed_match\|conjectured`) is first-class. |
| `extern_model.rank_model_candidates` | `(symbol, *, observed_seed, observed_returns, runtime_tags={})` | `CandidateRanking {ranked[], verdict ∈ RANKED/NO_CANDIDATE/INSUFFICIENT_OBSERVATION, why_top}`. Family-**agnostic** scorer (no PRNG logic): runs each registered family's `eval_sequence` over `observed_returns`, scores prefix match, reports per-candidate `mismatch` + the explicit reason #1 won. Advisory only — never auto-promotes a family to a closed finding. `register_family_hint` adds a family. |
| `setup_symex.lint_parity_inputs` | `(case_config, *, window, declared_inputs, observed_spec, sink_spec)` | `LintReport {findings[{level, code, detail, fix}], max_level}`. Extensible **invariant registry** (seed/sink/observed/mask wiring checks); non-blocking by default, `ERROR` rides above the run in `per_step`. Also auto-run inside `drive()` after `locate_boundary`. Adding a check = one invariant entry. |
| `observation_planner.suggest_observations` / `.run_plan` | `suggest_observations(prov, items, *, rules=DEFAULT_RULES)`; `run_plan(adapter, input_bytes, plan)` | `list[ObservePoint]` / `RerunResult`. Heuristic **rule registry** (`write_chain` / `extern_call` / `boundary_copy`) proposes the next observe points from code shape, with reason + heuristic tag. `ProvenanceResult.observation_plan` is attached ALONGSIDE `next_watch` only when `trace_provenance(..., plan_observations=True)`. Unmatched gap stays in `next_watch` — plan never silently hides it. (rules live in `engine._rules`.) |
| `execution_bundle.capture_bundle` | `(adapter, input_bytes, observation_spec, *, derived=DERIVED_NONE, exec_identity=None)` | `CaptureBundle {exec_identity, input, output, observations{name→...}, derived, same_execution}`. Named observation spec + a **declarative** `{field: extractor(bundle)}` derived map → one stamped same-execution evidence bundle. `to_dict()` / `write_json()` (reuses `export_stamp` header). A failing extractor ⇒ `null` + `derived_errors[name]`; cross-exec ⇒ `same_execution=false` — never a forged field. |
| `advisor.evidence_state` / `.advise` | `evidence_state(*, phase_run=None, ledger=None, claims=None, progress=None)`; `advise(goal_spec, evidence)` | `EvidenceState` (read-only aggregation of `closure_classification` + `cvd_ledger` + `authority_projection` + `progress`, with a `sources` provenance map) / `list[Advisory]`. Rule registry; the seed over-investment rule fires a NON-blocking rebalance `SUGGEST`. Cannot block/re-route a run; an unreadable signal ⇒ `null` + `sources` gap, never fabricated. |
| `next_actions.suggest_next_actions` | `(report_kind, verdict, reasons)` | `tuple[dict{helper, why, example}]`. Declarative `(kind, verdict, reason-predicate) → helper` registry. Seed: parity `UNCLOSABLE`(distinct<floor) → `real_gold.collect_real_gold`. Surfaced as additive `ParityVectorReport.next_actions` in `to_dict` (empty when unmapped — verdict/reason text byte-for-byte unchanged). |
| `related_helpers.related_helpers` | `(name)` | `list[str]` next-layer helpers for a CLI command / API entry point. One declarative `RELATED_HELPERS` map, two consumers (CLI `--verbose`/`UTOV_DEBUG` prints `ℹ related: …`; API lookup). `lint_related_helpers()` catches a relation naming a symbol that exists nowhere — no silent bad link. |

## 7. Versioning

Wire shapes and method names are stable across the canonical pass set
(plugin / binop / unary / imm / ext / bfx / ch / triton / sigma /
algorithm). When new passes land they get a new pass token; existing
tokens never change semantics.

Schema migrations (`source`, `batch_id`, `finding_groups`) are
idempotent — old `findings.sqlite` files keep working; missing fields
read as `NULL` / `'unknown'`.
