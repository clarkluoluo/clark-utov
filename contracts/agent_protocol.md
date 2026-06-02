# Agent ↔ Engine wire protocol v1

> Spec for external agents driving clark-utov via `utov agent-serve`.
> Companion to `contracts/runner_interface.md`.

## 0. Why this exists

`clark-utov` is an **automation-and-ledger** tool an agent uses. The agent
is the decision maker — it judges what an algorithm is, when to override
a verifier verdict, when to give up on a stuck point. The engine's job is
to batch through every deterministic question the verifier can answer
(plugin fingerprints / handler-semantic shapes / σ-Σ idioms / algorithm
template fits) and to surface the rest as structured stuck-point data
plus a decision surface (`override_verdict`, `batch_override_verdict`,
`promote_to_finding`, `rerun_from`). The findings table is the ledger
where every decision lands with provenance (`source` field) and audit
trail (`interventions` table).

Two driving modes exist:

- **Script mode** (`utov pipeline`) — the engine drives itself end-to-end.
  If S6 LLM is wanted, the engine calls DeepSeek/MiMo with its own API key
  (the **default is `--llm-backend none`** — the engine does *not* call
  any LLM unless the user explicitly opts in). The engine's S6 path is
  for demos and unattended batch runs.
- **Agent mode** (`utov agent-serve`) — an external **driving agent** spawns
  the engine and talks to it over stdio. The agent calls the engine's
  batch and ledger surface (§3.2) for everything deterministic, and answers
  any `llm_request` the engine sends back (§4) using whatever it likes:
  its own NL LLM, a deterministic Python rule pack, a file-handoff to a
  human, etc. The engine pays $0 — all LLM cost (if any) is on the agent.
  See §4.0 for the three first-class `llm_request`-answering kinds.

This file specifies the wire format an agent uses to drive the engine.

## 1. Channel

- One subprocess: `utov agent-serve --runner-cmd '<spawn cmd>' --input ...`
- All communication is NDJSON (one JSON object per line) over the engine's
  **stdin / stdout**.
- stderr is for human-readable diagnostics; an agent should not parse it
  except for the "engine ready" sentinel.
- The agent is the **client / initiator** for tool calls.
- The engine is the **client / initiator** for LLM delegation.

## 2. Bootstrap

```
[Engine on stderr]: engine ready
```

After this line, the agent may begin sending tool requests.

## 3. Tool calls (agent → engine)

### 3.1 Request shape

```json
{"id": 1, "method": "<name>", "params": {...}?}
```

- `id`: integer or string, agent's correlation token. Echoed back in response.
- `method`: one of §3.2.
- `params`: per-method (see §3.2).

### 3.2 Methods

| Method | params | result |
|---|---|---|
| `metadata` | — | `{target_name, arch, algo_entry_pc, algo_exit_pc, input_length, output_length, algo_symbol}` |
| `list_stages` | — | `["s1", "s1b", "s2", "s3", "s4", "s5"]` |
| `run_stage` | `{name, ...}` | stage summary dict |
| `run_pipeline` | `{stages?: [...]}` | list of stage summaries |
| `get_hypotheses` | `{status?, kind?, source?}` | list of hyp dicts |
| `submit_hypothesis` | `{kind, subject, payload, confidence?, parent_id?, source?}` | `{hyp_id}` |
| `promote_to_finding` | `{hyp_id, verifier_strategy?, stage?}` | `{finding_id}` |
| `verify_plugin_findings` | — | **Standard verify shape** `{checked, passed, failed, inconclusive, promoted}` — mechanically verify pending S1.5 plugin algo-signature hypotheses against the trace (no LLM); promote passes. Idempotent. |
| `verify_handler_binops` | — | Standard verify shape + `stage:"s5-verify"` — BR-4 §1 reg-reg-reg ARM binop pass; dedupes by PC. |
| `verify_handler_unaries` | — | Standard verify shape + `stage:"s5-verify-unary"` — 0526Plan C5.4/5 MOV/MVN/NEG/SXTW/UXTW/REV/CLZ unary pass. |
| `verify_handler_imm_binops` | — | Standard verify shape + `stage:"s5-verify-imm"` — 0526Plan C5.1/2/3 reg-imm binop (`add x?, x?, #imm`, `ror w?, w?, #N`, etc). |
| `verify_handler_extended_binops` | — | Standard verify shape + `stage:"s5-verify-ext"` — 0526Plan C5.7 shifted/extended-register binops (`add x4, x5, w6, sxtw #3`). |
| `verify_handler_bfx` | — | Standard verify shape + `stage:"s5-verify-bfx"` — 0526Plan C5.6 bit-field extract (`ubfx`, `sbfx`). |
| `verify_handler_ch_idioms` | — | Standard verify shape + `stage:"s5-verify-ch"` — 0526Plan C3 SHA-2 Ch(x,y,z) idiom. **BR-8 #2**: also matches `(x∧y)∣(¬x∧z)` via `and / bic / orr` (TC1 SHA-256 shape; gap-tolerant). |
| `verify_handler_maj_idioms` | — | Standard verify shape + `stage:"s5-verify-maj"` — **BR-8 #2** SHA-2 Maj(a,b,c) 3-insn idiom (`eor t,b,c ; and t,t,a ; eor d,t,b∧c`), verifier op="MAJ". |
| `verify_triton_simplifications` | — | `{stage:"s5-verify-triton", checked, passed, promoted, decode_failed, model_mismatch, no_dst}` — 0526Plan C1 Triton symbolic-execution verifier (skips PCs the deterministic passes already covered). `model_mismatch` counts instructions whose fresh-context Triton evaluation diverges from the trace (status flags / memory state Triton can't model in isolation); they're skipped silently, not failed. Includes `skipped_reason` field when Triton bindings aren't importable. |
| `verify_sigma_idioms` | — | `{stage:"s5-fold-sigma", checked, matched, algebra_mismatch, promoted, linked_members}` — 0526Plan C4 layer-1 SHA-2 σ/Σ fold-idiom; binds 3 layer-0 findings under one fold finding via `finding_groups`. **BR-8 #1** adds Phase 3 DFG-grouped scan keyed by input register so ILP-scheduled gaps and `dst==input` final-writes still match. |
| `verify_algorithm_templates` | — | `{stage:"s5-algorithm-fit", matched_algorithms, promoted, io_test}` — 0526Plan E1 layer-2 SHA-256/SHA-512 anchor-set fit; promotes `kind=algorithm_identified` findings. `io_test` flags whether the IO-equivalence rerun ran or was skipped (file-mode runners can't multi-input rerun yet). |
| `self_rescan_missing_anchors` | — | `{stage:"s5-anchor-rescan", checked, missing_before, sigma_promoted, ch_promoted, maj_promoted, refit, still_missing}` — **BR-8 #3** re-run σ/Σ + Ch + Maj when an `algorithm_identified` finding still has missing anchors, then call `recompute_algorithm_fits`. Idempotent; no-op when nothing is missing. |
| `dataflow_query` | `{kind, input_reg?, target_reg?, dst_reg?, from_pc?, within_pc_range?, max_depth?, max_results?}` | list of instruction dicts | **BR-8 #4** agent-facing trace queries. `kind` is one of: `rotations_on_input` (every `ror|lsr` and shifted-eor operand on `input_reg`), `xor_chain_to` (backward eor chain into `target_reg`), `producer_chain` (read-side BFS rooted at `dst_reg`), `boolean_subgraph` (and/orr/eor/bic/mvn within `within_pc_range`). Avoids the agent having to grep `s3_dfg.jsonl` by hand. |
| `emit_pseudocode` | `{format?}` (text / markdown) | `{text}` or `{error}` — **FEATURE-REQUEST-1** Tier 1 emitter. Reads `findings.sqlite` + `meta.json` + `s3_dfg.jsonl` for the current run and renders an `algorithm_identified` finding as paste-and-read pseudocode: header (algorithm + evidence_score + anchors), constants (IV + K), σ/Σ idioms with PCs + register assignments + observed loop counts, generic algorithm body, diagnostic notes (missing Ch/Maj, agent_override audit). On unsupported algorithm / missing finding, the dict carries an `error` field instead of `text`. `preprocess_batch` also auto-drops `<run_dir>/pseudocode.md` whenever an `algorithm_identified` is promoted (no need to call this RPC explicitly). |
| `preprocess_batch` | `{passes?: [str]}` | `{batch_id, ran, results, totals, next_step_hints}` — 0527 one-call deterministic chain. `passes` defaults to the full canonical chain `[plugin, binop, unary, imm, ext, bfx, ch, maj, triton, sigma, algorithm, rescan]`; unknown names raise. Every finding promoted during the call carries the returned `batch_id`. `results[pass_name]` is the same dict each individual `verify_*` method returns. `totals = {promoted, by_source, by_kind, matched_algorithms}`. `next_step_hints` is a short list of agent-UX observations (e.g. "algorithm_identified: SHA-512"). |
| `discard_batch` | `{batch_id, sources?: [str], kinds?: [str], reason?, actor?}` | `{batch_id, candidate_count, discarded, hyp_ids, errors}` — 0527 bulk-fail the hypotheses behind a preprocess batch via `override_verdict("fail")`. Append-only ledger preserved (rows stay; status flips); each per-hyp override writes an `interventions` audit row. Filters: `sources` (e.g. `["s5_triton"]` to discard only Triton-net findings, keep layer-0); `kinds` (e.g. `["fold_idiom"]`). |
| `get_findings` | `{source?, stage?, kind?, subject_like?, limit?}` | list of finding dicts — 0526Plan B1 query surface. Symmetric to `get_hypotheses`. |
| `stuck_statistics` | `{max_points?, cluster_gap?}` | `{total, by_mnemonic, by_verifiable_shape, by_pc_cluster}` — 0526Plan B2 structured stuck-point breakdown so agents don't groupby a 12K flat list. |
| `read_trace_window` | `{idx_from, idx_to}` | list of instruction dicts |
| `localize_divergence` | `{good_input_hex, bad_input_hex, resync_look_ahead?}` | `{divergence, candidates, resync_at}` — capability_request.md §P1-3 first-class differential localiser. Engine calls runner `get_trace` on both inputs, walks them in lock-step, returns the first material divergence (kind ∈ `pc / regs_write / mem / length`) plus ranked `CandidateHypothesis[]` (each shaped for `submit_hypothesis`). `resync_at` is the (good_idx, bad_idx) pair where traces realign — useful for a follow-up byte-graft. File-mode runners raise `NotImplementedError`; agents in that mode call `engine.localize.localize_divergence` directly on two pre-recorded `Instruction` lists. |
| `invalidate_cluster` | `{parent_finding_id, reason, actor?}` | `{cascade_id, parent_finding_id, parent_hyp_id, member_hyp_ids, skipped_member_finding_ids, parent_was_already_failed}` — capability_request.md §P1-4 cluster cascade. Flips the parent's origin hyp to `failed` and every `finding_groups`-linked member's origin hyp to `pending`; each per-hyp `force_status` lands in `interventions` tagged with the shared `cascade_id` so the entire roll-back can be replayed. |
| `static_tool` | `{tool, args: []}` | `{exit_code, stdout, stderr, available}` |
| `is_safe_to_interrupt` | — | `{safe: bool}` |
| `checkpoint` | — | `"ok"` |
| `pause` | `{reason, hint?}` | `"ok"` |
| `resume` | `{reason?, actor?}` | `"ok"` |
| `shutdown` | — | `"ok"` (engine exits) |

### 3.3.1 S6 LLM hypothesis methods (separate from `run_stage`)

`run_stage("s6")` would need a `StuckContext` dataclass which JSON-RPC
can't carry. Agents use these dedicated methods instead:

| Method | params | result |
|---|---|---|
| `s6_find_stuck_points` | `{max_points?: int}` | list of `{parent_hyp_id, kind_hint, summary, snippet, expected_output?}` |
| `s6_propose_and_verify` | `{stuck_context: {kind_hint, summary, snippet, parent_hyp_id?, expected_output?}, input_state?, expected_output_state?, n?}` | `{stage:"s6", candidates, passed, failed, pending}` |
| `s6_auto_loop` | `{max_points?: int, n?: int}` | `{processed, passed, failed, pending}` — mirrors `script_mode` batch loop. Budget enforced via LLMClient. |

Notes:
- `s6_find_stuck_points` reads `stage_outputs/s5_simplified.jsonl` and emits
  one candidate per "unrecognised survivor" instruction. Agent picks which
  to work on; or feeds the whole list to `s6_auto_loop`.
- `max_points` is **optional**. Default = no cap; rely on Budget (set on
  the LLMClient bound to the engine) to enforce spend ceiling.
- LLM call routing depends on backend wiring: with `agent-serve`, our
  `DelegatedBackend` sends `llm_request` to YOU; with no agent-binding the
  engine uses `DirectBackend` and calls DeepSeek directly.

### 3.4 Intervention API (PLAN §15 — agent operability + 留痕)

Every method below records a row in the `interventions` audit table.

| Method | params | result |
|---|---|---|
| `override_verdict` | `{hyp_id, new_verdict: pass\|fail\|inconclusive, reason, actor?}` | `"ok"` |
| `batch_override_verdict` | `{hyp_ids: [int], new_verdict, reason, actor?}` | `{total, ok: [hyp_id], errors: {hyp_id: msg}}` — 0526Plan B3, single-call mass override. |
| `force_status` | `{hyp_id, new_status: pending\|verifying\|passed\|failed\|abandoned, reason, actor?}` | `"ok"` |
| `inject_finding` | `{kind, subject, payload, reason, verifier_strategy?, actor?}` | `{finding_id}` |
| `add_tag` | `{hyp_id, axis, value, reason, actor?}` | `"ok"` |
| `add_dependency` | `{from_hyp_id, to_hyp_id, kind?, reason, actor?}` | `"ok"` |
| `rerun_from_stage` | `{stage: s1\|s1b\|s2\|s3\|s4\|s5\|s6, reason, actor?}` | `{cascade_stages, stage_files_deleted, hyps_abandoned_count, findings_deleted}` |
| `list_interventions` | `{limit?, action?, actor?}` | list of intervention rows |

`rerun_from_stage` cascades:
1. Removes the target stage and downstream from `stage_state` (forces re-run)
2. Deletes stage_output files for the cascade range
3. Marks every hyp `created_in_stage IN cascade` as `abandoned`
4. DELETEs findings whose `origin_hyp_id` is in the abandoned set

The intervention is logged with full before/after stage_state snapshots in `hyp_payloads`.

### 3.3 Response shape

Success:
```json
{"id": 1, "result": {...}, "methodology": {...}?}
```
Error:
```json
{"id": 1, "error": {"code": -32000, "message": "...", "data": {"methodology": {...}?}}}
```

`methodology` (sibling of `result`, added by the discipline wrapper —
PLAN §12.3 / `engine.discipline_wrapper`) carries the runtime anti-drift
payload:

```json
{
  "footer":  "[clark-utov ✓ <method> 完成]\n方法论自检:\n  □ …",
  "card":    "[clark-utov 周期性纪律提醒 · 第 N 步] …",
  "prompts": ["需要我帮你登记进账本吗?…", "检测到和 finding#42 矛盾,…"],
  "alerts":  ["距上次盘整 N 步且期间 M 次失败;主动建议盘整。"],
  "intercepted": true,
  "intercepted_reason": "拒绝调用 inject_finding:payload 含未入账本数据。…"
}
```

`footer` is present on every call (P0 acceptance #1). `card` appears
every `UTOV_METHODOLOGY_INTERVAL` calls (default 15; P0 #2). `prompts`
are context-sensitive reverse questions: evidence_class request after
a verdict; contradiction prompt when the result carries
`invariants_failed` or `contradicts_finding_id`; checkpoint suggestion
when the same method failed N times in a row; multi-candidate warning
when params set `multi_candidate: true`. `alerts` are runtime
interceptions the agent must acknowledge (3+ verifier bypasses, no
recent checkpoint with accumulated failures). When
`intercepted: true`, the dispatch was NOT executed — the envelope
arrives as a JSON-RPC `error` with code `-32001`.

Killable: setting `UTOV_METHODOLOGY=off` in the engine's environment
suppresses the wrapper entirely; envelopes degrade to the original
`{"id": …, "result": …}` shape with no `methodology` key.

`profile` (sibling of `methodology`, added by the discipline wrapper
when the engine is loaded with a v0.3.0 profile — PLAN §19 /
`engine.profile`) advertises which judgement-semantics profile the
engine is running under, so agents / loggers / downstream tools can
branch on it:

```json
{
  "name":  "vmp_algorithm_extraction",
  "chain": ["base", "vmp_algorithm_extraction"]
}
```

- `name` is the leaf profile (the one the engine was constructed
  with — e.g. `vmp_algorithm_extraction`, `key_extraction`, or a
  user-supplied target-specific profile).
- `chain` is the resolved inheritance chain from base to leaf,
  inclusive. Always begins with `"base"`.

The key is omitted when the engine was constructed without a profile
(legacy callers and the existing v0.2.0-dev surface). Adding the key
is non-breaking: agents that don't consume it continue to work.

### 3.5 Phase observation primitives (engine.phase_*)

Three primitives addressing the general "key data is computed in a
phase outside the main observation window" problem (callee, library
load, callback, delayed init, child thread). Phase is treated as
*another source of trace*, not a specially-handled object:

| Primitive | Module | Auto-gate | Toggle |
|---|---|---|---|
| **phase_discovery** | `engine.phase_discovery.discover_phase(value_addr, source, …)` | yes — runs when params carry a value record with `landing_address` and a `DiscoveryDataSource` is wired | `UTOV_PHASE_DISCOVERY=off` |
| **phase_instrument** | `engine.phase_instrument.request_phase_instrument(phase_name, anchor, granularity, …)` | yes — auto-suggests a spec when discovery returns a boundary | `UTOV_PHASE_INSTRUMENT=off` |
| **phase_replay** | `engine.phase_replay.make_replayable_unit(result, entry_state, boundary)` / `synthesize_unit_from_instructions(…)` | no — invoked explicitly when a runner result lands | — |

Anchor types (v1, extensible via `engine.phase.register_anchor_type`):
`func_entry`, `addr_first_exec`, `memregion_first_access`. Each
describable purely in terms of PC/address; no runtime/OS signals
required. Future kinds (`libload_done`, `thread_start`) land alongside
runner-side signals.

Granularity levels (`engine.phase_instrument`): `sparse_sample`,
`pc_band`, `reg_delta`, `full_instruction`. Auto-suggestion defaults
to `full_instruction`.

Envelope siblings added by the discipline wrapper:

```json
{
  "phase_discovery": [
    {"value_addr": 12309240,
     "boundary": {"name": "...", "pc_range": [...], "region": [...], "anchor": {...}},
     "crosses_out": true,
     "chain": [...],
     "reason": "out_of_window_writer pc=0x..."}
  ],
  "phase_instrument_suggestions": [
    {"phase_name": "...",
     "spec": {"kind": "phase_instrument",
              "anchor": {...}, "granularity": "full_instruction", ...},
     "advisory": "..."}
  ]
}
```

`phase_discovery` only lists *crossing-out* results (the producer
lives outside the loaded window). `phase_instrument_suggestions`
emits one entry per actionable boundary; advisory-only entries
appear when no anchor can be resolved.

ReplayableUnit contract — invariant with the main trace:
`unit.jsonl_path` is a JSONL trace in the same schema as the main
trace (contracts/runner_interface.md §2.1), consumable directly by
`engine.runner_client.JsonlTraceReader`. The phase metadata + entry
register state + memory snapshot ride in a sidecar JSON file
(`unit.sidecar_path`) — main-pipeline stages that need only the
instruction stream ignore the sidecar entirely.

Future RPC dispatch (runner-side milestone, not in this commit):

| Method | params | result |
|---|---|---|
| `phase_discover` | `{value_addr, value_size?, phase_name?}` | `PhaseDiscoveryResult.to_dict()` |
| `phase_request_instrument` | `{phase_name, anchor: {...}, granularity?, regions?, max_steps?}` | `PhaseInstrumentSpec.to_dict()` — engine wires through to runner |
| `phase_make_replayable_unit` | `{instrument_result, entry_state, boundary}` | `ReplayableUnit.to_dict()` |

Until the runner side lands, callers use the Python API directly
and feed synthesized `PhaseInstrumentResult` shapes from JSONL
fixtures.

### 3.6 Block-cause routing (engine.block_cause)

Job-chain alignment for unresolved nodes. L1 mechanical work
(discovery / instrument / replay / pipeline / structural-diff) and
block-cause classification stay in the py layer; only class-3 (true
boundary) reaches the user, and only with clark-prepared decision
elements.

Three cause classes + five routing actions:

| Class | Signals | Action | Envelope surface |
|---|---|---|---|
| **collection_gap** + runner has capability | `phase_discovery.crosses_out=true`; 0 mem-write at landing addr; chain ends at read pole | `auto_collect` | `block_cause` carries a `rerun_request` (instrument spec + node + reason). Agent NOT prompted. |
| **collection_gap** + capability missing | (same signals) + `oracle.has(capability)=false` | `register_backlog` | `block_cause` carries a `backlog_entry`. Entry also appended to `<run_dir>/capability_backlog.jsonl`. Agent NOT prompted. |
| **recognition_gap** | `data_collected=true, pattern_recognised=false` | `escalate_l2` | hint to run layer-2 recognition (template fit / handler classification) |
| **strategy_gap** | recognised + `strategy_resolved=false` | `escalate_l3` | hint to run layer-3 strategy (paradigm switch / cross-node reasoning) |
| **true_boundary** | all L1/L2/L3 ran, still ambiguous | `escalate_user` | `decision_elements`: `missing[]` / `cost_estimate` / `success_probability` / `probability_basis` / `options[]` |

Envelope sibling shape (`block_cause` is a list — one entry per
classified node on the call):

```json
{
  "block_cause": [
    {
      "classification": {"class": "collection_gap", "signals": [...],
                          "reasoning": "...", "node_context": {...}|null},
      "action": "auto_collect" | "register_backlog" | "escalate_l2"
              | "escalate_l3" | "escalate_user",
      "rerun_request":     {...} | null,
      "backlog_entry":     {...} | null,
      "escalation_hint":   "...",
      "decision_elements": {...} | null
    }
  ]
}
```

Phase intermediates hidden by default. When the router is active
(`block_cause_router` set on the wrapper and `UTOV_BLOCK_CAUSE` not
off), the wrapper hides `phase_discovery` and
`phase_instrument_suggestions` envelope siblings — they are L1
intermediates fully redundant with `block_cause` for routing. Set
`UTOV_PHASE_DEBUG=1` to restore them alongside `block_cause` for
troubleshooting.

Backlog file. `<run_dir>/capability_backlog.jsonl` — append-only
JSONL. Each line:

```json
{"gap_kind": "needs_collection_capability",
 "missing_capability": "memregion_watch",
 "node_id": "...",
 "trigger_evidence": {"value_addr_hex": "0x...", "reason": "...", "boundary": {...}},
 "suggested_spec": {"kind": "phase_instrument", ...} | null,
 "timestamp": 1748466000.123,
 "run_dir": "..."}
```

Developers can `cat work/*/runs/*/capability_backlog.jsonl` to
aggregate missing-capability requests across all runs.

Capability vocabulary. `engine.block_cause.ANCHOR_TO_CAPABILITY`
maps anchor types (engine.phase.ANCHOR_*) to capability names. v1:

| Anchor type | Capability name |
|---|---|
| `func_entry` | `func_entry_hook` |
| `addr_first_exec` | `pc_first_exec_hook` |
| `memregion_first_access` | `memregion_watch` |

Runner adapters declare what they implement via
`RunnerAdapter.CAPABILITIES: frozenset[str]` (engine-side static
declaration). A runner can opt-in additional capabilities at
runtime by returning them in `metadata().capabilities: list[str]` —
runtime declarations are unioned with the static set.

Auto-rerun is contract only in this milestone. When the router
emits `action=auto_collect`, the `rerun_request` payload describes
*what* to feed to *which* pipeline entry, but actual rerun
execution lands alongside runner-side fulfilment of
`PhaseInstrumentSpec`. Callers that need the loop today install
their own dispatcher reading the router results.

### 3.7 Constant-provenance classifier (engine.constant_provenance)

Deterministic classifier for "is this value a true constant, an
appkey-keyed function, a session-level derivative, or a per-input
variable?" — generalises M3 (per-input axis) and the M1
"dimension-variability=0 means untested" check as a multi-dim
source determination.

Two orthogonal probes:

  * **rerun variability** — caller supplies observations on four
    axes (`same_session` / `new_session` / `new_appkey` /
    `new_per_input`). The classifier compares values across axes.
  * **producer dataflow** — caller supplies what kinds of inputs
    the producing instructions read (`static` / `appkey` / `time`
    / `random` / `session_token` / `input` / `unknown`). Covers
    the rerun probe's blindspot when a stable-rerun environment
    has locked the entropy source.

Cross-product verdict + ceiling:

| Category | Evidence class ceiling | Scope | Recommended action |
|---|---|---|---|
| `hardcoded_fixed` | A | universal | `auto_pin` (close the node) |
| `appkey_fixed_function` | A | per_appkey | `mark_dual_path` (recover f(appkey) → A, else pin current as B) |
| `session_level_derived` | B | per_session | `escalate_usage_decision` |
| `per_input_variable` | — | per_input | `treat_as_variable` |
| `undetermined` | B | unspecified | `request_more_observations` |

Conflict rule: if the dataflow says the producer reads
`time` / `random` / `session_token`, the verdict is
`session_level_derived` regardless of rerun stability — probe 2
overrides probe 1 on the entropy-locked blindspot. Stable reruns
*without* dataflow corroboration downgrade to `undetermined`
(honest answer rather than false-confident `hardcoded_fixed`).

Value records opt in by carrying ``rerun_observations`` and/or
``producer_dataflow`` on the params shape:

```json
{
  "values": [
    {
      "value_name": "session_key",
      "rerun_observations": [
        {"dimension": "same_session",  "value_hex": "aa"},
        {"dimension": "same_session",  "value_hex": "aa"},
        {"dimension": "new_session",   "value_hex": "bb"},
        {"dimension": "new_appkey",    "value_hex": "bb"},
        {"dimension": "new_per_input", "value_hex": "aa"}
      ],
      "producer_dataflow": {
        "producer_reads": ["static", "session_token"]
      }
    }
  ]
}
```

Envelope sibling shape:

```json
{
  "constant_provenance": [
    {"value_name": "session_key",
     "category": "session_level_derived",
     "evidence_class_ceiling": "B",
     "scope": "per_session",
     "recommended_action": "escalate_usage_decision",
     "rerun_analysis":   {...} | null,
     "dataflow":         {...} | null,
     "signals":          [...],
     "reasoning":        "..."}
  ]
}
```

Relationship to existing gates. `engine.value_provenance` caps
`hook` / `dump` / `io` / `snapshot` values at evidence_class B and
does not distinguish session-level from truly-fixed.
`engine.m3_bypass_block` aggregates per-input variability evidence
across calls — that's the per-input axis special case of this
module's rerun probe. Existing gates keep their narrower contracts;
`constant_provenance` is the unified primitive for anything that
needs the finer-grained verdict.

## 4. LLM delegation (engine → agent)

When the pipeline needs LLM-style reasoning (S6 / blue-team / rule extraction),
the engine writes:

```json
{"id": "llm-N", "type": "llm_request",
 "system_prompt": "...",
 "user_context":  "...",
 "schema":        { ...JSON Schema for one candidate hypothesis... },
 "n":             5}
```

The agent MUST answer with the matching id:

```json
{"id": "llm-N", "type": "llm_response",
 "hypotheses": [
   {"kind": "...", "subject": "...", "payload": {...},
    "confidence": 0.0, "rationale": "..."},
   ...
 ]}
```

Or, on failure:
```json
{"id": "llm-N", "type": "llm_error", "message": "..."}
```

### 4.0 What is an "agent"? (BR-4 §B)

`llm_request` is **a structured tool call returning JSON matching `schema`** —
NOT a natural-language question/answer turn. The engine doesn't care whether
the agent uses a neural network, a regex, or a lookup table to produce the
answer; the only contract is "given this `(system_prompt, user_context,
schema, n)`, return up to N candidates in the `hypotheses` shape (§4.2)".

This makes three implementation kinds equally first-class:

| Kind | How it answers | Typical use |
|---|---|---|
| **Natural-language LLM agent** | Forwards `system_prompt` + `user_context` to a remote LLM (Claude / GPT / DeepSeek), parses the JSON the LLM produces, sends `llm_response` | Original design — useful when stuck-point ranges over forms the engine's verifier model can't pre-classify |
| **In-process Python provider** | A Python callable `(req: dict) -> dict` invoked synchronously inside the agent process. Inspect `req["user_context"]` (the snippet text) and emit hypotheses by deterministic rules | Heuristic / disassembly-aware rule packs; cheap, fast, deterministic; e.g. the bundled `arm-heuristic` provider in `engine.driver` recovers SHA-256 reg-reg-reg XORs without any LLM |
| **File-handoff** | Write `req` to `/tmp/utov_llm/in/<id>.json`, poll `/tmp/utov_llm/out/<id>.json` for the answer | Human-in-the-loop, or bridge to an external service (Notion, Slack bot, browser) without Python integration |

The wire format is byte-identical across all three; the engine cannot tell
them apart. **`system_prompt` and `user_context` are advisory inputs**: useful
when the answerer is an NL model, opaque hints to other answerers. What
matters is the structured `payload` returned in each hypothesis — the
verifier is the system's only source of truth (PLAN §1.1) and it consumes
only `payload`, never `rationale` or `subject`.

#### How to write an in-process provider

```python
# my_provider.py
def claude_arm_provider(req: dict) -> dict:
    """ARM-disassembly-aware provider — answer only when the snippet is a
    shape verifier.check_handler_semantic accepts; empty list otherwise."""
    import re
    ctx = req.get("user_context") or ""
    m = re.search(r"Snippet:\s*\n(.+)", ctx)
    if not m:
        return {"id": req["id"], "type": "llm_response", "hypotheses": []}
    insn = m.group(1).strip()
    # ... rule-based classification ...
    return {"id": req["id"], "type": "llm_response",
            "hypotheses": [{
                "kind":       "handler_semantic",
                "subject":    f"binop@{insn}",
                "payload":    {"op": "XOR", "dst": "x4", "src": ["x1", "x2"]},
                "confidence": 0.9,
                "rationale":  f"ARM64 '{insn}' classified as XOR",
            }]}
```

Drive it without any wire / CLI handoff:

```python
from engine.driver import drive
drive(
    runner_cmd="java -jar runner.jar serve lib.so",
    input_hex="616263",
    provider=claude_arm_provider,    # ← any (req: dict) -> dict callable
    workflow="s6-loop",
)
```

In-process is **the** recommended path when the answer is mechanical (regex,
mnemonic lookup, deterministic disassembly). NL agents are appropriate when
the stuck-point's shape is genuinely ambiguous — and they go through the
same wire protocol with no special-casing.

### 4.1 Concurrency rule (important)

While a `run_pipeline` or `run_stage` request is in flight, the engine MAY
emit one or more `llm_request` messages. The agent MUST be prepared to
answer them BEFORE the outer tool call returns — otherwise the engine
deadlocks waiting for its LLM response.

Practical implementation for an agent:
1. After sending a tool request, enter a read loop.
2. For each line on stdin:
   - `type=="event"` → record / ignore.
   - `type=="llm_request"` → service it (answer using own LLM), send `llm_response`.
   - has `result` or `error` with matching id → that's the tool reply, exit read loop.

### 4.2 Schema for one hypothesis (S6 default)

```json
{
  "kind":       "string",    // "handler_semantic" | "algo_signature" | ...
  "subject":    "string",    // human-readable target ("handler@0x40006cc4")
  "payload":    {            // concrete predicate the verifier can check
    "op":   "XOR",
    "dst":  "x4",
    "src":  ["x1", "x2"]
  },
  "confidence": 0.0,         // [0..1] self-rating
  "rationale":  "string"     // for audit only — verifier ignores it
}
```

Blue-team review (`type=="llm_request"` with `schema` from `blue_team.py`)
expects:
```json
{
  "verdict":   "approve" | "challenge",
  "rationale": "string",
  "suggested_extra_inputs":       ["aabbcc..", ...],
  "suggested_observation_points": [{...}, ...]
}
```

## 5. Events (engine → agent, fire-and-forget)

```json
{"type": "event", "kind": "<EventKind>", "timestamp": <float>, "detail": {...}}
```

Notable kinds an agent should react to:

| Kind | Suggested agent reaction |
|---|---|
| `stage_done` | log progress; update internal state |
| `safe_interrupt_point` | safe to inject a `pause` or `shutdown` now |
| `pause_request` | pacing trouble — ask user whether to continue |
| `ask_user.budget_overrun` | offer raise-budget choice |
| `ask_user.degraded_result` | flag in final deliverable; runner is File mode |
| `ask_user.no_fingerprint` | offer P3 handler-semantic recovery |
| `ask_user.blue_team_needed` | offer to run blue-team review |
| `ask_user.backtrack_limit` | propose broader input set / mode switch |
| `ask_user.runner_slow` | offer mode downgrade |
| `discipline_reminder` | the engine is reminding its own LLM-prompt context; if you're routing prompts through your own LLM, append this verbatim to the system prompt |
| `budget_warn` | warn user budget X% consumed |
| `pipeline_done` | flush, summarize for user |

Events are advisory — the engine continues by default. An agent who wants to
**veto** continuation should call `pause` immediately, which causes the next
engine checkpoint to record `paused: true` and refuse further `run_pipeline`
calls until a `resume` flow (planned).

## 6. Versioning

This protocol is `v1`. Breaking changes will bump the version in this file's
H1 header and in the first `safe_interrupt_point` event's detail.
