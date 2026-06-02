# Release notes · clark-utov

## v0.5.0-dev · Task target-management mechanism (2026-05-29)

The reference-case follow-on landed. A new top-level object —
**Task** — promotes "task done" from agent-resident understanding
to a clark-declared artefact with an objective `done_criterion`
gate. The runner becomes a neutral workbench (used by the task, not
the parent of it); profiles are intent-class semantic sets, not
runner-class. Engine suite **752 passed / 1 skipped** (was 655 at
v0.4.0 baseline); zero regression.

### Architectural correction — runner is a workbench

Mid-cycle revision: the original "task is runner-bound" framing
flipped. A runner is a stable, reproducible execution environment;
it pre-supposes nothing about what the user wants from it. The same
The reference target runner can carry "restore sign algo" + "memory integrity
check" + "timing profile" as three independent tasks; their nodes,
findings, gates, and profiles do not mix.

Consequences shipped in v0.5.0:

* **Task declares runner usage, does not bind** — `uses_runner` is
  optional; `runner_capabilities` lists abilities the workbench must
  supply.
* **No "bound vs standalone" distinction** — every task uniformly
  declares (or omits) runner usage. The §4.2 `input_contract`
  becomes a general reusability declaration for any task referenced
  by another task's child list / call site.
* **Profile is intent-class** — `vmp_algorithm_extraction` is the
  semantic set for the *algorithm-extraction intent*, NOT for the
  VMP runner. Same runner + different profile is legal.
* **Attribution chain** — `finding → node → task`; runner not on the
  chain (it provides capabilities, not identity).

### The Task object model

```
engine/task/
  types.py            NodeRef / NodeState / TaskSpec
  done_criterion.py   CriterionItem AST: node_closed / child_done /
                      named_artefact / all_of / any_of + evaluator
                      returning (satisfied, gaps)
  loader.py           JSON loader; missing done_criterion fails load;
                      dangling-ref checks on every node_closed /
                      child_done atom
  gate.py             TaskGate — second-layer M1; refuses task-done
                      declarations and names every missing gap
  tree.py             TaskTree — parent / child indexing, recursive
                      done evaluation, compose-time contract check on
                      referenced children
  contract.py         InputContract + validate_contract_compose
                      (compose-time gap-listing)
  audit.py            TaskAuditLog (JSONL) + insert / replace /
                      delete; assert_done_criterion_unchanged post-
                      condition check
```

### Two-layer M1, strictly separated

* **Node M1** — the v0.4.0 ConjunctiveGate on per-archival-call
  basis (unchanged).
* **Task M1** — TaskGate evaluates `done_criterion` when the agent
  declares "task done"; refusal names every unmet gap.

When the caller supplies a probe context with the task-done
declaration, TaskGate also walks the v0.4.0 ConjunctiveGate against
those params — every mechanism probe fires at the task-level call
surface too (PLAN §20.1.3 invariant #1). Lock B from v0.4.0 covers
the new surface for free: the conjunctive gate consults the
import-time builtin registry, so inserted / replacing tasks
inherit floor enforcement without any per-task wiring.

### Composable work units (v2)

* **4.1 Parent / child tree** — children are part of the parent's
  spec; the parent's `done_criterion` may reference child done
  state via `child_done` atoms. Children referenced as done
  dependencies MUST carry an `input_contract` (compose-time check).
* **4.2 Reusable invocation** (renamed from "standalone task") —
  any task with an `input_contract` is callable from another task's
  child list; cross-task references go through the contract.
* **4.3 Insert / replace / delete with audit** — every structural
  op writes a `TaskAuditEntry` (op, target, detail, who, why, when)
  to a JSONL audit log; replacements must preserve the replaced
  id (no rename-via-replace); post-condition check enforces
  `done_criterion` immutability across any structural transformation.

### Profile-layer integration

New `task_templates` section (domain-only — base lint rejects).
Profile declares the list of intent-recommended reusable task specs;
`MergedProfile.task_template_for(name)` returns the raw spec for
:func:`engine.task.parse_task_spec` to materialise.

### Test footprint

| File | New tests |
|---|---|
| `tests/task/test_t1_loader_and_types.py` | 21 |
| `tests/task/test_t2_done_criterion.py` | 17 |
| `tests/task/test_t3_task_gate.py` | 9 |
| `tests/task/test_t4_tree.py` | 8 |
| `tests/task/test_t5_input_contract.py` | 9 |
| `tests/task/test_t6_audit.py` | 13 |
| `tests/task/test_t7_mechanism_floor.py` | 4 |
| `tests/task/test_t8_profile_integration.py` | 7 |
| `tests/task/test_t9_acceptance.py` | 9 |

97 net new assertions; one v0.4.0 test fixture extended (MergedProfile
constructor gains `task_templates=()`).

## v0.4.0-dev · field-experience extensions (2026-05-29)

11 items sourced from the reference-case audit landed this cycle —
seven base-mechanism extensions and four vmp-domain extensions.
Every item drove a real "本可下放却上抛" or "extrapolation past
observed boundary" incident in tc3; v0.4.0 turns each finding into
declarable / lock-able framework primitives. Engine suite
**655 passed / 1 skipped** (was 563 at v0.3.0 baseline). Zero
regression on the existing tests.

### What landed

**Base mechanism extensions**

| Tag | Item | Surface |
|---|---|---|
| B1 | ScopeBoundaryGate | `engine/scope_boundary_gate.py` + probe adapter + `base.json` mechanism entry |
| B2 | ScopeUpscaleGate  | `engine/scope_upscale_gate.py`  + probe adapter + `base.json` mechanism entry |
| B3 | M1 audit triage   | `SuccessAuditResult.triage` ∈ {`agent_self_resolved`, `user_decision_required`}; configurable `subjective_keys` |
| B4 | env_limit_rows carve-out | `env_limit_rows_excluded` + `adjusted_pass_rate` on `SuccessAuditResult`; opt-in via top-level count or per-sample flag |
| B5 | Allocator-noise filter | `DataflowReadKind.ALLOCATOR` + `NOISE_READS` + `DataflowSummary.meaningful_reads()`; allocator alone → UNDETERMINED |
| B6 | Dynamic-criterion lint | `lint_dynamic_criteria(profile)` flags gate rules pinning concrete offsets without scan-style descriptors |
| B7 | Gap-kind realignment | `GapKind` 3-way enum (`capability_gap` / `boundary_limit` / `analysis_incomplete`); router falls through fine-grained → broad when only the broad name is declared |

**vmp_algorithm_extraction domain extensions**

| Tag | Item | Surface |
|---|---|---|
| V1 | `task_level_fixed` state binding the new `closure_state_within_task` role; `task_bound` scope rule |
| V2 | `scope_order = [task_bound, env_bound, single_identity_bound, cross_env]` — declarable ordering consumed by B1 / B2 / V4 |
| V3 | `cap_mapping` JSON section: 5-way constant_provenance category → evidence-class id, moved out of Python module; lint blocks `cap_mapping` in base |
| V4 | `use_case_fork` domain probe — `current_context_reproduction` refuses cross-env claims; `cross_context_claim` demands dataflow proof |

### New mechanism probes — two more locked into base

`scope_boundary_gate` and `scope_upscale_gate` both carry
`mechanism = True` and join the conjunctive gate's force-included
set. Lock A / B / C apply uniformly:

- Lock A · subprofile cannot `disable:` either probe (registry merge
  raises ProfileMergeError).
- Lock B · `ConjunctiveGate` consults the import-time decorator
  registry + `cls.mechanism is True`; a tampered MergedProfile that
  zeroes `mechanism_probe_names` still runs both gates.
- Lock C · base lint rejects `scope_order` / `cap_mapping` in base
  profiles (they're domain semantics).

### Profile surface changes

| Field | Layer | Effect |
|---|---|---|
| `Profile.scope_order` | domain | Ordered scope vocabulary, narrowest first. `MergedProfile.scope_rank(scope)` returns the rank for ScopeBoundaryGate / ScopeUpscaleGate / UseCaseFork. |
| `Profile.cap_mapping` (`tuple[CapMappingEntry, ...]`) | domain | constant_provenance category → evidence-class id; CP probe consults `MergedProfile.cap_for_category` before falling back to the module table. |

Both are domain-only — base lint rejects them, `MergedProfile`
exposes read-only helpers (`scope_rank` / `cap_for_category`).

### Backward compatibility

Every v0.4.0 change is additive:

- Existing M1 callers see new fields in `to_dict()` but no behaviour
  change for already-passing test fixtures.
- `DataflowSummary.producer_reads` keeps its raw tuple; only the
  `reads_*` helpers consult `meaningful_reads()`.
- `BlockCauseClass` enum unchanged; `GapKind` is a new parallel
  vocabulary, profile may declare routing by either name.
- `MergedProfile` adds two fields (`scope_order`, `cap_mapping`); the
  one hand-rolled test fixture was updated to pass `()` defaults.
- CP probe still produces the module-table ceiling when
  `ctx.profile` is `None` or the profile declares no `cap_mapping`.

### Test footprint

| File | New tests |
|---|---|
| `tests/profile/test_v040_vmp_extensions.py` | 15 |
| `tests/profile/test_v040_dynamic_criterion_lint.py` | 8 |
| `tests/profile/test_v040_use_case_fork.py` | 13 |
| `tests/test_constant_provenance_allocator_noise.py` | 11 |
| `tests/test_block_cause_gap_kind_realignment.py` | 10 |
| `tests/test_m1_triage_and_env_limit.py` | 13 |
| `tests/test_scope_gates.py` | 22 |

92 net new assertions; four existing tests updated for the
mechanism-count loosening (5 → ≥ 5) plus the `MergedProfile`
constructor signature change.

### Still deferred — single-sample observations

Four candidate items from the original v0.4.0 list remain pending
second-target evidence (M1 achievement delegation, 4-way → 3-way
gap-class collapse with the existing enum removed, `bounded_by_runner_scope`
real load, mechanism baseline travelling to a second domain). They
move forward as v0.5.0-dev candidates — see `dev_doc/PLAN.md` §21
(后续路线) for the holding list.

## v0.3.0-dev · profile layer (2026-05-29)

Lifts the v0.2.0-dev framework gates out of hardcoded engine modules
into a **declarable, extensible profile layer** (PLAN §19 /
IMPL_PLAN §P1.0). utov goes from "a VMP tool" to "a general
exploration framework": adding a new target domain (key extraction,
weird custom protections, …) is now a profile-JSON edit, not an
engine source change.

The semantic / mechanism split is enforced structurally — domain
profiles freely declare evidence-class ordering, state vocabulary,
closure-gate composition, scope rules, and domain probes; base
mechanism (M1 observation≠closure, M3 false-block detection,
constant_provenance framework, value_provenance "observed must cap",
watch_first_write producer trace, §17 runner conformance) is locked
by **three independent doors** that subprofiles cannot pry open.

### Profile layer surface

| Module | Role |
|---|---|
| `engine.profile.types` | Frozen dataclasses (Profile / EvidenceClassSpec / StateSpec / ProbeSpec / GateSpec / RoutingRule / ScopeRule). |
| `engine.profile.loader` | JSON-file parser; `module:` / `disable:` syntax; structural validation. |
| `engine.profile.registry` | Inheritance chain assembly (`base → domain → user`); mechanism-lock enforcement at merge time; orphan-role detection. |
| `engine.profile.probe_runtime` | `Probe` ABC with `mechanism` class attribute; `ProbeContext` carrying state bindings + profile reference; builtin decorator registry; `resolve_probe_class` for `module:` dynamic import. |
| `engine.profile.probes.*` | Six builtin adapter modules — five `mechanism=True` (M1 / M3 / constant_provenance / value_provenance / watch_first_write) wrapping their existing `engine.*` implementations, plus `length_chain_check` (domain). |
| `engine.profile.state_machine` | Static role→state binding for ProbeContext (per §19.1 — base references roles, domain plugs states). |
| `engine.profile.gate_runtime` | `ConjunctiveGate` force-includes mechanism via class-attribute scan from bytecode (Lock B linchpin). Caches mechanism probe instances so M3's per-session detector survives cross-call. |
| `engine.profile.evidence_class_synth` | `most_restrictive_class_id` / `synth_node_cap` — profile-ordered cap synthesis (§19.3). Replaces the alphabetic-max placeholder in step-4 CP/VP probes. |
| `engine.profile.routing_runtime` | `RoutingTable` over `MergedProfile.routing_rules` + `lint_actions_against_known`; `BlockCauseRouter` consults it when supplied, falls back to legacy hardcoded mapping when not. |
| `engine.profile.lint` | Structural lint (base may only contain mechanism; domain may not declare mechanism) + source-literal scan (domain state names forbidden in kernel and base mechanism implementation files). |

### Shipped profiles

| File | Role |
|---|---|
| `engine/profiles/base.json` | Five mechanism probes registered; no states, no evidence classes, no routing (those are all domain). |
| `engine/profiles/vmp_algorithm_extraction.json` | First domain profile — A/B/C evidence classes, five canonical node states (`closed_form` binds `closure_state` role), `length_chain_check` domain probe, four routing rules (collection_gap / recognition_gap / strategy_gap / true_boundary), `env_fixed_observed → env_bound` scope. |
| `engine/profiles/key_extraction.json` | Stub for the second domain (§19.7 #3) — five new states with `key_verified` binding the same `closure_state` role; mechanism rules cross-domain reuse with zero kernel change. |
| `engine/profiles/weird_target_x.json` | Three-level inheritance demonstration (§19.7 #4) — `base → vmp → weird`, increment-only declaration. |

### The three locks — structurally implemented

**Lock A · load-time registry rejection.** `ProfileMergeError`
raised when a non-base profile redeclares a `mechanism: true` probe
name, claims `mechanism: true` itself, lists a mechanism name in
`disable:`, or leaves a `closure_state`-style role unbound after the
chain merges. Each of the five mechanism probes verified — total 10
adversarial assertions covering override + disable per probe.

**Lock B · runtime gate force-include.** `ConjunctiveGate` does not
consult `MergedProfile.mechanism_probe_names` — that field is
profile data and an attacker who skips the registry can set it to
`frozenset()`. The gate instead iterates the import-time builtin
registry (`list_builtin_probes()`) and filters on
`cls.mechanism is True` — a class attribute on the implementation,
unreachable to profile-data tampering. Verified by constructing a
`MergedProfile(probes=(), mechanism_probe_names=frozenset())`
directly and confirming all five mechanism verdicts still fire.

**Lock C · dual-side lint.** `base.json` may contain only
`mechanism: true` entries (no `node_states`, no `evidence_classes`,
no `routing_rules` — those are domain). Domain JSON may not contain
`mechanism: true`. Domain may not reference a role that no state in
the merged chain binds. Source scan rejects domain-state literals in
the kernel and in any base mechanism Python file — prevents
`if state == "closed_form":` from silently bypassing the role
indirection.

### Backward compatibility

The wire-in is opt-in via keyword-only arguments:

- `Core(profile_name: str | None = None)` — `None` keeps the
  ProfileRegistry unloaded.
- `DisciplineWrapper(profile: MergedProfile | None = None)` — `None`
  keeps the envelope shape identical to v0.2.0-dev (no `profile` key
  emitted by `to_dict()`).

The 502 v0.2.0-dev acceptance tests construct Core and the wrapper
without these kwargs and pass unchanged. New profile-aware tests
(185 of them, in `engine/tests/profile/`) supply the kwargs
explicitly.

### Production wire-in scope

Step 8 makes the profile reachable from `Core` and the wrapper but
does **not** migrate the wrapper's internal dispatch logic to consult
`MergedProfile.probes` for the M1/M3/CP/VP/WFW probes — those still
go through their existing hardcoded imports. The wire is connected;
the migration of the production code path to a registry-driven
dispatch is held for v0.4.0 (separate issue) to avoid step-8 also
becoming a wrapper-refactor (~700 LoC, broad regression surface).
For users building new domain profiles today, this is invisible: a
new profile is loadable, `Core.profile` resolves it, `Verdict` /
`ConjunctiveGate` / `RoutingTable` are usable end-to-end. The
the existing reference target production path is unchanged.

### Build / sync layout — downstream consumers please read

The engine source tree gained a new top-level sibling of the
`engine/engine/` Python package:

```
engine/
├── engine/        ← Python package (stages, core, profile/, ...)
└── profiles/      ← v0.3.0+ profile JSON (sibling, NOT inside the package)
```

`ProfileRegistry().PROFILES_DIR` resolves to `engine/profiles/` via
`Path(__file__).resolve().parents[2] / "profiles"` from the package's
`registry` module — i.e. **the dest layout must mirror the source
layout: `engine/engine/` *and* `engine/profiles/` side by side.**

Downstream consumers maintaining their own sync flow: add an
equivalent step. A launcher that puts only `engine/engine/` on
`PYTHONPATH` will load the package fine but `ProfileRegistry`
will raise `ProfileLoadError` on first profile lookup.

### Agent-protocol envelope addition

`agent_protocol.md` §3.3 documents a new optional envelope sibling:

```json
{"profile": {"name": "vmp_algorithm_extraction",
             "chain": ["base", "vmp_algorithm_extraction"]}}
```

Emitted by the wrapper when constructed with a profile; omitted
otherwise (keeps the v0.2.0-dev envelope schema for legacy agents).

### Field-experience coverage (v0.3.0-dev vs accumulated principles)

A post-v0.3.0 audit against the accumulated field-experience profile
material (18 distilled principles, base 9 / vmp_domain 5 /
single-sample 4) lands the coverage at:

- **5 fully covered** — constant_provenance dataflow-vs-rerun
  judgement, observed-must-cap principle (`value_provenance`),
  node_state vocabulary + role binding,
  observation→producer trace (`watch_first_write`), three-class
  evidence ordering for VMP.
- **4 partially covered** — pinned-doesn't-upscale principle (the
  ingredients live in base but the composing `ScopeUpscaleGate`
  isn't a single named gate yet); gap-class framework (block_cause
  has four classes, accumulated experience suggests three-class
  realignment); scope vocabulary (env_bound only, four other values
  pending); `constant_provenance`→evidence_class cap mapping (hardcoded
  in module, not declarative profile field).
- **8 not yet covered** — ScopeBoundaryGate, task-level-fixed state,
  M1 audit triage (deterministic-vs-subjective split), environment-limit
  carve-out for M1, dynamic-criterion lint, allocator-noise filter,
  use-case fork rule, the standalone ScopeUpscaleGate.
- **1 needs verify against second target** — whether
  `constant_provenance` already treats allocator side-effects as
  non-source.

The eight uncovered + four partial items form the
`v0.6.0-dev candidates` block in `dev_doc/PLAN.md` §21. The 4 single-sample
observations stay on the audit log until a second independent target
either confirms or refutes them.

The v0.3.0-dev shipping decision is consistent with the
"open semantic layer, hold mechanism baseline" principle: the
*structure* (registry / three locks / role indirection / synthesis /
gate runtime / routing) is in place; the *content* (additional gates,
states, mapping entries) accumulates as field evidence arrives.

### Tests

Profile suite: **185 / 185 green**.
- `test_base_domain_boundary_lint.py` (24)
- `test_mechanism_baseline_locked.py` (49 — Lock A × 5 + Lock B × 5 + probe behaviour × 5)
- `test_state_machine.py` (12)
- `test_evidence_class_synth.py` (14)
- `test_gate_runtime.py` (10)
- `test_routing_runtime.py` (13)
- `test_vmp_profile_regression.py` (15)
- `test_key_extraction_stub.py` (11)
- `test_weird_target_inheritance.py` (13)
- `test_module_import.py` (10)
- `test_core_profile.py` (6)
- `test_wrapper_profile.py` (8)

Full engine suite: **563 passed / 1 skipped** (was 502 baseline +
v0.2.0-dev; the profile layer is purely additive).

## v0.2.0-dev · capability + anti-drift (2026-05-28)

Closes the P0/P1/P2 capability gaps surfaced by the reference target partial
archive (`work-tc3-samples/work/legacy/reference-target_partial_archive/`) and adds
the runtime methodology-reinforcement wrapper.

### Capability (capability_request.md)

| Area | Module | What it adds |
|---|---|---|
| **P0-1** VMP-internal observation | `engine.capability` | `CodeHookRange` (PC-band step-callback) + `RegisterTrace` + `evaluate_hook_sanity` (M3 ≥3-input variability check). `RunnerAdapter.code_hook_range()` contract added; File-mode fallback synthesises traces from in-memory `Instruction` lists. |
| **P0-2** Main-VMP trace window | `CoreConfig.extra_trace_windows`, `s4_slice` | Configurable extra PC bands (e.g. `0x32302c..0x325708`) feed S4 as additional sinks so backward slices reach data flowing through main-VMP. Session-persisted in `session.json`. |
| **P1-1** Numeric claim guard (M1) | `engine.evidence` | `EvidenceClass` enum, `NumericClaim` default `pending_review`, `KNOWN_NEGATIVE_PATTERNS` (incl. `e9a86ab9`). |
| **P1-2** Parity invariants (M8) | `engine.invariants` | 5 default invariants auto-flag contradictions (`hook_valid_but_no_match`, `input_len_constant_but_many_vectors`, `pass_rate_without_evidence_class`, …). |
| **P1-3** Differential localisation | `engine.localize` + `Core.localize_divergence` + RPC | First-class three-layer diff: divergence point + ranked candidate hypotheses + resync point. |
| **P1-4** Cluster cascade | `engine.cluster` + `Core.invalidate_cluster` + RPC | Parent finding invalidated → every `finding_groups` member's origin hyp flips to `pending` under a shared `cascade_id`. |
| **P2-1** Number formatting | `engine.numfmt` | `as_hex`/`as_dec`/`parse_explicit` — base=10 rejects `0x`-prefixed strings. |
| **P2-2** Run manifest (M9) | `engine.manifest` | `run_manifest.json` per parity run with JVM flags + jar SHA-256 + hook RVAs; `heap.trace=true` blocked unless `DEBUG_HEAP=1`. |
| **P2-3** Verdict YAML (M10) | `engine.verdict` | Structured `confirmed / not_confirmed / invalidated / known_gaps`; bare "确认/passed" rejected; `archival_allowed` computed. |

### Anti-drift (PLAN §12.3 runtime form)

| Module | Role |
|---|---|
| `engine.methodology` | Text catalog (footer / periodic card / 6 prompts / 4 alerts) + `MethodologyConfig` + `MethodologyState`. |
| `engine.discipline_wrapper` | Wraps every JSON-RPC dispatch: pre-check interception (un-ledgered payload, forbidden keyword) → context-sensitive prompts (evidence_class, contradiction, high-rate success, streak, multi-candidate) → footer + every-Nth-step card + alerts (verifier-bypass count, no-recent-checkpoint). |
| `serve_mcp` | Attaches `methodology: {...}` as a sibling of `result` in every envelope. Killable via `UTOV_METHODOLOGY=off`; tunable via `UTOV_METHODOLOGY_INTERVAL`, `…_FAILURE_STREAK`, etc. |

5-point acceptance covered by `tests/test_methodology.py`:
footer-every-call · periodic card in step 15-20 · contradiction prompt
· streak prompt · un-ledgered intercept.

### Framework gates (M1/M3-class anti-drift primitives)

Five reference-target retro patterns previously caught by hand — now
framework-enforced through `discipline_wrapper.step()`. Each gate is
a standalone module with its own env toggle so debug isolation is
preserved. Every gate decorates the envelope with a structured
block and a one-line alert; failing claims are either rewritten
in-place before dispatch (downgrade) or refused outright (intercept).

| Gate | Module | Env toggle | What it catches |
|---|---|---|---|
| **M1 success audit** | `engine.m1_success_audit` | `UTOV_M1_AUDIT=off` | `target_success` / `archival_allowed=true` without dimension-coverage / overfit / scope / closure proof. Grades A/B/C; B auto-downgrades to `strong_partial`, C refuses. Acceptance: 94/94 prefix-fixed sample set → B. |
| **M3 bypass block** | `engine.m3_bypass_block` | `UTOV_M3_BYPASS=off` (N via `UTOV_M3_BYPASS_N`) | Same candidate block failing M3 variability under ≥ N distinct observation methods → flip to `suspected_bypass_block`; subsequent observation attempts on that block refused before dispatch. Single-method failure intentionally does not trigger (distinguishes obs bug from bypass). |
| **Value provenance** | `engine.value_provenance` | `UTOV_VALUE_PROVENANCE=off` | `source ∈ {hook, dump, io, snapshot}` without verified closed-form recompute → tag `observed`, cap evidence_class at B. `observation_parity=True` alone does not lift the ceiling (M1+ rule). |
| **Watch-first-write** | `engine.watch_first_write` | `UTOV_WATCH_FIRST_WRITE=off` (advisory-only via `UTOV_WATCH_FIRST_WRITE_AUTO_TRIGGER=off`) | Observed value at a concrete `landing_address` with no closed-form → auto-emit `watch_first_write(addr)` spec on the envelope so the runner can install a memory watchpoint and capture the producing PC + source bytes. |
| **Length-chain check** | `engine.length_chain_check` | `UTOV_LENGTH_CHAIN=off` (cap via `UTOV_LENGTH_CHAIN_MAX_MULTIPLE`) | Adjacent nodes on a declared `length_chain` must satisfy one of: equal / integer-multiple / hex 2:1 / strict base64 4:3 / explicit ratio / explicit delta / caller-whitelisted. Otherwise flag `length_mismatch_unexplained`. The reference target 32→21→16 both edges flagged. |

Wrapper plumbing additions: `DisciplineEnvelope` now carries
`m1_audit`, `m3_bypass`, `value_provenance`, `watch_suggestions`,
`length_chain` sibling fields next to the existing `methodology`
block. `DisciplineWrapper.__init__` accepts an opt-in config per
gate plus a shared `BypassBlockDetector` instance for multi-call
state.

### Phase observation primitives (any-phase capture + replay)

General capability for "key data is computed in a phase outside the
main observation window" — callee, library load, callback, delayed
init, child thread. Three composable primitives; core abstraction is
**phase is a trace source, not a special-cased object**. The
captured unit feeds the existing main pipeline (segmentation /
dedup / symex / slice / diff) without per-phase branching.

| Primitive | Module | What it does | Env toggle |
|---|---|---|---|
| **phase_discovery** | `engine.phase_discovery` | Walk backward from a key value's landing address; flag when the producer crosses out of the current trace window. Pluggable `DiscoveryDataSource` — default reads loaded trace + optional ledger probe; option-2/3 (runner RPC, value_provenance) plug without refactor. Auto-runs when params carry a value record with `landing_address` and a source provider is wired. | `UTOV_PHASE_DISCOVERY=off` |
| **phase_instrument** | `engine.phase_instrument` | Spec for "where to hook + how much to capture", decoupled from main task timing. v1 anchors: `func_entry` / `addr_first_exec` / `memregion_first_access` (all PC/address-only; runner-OS-event anchors deferred to runner-side milestone). Granularity ladder: `sparse_sample` / `pc_band` / `reg_delta` / `full_instruction`; auto-suggestion defaults to `full_instruction`. Anchor type registry is extensible — `register_anchor_type()` for future kinds. | `UTOV_PHASE_INSTRUMENT=off` |
| **phase_replay** | `engine.phase_replay` | Wrap a captured phase as a `ReplayableUnit` — JSONL trace in the same schema as the main trace (consumable by `JsonlTraceReader`) plus a sidecar carrying entry register state + memory snapshot + phase metadata. Sidecar is optional consumer information; stages needing only the instruction stream ignore it. | — |

Composition flow: `key_value → phase_discovery (locate phase) →
phase_instrument (full observation spec) → phase_replay (wrap unit)
→ JsonlTraceReader (main pipeline interface)`.

Wrapper plumbing: `DisciplineEnvelope` adds `phase_discovery` and
`phase_instrument_suggestions` sibling fields.
`DisciplineWrapper.__init__` accepts a
`phase_discovery_source_provider: (core, method, params) → DiscoveryDataSource | None`
factory; without a provider, the wrapper does not auto-walk producer
chains (the primitives remain reachable via direct calls). Engine
side ships specs + auto-suggestions only; runner-side fulfilment
(memory-watchpoint streaming + anchor install) lands in a follow-up
milestone — until then end-to-end runs use JSONL fixtures driving
`synthesize_unit_from_instructions`.

### Block-cause routing (job-chain alignment)

The reference target R10-R20 retro: L1 mechanical work was leaking out of the
py layer and being handed to the agent; the agent then escalated a
class-1 collection gap ("we don't have the bytes") to the user as a
three-way choice. Job-chain misaligned. Fix is a single L1 router
between `phase_discovery` and the envelope — clark owns class 1 and
class 2, only class 3 (true boundary) reaches the user, and only
with clark-prepared decision elements.

Three cause classes, five actions. Module: `engine.block_cause`.

| Class | Action | Owner |
|---|---|---|
| **collection_gap** + runner has capability | `auto_collect` (emit `RerunRequest` from the instrument spec) | clark, automatic |
| **collection_gap** + capability missing | `register_backlog` (append to `<run_dir>/capability_backlog.jsonl`) | clark, automatic |
| **recognition_gap** | `escalate_l2` (template fit / handler classification) | clark + L2 |
| **strategy_gap** | `escalate_l3` (paradigm switch / cross-node reasoning) | clark + L3 / agent |
| **true_boundary** | `escalate_user` (with `DecisionElements`) | clark prepares, user decides |

Phase intermediates hidden by default. When the router is wired,
the envelope no longer surfaces `phase_discovery` /
`phase_instrument_suggestions` — those are L1 intermediates fully
redundant with `block_cause` for routing decisions.
`UTOV_PHASE_DEBUG=1` restores them for troubleshooting;
`UTOV_BLOCK_CAUSE=off` falls back to the prior raw-surfacing path.

Capability declaration. `RunnerAdapter.CAPABILITIES: frozenset[str]`
is the engine-side static declaration; `metadata().capabilities`
(optional field on `TargetMeta`-shaped objects) lets a runner
opt-in additional capabilities at runtime — runtime is unioned
with static. Anchor → capability vocabulary lives in
`engine.block_cause.ANCHOR_TO_CAPABILITY` (3 v1 entries:
`func_entry_hook`, `pc_first_exec_hook`, `memregion_watch`).

Backlog file. `<run_dir>/capability_backlog.jsonl` —
append-only. Each line carries the missing capability name, the
triggering node, the discovery evidence, and the suggested
instrument spec the runner would need to fulfil. Developers can
`cat work/*/runs/*/capability_backlog.jsonl` to aggregate across
runs.

Auto-rerun is contract-only this milestone. The router emits
`RerunRequest(instrument_spec, triggered_by_node, reason)` on the
class-1 auto-collect path; actual rerun execution lands alongside
runner-side `PhaseInstrumentSpec` fulfilment.

### Constant-provenance classifier (M1/M3 unified generalisation)

The reference target prefix + template both fell into the session-level
constant trap because nothing automatically distinguished "the
bytes are the same every time we look in this session" from "the
bytes are computed once per session and reused". Manual M1 audit
rescued. Solidified into a deterministic capability.

Module: `engine.constant_provenance`. Two orthogonal probes:
**rerun variability** (four axes: same_session / new_session /
new_appkey / new_per_input — M3 generalised) and **producer
dataflow** (kinds of reads: static / appkey / time / random /
session_token / input / unknown — covers rerun probe's
entropy-locked-environment blindspot).

Five categories with evidence-class ceiling + recommended action:

| Category | Ceiling | Scope | Action |
|---|---|---|---|
| `hardcoded_fixed` | A | universal | `auto_pin` |
| `appkey_fixed_function` | A | per_appkey | `mark_dual_path` |
| `session_level_derived` | B | per_session | `escalate_usage_decision` |
| `per_input_variable` | — | per_input | `treat_as_variable` |
| `undetermined` | B | unspecified | `request_more_observations` |

Conflict rule. Dataflow reading any session-entropy source
(`time` / `random` / `session_token`) flips the verdict to
`session_level_derived` regardless of rerun stability — the rerun
probe alone is fooled by stable test environments. Stable reruns
without dataflow corroboration honestly downgrade to
`undetermined` (don't claim what you can't prove).

Wrapper plumbing. New envelope sibling `constant_provenance`
(list of verdicts, one per value record on the call that carried
`rerun_observations` and/or `producer_dataflow`). Independent
toggle `UTOV_CONSTANT_PROVENANCE=off`. Existing M1/M3/value_provenance
gates keep their narrower contracts; `constant_provenance` is the
unified primitive for callers that need the finer-grained verdict.

### Protocol

`contracts/agent_protocol.md` §3.3 — envelope adds the `methodology`
sibling; new RPCs `localize_divergence`, `invalidate_cluster`.

### Tests

378 passed / 1 pre-existing skip. New constant-provenance suites:
`test_constant_provenance` (18 — each category, dataflow override,
undetermined safe-default), `test_constant_provenance_wrapper`
(5 — envelope surface + toggle). Earlier this milestone:
block-cause suites:
`test_block_cause` (22 — classifier, oracle, backlog, router by
class), `test_block_cause_wrapper` (6 — envelope cleanup, debug
toggle, regression test for class-1 never surfacing to agent).
New phase-primitive suites:
`test_phase` (8), `test_phase_discovery` (8),
`test_phase_instrument` (12), `test_phase_replay` (6),
`test_phase_pipeline_e2e` (3 — two scenarios: callee-internal +
libload-period, plus target-agnostic capability check),
`test_phase_wrapper` (6). Earlier gate suites:
`test_m1_success_audit` (11), `test_m3_bypass_block` (20),
`test_value_provenance` (10), `test_watch_first_write` (13),
`test_length_chain` (13). Previously added: `test_capability`,
`test_cluster_cascade`, `test_evidence`, `test_invariants`,
`test_localize`, `test_manifest`, `test_methodology`, `test_numfmt`,
`test_trace_window`, `test_verdict`.

---

## v0.1.0-partial · `first-real-target-partial` (2026-05-28)

**First real-target partial recovery.**

### Pipeline (verified under pressure)

| Stage | Outcome |
|-------|---------|
| Wide native trace | ~**95k** insns → S4 slice **~2964** (~97% reduction) |
| S5 simplify | **207** instr → **0** logical stuck (189 symex + 8 cluster) |
| Pseudocode / vmtrace | 32-round main VMP · three-leg macro `0x39→0x54→0x22→0x2a` |
| I/O oracle | **93/93** |
| Triton symbolic gate | Handler + full-algorithm checks (symbolic domain) |
| R1 digest hook | **79/79** @ `0xb7bb0` protocol (+ offline `0x31c4b0`) |
| R2 SM3 body | **Suspended** — capability gap (no varying block at PLT memcpy) |

### Methodology validated

1. Lock entry/exit (I/O vs digest vs wrong export PC)
2. Ledger-first (invalidate `0x32350c` constant hook)
3. Backtrack on contradiction (reconciliation before R1)
4. Diff to localize (random_007 → SignRunner, not input)
5. Re-start at observation layer (R2 VMP enum → suspend, not tune `sm3_gmt`)

### Gaps handed to dev

- **M-R2 P0**: compress_leg / x22 observation + main-VMP trace window `0x32302c..708`
- See `work-tc3-samples/work/legacy/reference-target_partial_archive/handoff_to_dev.md`

### Archive

`work-tc3-samples/work/legacy/reference-target_partial_archive/final_archive.md`

---

## Prior

- P0+P1 engine: S1–S5, script_mode, agent_mode, hypothesis ledger, conformance gate (pre–real-target).
