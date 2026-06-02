"""Set-up symex primitive — the four contracts + dual-mode + guard-railed template.

These tests pin the contract behaviour the VMP cipher-body case (163511) forced
out: provenance-located boundaries (no assumed addresses), complete entry state,
symbol-preserving hybrid classification, mem[] backing detection, the
forward→backward mode switch, and the template's explicit agent checkpoints.
"""

from __future__ import annotations

import pytest

from engine.setup_symex import (
    AddressClosureReport,
    BoundaryEnd,
    BoundaryNotProvenanceLocated,
    BoundaryRole,
    Checkpoint,
    ConcreteBackingConflict,
    EntryStateSpec,
    IncompleteEntryState,
    LOCATED_ASSUMED,
    LOCATED_DFG,
    LOCATED_WATCH,
    OpacitySignals,
    SetupSymexConfig,
    SetupSymexDisabled,
    SymexMode,
    ParityVector,
    ParityVectorReport,
    audit_address_closure,
    bind_boundary,
    build_concrete_backing,
    build_setup_symex_plan,
    check_mem_backing,
    check_parity_vectors,
    check_seed_independence,
    classify_hybrid_step,
    emit_python,
    estimate_opacity,
    locate_boundary,
    pick_mode,
    seed_entry_state,
)
from engine.types import Instruction, MemOp, MemSnapshot


# --- helpers ----------------------------------------------------------------

def ins(idx, pc, mnem, *, reads=None, writes=None, mem=()):
    return Instruction(
        idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
        regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem),
    )


ON = SetupSymexConfig()  # enabled, default thresholds


# --- Contract 1: boundary via provenance ------------------------------------

def test_locate_boundary_builds_two_watch_specs():
    plan = locate_boundary(seed_hint_addr=0x1000, sink_hint_addr=0x2000, cfg=ON)
    assert plan.seed_watch.addr == 0x1000
    assert plan.sink_watch.addr == 0x2000
    d = plan.to_dict()
    assert d["kind"] == "setup_symex_boundary_plan"
    assert d["seed_watch"]["kind"] == "watch_first_write"


def test_bind_boundary_accepts_provenance_located():
    end = BoundaryEnd(BoundaryRole.SINK, 0xe4fff2e0, LOCATED_WATCH, "cipher")
    assert bind_boundary(end) is end
    assert bind_boundary(
        BoundaryEnd(BoundaryRole.SEED, 0x1234, LOCATED_DFG)
    ).located_via == LOCATED_DFG


def test_bind_boundary_rejects_assumed_address():
    # The case bound sink/pointer/input to assumed addresses three times — the
    # guardrail must refuse so the agent goes back to locate_boundary.
    bad = BoundaryEnd(BoundaryRole.SINK, 0x12316078, LOCATED_ASSUMED, "carrier")
    with pytest.raises(BoundaryNotProvenanceLocated):
        bind_boundary(bad)
    with pytest.raises(BoundaryNotProvenanceLocated):
        bind_boundary(BoundaryEnd(BoundaryRole.SEED, 0x1, "", ""))


# --- Contract 2: complete entry state ---------------------------------------

def test_seed_entry_state_symbolizes_full_reg_file_by_default():
    spec = seed_entry_state(
        entry_pc=0x6fdf0, reg_file=["x0", "x1", "x6", "x8"],
        pointed_buffers=[(0x1000, 16)], cfg=ON,
    )
    assert isinstance(spec, EntryStateSpec)
    assert spec.symbolic_regs == ("x0", "x1", "x6", "x8")
    assert spec.pointed_buffers == ((0x1000, 16),)


def test_seed_entry_state_subset_must_be_in_reg_file():
    seed_entry_state(entry_pc=0x10, reg_file=["x0", "x1"], symbolic_regs=["x0"], cfg=ON)
    with pytest.raises(IncompleteEntryState):
        seed_entry_state(entry_pc=0x10, reg_file=["x0"], symbolic_regs=["x9"], cfg=ON)


def test_seed_entry_state_rejects_empty_reg_file():
    with pytest.raises(IncompleteEntryState):
        seed_entry_state(entry_pc=0x10, reg_file=[], cfg=ON)


# --- Contract 3: symbol-preserving hybrid -----------------------------------

def test_hybrid_must_symbolize_when_reading_symbolic_reg():
    d = classify_hybrid_step(
        ins(5, 0x100, "add x2, x1, x3", reads={"x1": 7, "x3": 9}, writes={"x2": 16}),
        symbolic_regs=["x1"],
    )
    assert d.must_symbolize is True
    assert "x1" in d.reason


def test_hybrid_concrete_sync_when_no_symvar_touched():
    d = classify_hybrid_step(
        ins(6, 0x104, "add x2, x4, x5", reads={"x4": 1, "x5": 2}, writes={"x2": 3}),
        symbolic_regs=["x1"],
    )
    assert d.must_symbolize is False
    assert "concrete-sync is sound" in d.reason


def test_hybrid_load_at_symbolic_ea_must_symbolize():
    # The "decode-ok load reads back concrete 0" trap: even with no symbolic
    # register read, a load at a symbolic EA carries the dependency.
    d = classify_hybrid_step(
        ins(7, 0x108, "ldr w21, [x1, x9]", reads={"x1": 0, "x9": 0},
            mem=[MemOp("r", 0xdead00, 0x41, 4)]),
        symbolic_regs=["x7"], symbolic_addrs=[0xdead00],
    )
    assert d.must_symbolize is True
    assert "real EA" in d.reason


# --- Contract 4: mem[] backing ----------------------------------------------

def test_mem_backing_sufficient_when_loads_carry_operands():
    items = [
        ins(0, 0x6ff50, "ldr w21, [x1, x9]", mem=[MemOp("r", 0x100, 0x41, 4)]),
        ins(1, 0x6ff54, "str w21, [x2]", mem=[MemOp("w", 0x200, 0x41, 4)]),
        ins(2, 0x6ff58, "add x0, x0, 1"),  # not memory-class, ignored
    ]
    rep = check_mem_backing(items, window=(0x6ff50, 0x6ffa4))
    assert rep.sufficient is True
    assert rep.mem_class_steps == 2 and rep.backed_steps == 2
    assert rep.backing_rate == 1.0


def test_mem_backing_blind_leg_detected():
    # The staging ring load with NO mem[] = the 57/65 degradation root.
    items = [
        ins(0, 0x6ff50, "ldr w21, [x1, x9]"),  # empty mem[] -> blind
        ins(1, 0x6ff54, "str w21, [x2]", mem=[MemOp("w", 0x200, 0x41, 4)]),
    ]
    rep = check_mem_backing(items, window=(0x6ff50, 0x6ffa4))
    assert rep.sufficient is False
    assert rep.blind_pcs == (0x6ff50,)
    assert "BLIND" in rep.advisory


def test_mem_backing_ignores_outside_window():
    items = [ins(0, 0x9999, "ldr w0, [x1]")]  # no mem[], but outside window
    rep = check_mem_backing(items, window=(0x1000, 0x2000))
    assert rep.mem_class_steps == 0 and rep.sufficient is True


# --- Dual mode: forward vs backward-alias -----------------------------------

def test_pick_mode_stays_forward_when_propagated():
    sig = OpacitySignals(sym_propagated=True)
    dec = pick_mode(sig, cfg=ON)
    assert dec.mode is SymexMode.FORWARD


def test_pick_mode_switches_backward_on_dispatch():
    # forward didn't reach the sink AND the path is opaque (dispatch) -> switch.
    sig = OpacitySignals(sym_propagated=False, slice_steps=2000,
                         indirect_branch_density=0.30)
    dec = pick_mode(sig, cfg=ON)
    assert dec.mode is SymexMode.BACKWARD_ALIAS
    assert "opaque" in dec.reason


def test_pick_mode_switches_backward_on_concrete_overwrite():
    sig = OpacitySignals(sym_propagated=False, concrete_overwrite_rate=0.80)
    assert pick_mode(sig, cfg=ON).mode is SymexMode.BACKWARD_ALIAS


def test_pick_mode_stays_forward_when_not_opaque():
    # not propagated yet, but no opacity signal crossed -> keep trying forward.
    sig = OpacitySignals(sym_propagated=False, slice_steps=10,
                         indirect_branch_density=0.0)
    assert pick_mode(sig, cfg=ON).mode is SymexMode.FORWARD


def test_estimate_opacity_counts_indirect_branches():
    items = [
        ins(0, 0x10, "br x8"),
        ins(1, 0x14, "blr x9"),
        ins(2, 0x18, "add x0, x0, 1"),
        ins(3, 0x1c, "ret"),
    ]
    sig = estimate_opacity(items, sym_propagated=False)
    assert sig.slice_steps == 4
    assert sig.indirect_branch_density == pytest.approx(0.5)


# --- emit -------------------------------------------------------------------

def test_emit_python_carries_hard_parity_gate():
    intent = emit_python(mode=SymexMode.BACKWARD_ALIAS,
                         expr_source="cipher = alias_map(seed)",
                         inputs=["nonce", "dm", "biz", "key"], parity_min=8)
    assert intent.parity_min == 8
    assert intent.to_dict()["kind"] == "setup_symex_emit"


def test_emit_python_rejects_empty_expr():
    with pytest.raises(ValueError):
        emit_python(mode=SymexMode.FORWARD, expr_source="   ", inputs=[])


# --- The guard-railed template ----------------------------------------------

def test_template_orders_contracts_and_surfaces_checkpoints():
    plan = build_setup_symex_plan()
    orders = [s.order for s in plan.steps]
    assert orders == sorted(orders)  # strictly ordered skeleton
    names = [s.name for s in plan.steps]
    # the four contracts + dual-mode are all present as guardrail steps
    assert "locate_boundary" in names and "seed_entry_state" in names
    assert "pick_mode" in names and "check_mem_backing" in names
    assert "classify_hybrid_steps" in names and "emit_python" in names


def test_template_checkpoints_are_the_two_real_judgments():
    plan = build_setup_symex_plan()
    cps = {c.name for c in plan.checkpoints}
    assert cps == {"alias_vs_compute", "which_static"}
    for c in plan.checkpoints:
        assert isinstance(c, Checkpoint)
        assert c.to_dict()["is_judgment"] is True


def test_template_guardrail_steps_are_not_judgments():
    plan = build_setup_symex_plan()
    guardrails = [s for s in plan.steps if not s.is_judgment]
    # guardrail steps carry an auto-enforced rule, no checkpoint
    assert all(s.guardrail and s.checkpoint is None for s in guardrails)
    assert any("determinism" in plan.determinism_note.lower() for _ in [0])


def test_template_serializes_with_checkpoints():
    d = build_setup_symex_plan().to_dict()
    assert d["kind"] == "setup_symex_plan"
    assert len(d["checkpoints"]) == 2
    assert d["determinism_note"]


# --- env toggle -------------------------------------------------------------

def test_disabled_toggle_blocks_primitive():
    off = SetupSymexConfig.from_env({"UTOV_SETUP_SYMEX": "off"})
    assert off.enabled is False
    with pytest.raises(SetupSymexDisabled):
        locate_boundary(seed_hint_addr=0x1, sink_hint_addr=0x2, cfg=off)
    with pytest.raises(SetupSymexDisabled):
        seed_entry_state(entry_pc=0x1, reg_file=["x0"], cfg=off)


# --- Contract 2 backing arm: concrete backing of the address closure ---------
#
# Fixture shape: hook 20260531-161157-hash2236-regs, key EA @0x2236b4 of the
# cipher-body hash loop (case 163511). The loop's loads read hash state / table
# via base registers x20/x24/x25; their concrete values come from the snapshot:
#   x20=0x1230ed40  x24=0x13082500  x25=0x130825a0  x27=0xffffffb5
# (target-specific values stay here in the test, never in the primitive.)

_HASH_REGS = {"x20": 0x1230ED40, "x24": 0x13082500, "x25": 0x130825A0}


def _hash_loop():
    """The hash loop window 0x2236b4..e0 — loads off base regs x20/x24/x25.

    Each load DOES carry a mem[] operand (so check_mem_backing passes), yet the
    base registers are live-in to the window and were never injected as concrete
    backing — the exact 0/3 → 2/2 gap the spec calls out."""
    return [
        ins(0, 0x2236B4, "ldrb w0, [x24, 0x2a]", reads={"x24": 0x13082500},
            writes={"w0": 0x41}, mem=[MemOp("r", 0x1308252A, 0x41, 1)]),
        ins(1, 0x2236B8, "ldr w1, [x20]", reads={"x20": 0x1230ED40},
            writes={"w1": 0x7C}, mem=[MemOp("r", 0x1230ED40, 0x7C, 4)]),
        ins(2, 0x2236BC, "ldr x2, [x25]", reads={"x25": 0x130825A0},
            writes={"x2": 0x99}, mem=[MemOp("r", 0x130825A0, 0x99, 8)]),
        ins(3, 0x2236C0, "mul w0, w0, w27", reads={"w0": 0x41, "w27": 0xFFFFFFB5},
            writes={"w0": 0x68}),
    ]


def test_build_concrete_backing_exposes_backed_regs_and_addrs():
    backing = build_concrete_backing(
        reg_values=_HASH_REGS,
        mem=[MemSnapshot(addr=0x13082500, data=b"\x00" * 0x40, label="table")],
    )
    assert backing.backed_regs == frozenset({"x20", "x24", "x25"})
    assert 0x13082500 in backing.backed_addrs
    assert 0x1308253F in backing.backed_addrs       # last byte of the 0x40 region
    assert backing.to_dict()["kind"] == "setup_symex_concrete_backing"


def test_seed_entry_state_attaches_concrete_backing():
    backing = build_concrete_backing(reg_values=_HASH_REGS)
    spec = seed_entry_state(
        entry_pc=0x2236A4, reg_file=["x0", "x20", "x24", "x25", "x27"],
        symbolic_regs=["x0"], concrete_backing=backing,
    )
    assert spec.concrete_backing is backing
    assert spec.to_dict()["concrete_backing"]["reg_values"]["x24"] == "0x13082500"
    assert "concrete backing injected" in spec.note


def test_entry_dict_internalizes_concrete_regs_and_mem_for_runner():
    # The backing flow is internalized: the entry dict the runner consumes carries
    # the pinned base values AND the bytes the pointed region holds — so the runner
    # upfront-seeds Triton without the agent injecting a decoder/hook in pipeline.
    backing = build_concrete_backing(
        reg_values={"x24": 0x13065EA0},
        mem=[MemSnapshot(addr=0x13065F48, data=b"\xe9\x93\x87\x38", label="slot")],
    )
    spec = seed_entry_state(
        entry_pc=0x13065F00, reg_file=["x8", "x24"],
        symbolic_regs=["x8"], concrete_backing=backing,
    )
    d = spec.to_dict()
    # concrete_regs: ints (the runner's reset does int(v)), the pinned base only.
    assert d["concrete_regs"] == {"x24": 0x13065EA0}
    # concrete_mem: the region bytes the runner injects via setConcreteMemoryAreaValue.
    assert d["concrete_mem"] == [
        {"addr": 0x13065F48, "addr_hex": "0x13065f48", "size": 4,
         "data_hex": "e9938738"}]


def test_entry_dict_has_empty_concrete_arms_without_backing():
    # No backing → empty concrete arms (idempotent; the symbolize-only default).
    spec = seed_entry_state(entry_pc=0x10, reg_file=["x0", "x1"], symbolic_regs=["x0"])
    d = spec.to_dict()
    assert d["concrete_regs"] == {} and d["concrete_mem"] == []


def test_seed_entry_state_rejects_symbolic_and_pinned_same_reg():
    # x24 cannot be BOTH the symbolic input and a concretely pinned pointer.
    backing = build_concrete_backing(reg_values={"x24": 0x13082500})
    with pytest.raises(ConcreteBackingConflict):
        seed_entry_state(
            entry_pc=0x2236A4, reg_file=["x0", "x24"],
            symbolic_regs=["x0", "x24"], concrete_backing=backing,
        )


# --- Contract 4 backing arm: address-computation closure audit ---------------

def test_address_closure_blind_when_base_regs_unbacked():
    # The killer contrast: mem[] backing PASSES (every load carries an operand)
    # yet the closure is BLIND because x20/x24/x25 have no concrete value.
    items = _hash_loop()
    mem_rep = check_mem_backing(items, window=(0x2236B4, 0x2236E0))
    assert mem_rep.sufficient is True and mem_rep.backing_rate == 1.0

    rep = audit_address_closure(items, window=(0x2236B4, 0x2236E0))
    assert isinstance(rep, AddressClosureReport)
    assert rep.sufficient is False
    assert set(rep.unbacked_roots) == {"x20", "x24", "x25"}
    assert set(rep.blind_pcs) == {0x2236B4, 0x2236B8, 0x2236BC}
    assert "BLIND" in rep.advisory


def test_address_closure_backed_when_closure_injected():
    # Inject the snapshot's concrete base-register values -> closure resolves.
    items = _hash_loop()
    backing = build_concrete_backing(reg_values=_HASH_REGS)
    rep = audit_address_closure(items, window=(0x2236B4, 0x2236E0), backing=backing)
    assert rep.sufficient is True
    assert rep.unbacked_roots == ()
    assert rep.blind_pcs == ()
    assert "backed" in rep.advisory


def test_address_closure_recurses_to_base_registers():
    # EA base x24 is COMPUTED in-window (add x24, x20, x6) — the closure must
    # recurse past x24 to its live-in roots x20 + x6, not stop at x24.
    items = [
        ins(0, 0x2236A8, "add x24, x20, x6", reads={"x20": 0x1230ED40, "x6": 0x2A},
            writes={"x24": 0x13082500}),
        ins(1, 0x2236B4, "ldrb w0, [x24, 0x2a]", reads={"x24": 0x13082500},
            writes={"w0": 0x41}, mem=[MemOp("r", 0x1308252A, 0x41, 1)]),
    ]
    rep = audit_address_closure(items, window=(0x2236A8, 0x2236E0))
    assert rep.unbacked_roots == ("x20", "x6")     # recursed past the computed x24
    # backing the live-in roots (not x24) closes it
    backed = audit_address_closure(
        items, window=(0x2236A8, 0x2236E0),
        backed_regs={"x20", "x6"},
    )
    assert backed.sufficient is True


def test_address_closure_flags_chained_load_with_no_pointed_memory():
    # Base x24 is itself loaded from [x25] with NO mem[] -> both the pointed
    # bytes (mem@pc) and the deeper pointer x25 are blind roots.
    items = [
        ins(0, 0x2236A8, "ldr x24, [x25]", reads={"x25": 0x130825A0},
            writes={"x24": 0x13082500}),                       # empty mem[] -> blind
        ins(1, 0x2236B4, "ldrb w0, [x24, 0x2a]", reads={"x24": 0x13082500},
            writes={"w0": 0x41}, mem=[MemOp("r", 0x1308252A, 0x41, 1)]),
    ]
    rep = audit_address_closure(items, window=(0x2236A8, 0x2236E0))
    assert rep.sufficient is False
    assert "x25" in rep.unbacked_roots
    assert any(r.startswith("mem@") for r in rep.unbacked_roots)


def test_seed_backing_feeds_address_closure_audit():
    # ② provides the backing, ① consumes it: the contract-2 arm closes the
    # contract-4 closure. End-to-end of the spec's 0/3 -> 2/2.
    backing = build_concrete_backing(reg_values=_HASH_REGS)
    spec = seed_entry_state(
        entry_pc=0x2236A4, reg_file=["x0", "x20", "x24", "x25", "x27"],
        symbolic_regs=["x0"], concrete_backing=backing,
    )
    rep = audit_address_closure(
        _hash_loop(), window=(0x2236B4, 0x2236E0),
        backing=spec.concrete_backing,
    )
    assert rep.sufficient is True


# --- C4 unified backing criterion (check_mem_backing == audit_address_closure) ---
#
# Fixture shape: F0 staging window 0xe4fff200 (run 164919-f0aligned). A reg-trace
# runner that does NOT record store EAs, so every memory step carries an EMPTY
# mem[]; backing for the base regs (x16/x20) comes from a same-execution
# snapshot. Pre-fix, audit_address_closure judged it backed while
# check_mem_backing's mem[]-only rate stayed ~0% → false-fail blocking emit.

_F0_EXEC = "libEncryptor.so|f0aligned|run-164919"


def _reg_trace_window():
    """F0 staging — loads/stores with empty mem[]; base regs are live-in."""
    return [
        ins(0, 0xE4FFF200, "ldr w0, [x16]", reads={"x16": 0x12340000}),
        ins(1, 0xE4FFF204, "str w0, [x20, 0x8]", reads={"x20": 0x12350000, "w0": 0}),
        ins(2, 0xE4FFF208, "ldr w1, [x16, 0x4]", reads={"x16": 0x12340000}),
    ]


def test_check_mem_backing_agrees_with_closure_under_snapshot():
    win = (0xE4FFF200, 0xE4FFF2FF)
    items = _reg_trace_window()
    # No backing: the trace carries no mem[] -> both checks judge blind. Their
    # agreement here is the baseline; the gap was disagreement under a snapshot.
    assert check_mem_backing(items, window=win).sufficient is False
    assert audit_address_closure(items, window=win).sufficient is False

    # Same-execution snapshot backs the base regs -> BOTH now judge backed (the
    # unify fix: check_mem_backing no longer false-fails on missing mem[]).
    backing = build_concrete_backing(
        reg_values={"x16": 0x12340000, "x20": 0x12350000}, exec_id=_F0_EXEC)
    mem = check_mem_backing(items, window=win, backing=backing, trace_exec_id=_F0_EXEC)
    clo = audit_address_closure(items, window=win, backing=backing, trace_exec_id=_F0_EXEC)
    assert mem.sufficient is True and clo.sufficient is True       # consistent verdict
    assert mem.snapshot_backed_steps == 3                          # all backed via snapshot
    assert mem.backing_rate == 1.0
    assert "snapshot/hook backing" in mem.advisory


def test_determinism_guard_rejects_cross_execution_snapshot():
    win = (0xE4FFF200, 0xE4FFF2FF)
    items = _reg_trace_window()
    stale = build_concrete_backing(
        reg_values={"x16": 0x12340000, "x20": 0x12350000}, exec_id="OTHER-RUN")
    # The trace asserts its own exec id; the snapshot is from a different run.
    # Guard: a cross-execution snapshot must NOT mask the blind legs (no false-pass).
    mem = check_mem_backing(items, window=win, backing=stale, trace_exec_id=_F0_EXEC)
    clo = audit_address_closure(items, window=win, backing=stale, trace_exec_id=_F0_EXEC)
    assert mem.sufficient is False and clo.sufficient is False
    assert mem.snapshot_backed_steps == 0
    # Unscoped (caller asserts no trace_exec_id) -> legacy behaviour, counts.
    assert check_mem_backing(items, window=win, backing=stale).sufficient is True


# ---------------------------------------------------------------------------
# Per-handler / window MULTI-VECTOR parity (cross-run) — the gate that refuses
# to stamp a transform EXACT off a tautological 1/1 (handler10 lesson).
# ---------------------------------------------------------------------------


def _vec(key, *, ok=True, exec_id=None, derived=False):
    """A parity vector that matches (ok) or not, with optional execution id.

    ``observed`` is keyed off ``input_key`` so a cohort of distinct inputs has
    distinct observed outputs — a REAL recovery's output varies with its input
    (the observed-variance gate's premise). A matching vector predicts that same
    per-input output; a non-matching one predicts a fixed wrong value."""
    observed = f"out-{key}"
    return ParityVector(input_key=key, observed=observed,
                        predicted=(observed if ok else "WRONG"),
                        exec_id=exec_id, derived_from=derived)


def test_parity_block_on_tautological_single_vector():
    # The whole bug: one vector, and it IS the trace the transform was derived
    # from. counted=0 -> nothing independent confirms it -> BLOCK, never exact.
    r = check_parity_vectors([_vec("carrier-A", derived=True)],
                             window=(0x1000, 0x10FF), min_vectors=3)
    assert isinstance(r, ParityVectorReport)
    assert r.verdict == "BLOCK" and r.sufficient is False
    assert r.counted == 0 and r.independent_pass == 0
    assert "tautological" in " ".join(r.reasons)


def test_parity_unclosable_below_independent_floor():
    # Two genuinely distinct passing vectors, floor is 3. The independent side
    # carries only 2 distinct observed outputs (< min_vectors), so NO F can
    # EXACT-close: this is the cohort being too shallow (need output-diverse
    # seeds), judged FIRST as UNCLOSABLE — not an F-error BLOCK. (F0 @1769: a
    # distinct=2 independent side under min=3 must not pass.)
    r = check_parity_vectors([_vec("A", exec_id="r1"), _vec("B", exec_id="r2")],
                             window=(0x1000, 0x10FF), min_vectors=3)
    assert r.verdict == "UNCLOSABLE" and r.sufficient is False
    assert r.independent_pass == 2 and r.min_vectors == 3
    assert r.observed_distinct == 2 and r.to_dict()["independent_observed_distinct"] == 2
    assert "no F can EXACT-close" in " ".join(r.reasons)
    assert "fix the cohort" in r.advisory


def test_parity_exact_with_enough_independent_cross_run_vectors():
    vs = [_vec("A", exec_id="r1"), _vec("B", exec_id="r2"), _vec("C", exec_id="r3")]
    r = check_parity_vectors(vs, window=(0x1000, 0x10FF), min_vectors=3)
    assert r.verdict == "EXACT" and r.sufficient is True
    assert r.independent_pass == 3 and r.determinism_ok and r.determinism_seen
    assert "holds beyond the trace" in r.advisory


def test_parity_unclosable_when_observed_constant_across_varying_inputs():
    # The F0 @1769 shape: input varied (3 distinct counted vectors, predicted all
    # match), but the INDEPENDENT-side observed COLLAPSES to one constant value ->
    # observed_distinct=1 < min=3 -> UNCLOSABLE, judged FIRST and INDEPENDENTLY of
    # the (fully passing) match floor. No F closes a collapsed independent side: it
    # is a COHORT defect, not an F defect. This is the merged successor of the old
    # DEGENERATE (which was distinct==1; now a strict subset of distinct<min).
    vs = [ParityVector(input_key=k, observed="C", predicted="C", exec_id=e)
          for k, e in (("A", "r1"), ("B", "r2"), ("C", "r3"))]
    r = check_parity_vectors(vs, window=(0x1000, 0x10FF), min_vectors=3)
    assert r.verdict == "UNCLOSABLE"
    assert r.sufficient is False              # UNCLOSABLE does NOT close
    assert r.observed_distinct == 1
    assert r.independent_pass == 3 and r.determinism_ok    # predicted DID all match
    assert "no F can EXACT-close" in " ".join(r.reasons)
    assert "fix the cohort" in r.advisory
    assert r.to_dict()["observed_distinct"] == 1
    assert r.to_dict()["independent_observed_distinct"] == 1
    assert r.to_dict()["verdict"] == "UNCLOSABLE"


def test_parity_unclosable_judged_independently_of_mismatch():
    # Cohort collapsed AND predicted is wrong: closability is judged FIRST, so the
    # verdict is UNCLOSABLE (fix the cohort), not BLOCK (white-tune F). The reason
    # tells the consumer the cohort is the problem before any F reason.
    vs = [ParityVector(input_key=k, observed="C", predicted="WRONG", exec_id=e)
          for k, e in (("A", "r1"), ("B", "r2"), ("C", "r3"))]
    r = check_parity_vectors(vs, window=(0x1000, 0x10FF), min_vectors=3)
    assert r.verdict == "UNCLOSABLE"
    assert r.observed_distinct == 1
    assert "no F can EXACT-close" in r.reasons[0]   # cohort reason comes FIRST


def test_parity_constant_observed_two_vectors_is_unclosable():
    # Observed is constant AND only 2 vectors -> independent side collapses to 1
    # distinct (< min=3) -> UNCLOSABLE (the cohort is too shallow / collapsed), not
    # a half-floor BLOCK: the gate is the OUTPUT-side closability check.
    vs = [ParityVector(input_key="A", observed="C", predicted="C", exec_id="r1"),
          ParityVector(input_key="B", observed="C", predicted="C", exec_id="r2")]
    r = check_parity_vectors(vs, window=(0x1000, 0x10FF), min_vectors=3)
    assert r.verdict == "UNCLOSABLE"
    assert r.independent_pass == 2 and r.observed_distinct == 1


def test_parity_repeated_input_is_not_independent():
    # The same input replayed cannot stand in for an independent vector: counted=1,
    # so the independent side carries 1 distinct observed (< min=3) -> UNCLOSABLE
    # (the cohort never supplied independent diversity).
    vs = [_vec("A", exec_id="r1"), _vec("A", exec_id="r2"), _vec("A", exec_id="r3")]
    r = check_parity_vectors(vs, window=(0x1000, 0x10FF), min_vectors=3)
    assert r.verdict == "UNCLOSABLE"
    assert r.counted == 1 and r.independent_pass == 1 and r.observed_distinct == 1


def test_parity_determinism_rejects_mixed_execution():
    # Three distinct passing inputs, but two share one execution -> one run's
    # observed output was reused for another input (cross-run mixing) -> BLOCK.
    vs = [_vec("A", exec_id="r1"), _vec("B", exec_id="r1"), _vec("C", exec_id="r2")]
    r = check_parity_vectors(vs, window=(0x1000, 0x10FF), min_vectors=3)
    assert r.verdict == "BLOCK" and r.determinism_ok is False
    assert "cross-run mixing" in r.advisory


def test_parity_trace_exec_id_excludes_the_deriving_vector():
    # A vector from the deriving execution is tautological even without the flag:
    # DERIV is dropped, so only B, C count (2 distinct observed < min=3). The
    # excluded deriving vector means the INDEPENDENT side is too shallow to close
    # -> UNCLOSABLE (the exclusion is the whole point; the verdict reflects the
    # cohort, post-exclusion, being below the diversity floor).
    vs = [_vec("A", exec_id="DERIV"), _vec("B", exec_id="r2"), _vec("C", exec_id="r3")]
    r = check_parity_vectors(vs, window=(0x1000, 0x10FF), min_vectors=3,
                             trace_exec_id="DERIV")
    assert r.verdict == "UNCLOSABLE"      # only B, C count -> 2 distinct < 3
    assert r.counted == 2 and r.independent_pass == 2 and r.observed_distinct == 2


def test_parity_trace_exec_id_excluded_then_diverse_cohort_exact():
    # Same exclusion, but the independent side is genuinely diverse (B, C, D = 3
    # distinct observed, all match) -> EXACT. Confirms exclusion + a deep-enough
    # cohort still closes (invariant 7: the EXACT path is unchanged when the
    # independent side really carries >= min distinct outputs).
    vs = [_vec("A", exec_id="DERIV"), _vec("B", exec_id="r2"),
          _vec("C", exec_id="r3"), _vec("D", exec_id="r4")]
    r = check_parity_vectors(vs, window=(0x1000, 0x10FF), min_vectors=3,
                             trace_exec_id="DERIV")
    assert r.verdict == "EXACT" and r.sufficient is True
    assert r.counted == 3 and r.independent_pass == 3 and r.observed_distinct == 3


def test_parity_f0_1769_independent_side_collapses_is_unclosable():
    # ACCEPTANCE — the live F0 @1769 shape: 4 vectors, the main/deriving trace plus
    # 3 INDEPENDENT ones whose observed ALL collapse to the same constant
    # (0x745ee0f5). The full cohort distinct=2 (the main differs) would sneak past a
    # naive ">1" gate, but the INDEPENDENT side distinct=1 < min=3 → UNCLOSABLE.
    # Even a perfectly matching F cannot close it; the fix is the cohort (the 4 seeds
    # do not diverge the output at this staging point), not F.
    vs = [
        ParityVector(input_key="main", observed="0xdeadbeef", predicted="0xdeadbeef",
                     exec_id="MAIN", derived_from=True),       # the deriving trace
        ParityVector(input_key="s1", observed="0x745ee0f5", predicted="0x745ee0f5",
                     exec_id="r1"),
        ParityVector(input_key="s2", observed="0x745ee0f5", predicted="0x745ee0f5",
                     exec_id="r2"),
        ParityVector(input_key="s3", observed="0x745ee0f5", predicted="0x745ee0f5",
                     exec_id="r3"),
    ]
    r = check_parity_vectors(vs, window=(0x1769, 0x1851), min_vectors=3,
                             trace_exec_id="MAIN")
    assert r.verdict == "UNCLOSABLE"
    # independent side (main excluded): 3 vectors, all 0x745ee0f5 → distinct=1.
    assert r.counted == 3 and r.independent_pass == 3
    assert r.observed_distinct == 1
    assert r.to_dict()["independent_observed_distinct"] == 1
    assert "no F can EXACT-close" in " ".join(r.reasons)


def test_parity_stands_down_with_no_evidenced_observed():
    # Invariant 8 stand-down: counted vectors carry NO exec_id (the scalar m/n gold
    # FALLBACK shape — observed is a placeholder, not a real oracle reading). With no
    # evidenced observed there is NO real signal → the gate is SILENT: it does NOT
    # fabricate an UNCLOSABLE from a placeholder. EXACT/BLOCK byte-for-byte as before
    # (invariant 7) — here matched + floor met with no exec_id → EXACT.
    vs = [ParityVector(input_key=k, observed="P", predicted="P")  # no exec_id
          for k in ("A", "B", "C")]
    r = check_parity_vectors(vs, window=(0x1000, 0x10FF), min_vectors=3)
    assert r.observed_distinct == 0          # no evidenced vectors → no real signal
    assert r.verdict == "EXACT"              # gate stood down (invariant 7/8)
    assert r.sufficient is True


def test_parity_report_records_per_vector_detail():
    # #1: every supplied vector gets {input_key, observed, predicted, matches} so the
    # agent can see the "F wrong" shape (supplied >= need but matched < need) per row.
    vs = [_vec("A", ok=True,  exec_id="r1"),
          _vec("B", ok=False, exec_id="r2"),
          _vec("C", ok=True,  exec_id="r3")]
    r = check_parity_vectors(vs, window=(0x1000, 0x10FF), min_vectors=3)
    assert len(r.vectors) == 3
    by_key = {v["input_key"]: v for v in r.vectors}
    assert by_key["A"]["matches"] is True
    assert by_key["B"]["matches"] is False
    assert by_key["B"]["observed"] == "out-B" and by_key["B"]["predicted"] == "WRONG"
    # matches == (observed == predicted), the real comparison (invariant 8).
    assert all(v["matches"] == (v["observed"] == v["predicted"]) for v in r.vectors)
    d = r.to_dict()
    assert d["vectors"] == [dict(v) for v in r.vectors]


def test_parity_report_vectors_capped_in_to_dict():
    # Invariant 4: a wide cohort's per-vector detail is capped to a sample + count.
    vs = [_vec(f"k{i}", exec_id=f"r{i}") for i in range(20)]
    r = check_parity_vectors(vs, window=(0x1000, 0x10FF), min_vectors=3)
    assert len(r.vectors) == 20             # full detail kept on the object
    d = r.to_dict()
    assert d["vectors"]["_trimmed_list"] is True
    assert d["vectors"]["count"] == 20
    assert len(d["vectors"]["sample"]) == 8


# --- handler10 fixture: incomplete vs complete window transform ---------------
#
# handler10's transform stopped at idx107 and dropped the idx107->idx113 x8
# update. It matched the trace it was derived from (input A) but diverges on any
# other carrier. Modelled abstractly: the OUTPUT carries that x8 update or not.

_H10_INPUTS = (("A", "DERIV"), ("B", "run-B"), ("C", "run-C"), ("D", "run-D"))


def _h10_observed(x):
    """The oracle (complete transform): includes the dropped x8 update."""
    return f"{x}-x8"


def _h10_incomplete(x):
    """handler10 as recovered: stops early, agrees with the oracle ONLY on the
    deriving input A (which is why 1/1 passed)."""
    return f"{x}-x8" if x == "A" else f"{x}-stop107"


def _h10_complete(x):
    """The transform after the window is completed to idx113."""
    return f"{x}-x8"


def _h10_vectors(transform):
    return [ParityVector(input_key=x, observed=_h10_observed(x),
                        predicted=transform(x), exec_id=ex,
                        derived_from=(x == "A"))
            for x, ex in _H10_INPUTS]


def test_handler10_incomplete_transform_blocks_under_multivector():
    # ACCEPTANCE: the incomplete transform that passed 1/1 must BLOCK now.
    r = check_parity_vectors(_h10_vectors(_h10_incomplete),
                             window=(0x953b89cc, 0xfb9881b1), min_vectors=3)
    assert r.verdict == "BLOCK" and r.sufficient is False
    # A is excluded (deriving); B/C/D all mismatch -> 0 independent passes.
    assert r.independent_pass == 0
    assert set(r.mismatches) == {"B", "C", "D"}
    assert "wrong or incomplete" in " ".join(r.reasons)


def test_handler10_completed_transform_passes_under_multivector():
    # ACCEPTANCE: completing the window (to idx113) makes it pass on every
    # independent cross-run vector.
    r = check_parity_vectors(_h10_vectors(_h10_complete),
                             window=(0x953b89cc, 0xfb9881b1), min_vectors=3)
    assert r.verdict == "EXACT" and r.sufficient is True
    assert r.independent_pass == 3 and r.determinism_ok   # B, C, D (A excluded)


# ---------------------------------------------------------------------------
# Memory arm of the live-in — external memory inputs (handler11 symbolic=0 fix).
# ---------------------------------------------------------------------------

from engine.setup_symex import (  # noqa: E402
    MemLiveIn,
    derive_window_mem_live_in,
    derive_window_symbolic_regs,
)


def _insm(idx, pc, mnem, *, mem=()):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read={}, regs_write={}, mem=tuple(mem))


def test_mem_live_in_detects_external_load_excludes_internal():
    items = [
        _insm(0, 0x1000, "ldr x0, [x9]",  mem=[MemOp("r", 0x9000, 0, 8)]),   # external
        _insm(1, 0x1004, "str x0, [x10]", mem=[MemOp("w", 0xA000, 0, 8)]),
        _insm(2, 0x1008, "ldr x1, [x10]", mem=[MemOp("r", 0xA000, 0, 8)]),   # internal
    ]
    ml, info = derive_window_mem_live_in(items, window=(0x1000, 0x10FF))
    assert len(ml) == 1 and isinstance(ml[0], MemLiveIn)
    assert ml[0].addr == 0x9000 and ml[0].size == 8 and ml[0].src_idx == 0
    assert info["empty"] is False and info["kind"] == "setup_symex_mem_live_in"


def test_mem_live_in_partial_write_leaves_external_run():
    # bytes 0x9000..0x9003 written in-window; 0x9004..0x9007 external -> one run.
    items = [
        _insm(0, 0x1000, "str w0, [x9]",      mem=[MemOp("w", 0x9000, 0, 4)]),
        _insm(1, 0x1004, "ldr x1, [x9]",      mem=[MemOp("r", 0x9000, 0, 8)]),
    ]
    ml, _ = derive_window_mem_live_in(items, window=(0x1000, 0x10FF))
    assert [(m.addr, m.size) for m in ml] == [(0x9004, 4)]


def test_mem_live_in_empty_when_no_loads():
    ml, info = derive_window_mem_live_in([_insm(0, 0x1000, "add x0, x1, x2")],
                                         window=(0x1000, 0x10FF))
    assert ml == () and info["empty"] is True


def test_mem_live_in_idx_window():
    items = [
        _insm(5, 0x2000, "ldr x0, [x9]", mem=[MemOp("r", 0x9000, 0, 8)]),    # idx in
        _insm(9, 0x2008, "ldr x1, [x9]", mem=[MemOp("r", 0x8000, 0, 8)]),    # idx out
    ]
    ml, _ = derive_window_mem_live_in(items, window=(5, 6), window_is_idx=True)
    assert [m.addr for m in ml] == [0x9000]


# --- idx vs equivalent pc window: identical live-in / backing / closure -------
# 验收 item 3: for the SAME items, an idx window and a pc window that bound the
# same instruction subset must give identical live-in / backing / closure / mem
# selections. (The drive threads ONE window_kind to every step; this proves the
# two bases agree when they select the same steps — no off-by-basis divergence.)

def _eq_items():
    # idx == execution order; pcs chosen so (pc 0x1004..0x1008) selects EXACTLY
    # the same two steps as (idx 1..2). idx 0 and idx 3 are outside both windows.
    return [
        ins(0, 0x1000, "mov x9, x0", reads={"x0": 0}, writes={"x9": 0}),
        ins(1, 0x1004, "ldr x1, [x9]", reads={"x9": 0}, writes={"x1": 0},
            mem=[MemOp("r", 0x9000, 0, 8)]),
        ins(2, 0x1008, "add x2, x1, x4", reads={"x1": 0, "x4": 0}, writes={"x2": 0}),
        ins(3, 0x100C, "ldr x3, [x20]", reads={"x20": 0}, writes={"x3": 0},
            mem=[MemOp("r", 0x8000, 0, 8)]),
    ]


def test_idx_window_matches_equivalent_pc_window():
    items = _eq_items()
    idx_win, pc_win = (1, 2), (0x1004, 0x1008)

    # live-in registers
    li_idx, info_idx = derive_window_symbolic_regs(
        items, window=idx_win, window_is_idx=True)
    li_pc, info_pc = derive_window_symbolic_regs(items, window=pc_win)
    assert li_idx == li_pc
    assert info_idx["window_basis"] == "idx" and info_pc["window_basis"] == "pc"
    assert info_idx["n_window_items"] == info_pc["n_window_items"] == 2

    # memory live-in
    ml_idx, _ = derive_window_mem_live_in(items, window=idx_win, window_is_idx=True)
    ml_pc, _ = derive_window_mem_live_in(items, window=pc_win)
    assert [m.addr for m in ml_idx] == [m.addr for m in ml_pc] == [0x9000]

    # backing + closure
    back_idx = check_mem_backing(items, window=idx_win, window_is_idx=True)
    back_pc = check_mem_backing(items, window=pc_win)
    assert back_idx.mem_class_steps == back_pc.mem_class_steps
    assert back_idx.blind_pcs == back_pc.blind_pcs
    assert back_idx.sufficient == back_pc.sufficient

    clo_idx = audit_address_closure(items, window=idx_win, window_is_idx=True)
    clo_pc = audit_address_closure(items, window=pc_win)
    assert clo_idx.blind_pcs == clo_pc.blind_pcs
    assert clo_idx.unbacked_roots == clo_pc.unbacked_roots
    assert clo_idx.sufficient == clo_pc.sufficient


def test_idx_window_excludes_pc_aliased_occurrence():
    # The whole point of idx: a handler jumps OUT of its address band, so a pc
    # band pulls in another occurrence at the same pc. idx isolates the segment.
    items = [
        ins(0, 0x1004, "ldr x1, [x20]", reads={"x20": 0}, writes={"x1": 0},
            mem=[MemOp("r", 0x9000, 0, 8)]),          # segment A (idx 0)
        ins(1, 0x1004, "ldr x1, [x20]", reads={"x20": 0}, writes={"x1": 0},
            mem=[MemOp("r", 0x9000, 0, 8)]),          # segment B re-visit, same pc
    ]
    # idx window (0,0) takes only the first occurrence; pc window pulls in both.
    back_idx = check_mem_backing(items, window=(0, 0), window_is_idx=True)
    back_pc = check_mem_backing(items, window=(0x1004, 0x1004))
    assert back_idx.mem_class_steps == 1
    assert back_pc.mem_class_steps == 2


# --- seed-independence gate (pre-symex): the seed must VARY across the cohort --
# Subject is the SEED (the symbolized recovery variable), not hard-wired "input":
# F may be F(input), F(nonce), F(input,nonce) — the gate asks whether WHATEVER we
# symbolized varies across the cohort. All synthetic values; zero case-specific.

def test_seed_independence_blocks_when_all_seeds_constant():
    # F0 false-EXACT root: every symbolized seed is the same value in every vector
    # → symbolizing a constant → BLOCK before wasting a symex run.
    rep = check_seed_independence(
        {"x8": [0xAA, 0xAA, 0xAA], "carrier": [7, 7, 7]}, min_vectors=3)
    assert rep.blocked and rep.verdict == "BLOCK" and not rep.sufficient
    assert set(rep.constant_seeds) == {"x8", "carrier"} and rep.varying_seeds == ()
    assert "not driven by the recovery variable" in rep.advisory


def test_seed_independence_passes_when_a_seed_varies():
    # A seed that takes >= 2 distinct values across the cohort → F is exercised → OK.
    rep = check_seed_independence(
        {"x8": [1, 2, 3], "carrier": [9, 9, 9]}, min_vectors=3)
    assert rep.sufficient and rep.verdict == "OK"
    assert rep.varying_seeds == ("x8",)
    # the constant one is surfaced as concrete backing, not a symbolic input.
    assert rep.constant_seeds == ("carrier",)
    assert "CONCRETE backing" in rep.advisory


def test_seed_independence_nonce_only_function_is_not_falsely_blocked():
    # dev addendum M1: an F(nonce) with NO external input must NOT error-BLOCK just
    # because "input" didn't vary — the seed here IS the nonce, and it varies.
    rep = check_seed_independence({"nonce": [0x11, 0x22, 0x33]}, min_vectors=3)
    assert rep.sufficient and rep.varying_seeds == ("nonce",)


def test_seed_independence_distinct_vector_count_ignores_repeats():
    # dev addendum M3: a repeated seed assignment is not extra independent evidence;
    # distinct_vector_count counts genuinely-different vectors.
    rep = check_seed_independence({"x8": [5, 5, 7]}, min_vectors=2)
    assert rep.sufficient and rep.distinct_vector_count == 2   # {5, 7}, not 3


def test_seed_independence_insufficient_cohort_is_surfaced_not_silent():
    rep = check_seed_independence({"x8": [3]}, min_vectors=3)
    assert rep.verdict == "INSUFFICIENT" and not rep.sufficient and not rep.blocked
    assert "undecidable" in rep.advisory
    assert rep.to_dict()["kind"] == "setup_symex_seed_independence"
