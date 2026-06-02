"""Phase 0 opaque-staging diagnosis + Phase 1/2(i) staging-interval logic.

Synthetic, zero case-specific: every concrete address / offset / idx is a fixture
constant chosen only to exercise a mechanism (verdict / byte map / ea_symbolic /
ea_varies_cohort), never a real target coordinate. Asserts validate the mechanism,
not a case solution.
"""

from __future__ import annotations

from engine.opaque_staging import (
    BlindLoad,
    CohortStagingAdvisory,
    EaVaryingSite,
    PointerChainSpec,
    StagingByte,
    StagingDiagnosis,
    VERDICT_INCONCLUSIVE,
    VERDICT_KNOWN_ADDR,
    VERDICT_SYMBOLIC_ADDRESS,
    cohort_staging_advisory,
    diagnose_opaque_staging,
    resolve_staging_address,
)
from engine.types import Instruction, MemOp


def ins(idx, pc, mnem, *, reads=None, writes=None, mem=()):
    return Instruction(
        idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
        regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem),
    )


# Synthetic coordinates (fixture-only — never a real target address).
_BASE = 0x10000          # a concrete staging base pointer value
_OFF = 0x20
_STAGING = _BASE + _OFF  # the staging landing address


# --- Form A: known_addr (concrete base + constant offset, EA constant) -------

def test_form_a_known_addr_ea_not_symbolic():
    # x10 is a concrete base live-in; the store lands at [x10] and a later ldr
    # reads [x10] back. The load's EA backtraces to a concrete base (no symbolic
    # input, no chained load) → known_addr (→ Phase 1).
    items = [
        ins(0, 0x1000, "str x8, [x10]", reads={"x8": 0x41, "x10": _STAGING},
            mem=[MemOp("w", _STAGING, 0x41, 8)]),
        ins(1, 0x1004, "ldr x9, [x10]", reads={"x10": _STAGING},
            writes={"x9": 0x41}, mem=[MemOp("r", _STAGING, 0x41, 8)]),
    ]
    diag = diagnose_opaque_staging(
        items, window=(0, 1), window_is_idx=True, symbolic_inputs=("x8",))
    assert diag.verdict == VERDICT_KNOWN_ADDR
    assert diag.routes_to_phase == 1
    # the load IS a blind leg (x10 is an un-backed live-in) but its EA is NOT symbolic
    assert diag.blind_loads
    assert all(not bl.ea_symbolic for bl in diag.blind_loads)


# --- Form B: symbolic_address (EA taint from a symbolic input) ---------------

def test_form_b_symbolic_address_ea_from_symbolic_input():
    # x10 (the EA register) is COMPUTED from the symbolic input x8 → ea_symbolic.
    items = [
        ins(0, 0x1000, "add x10, x8, x11", reads={"x8": 1, "x11": _BASE},
            writes={"x10": _STAGING}),
        ins(1, 0x1004, "ldr x9, [x10]", reads={"x10": _STAGING},
            writes={"x9": 0x41}, mem=[MemOp("r", _STAGING, 0x41, 8)]),
    ]
    diag = diagnose_opaque_staging(
        items, window=(0, 1), window_is_idx=True, symbolic_inputs=("x8",))
    assert diag.verdict == VERDICT_SYMBOLIC_ADDRESS
    assert diag.routes_to_phase == 2
    assert any(bl.ea_symbolic for bl in diag.blind_loads)


def test_form_b_symbolic_address_ea_via_pointer_chain_load():
    # x10 (the EA register) is itself LOADED from memory (a pointer chain): the
    # address came out of memory → ea_symbolic even with no symbolic_inputs.
    items = [
        ins(0, 0x1000, "ldr x10, [x12]", reads={"x12": _BASE},
            writes={"x10": _STAGING}, mem=[MemOp("r", _BASE, _STAGING, 8)]),
        ins(1, 0x1004, "ldr x9, [x10]", reads={"x10": _STAGING},
            writes={"x9": 0x41}, mem=[MemOp("r", _STAGING, 0x41, 8)]),
    ]
    diag = diagnose_opaque_staging(items, window=(0, 1), window_is_idx=True)
    assert diag.verdict == VERDICT_SYMBOLIC_ADDRESS
    bl = next(b for b in diag.blind_loads if b.pc == 0x1004)
    assert bl.ea_symbolic
    assert "pointer chain" in bl.reason


# --- Form C: cohort (2 traces, EA value varies across vectors) ---------------

def _cohort_trace(staging_addr: int, val: int):
    return [
        ins(0, 0x1000, "ldr x9, [x10]", reads={"x10": staging_addr},
            writes={"x9": val}, mem=[MemOp("r", staging_addr, val, 8)]),
    ]


def test_form_c_cohort_ea_varies_marks_symbolic_address():
    # Single-trace the EA reg is a plain live-in (would diagnose known_addr), but
    # the cohort shows the LOAD's actual EA varies across vectors → symbolic_address.
    base = _cohort_trace(_STAGING, 0xAA)
    coh = [_cohort_trace(_STAGING, 0xAA), _cohort_trace(_STAGING + 8, 0xBB)]
    diag = diagnose_opaque_staging(
        base, window=(0, 0), window_is_idx=True, cohort_traces=coh)
    assert diag.verdict == VERDICT_SYMBOLIC_ADDRESS
    bl = diag.blind_loads[0]
    assert bl.ea_varies_cohort is True


def test_form_c_cohort_ea_constant_stays_known_addr():
    base = _cohort_trace(_STAGING, 0xAA)
    coh = [_cohort_trace(_STAGING, 0xAA), _cohort_trace(_STAGING, 0xBB)]
    diag = diagnose_opaque_staging(
        base, window=(0, 0), window_is_idx=True, cohort_traces=coh)
    assert diag.verdict == VERDICT_KNOWN_ADDR
    assert diag.blind_loads[0].ea_varies_cohort is False


# --- Form D: opaque-branch backed staging (Phase 0b — the new gate) ----------
# Replica of the verifier's ``opaque`` branch (backing_ok=True): EVERY load is
# backed (concrete trace value) so audit_address_closure reports NO un-backed
# legs — the old unbacked-legs gate finds nothing and would diagnose inconclusive.
# But one load's EA is input/pointer-derived (it reads back a symbolic store's
# landing), so it SHOULD carry the symbol. The DFG-derived candidate gate (Phase
# 0b) must still select it → verdict is NOT inconclusive. _SP is the stack base
# the closure resolver treats as backed, so the legs come out backed (the
# opaque-branch contrast), never an un-backed live-in.

_SP = 0x7000          # a concrete stack frame slot (fixture-only)


def test_form_d_opaque_branch_backed_staging_selected_by_dfg_gate():
    from engine.setup_symex import audit_address_closure

    # x10 (the staging base) is LOADED from the stack (a pointer chain), then the
    # staging load reads [x10]. All mem ops are backed → no un-backed leg.
    items = [
        ins(0, 0x1000, "str x8, [sp]", reads={"x8": _STAGING},
            mem=[MemOp("w", _SP, _STAGING, 8)]),
        ins(1, 0x1004, "ldr x10, [sp]", writes={"x10": _STAGING},
            mem=[MemOp("r", _SP, _STAGING, 8)]),
        ins(2, 0x1008, "ldr x9, [x10]", reads={"x10": _STAGING},
            writes={"x9": 0x41}, mem=[MemOp("r", _STAGING, 0x41, 8)]),
    ]
    # the OLD gate is empty: every address closure is backed (sp base / chained).
    closure = audit_address_closure(items, window=(0, 2), window_is_idx=True)
    assert closure.sufficient
    assert all(leg.backed for leg in closure.legs)   # NO un-backed leg
    # the NEW gate still picks the staging load and routes it (not inconclusive).
    diag = diagnose_opaque_staging(items, window=(0, 2), window_is_idx=True)
    assert diag.verdict == VERDICT_SYMBOLIC_ADDRESS
    bl = next(b for b in diag.blind_loads if b.pc == 0x1008)
    assert bl.ea_symbolic
    assert "dfg_staging" in bl.reason and "pointer chain" in bl.reason


def test_form_d_backed_staging_ea_from_symbolic_input_known_addr_when_constant():
    # EA derives from a symbolic input that is ALSO backed (so no un-backed leg),
    # and is constant across the (single) trace → still selected by the DFG gate,
    # routed symbolic_address because the root is a symbolic input.
    items = [
        ins(0, 0x1000, "add x10, x8, x11", reads={"x8": 1, "x11": _BASE},
            writes={"x10": _STAGING}),
        ins(1, 0x1004, "ldr x9, [x10]", reads={"x10": _STAGING},
            writes={"x9": 0x41}, mem=[MemOp("r", _STAGING, 0x41, 8)]),
    ]
    diag = diagnose_opaque_staging(
        items, window=(0, 1), window_is_idx=True, symbolic_inputs=("x8",))
    assert diag.verdict == VERDICT_SYMBOLIC_ADDRESS
    assert any(b.ea_symbolic for b in diag.blind_loads)


def test_form_d_cohort_ea_varies_reinforces_symbolic_address():
    # Backed pointer-chain staging load whose actual EA varies across the cohort:
    # both the DFG gate (pointer chain) and the cohort (EA varies) agree → P2.
    def trace(staging):
        return [
            ins(0, 0x1000, "ldr x10, [sp]", writes={"x10": staging},
                mem=[MemOp("r", _SP, staging, 8)]),
            ins(1, 0x1004, "ldr x9, [x10]", reads={"x10": staging},
                writes={"x9": 0x41}, mem=[MemOp("r", staging, 0x41, 8)]),
        ]
    coh = [trace(_STAGING), trace(_STAGING + 8)]
    diag = diagnose_opaque_staging(
        trace(_STAGING), window=(0, 1), window_is_idx=True, cohort_traces=coh)
    assert diag.verdict == VERDICT_SYMBOLIC_ADDRESS
    bl = next(b for b in diag.blind_loads if b.pc == 0x1004)
    assert bl.ea_symbolic and bl.ea_varies_cohort is True


def test_form_d_pointer_chain_narrows_candidate_origin():
    # Supplying the chain's load-base register annotates the matching candidate as
    # a chain hit (the optional narrowing) WITHOUT changing the verdict — the DFG
    # scan is the backbone, the chain only re-orders / annotates.
    items = [
        ins(0, 0x1004, "ldr x10, [sp]", writes={"x10": _STAGING},
            mem=[MemOp("r", _SP, _STAGING, 8)]),
        ins(1, 0x1008, "ldr x9, [x10]", reads={"x10": _STAGING},
            writes={"x9": 0x41}, mem=[MemOp("r", _STAGING, 0x41, 8)]),
    ]
    chain = PointerChainSpec(store_base_regs=("x10",), load_base_regs=("x10",))
    diag = diagnose_opaque_staging(
        items, window=(0, 1), window_is_idx=True, pointer_chain=chain)
    assert diag.verdict == VERDICT_SYMBOLIC_ADDRESS
    bl = next(b for b in diag.blind_loads if b.pc == 0x1008)
    assert "dfg_staging_chain" in bl.reason   # the chain narrowing tagged it


# --- Form E: negative (all loads concrete base + const, no input-derived EA) --

def test_form_e_no_input_derived_load_is_inconclusive_not_false_positive():
    # Every load reads off the stack base with a constant displacement — fully
    # backed, EA not input-derived. The DFG gate finds NO candidate and the
    # unbacked gate is empty → inconclusive + a note (never a silent false split).
    items = [
        ins(0, 0x1000, "ldr x9, [sp]", writes={"x9": 0x41},
            mem=[MemOp("r", _SP, 0x41, 8)]),
        ins(1, 0x1004, "ldr x8, [sp]", writes={"x8": 0x42},
            mem=[MemOp("r", _SP + 8, 0x42, 8)]),
    ]
    diag = diagnose_opaque_staging(items, window=(0, 1), window_is_idx=True)
    assert diag.verdict == VERDICT_INCONCLUSIVE
    assert not diag.blind_loads
    assert diag.reasons   # a note, never silent


# --- byte-level staging map (the localize side, sub-case 4) ------------------

def test_cohort_staging_byte_map_varies_and_idx():
    # store at idx0 lands at _STAGING; loaded back at idx1. Two cohort vectors give
    # the stored byte a different value → StagingByte.varies_cohort True, with idx.
    def trace(v):
        return [
            ins(0, 0x1000, "str w8, [x10]", reads={"w8": v, "x10": _STAGING},
                mem=[MemOp("w", _STAGING, v, 1)]),
            ins(1, 0x1004, "ldr w9, [x10]", reads={"x10": _STAGING},
                writes={"w9": v}, mem=[MemOp("r", _STAGING, v, 1)]),
        ]
    coh = [trace(0x11), trace(0x22)]
    diag = diagnose_opaque_staging(
        trace(0x11), window=(0, 1), window_is_idx=True, cohort_traces=coh)
    sb = next(b for b in diag.staging_bytes if b.addr == _STAGING)
    assert sb.varies_cohort is True
    assert sb.store_idx == 0 and sb.load_idx == 1


def test_single_trace_staging_byte_varies_is_none_not_false():
    # No cohort → varies_cohort must be None (cannot tell), never a silent False.
    items = [
        ins(0, 0x1000, "str w8, [x10]", reads={"w8": 1, "x10": _STAGING},
            mem=[MemOp("w", _STAGING, 1, 1)]),
    ]
    diag = diagnose_opaque_staging(items, window=(0, 0), window_is_idx=True)
    sb = next(b for b in diag.staging_bytes if b.addr == _STAGING)
    assert sb.varies_cohort is None
    assert sb.store_idx == 0


# --- inconclusive (no blind leg / insufficient evidence, never silent) -------

def test_inconclusive_when_no_blind_load():
    # Every load EA is backed (x10 written by an in-window mov from a backed const):
    # actually here there is simply no memory op at all → no blind leg.
    items = [ins(0, 0x1000, "add x0, x1, x2", reads={"x1": 1, "x2": 2},
                 writes={"x0": 3})]
    diag = diagnose_opaque_staging(items, window=(0, 0), window_is_idx=True)
    assert diag.verdict == VERDICT_INCONCLUSIVE
    assert diag.reasons   # a note, never silent


# --- to_dict shape + invariant 4 (big byte list digested) --------------------

def test_to_dict_shape_and_routing():
    diag = StagingDiagnosis(
        window=(0, 5), window_is_idx=True, verdict=VERDICT_KNOWN_ADDR,
        blind_loads=(BlindLoad(idx=1, pc=0x1004, ea_regs=("x10",),
                               ea_symbolic=False),),
        staging_bytes=(StagingByte(addr=_STAGING, store_idx=0, load_idx=1),),
        reasons=("r",))
    d = diag.to_dict()
    assert d["kind"] == "opaque_staging_diagnosis"
    assert d["verdict"] == VERDICT_KNOWN_ADDR and d["routes_to_phase"] == 1
    assert d["window_basis"] == "idx"
    assert isinstance(d["staging_bytes"], list)


def test_to_dict_digests_large_byte_list():
    many = tuple(StagingByte(addr=_STAGING + i, store_idx=i) for i in range(50))
    diag = StagingDiagnosis(window=(0, 50), window_is_idx=True,
                            verdict=VERDICT_KNOWN_ADDR, staging_bytes=many)
    d = diag.to_dict()
    assert d["staging_bytes"]["_trimmed_list"] is True
    assert d["staging_bytes"]["count"] == 50
    assert "sha1" in d["staging_bytes"] and len(d["staging_bytes"]["sample"]) == 8


# --- resolve_staging_address (Phase 1/2(i) landing from the trace) -----------

def test_resolve_staging_address_from_pointer_chain():
    items = [
        ins(0, 0x1000, "str x8, [x10]", reads={"x8": 0x41, "x10": _STAGING},
            mem=[MemOp("w", _STAGING, 0x41, 8)]),
        ins(1, 0x1004, "str x8, [x11]", reads={"x8": 0x41, "x11": 0x99999},
            mem=[MemOp("w", 0x99999, 0x41, 8)]),  # different base — not the chain
    ]
    chain = PointerChainSpec(store_base_regs=("x10",), load_base_regs=("x10",))
    out = resolve_staging_address(items, chain, window=(0, 1), window_is_idx=True)
    assert out == [(_STAGING, 8)]   # only the x10 store, by the chain's named base


def test_resolve_staging_address_none_chain_returns_empty():
    items = [ins(0, 0x1000, "str x8, [x10]", reads={"x10": _STAGING},
                 mem=[MemOp("w", _STAGING, 1, 8)])]
    assert resolve_staging_address(items, None) == []


def test_resolve_staging_address_size_override():
    items = [ins(0, 0x1000, "str x8, [x10]", reads={"x10": _STAGING},
                 mem=[MemOp("w", _STAGING, 1, 8)])]
    chain = PointerChainSpec(store_base_regs=("x10",), store_size=4)
    assert resolve_staging_address(items, chain) == [(_STAGING, 4)]


# --- derive_pointer_chain (坎3: self-produce the shape from the diagnosis) -----

def test_derive_pointer_chain_self_produces_store_and_load_base():
    # A staging window: store [x10] → later ldr [x10] whose EA is input-derived
    # (pointer-chain load). The diagnosis routes symbolic_address; derive_pointer_
    # chain reads the STORE's EA base (x10) from the diagnosed staging store and the
    # LOAD's EA base (x10) from the blind load — NO caller-supplied shape.
    from engine.opaque_staging import derive_pointer_chain
    items = [
        ins(0, 0x1000, "str x8, [x10]", reads={"x8": 0x41, "x10": _STAGING},
            mem=[MemOp("w", _STAGING, 0x41, 8)]),
        ins(1, 0x1004, "ldr x10, [sp]", writes={"x10": _STAGING},
            mem=[MemOp("r", _SP, _STAGING, 8)]),
        ins(2, 0x1008, "ldr x9, [x10]", reads={"x10": _STAGING},
            writes={"x9": 0x41}, mem=[MemOp("r", _STAGING, 0x41, 8)]),
    ]
    diag = diagnose_opaque_staging(items, window=(0, 2), window_is_idx=True)
    assert diag.verdict == VERDICT_SYMBOLIC_ADDRESS
    derived = derive_pointer_chain(diag, items, window=(0, 2), window_is_idx=True)
    assert derived is not None
    assert "x10" in derived.store_base_regs   # from the staging store's [x10]
    assert "x10" in derived.load_base_regs    # from the blind load's EA
    # the derived shape, fed back, resolves the staging landing from the trace.
    out = resolve_staging_address(items, derived, window=(0, 2), window_is_idx=True)
    assert (_STAGING, 8) in out


def test_derive_pointer_chain_none_when_no_staging_store():
    # No window store lands a staging byte (load-only window) → no store base reg →
    # None (invariant 8: never fabricate a shape).
    from engine.opaque_staging import derive_pointer_chain
    items = [
        ins(0, 0x1004, "ldr x10, [sp]", writes={"x10": _STAGING},
            mem=[MemOp("r", _SP, _STAGING, 8)]),
        ins(1, 0x1008, "ldr x9, [x10]", reads={"x10": _STAGING},
            writes={"x9": 0x41}, mem=[MemOp("r", _STAGING, 0x41, 8)]),
    ]
    diag = diagnose_opaque_staging(items, window=(0, 1), window_is_idx=True)
    derived = derive_pointer_chain(diag, items, window=(0, 1), window_is_idx=True)
    assert derived is None


# --- Phase 3 primitive: cohort_staging_advisory (EA variance, cohort-only) ----

def test_cohort_staging_advisory_surfaces_store_and_load_ea_variance():
    # Same store/load PC, different EA per vector → an EaVaryingSite per (pc, rw).
    a = [ins(0, 0x1000, "str x8, [x10]", mem=[MemOp("w", 0xA000, 0x41, 8)]),
         ins(1, 0x1004, "ldr x9, [x10]", mem=[MemOp("r", 0xA000, 0x41, 8)])]
    b = [ins(0, 0x1000, "str x8, [x10]", mem=[MemOp("w", 0xB000, 0x41, 8)]),
         ins(1, 0x1004, "ldr x9, [x10]", mem=[MemOp("r", 0xB000, 0x41, 8)])]
    adv = cohort_staging_advisory([a, b])
    assert isinstance(adv, CohortStagingAdvisory)
    assert adv.n_cohort == 2
    assert len(adv.ea_varying_sites) == 2
    by_rw = {s.rw: s for s in adv.ea_varying_sites}
    assert by_rw["w"].pc == 0x1000 and by_rw["w"].idx == 0
    assert by_rw["r"].pc == 0x1004 and by_rw["r"].idx == 1
    assert by_rw["w"].n_distinct_ea == 2
    assert set(by_rw["w"].sample_eas) == {0xA000, 0xB000}


def test_cohort_staging_advisory_constant_ea_is_empty_not_false():
    # Same EA across vectors → not an EA-varying site → empty advisory + note.
    a = [ins(0, 0x1000, "ldr x9, [x10]", mem=[MemOp("r", 0xA000, 1, 8)])]
    b = [ins(0, 0x1000, "ldr x9, [x10]", mem=[MemOp("r", 0xA000, 2, 8)])]
    adv = cohort_staging_advisory([a, b])
    assert adv.ea_varying_sites == ()
    assert "genuinely invisible" in adv.note or "no store/load" in adv.note


def test_cohort_staging_advisory_single_trace_is_empty_with_note():
    a = [ins(0, 0x1000, "ldr x9, [x10]", mem=[MemOp("r", 0xA000, 1, 8)])]
    adv = cohort_staging_advisory([a])
    assert adv.ea_varying_sites == () and adv.n_cohort == 1
    assert "need >= 2" in adv.note


def test_cohort_staging_advisory_ignore_addrs_controls_out_coupling_axis():
    # A coupling-axis address present in only some vectors must not, alone, make a
    # varying site; ignore_addrs scopes it out.
    a = [ins(0, 0x1000, "ldr x9, [x10]", mem=[MemOp("r", 0xA000, 1, 8)])]
    b = [ins(0, 0x1000, "ldr x9, [x10]", mem=[MemOp("r", 0xC000, 1, 8)])]
    raw = cohort_staging_advisory([a, b])
    assert len(raw.ea_varying_sites) == 1            # uncontrolled: A000 vs C000
    scoped = cohort_staging_advisory([a, b], ignore_addrs=(0xA000, 0xC000))
    assert scoped.ea_varying_sites == ()             # both ignored → no variance


def test_cohort_staging_advisory_region_filters():
    # region scopes which idx band is collected.
    a = [ins(0, 0x1000, "ldr x9, [x10]", mem=[MemOp("r", 0xA000, 1, 8)]),
         ins(5, 0x2000, "ldr x9, [x11]", mem=[MemOp("r", 0xD000, 1, 8)])]
    b = [ins(0, 0x1000, "ldr x9, [x10]", mem=[MemOp("r", 0xB000, 1, 8)]),
         ins(5, 0x2000, "ldr x9, [x11]", mem=[MemOp("r", 0xD000, 1, 8)])]
    full = cohort_staging_advisory([a, b])
    assert len(full.ea_varying_sites) == 1 and full.ea_varying_sites[0].pc == 0x1000
    out = cohort_staging_advisory([a, b], region=(5, 5))   # 0x2000 only — constant
    assert out.ea_varying_sites == ()


def test_cohort_staging_advisory_to_dict_shape_and_invariant4():
    a = [ins(0, 0x1000, "ldr x9, [x10]", mem=[MemOp("r", 0xA000, 1, 8)])]
    b = [ins(0, 0x1000, "ldr x9, [x10]", mem=[MemOp("r", 0xB000, 1, 8)])]
    d = cohort_staging_advisory([a, b]).to_dict()
    assert d["kind"] == "cohort_staging_advisory"
    assert d["n_cohort"] == 2
    site = d["ea_varying_sites"][0]
    assert site["pc"] == "0x1000" and site["rw"] == "r"
    assert set(site["sample_eas"]) == {"0xa000", "0xb000"}


# --- Cross-cut Part B: regs_write coverage self-check + regs_read fallback -----

def test_form_h_read_only_ea_not_false_known_addr():
    # Form H: the EA register's value is observed only on the READ side; the
    # window has NO regs_write at all (regs_write-coverage = 0). The EA backtrace
    # would otherwise call x10 a concrete-base live-in → a FALSE known_addr. The
    # coverage self-check must downgrade to inconclusive (not a false known_addr).
    items = [
        ins(0, 0x1000, "str x8, [x10]", reads={"x8": 0x41, "x10": 0xA000},
            mem=[MemOp("w", 0xA000, 0x41, 8)]),
        ins(1, 0x1004, "ldr x9, [x10]", reads={"x10": 0xA000},
            mem=[MemOp("r", 0xA000, 0x41, 8)]),   # NB: no writes — read-only values
    ]
    diag = diagnose_opaque_staging(items, window=(0, 1), window_is_idx=True)
    assert diag.verdict == VERDICT_INCONCLUSIVE
    assert any("coverage" in r for r in diag.reasons)


def test_form_h_control_with_regs_write_keeps_original_verdict():
    # Control: same shape but the load DOES populate regs_write → coverage healthy
    # → the new logic does NOT trigger; the original known_addr verdict stands.
    items = [
        ins(0, 0x1000, "str x8, [x10]", reads={"x8": 0x41, "x10": 0xA000},
            mem=[MemOp("w", 0xA000, 0x41, 8)]),
        ins(1, 0x1004, "ldr x9, [x10]", reads={"x10": 0xA000},
            writes={"x9": 0x41}, mem=[MemOp("r", 0xA000, 0x41, 8)]),
    ]
    diag = diagnose_opaque_staging(items, window=(0, 1), window_is_idx=True)
    assert diag.verdict == VERDICT_KNOWN_ADDR     # coverage 50% > 5% → unchanged
    assert not any("coverage" in r for r in diag.reasons)


def test_form_i_low_coverage_window_downgrades_known_addr():
    # Form I: a window dominated by instructions with regs_write={} (coverage below
    # threshold) → a would-be known_addr is downgraded to inconclusive + note.
    items = [ins(i, 0x1000 + 4 * i, "nop") for i in range(20)]
    # the single staging load (read-only EA) at the end.
    items.append(ins(20, 0x2000, "ldr x9, [x10]", reads={"x10": 0xA000},
                     mem=[MemOp("r", 0xA000, 0x41, 8)]))
    diag = diagnose_opaque_staging(items, window=(0, 20), window_is_idx=True)
    assert diag.verdict == VERDICT_INCONCLUSIVE   # coverage 0/21 < 5%
    assert any("coverage" in r for r in diag.reasons)


def test_form_i_coverage_threshold_is_parameterized():
    # The threshold is a signature parameter: the SAME window flips verdict with it.
    items = [ins(0, 0x1000, "str x8, [x10]", reads={"x8": 1, "x10": 0xA000},
                 mem=[MemOp("w", 0xA000, 1, 8)]),
             ins(1, 0x1004, "ldr x9, [x10]", reads={"x10": 0xA000},
                 writes={"x9": 1}, mem=[MemOp("r", 0xA000, 1, 8)])]
    # coverage = 1/2 = 50%.
    lax = diagnose_opaque_staging(items, window=(0, 1), min_regs_write_coverage=0.05)
    assert lax.verdict == VERDICT_KNOWN_ADDR
    strict = diagnose_opaque_staging(items, window=(0, 1), min_regs_write_coverage=0.75)
    assert strict.verdict == VERDICT_INCONCLUSIVE


def test_low_coverage_does_not_suppress_symbolic_address_finding():
    # Honesty in the OTHER direction: a positive symbolic-address finding (EA taint
    # from a symbolic input) is evidence-positive and must NOT be masked by the
    # coverage gate — only the false-negative-risk known_addr is downgraded.
    items = [ins(i, 0x1000 + 4 * i, "nop") for i in range(20)]
    items.append(ins(20, 0x2000, "ldr x9, [x10]", reads={"x10": 0xA000},
                     writes={"x9": 1}, mem=[MemOp("r", 0xA000, 0x41, 8)]))
    diag = diagnose_opaque_staging(
        items, window=(0, 20), window_is_idx=True, symbolic_inputs=("x10",))
    assert diag.verdict == VERDICT_SYMBOLIC_ADDRESS
