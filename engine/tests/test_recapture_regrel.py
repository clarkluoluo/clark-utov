"""B6: next_watch reg-relative upgrade — three-tier classification.

When ``observe_points_from_provenance`` is handed the trace ``items``, every
NEEDS_OBSERVATION gap is classified by its PC's addressing mnemonic + same-run
register row into:

  ① REGREL_UPGRADED   -> a ``mem_regrel`` watch (base_reg+offset[,index,scale]),
                         NO concrete addr (the runner resolves the live EA at hook
                         time; the stale run-local addr never crosses runs).
  ② NEEDS_REG_SNAPSHOT-> a register-observe directive (capture addressing regs
                         at the PC, then re-decompose) — never a bare concrete mem.
  ③ UNSTABLE_CONCRETE -> an explicit unstable status with a reason — never reused
                         as a cross-run watch.

Plus the GLOBAL invariant (A8④ / invariant 5): the output carries NO bare
``mem: 0x…`` silent point. Decomposition itself (``decompose_addressing``) is
unit-tested across every addressing form so this is principle, not curve-fit.
"""

from __future__ import annotations

from engine.aarch64_mem import AddrDecomposition, decompose_addressing
from engine.oracle_provenance import ProvenanceVerdict, trace_provenance
from engine.recapture import observe_points_from_provenance
from engine.runner_client import ObservePoint, RegRelWatch
from engine.types import Instruction, MemOp


def _ins(idx, pc, mnem, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read=dict(reads or {}),
                       regs_write=dict(writes or {}), mem=mem)


OUT = 0x72b18
EXPECTED = bytes([0x34, 0x15, 0x5f, 0xe9])


# ============================================================================
# decompose_addressing — unit, every addressing FORM (principle not curve-fit)
# ============================================================================

def test_decompose_simple_offset_proves_base_plus_imm():
    # [x19, #0x38] with x19 in the reg row and A == x19 + 0x38 -> ①
    x19 = 0x200000
    d = decompose_addressing("ldr w0, [x19, #0x38]", x19 + 0x38, {"x19": x19})
    assert d.verdict is AddrDecomposition.REGREL_UPGRADED
    assert d.base_reg == "x19" and d.offset == 0x38
    assert d.index is None and d.scale == 0
    assert d.width == 4                     # w-reg load -> 4 bytes


def test_decompose_bare_base_offset_zero():
    # [x22] -> base+offset with offset 0
    x22 = 0x300000
    d = decompose_addressing("ldr x0, [x22]", x22, {"x22": x22})
    assert d.verdict is AddrDecomposition.REGREL_UPGRADED
    assert d.base_reg == "x22" and d.offset == 0
    assert d.width == 8                     # x-reg load -> 8 bytes


def test_decompose_register_offset_uxtw_scale():
    # [x11, w9, uxtw #3] : A == x11 + (w9 << 3); both regs live -> ① with index/scale
    x11, w9 = 0x100000, 5
    A = x11 + (w9 << 3)
    d = decompose_addressing("ldr x9, [x11, w9, uxtw #3]", A, {"x11": x11, "w9": w9})
    assert d.verdict is AddrDecomposition.REGREL_UPGRADED
    assert d.base_reg == "x11" and d.index == "w9" and d.scale == 3
    assert d.offset == 0                    # register-offset form has no imm offset


def test_decompose_register_offset_lsl_scale():
    x1, x2 = 0x500, 4
    d = decompose_addressing("ldr x0, [x1, x2, lsl #2]", x1 + (x2 << 2),
                             {"x1": x1, "x2": x2})
    assert d.verdict is AddrDecomposition.REGREL_UPGRADED
    assert d.base_reg == "x1" and d.index == "x2" and d.scale == 2


def test_decompose_missing_base_reg_needs_snapshot():
    # decomposable [x19,#0x38] but the reg row has no x19 value -> ②
    d = decompose_addressing("ldr w0, [x19, #0x38]", 0x200038, {})
    assert d.verdict is AddrDecomposition.NEEDS_REG_SNAPSHOT
    assert d.needed_regs == ("x19",)
    assert d.base_reg == "x19"              # structure still named


def test_decompose_register_offset_missing_index_needs_snapshot():
    x11 = 0x100000
    d = decompose_addressing("ldr x9, [x11, w9, uxtw #3]", x11 + 40, {"x11": x11})
    assert d.verdict is AddrDecomposition.NEEDS_REG_SNAPSHOT
    assert d.needed_regs == ("w9",)


def test_decompose_pre_index_writeback_unstable():
    d = decompose_addressing("ldr x0, [x1, #0x10]!", 0x1010, {"x1": 0x1000})
    assert d.verdict is AddrDecomposition.UNSTABLE_CONCRETE
    assert "pre-index" in d.reason


def test_decompose_post_index_writeback_unstable():
    d = decompose_addressing("ldr x0, [x1], #0x10", 0x1000, {"x1": 0x1000})
    assert d.verdict is AddrDecomposition.UNSTABLE_CONCRETE
    assert "post-index" in d.reason


def test_decompose_pc_relative_literal_unstable():
    # no bracketed form (literal/PC-relative) -> no base reg -> ③
    d = decompose_addressing("ldr x0, =0x40000", 0x40000, {})
    assert d.verdict is AddrDecomposition.UNSTABLE_CONCRETE
    assert "PC-relative" in d.reason or "literal" in d.reason


def test_decompose_addr_mismatch_with_reg_row_is_unstable():
    # reg row present but addr does NOT reconstruct -> ③ (never a false ① upgrade)
    d = decompose_addressing("ldr w0, [x19, #0x38]", 0xDEADBEEF, {"x19": 0x200000})
    assert d.verdict is AddrDecomposition.UNSTABLE_CONCRETE
    assert "mismatch" in d.reason


def test_decompose_register_offset_mismatch_is_unstable():
    x11, w9 = 0x100000, 5
    d = decompose_addressing("ldr x9, [x11, w9, uxtw #3]", 0x999, {"x11": x11, "w9": w9})
    assert d.verdict is AddrDecomposition.UNSTABLE_CONCRETE


def test_decompose_non_memory_mnemonic_unstable():
    d = decompose_addressing("add x0, x1, x2", 0, {})
    assert d.verdict is AddrDecomposition.UNSTABLE_CONCRETE


def test_decompose_lsr_shift_is_not_valid_address_shift():
    x1, x2 = 0x500, 4
    d = decompose_addressing("ldr x0, [x1, x2, lsr #2]", x1 + (x2 >> 2),
                             {"x1": x1, "x2": x2})
    assert d.verdict is AddrDecomposition.UNSTABLE_CONCRETE


# ============================================================================
# observe_points_from_provenance(items=...) — end-to-end three tiers
# ============================================================================

def _needs_obs_prov(trace):
    prov = trace_provenance(trace, EXPECTED, sink_base=OUT)
    assert prov.verdict is ProvenanceVerdict.NEEDS_OBSERVATION
    return prov


def test_tier1_regrel_upgraded_simple_offset_no_concrete_ea():
    # idx0 loads from [x19,#0x38] (an un-captured addr); idx1 stores wrong value to
    # sink -> NEEDS_OBSERVATION gap at the LOAD pc. With items, it upgrades to ①.
    x19 = 0x200000
    UNK = x19 + 0x38
    pc_load = 0x70ec4
    trace = [
        _ins(0, pc_load, "ldr w8, [x19, #0x38]", reads={"x19": x19}, writes={"x8": 0},
             mem=(MemOp("r", UNK, 0, 4),)),
        _ins(1, 0x70ec8, "str w8, [x10]", reads={"x8": 0, "x10": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]
    prov = _needs_obs_prov(trace)
    pre = observe_points_from_provenance(prov, items=trace)

    # ① one reg-relative ObservePoint, NO concrete mem anywhere.
    pre.assert_no_bare_concrete()
    regrel = [op for op in pre.observe_points if op.mem_regrel]
    assert len(regrel) == 1
    op = regrel[0]
    assert op.pc == pc_load and op.when == "before"
    assert not op.mem                                  # invariant 1/2: no concrete EA
    w = op.mem_regrel[0]
    assert isinstance(w, RegRelWatch)
    assert w.base_reg == "x19" and w.offset == 0x38 and w.index is None
    assert w.width == 4 and w.pc == pc_load
    assert pre.reg_snapshot_directives == ()
    assert pre.unstable_points == ()


def test_tier1_register_offset_carries_index_scale():
    # acceptance: ldr x9,[x11,w9,uxtw#3] with x11/w9 in reg row -> mem_regrel with
    # base/index/scale, no ea.
    x11, w9 = 0x100000, 7
    UNK = x11 + (w9 << 3)
    pc = 0x706c4
    trace = [
        _ins(0, pc, "ldr x8, [x11, w9, uxtw #3]", reads={"x11": x11, "w9": w9},
             writes={"x8": 0}, mem=(MemOp("r", UNK, 0, 8),)),
        _ins(1, 0x706c8, "str w8, [x10]", reads={"x8": 0, "x10": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]
    prov = _needs_obs_prov(trace)
    pre = observe_points_from_provenance(prov, items=trace)
    pre.assert_no_bare_concrete()
    w = [op for op in pre.observe_points if op.mem_regrel][0].mem_regrel[0]
    assert w.base_reg == "x11" and w.index == "w9" and w.scale == 3
    assert not any(op.mem for op in pre.observe_points)


def test_tier2_needs_reg_snapshot_emits_directive_not_bare_concrete():
    # Same [x19,#0x38] form but the LOAD's reg row is MISSING x19 -> ②: a
    # register-observe directive + a regs ObservePoint, never a bare mem.
    UNK = 0x90000
    pc_load = 0x70ec4
    trace = [
        _ins(0, pc_load, "ldr w8, [x19, #0x38]", reads={}, writes={"x8": 0},
             mem=(MemOp("r", UNK, 0, 4),)),
        _ins(1, 0x70ec8, "str w8, [x10]", reads={"x8": 0, "x10": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]
    prov = _needs_obs_prov(trace)
    pre = observe_points_from_provenance(prov, items=trace)

    pre.assert_no_bare_concrete()
    assert len(pre.reg_snapshot_directives) == 1
    d = pre.reg_snapshot_directives[0]
    assert d.pc == pc_load and d.needed_regs == ("x19",)
    # a register-observe ObservePoint is armed (rides the B2 recapture mechanism)
    regs_pts = [op for op in pre.observe_points if op.capture == ("regs",)]
    assert len(regs_pts) == 1
    assert regs_pts[0].pc == pc_load and regs_pts[0].regs == ("x19",)
    # explicitly NO concrete mem and NO mem_regrel (can't prove yet)
    assert all(not op.mem for op in pre.observe_points)
    assert all(not op.mem_regrel for op in pre.observe_points)
    # the directive is serializable / explicit
    assert d.to_dict()["status"] == "NEEDS_REG_SNAPSHOT"


def test_tier3_unstable_concrete_pre_index_writeback():
    # A pre-index writeback load: structurally unstable -> ③, no watch emitted.
    UNK = 0x91000
    pc_load = 0x70f10
    trace = [
        _ins(0, pc_load, "ldr w8, [x19, #0x8]!", reads={"x19": UNK - 8}, writes={"x8": 0},
             mem=(MemOp("r", UNK, 0, 4),)),
        _ins(1, 0x70f14, "str w8, [x10]", reads={"x8": 0, "x10": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]
    prov = _needs_obs_prov(trace)
    pre = observe_points_from_provenance(prov, items=trace)

    pre.assert_no_bare_concrete()
    assert len(pre.unstable_points) == 1
    u = pre.unstable_points[0]
    assert u.pc == pc_load and "pre-index" in u.reason
    assert u.to_dict()["status"] == "UNSTABLE_CONCRETE"
    # nothing was emitted as a watch for an unstable point
    assert all(not op.mem and not op.mem_regrel for op in pre.observe_points)
    assert pre.reg_snapshot_directives == ()


def test_global_no_bare_concrete_across_mixed_gaps():
    # Mixed: one ① (provable), one ② (missing reg), one ③ (writeback). All three
    # loads feed the sink store (so all become gaps in the producer chain). The
    # output must contain NO bare concrete mem point — the global A8④ assertion.
    x19 = 0x200000
    UNK1 = x19 + 0x10           # ① byte 0
    UNK2 = 0x95000              # ② byte 1 (no reg row)
    UNK3 = 0x96000              # ③ byte 2 (post-index)
    trace = [
        _ins(0, 0x70a00, "ldrb w8, [x19, #0x10]", reads={"x19": x19}, writes={"x8": 0},
             mem=(MemOp("r", UNK1, 0, 1),)),
        _ins(1, 0x70a04, "ldrb w9, [x20, #0x4]", reads={}, writes={"x9": 0},
             mem=(MemOp("r", UNK2, 0, 1),)),
        _ins(2, 0x70a08, "ldrb w11, [x21], #0x8", reads={"x21": UNK3}, writes={"x11": 0},
             mem=(MemOp("r", UNK3, 0, 1),)),
        # each byte of the sink is stored from a distinct load (x8/x9/x11), so the
        # producer backtrace reaches all three loads -> three gaps, one per tier.
        _ins(3, 0x70a0c, "strb w8, [x10]", reads={"x8": 0, "x10": OUT},
             mem=(MemOp("w", OUT, 0, 1),)),
        _ins(4, 0x70a10, "strb w9, [x10]", reads={"x9": 0, "x10": OUT + 1},
             mem=(MemOp("w", OUT + 1, 0, 1),)),
        _ins(5, 0x70a14, "strb w11, [x10]", reads={"x11": 0, "x10": OUT + 2},
             mem=(MemOp("w", OUT + 2, 0, 1),)),
    ]
    prov = _needs_obs_prov(trace)
    pre = observe_points_from_provenance(prov, items=trace)

    pre.assert_no_bare_concrete()                       # the global invariant
    # each tier represented
    assert any(op.mem_regrel for op in pre.observe_points)          # ①
    assert len(pre.reg_snapshot_directives) == 1                    # ②
    assert len(pre.unstable_points) == 1                            # ③
    # and absolutely no bare concrete mem leaked
    assert all(not op.mem for op in pre.observe_points)


def test_legacy_path_without_items_unchanged():
    # invariant 7: omitting items keeps the OLD within-run behaviour (concrete mem
    # ObservePoints) — used by the B2 within-run recapture loop.
    UNK = 0x9000
    trace = [
        _ins(0, 0x70000, "ldr x8, [x9]", reads={"x9": UNK}, writes={"x8": 0},
             mem=(MemOp("r", UNK, 0, 4),)),
        _ins(1, 0x70004, "str x8, [x10]", reads={"x8": 0, "x10": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]
    prov = _needs_obs_prov(trace)
    pre = observe_points_from_provenance(prov)          # no items
    assert len(pre.observe_points) == 1
    assert pre.observe_points[0].mem == ((UNK, 4),)     # concrete, as before
    assert pre.reg_snapshot_directives == ()
    assert pre.unstable_points == ()


def test_gap_pc_absent_from_items_becomes_explicit_directive():
    # If the gap names a PC not present in the supplied items, we can't inspect its
    # mnemonic -> explicit NEEDS_REG_SNAPSHOT directive, never a bare concrete watch.
    UNK = 0x9000
    trace = [
        _ins(0, 0x70000, "ldr x8, [x9]", reads={"x9": UNK}, writes={"x8": 0},
             mem=(MemOp("r", UNK, 0, 4),)),
        _ins(1, 0x70004, "str x8, [x10]", reads={"x8": 0, "x10": OUT},
             mem=(MemOp("w", OUT, 0, 4),)),
    ]
    prov = _needs_obs_prov(trace)
    # decompose against a DIFFERENT items list that lacks the gap PC.
    other = [_ins(0, 0xAAAA, "nop", reads={})]
    pre = observe_points_from_provenance(prov, items=other)
    pre.assert_no_bare_concrete()
    assert len(pre.reg_snapshot_directives) == 1
    assert pre.reg_snapshot_directives[0].pc == 0x70000


# ============================================================================
# serialization round-trip: mem_regrel (incl. index/scale) survives the wire
# ============================================================================

def test_regrel_watch_wire_roundtrip_with_index_scale():
    from engine.recapture import RecaptureSpec
    from engine.runner_client import SubprocessRunnerAdapter
    op = ObservePoint(
        pc=0x706c4, when="before", capture=("mem",),
        mem_regrel=(
            RegRelWatch(base_reg="x19", offset=0x38, width=4, pc=0x706c4),
            RegRelWatch(base_reg="x11", offset=0, width=8, pc=0x706c4,
                        index="w9", scale=3),
        ))
    # JSON-RPC wire path
    wire = SubprocessRunnerAdapter._serialize_observe_point(op)
    assert "mem" in wire and wire["mem"] == []          # no concrete mem
    mr = wire["mem_regrel"]
    assert mr[0] == {"base_reg": "x19", "offset": 0x38, "width": 4,
                     "pc": "0x706c4", "kind": "read"}
    # plain base+offset: NO index/scale keys (invariant 7 byte-for-byte)
    assert "index" not in mr[0] and "scale" not in mr[0]
    assert mr[1]["index"] == "w9" and mr[1]["scale"] == 3

    # recapture spec serializer uses the SAME shape (construct symmetry)
    spec = RecaptureSpec(input=b"\x01", observe_points=[op])
    sp_pt = spec.to_dict()["observe_points"][0]
    assert sp_pt["mem_regrel"] == mr
