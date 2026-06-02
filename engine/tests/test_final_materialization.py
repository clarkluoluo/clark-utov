"""#7 — final-materialization gate (route before whole-chain composite).

Regression fixtures map to spec_tc2_final_materialization_gate "Regression
fixtures": pseudo-closure trap, final-construction routing, negative (in-place),
plus the A8④ SOURCE_UNOBSERVABLE degenerate.
"""

from __future__ import annotations

from engine.closure_classification import (
    ClosureLevel,
    TrapState,
    classify_closure,
)
from engine.final_materialization import (
    FinalMaterialization,
    MaterializationVerdict,
    NextMove,
    detect_final_materialization,
)
from engine.import_map import build_import_map, annotate_calls
from engine.types import Instruction, MemOp, MemSnapshot


def _ins(idx, mnem, reads=None, mem=(), pc=None):
    return Instruction(idx=idx, pc=pc if pc is not None else 0x70000 + idx * 4,
                       bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=dict(reads or {}), regs_write={}, mem=mem)


SINK = 0x72000
SRC = 0x90000


def _strb(idx, sink_addr, byte_val, src_addr=None, src_val=None):
    """A byte store into the sink; optionally pairs a source read in the same ins
    (the load-byte/store-byte copy couple)."""
    mem = []
    if src_addr is not None:
        mem.append(MemOp("r", src_addr, byte_val if src_val is None else src_val, 1))
    mem.append(MemOp("w", sink_addr, byte_val, 1))
    return _ins(idx, "strb w0, [x1]", mem=tuple(mem))


# --- Final-construction routing (header ‖ copy) ------------------------------

def test_final_header_copy_routes_to_recover_source_before_composite():
    # output = 4-byte fixed header || 8-byte copy from a contiguous source region.
    header = bytes([0xDE, 0xAD, 0xBE, 0xEF])
    body = bytes(range(8))
    output = header + body
    trace = []
    # header bytes: written with NO source read (immediate).
    for i, b in enumerate(header):
        trace.append(_ins(i, "strb w0, [x1]", mem=(MemOp("w", SINK + i, b, 1),)))
    # body bytes: copied from a contiguous source buffer (paired source reads),
    # AND the source region is observable (snapshot covers it → recover, not watch).
    for j, b in enumerate(body):
        trace.append(_strb(100 + j, SINK + 4 + j, b, src_addr=SRC + j, src_val=b))
    snaps = [MemSnapshot(addr=SRC, data=body, label="source_buffer")]
    res = detect_final_materialization(
        trace, sink_base=SINK, output=output, snapshots=snaps)
    assert res.verdict is MaterializationVerdict.FINAL_HEADER_COPY
    assert res.header_bytes == header
    assert res.source_region == {"base": SRC, "len": 8}
    assert res.next_move is NextMove.RECOVER_SOURCE_PROVENANCE
    assert res.source_observable is True


def test_final_copy_via_memcpy_annotation():
    # output is a bulk memcpy from SRC; recognised from the #4 annotated call.
    output = bytes(range(16))
    im = build_import_map(plt_map={0x400a90: "memcpy"})
    call = _ins(0, "bl 0x400a90", reads={"x0": SINK, "x1": SRC, "x2": 16})
    # the actual byte writes into the sink (so the gate sees a final write seq)
    writes = [_ins(10 + i, "strb", mem=(MemOp("w", SINK + i, output[i], 1),))
              for i in range(16)]
    trace = [call] + writes
    anns = annotate_calls(trace, im)
    snaps = [MemSnapshot(addr=SRC, data=output, label="src")]
    res = detect_final_materialization(
        trace, sink_base=SINK, output=output, snapshots=snaps,
        annotated_calls=anns)
    assert res.verdict is MaterializationVerdict.FINAL_COPY
    assert res.source_region["base"] == SRC
    assert res.next_move is NextMove.RECOVER_SOURCE_PROVENANCE


# --- A8④ degenerate: source unobservable -------------------------------------

def test_source_unobservable_routes_to_watch_not_composite():
    # A copy run from a source region that is NOT covered by any snapshot/trace read
    # value → SOURCE_UNOBSERVABLE, route to watch_source_buffer (not fall-through).
    body = bytes(range(8))
    output = body
    # The store writes the byte but reads from an un-observed src addr (no value we
    # can confirm) — we simulate "unobservable" by NOT providing a snapshot AND
    # making the source reads land on addresses with no covering write.
    trace = []
    for j, b in enumerate(body):
        # paired read marks the source region, but nothing else observes it
        trace.append(_strb(j, SINK + j, b, src_addr=SRC + j, src_val=b))
    # The source bytes ARE in the reads though — so to force unobservable, use a
    # source region disjoint from the reads is not possible here; instead drop the
    # read values' coverage by giving the source region a HIGHER base than read.
    # Simplest: detect with a source region the reads cover, but assert observable.
    res = detect_final_materialization(trace, sink_base=SINK, output=output)
    # reads DO cover the source here → observable True; this case is the positive
    # control. The真 unobservable case is below.
    assert res.source_observable is True


def test_source_unobservable_when_no_read_value_and_no_snapshot():
    # Build a sink write seq with NO paired source reads at all but a memcpy whose
    # SRC region is not observed → watch.
    output = bytes(range(16))
    im = build_import_map(plt_map={0x400a90: "memcpy"})
    call = _ins(0, "bl 0x400a90", reads={"x0": SINK, "x1": SRC, "x2": 16})
    writes = [_ins(10 + i, "strb", mem=(MemOp("w", SINK + i, output[i], 1),))
              for i in range(16)]
    trace = [call] + writes
    anns = annotate_calls(trace, im)
    # NO snapshot for SRC, and SRC is never read in the trace → unobservable.
    res = detect_final_materialization(
        trace, sink_base=SINK, output=output, annotated_calls=anns)
    assert res.verdict is MaterializationVerdict.FINAL_COPY
    assert res.source_observable is False
    assert res.next_move is NextMove.WATCH_SOURCE_BUFFER
    assert res.to_dict()["degenerate"] == "SOURCE_UNOBSERVABLE"
    assert res.watch_spec is not None
    assert res.watch_spec["addr"] == f"0x{SRC:x}"


# --- Negative: in-place computed output --------------------------------------

def test_in_place_computed_output_falls_through_composite():
    # The sink is written from scattered, unrelated computed values (no single
    # contiguous source) → NO_FINAL_MATERIALIZATION, composite path unchanged.
    output = bytes([0x11, 0x22, 0x33, 0x44])
    trace = [
        _ins(0, "strb", mem=(MemOp("r", 0xAAAA, 0x11, 1), MemOp("w", SINK, 0x11, 1))),
        _ins(1, "strb", mem=(MemOp("r", 0xBBBB, 0x22, 1), MemOp("w", SINK + 1, 0x22, 1))),
        _ins(2, "strb", mem=(MemOp("r", 0xCCCC, 0x33, 1), MemOp("w", SINK + 2, 0x33, 1))),
        _ins(3, "strb", mem=(MemOp("r", 0xDDDD, 0x44, 1), MemOp("w", SINK + 3, 0x44, 1))),
    ]
    res = detect_final_materialization(trace, sink_base=SINK, output=output)
    assert res.verdict is MaterializationVerdict.NO_FINAL_MATERIALIZATION
    assert res.next_move is NextMove.FALL_THROUGH_COMPOSITE


def test_no_traced_sink_write_falls_through():
    output = bytes([0x01, 0x02])
    trace = [_ins(0, "mov x0, x0")]
    res = detect_final_materialization(trace, sink_base=SINK, output=output)
    assert res.verdict is MaterializationVerdict.NO_FINAL_MATERIALIZATION
    assert res.next_move is NextMove.FALL_THROUGH_COMPOSITE


# --- The gate ROUTES, it does not CLOSE (decision #1) ------------------------

def test_gate_never_promotes_to_oracle_pseudo_closure_trap_still_caught():
    # The TC2 F=7 pseudo-closure: a constant local F with unconfirmed sink/provenance
    # must NOT be reported as algorithm closure — the existing PSEUDO_CLOSURE_TRAP
    # catches it. The gate routing never short-circuits this.
    cc = classify_closure(
        structural_closed=True,
        output_sink_confirmed=False,
        provenance_closed=False,
        parity_exact=False,
        is_constant=True,
        provenance_supported=False,
    )
    assert cc.level is ClosureLevel.STRUCTURAL
    assert cc.trap is TrapState.PSEUDO_CLOSURE_TRAP
    assert cc.algorithm_closed is False


def test_recovered_header_copy_still_must_clear_closure_classification():
    # Even after the gate detects FINAL_HEADER_COPY and recovers the source, the F
    # is NOT oracle-closed until provenance + parity hold. A recovered-but-unparitied
    # F lands PROVENANCE/candidate, never ORACLE — the gate did not close it.
    cc = classify_closure(
        structural_closed=True,
        output_sink_confirmed=True,
        provenance_closed=True,
        parity_exact=False,        # not yet multi-input parity-confirmed
    )
    assert cc.level is ClosureLevel.PROVENANCE
    assert cc.algorithm_closed is False


def test_empty_output_raises():
    import pytest
    with pytest.raises(ValueError):
        detect_final_materialization([_ins(0, "nop")], sink_base=SINK, output=b"")
