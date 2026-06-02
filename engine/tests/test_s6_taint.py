"""S6-taint: forward dynamic taint propagation. Synthetic concrete traces only
(no real binary). Pins: labels propagate reg+mem forward to sinks, handler PCs
recorded, the source->sink summary, and honest COULD_NOT_CLOSE breakpoints for
reads of never-stored memory / unparseable ops. Everything is parameterised via
ctx — no target address lives in the stage.
"""

from __future__ import annotations

import inspect

from engine.stages import s6_taint
from engine.stages.s6_taint import propagate_taint
from engine.types import Instruction, MemOp


def _ins(idx, mnem, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=0x70000 + idx * 4, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read=dict(reads or {}),
                       regs_write=dict(writes or {}), mem=mem)


def test_label_propagates_from_seeded_memory_through_to_sink():
    # seed "key" on memory [0x1000, 8); it flows ldr -> add -> str-to-sink.
    SRC, OUT = 0x1000, 0x2000
    trace = [
        _ins(0, "ldr x8, [x10]", reads={"x10": SRC}, writes={"x8": 0xAB},
             mem=(MemOp("r", SRC, 0xAB, 8),)),
        _ins(1, "add x9, x8, x11", reads={"x8": 0xAB, "x11": 1}, writes={"x9": 0xAC}),
        _ins(2, "str x9, [x12]", reads={"x9": 0xAC, "x12": OUT},
             mem=(MemOp("w", OUT, 0xAC, 8),)),
    ]
    res = propagate_taint(trace, {"key": {"mem": [[SRC, 8]]}}, sinks=[2])

    assert res.status == "closed"             # every read resolved
    assert len(res.sink_bytes) == 8           # 8 tainted output bytes
    first = res.sink_bytes[0]
    assert first["addr"] == f"0x{OUT:x}"
    assert first["source_labels"] == ["key"]
    # handler PCs = the chain that carried the taint: ldr(0), add(1), str(2)
    assert first["handler_pcs"] == ["0x70000", "0x70004", "0x70008"]
    assert res.by_source["key"][0] == f"0x{OUT:x}"


def test_register_seed_propagates_to_register_sink():
    res = propagate_taint(
        [
            _ins(0, "mov x1, x20", reads={"x20": 7}, writes={"x1": 7}),
            _ins(1, "eor x2, x1, x3", reads={"x1": 7, "x3": 9}, writes={"x2": 14}),
        ],
        {"nonce": {"regs": ["x20"]}},
        sinks=[1],
    )
    assert res.status == "closed"
    assert any(r["reg"] == "x2" and r["source_labels"] == ["nonce"]
               for r in res.sink_regs)
    assert res.by_source["nonce"] == ["reg:x2"]


def test_unresolved_memory_read_is_marked_could_not_close():
    # idx0 loads from memory never seeded and never stored in-trace.
    UNK = 0x5000
    res = propagate_taint(
        [_ins(0, "ldr x8, [x10]", reads={"x10": UNK}, writes={"x8": 0},
              mem=(MemOp("r", UNK, 0, 8),))],
        {"key": {"mem": [[0x1000, 8]]}},
        sinks=[],
    )
    assert res.status == "could_not_close"
    bps = res.breakpoints
    assert len(bps) == 1
    assert bps[0]["kind"] == "unresolved_mem_read"
    assert bps[0]["addr"] == f"0x{UNK:x}"
    assert bps[0]["idx"] == 0


def test_unparseable_mem_op_is_marked():
    res = propagate_taint(
        [_ins(0, "ldr x8, [x10]", reads={"x10": 0x1000},
              mem=(MemOp("r", 0x1000, 0, 0),))],   # size 0 -> unparseable
        {}, sinks=[],
    )
    assert res.status == "could_not_close"
    assert res.breakpoints[0]["kind"] == "unparseable_mem_op"


def test_partial_hole_still_propagates_resolved_bytes_and_flags_the_hole():
    # read spans 16 bytes; only the first 8 are seeded -> label still flows,
    # AND the hole over the unseeded half is flagged.
    res = propagate_taint(
        [
            _ins(0, "ldr x8, [x10]", reads={"x10": 0x1000}, writes={"x8": 0},
                 mem=(MemOp("r", 0x1000, 0, 16),)),
            _ins(1, "str x8, [x12]", reads={"x8": 0, "x12": 0x3000},
                 mem=(MemOp("w", 0x3000, 0, 8),)),
        ],
        {"key": {"mem": [[0x1000, 8]]}},     # only first 8 of the 16 read
        sinks=[1],
    )
    assert res.status == "could_not_close"
    assert any(b["kind"] == "unresolved_mem_read" for b in res.breakpoints)
    assert res.sink_bytes and res.sink_bytes[0]["source_labels"] == ["key"]


def test_clean_trace_has_closed_status():
    res = propagate_taint(
        [_ins(0, "add x0, x1, x2", reads={"x1": 1, "x2": 2}, writes={"x0": 3})],
        {"k": {"regs": ["x1"]}}, sinks=[0],
    )
    assert res.status == "closed"
    assert res.breakpoints == []


def test_stage_is_fully_parameterised_no_hardcoded_address():
    # the stage source must not contain a literal hex address — seeds/sinks are
    # all supplied through ctx.
    import re
    src = inspect.getsource(s6_taint)
    # allow size literals / 0x0-style in docstring examples only by checking the
    # code body has no 0x<addr> outside comments/docstrings: simplest robust
    # check — no 4+ hex-digit literal addresses anywhere in the module source.
    big_hex = re.findall(r"0x[0-9a-fA-F]{4,}", src)
    assert big_hex == [], f"unexpected hardcoded address literal(s): {big_hex}"
