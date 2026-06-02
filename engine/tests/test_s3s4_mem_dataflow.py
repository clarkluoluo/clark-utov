"""S0.5 + S3 mem-deps + S4 mem-aware slice: trace cipher back from its memory
store to the value source, on a concrete trace with empty regs_write.

Scenario (the confirmed break): the trace only carries the PRE-execution
register snapshot (`regs_read`); `regs_write` is empty; the cipher byte reaches
the outparam buffer through memory (store → load → store), so the value source
is a pure memory store that writes NO register. Register-only slicing can never
reach it; the memory-dependency edge is what pulls it in.
"""

from __future__ import annotations

from engine.stages.s0_5_normalize import reconstruct_regs_write
from engine.stages.s3_triton import build_dfg
from engine.stages.s4_slice import outparam_sinks, slice_backward
from engine.types import Instruction, MemOp

X24 = 0x9000  # an external entry register, constant across the trace
SRC = 0x1000  # where the cipher byte is first materialised in memory
OUT = 0x2000  # caller-supplied output buffer
CIPHER = 0xAB


def _ins(idx, mnem, prestate, mem=()):
    return Instruction(idx=idx, pc=0x70000 + idx * 4, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read=dict(prestate), regs_write={}, mem=mem)


def _trace():
    """idx2 stores the cipher to memory (the VALUE SOURCE, writes no register);
    idx3 loads it into x8; idx6 stores x8 to the outparam buffer (the SINK)."""
    base = {"x24": X24}
    return [
        _ins(0, "mov x10, #0x1000", base),
        _ins(1, "mov x20, #0xab", {**base, "x10": SRC}),
        _ins(2, "str x20, [x10]", {**base, "x10": SRC, "x20": CIPHER},
             mem=(MemOp("w", SRC, CIPHER, 8),)),                       # VALUE SOURCE
        _ins(3, "ldr x8, [x10]", {**base, "x10": SRC, "x20": CIPHER},
             mem=(MemOp("r", SRC, CIPHER, 8),)),                       # load cipher → x8
        _ins(4, "mov x9, #0x2000", {**base, "x10": SRC, "x20": CIPHER, "x8": CIPHER}),
        _ins(5, "mov x6, #0", {**base, "x10": SRC, "x20": CIPHER, "x8": CIPHER, "x9": OUT}),
        _ins(6, "str x8, [x9, x6]",
             {**base, "x10": SRC, "x20": CIPHER, "x8": CIPHER, "x9": OUT, "x6": 0},
             mem=(MemOp("w", OUT, CIPHER, 8),)),                       # OUTPARAM SINK
        _ins(7, "ldr w8, [x24, #0x84]",
             {**base, "x10": SRC, "x20": CIPHER, "x8": CIPHER, "x9": OUT, "x6": 0},
             mem=(MemOp("r", X24 + 0x84, 0, 4),)),                     # VMP step (last)
    ]


VALUE_SOURCE_IDX = 2
LOAD_IDX = 3
SINK_IDX = 6


def _reg_only_slice(dfg, sinks):
    """Old behaviour: backward BFS over reg_deps ONLY (no mem edges)."""
    by_idx = {n.idx: n for n in dfg}
    keep, stack = set(), list(sinks)
    while stack:
        cur = stack.pop()
        if cur in keep:
            continue
        keep.add(cur)
        node = by_idx.get(cur)
        if node is None:
            continue
        for p in node.reg_deps.values():
            if p is not None and p not in keep:
                stack.append(p)
    return keep


def test_regs_write_reconstructed_for_the_load():
    norm = reconstruct_regs_write(_trace())
    # idx3 `ldr x8,[x10]` had empty regs_write; frame-diff rebuilds {x8: CIPHER}
    assert norm[LOAD_IDX].regs_write == {"x8": CIPHER}
    # idx2 `str ...` writes memory, NOT a register — stays empty
    assert norm[VALUE_SOURCE_IDX].regs_write == {}


def test_mem_dep_edge_links_load_to_value_source():
    dfg = build_dfg(reconstruct_regs_write(_trace()))
    assert dfg[LOAD_IDX].mem_deps == (VALUE_SOURCE_IDX,)


def test_value_source_is_pulled_into_slice_only_via_memory():
    norm = reconstruct_regs_write(_trace())
    dfg = build_dfg(norm)

    # BEFORE (reg-only): the value source writes no register, so it is
    # unreachable — the slice from the sink does NOT contain it.
    reg_only = _reg_only_slice(dfg, [SINK_IDX])
    assert VALUE_SOURCE_IDX not in reg_only

    # AFTER (mem-aware): following mem_deps pulls the value source in.
    kept = slice_backward(dfg, [SINK_IDX])
    assert VALUE_SOURCE_IDX in kept
    assert LOAD_IDX in kept

    # the value source is reachable ONLY through memory — it is never a
    # register-dependency producer anywhere in the graph.
    reg_producers = {p for n in dfg for p in n.reg_deps.values() if p is not None}
    assert VALUE_SOURCE_IDX not in reg_producers


def test_reconstruct_is_idempotent_on_populated_trace():
    populated = [
        Instruction(idx=0, pc=0x10, bytes_=b"\x00\x00\x00\x00", mnemonic="mov x0, #1",
                    regs_read={}, regs_write={"x0": 1}, mem=()),
        Instruction(idx=1, pc=0x14, bytes_=b"\x00\x00\x00\x00", mnemonic="add x1, x0, #2",
                    regs_read={"x0": 1}, regs_write={"x1": 3}, mem=()),
    ]
    out = reconstruct_regs_write(populated)
    assert [i.regs_write for i in out] == [{"x0": 1}, {"x1": 3}]
    assert out == populated  # unchanged objects


def test_outparam_sinks_picks_the_contiguous_output_buffer():
    # four adjacent 8-byte stores form the output buffer; a lone unrelated
    # store sits elsewhere and must not be chosen.
    items = [
        _ins(0, "str x0, [x9]", {}, mem=(MemOp("w", OUT + 0, 0, 8),)),
        _ins(1, "str x1, [x9, #8]", {}, mem=(MemOp("w", OUT + 8, 0, 8),)),
        _ins(2, "str x2, [x9, #16]", {}, mem=(MemOp("w", OUT + 16, 0, 8),)),
        _ins(3, "str x3, [x9, #24]", {}, mem=(MemOp("w", OUT + 24, 0, 8),)),
        _ins(4, "str x5, [x12]", {}, mem=(MemOp("w", 0x90000, 0, 8),)),  # lone, far away
    ]
    assert outparam_sinks(items) == [0, 1, 2, 3]


def test_outparam_sinks_empty_without_mem_writes():
    items = [_ins(0, "add x0, x1, x2", {})]
    assert outparam_sinks(items) == []
