"""Item ② — build_dfg regs_read fallback (coarse inferred-interval producers).

Synthetic, zero case-specific. Verifies: (a) a regs_write-populated trace is
byte-for-byte unchanged (no inferred edges, no new s3_dfg key); (b) a window with
empty regs_write but changing regs_read values yields coarse inferred-interval
producers marked low-confidence; (c) Phase 0 does NOT emit a false known_addr on
the regs_write-empty form (inconclusive instead).
"""

from __future__ import annotations

from engine.opaque_staging import VERDICT_KNOWN_ADDR, VERDICT_INCONCLUSIVE, diagnose_opaque_staging
from engine.stages.s3_triton import InferredProducer, build_dfg
from engine.types import Instruction, MemOp


def ins(idx, pc, mnem="nop", *, reads=None, writes=None, mem=()):
    return Instruction(
        idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
        regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


_STAGING = 0x10020


def test_regs_write_populated_no_inferred_edges():
    # regs_write present everywhere -> fallback never triggers, no inferred edges.
    items = [
        ins(0, 0x1000, "mov x1,#1", writes={"x1": 1}),
        ins(1, 0x1004, "add x2,x1,#1", reads={"x1": 1}, writes={"x2": 2}),
        ins(2, 0x1008, "add x3,x2,#1", reads={"x2": 2}, writes={"x3": 3}),
    ]
    nodes = build_dfg(items)
    assert all(not n.reg_deps_inferred for n in nodes)
    # original reg_deps unchanged: x1 produced by idx0, x2 by idx1.
    assert nodes[1].reg_deps["x1"] == 0
    assert nodes[2].reg_deps["x2"] == 1


def test_regs_read_change_yields_inferred_interval():
    # x9 is NEVER in any regs_write, but its regs_read value changes between reads
    # -> an inferred-interval producer (coarse, low confidence).
    items = [
        ins(0, 0x1000, "use x9", reads={"x9": 0x100}),
        ins(1, 0x1004, "nop"),
        ins(2, 0x1008, "use x9", reads={"x9": 0x200}),   # value changed
    ]
    nodes = build_dfg(items)
    assert "x9" not in nodes[0].reg_deps_inferred       # first read: no prior
    inf = nodes[2].reg_deps_inferred.get("x9")
    assert isinstance(inf, InferredProducer)
    assert inf.confidence == "inferred"
    assert inf.at_idx == 2 and inf.after_idx == 0       # interval (0, 2]
    # reg_deps stays None (external) for the original consumers — additive.
    assert nodes[2].reg_deps["x9"] is None


def test_regs_read_constant_no_inferred():
    # value does NOT change across reads -> no inferred producer (nothing wrote it).
    items = [
        ins(0, 0x1000, "use x9", reads={"x9": 0x100}),
        ins(1, 0x1004, "use x9", reads={"x9": 0x100}),
    ]
    nodes = build_dfg(items)
    assert all(not n.reg_deps_inferred for n in nodes)


def test_phase0_no_false_known_addr_on_empty_regs_write():
    # tc4-form: regs_write empty across the window; a load reads a staging address.
    # The EA backtrace would say "concrete base" (no producers visible) -> WITHOUT
    # the gate that is a false known_addr. The regs_write-coverage gate downgrades
    # it to inconclusive (honest), never a false known_addr.
    items = [
        ins(0, 0x1000, "str x8,[x10]", reads={"x8": 0x41, "x10": _STAGING},
            mem=[MemOp("w", _STAGING, 0x41, 8)]),
        ins(1, 0x1004, "ldr x9,[x10]", reads={"x10": _STAGING},
            mem=[MemOp("r", _STAGING, 0x41, 8)]),   # NO regs_write anywhere
    ]
    diag = diagnose_opaque_staging(items, window=(0, 1), window_is_idx=True)
    assert diag.verdict == VERDICT_INCONCLUSIVE
    assert diag.verdict != VERDICT_KNOWN_ADDR


def test_phase0_known_addr_when_regs_write_populated_control():
    # Control: same shape but regs_write populated -> original known_addr path,
    # proving the fallback/gate only fires on the regs_write-empty form.
    # x10 is a concrete live-in base (no in-window producer); regs_write IS
    # populated (the load writes x9) so the coverage gate does not fire. EA
    # backtraces to a concrete base -> known_addr (the proven form-A shape).
    items = [
        ins(0, 0x1000, "str x8,[x10]", reads={"x8": 0x41, "x10": _STAGING},
            writes={"x8": 0x41}, mem=[MemOp("w", _STAGING, 0x41, 8)]),
        ins(1, 0x1004, "ldr x9,[x10]", reads={"x10": _STAGING},
            writes={"x9": 0x41}, mem=[MemOp("r", _STAGING, 0x41, 8)]),
    ]
    diag = diagnose_opaque_staging(items, window=(0, 1), window_is_idx=True,
                                   symbolic_inputs=("x8",))
    assert diag.verdict == VERDICT_KNOWN_ADDR
