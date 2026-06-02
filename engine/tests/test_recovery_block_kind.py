"""§需求1/3 — terminal block_kind 5-class split + symbol_trace evidence.

Pins the additive ``block_kind`` field that stops the opaque MERGE POINT from
塌成 a flat "opaque": F0-shape (opaque staging) and TC2-shape (symbol off the
output path) must surface DIFFERENT block kinds, the degenerate tail still emits a
verdict with ``excluded``, and a symbol_not_on_output_path block carries a
``symbol_trace`` localizing where the symbolic chain broke.

Synthetic shapes only — no case addresses / values / handler ids (invariant 2/6).
The decision tree is deterministic (invariant 8); these test it directly on
already-structured signals + end-to-end through the Verifier for the two anchors.
"""

from __future__ import annotations

from engine.cvd import CvdState, VStatus
from engine.cvd_recovery import (
    BLOCK_EMIT_PICKED_CONSTANT,
    BLOCK_OPAQUE_STAGING,
    BLOCK_SYMBOL_NOT_ON_OUTPUT_PATH,
    BLOCK_UNDETERMINED_CONSTANT,
    BLOCK_WINDOW_BOUNDARY_MISMATCH,
    RecoverWindowVerifier,
    _classify_block_kind,
    _dfg_symbol_trace,
)
from engine.opaque_staging import (
    BlindLoad,
    StagingDiagnosis,
    VERDICT_KNOWN_ADDR,
    VERDICT_SYMBOLIC_ADDRESS,
)
from engine.setup_symex import CaseConfig, DriveResult, build_concrete_backing
from engine.types import Instruction, MemOp


# --------------------------------------------------------------------------- #
# Builders — a minimal DriveResult + trace, parameterised per block-kind shape.
# --------------------------------------------------------------------------- #

def _ins(idx, pc, mnem, *, reads=None, writes=None, mem=()):
    return Instruction(idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic=mnem,
                       regs_read=reads or {}, regs_write=writes or {}, mem=tuple(mem))


def _mem_r(addr, size, val):
    return MemOp(rw="r", addr=addr, size=size, val=val)


def _result(*, emitted_F=None, symbolic_forwards=0, symbolized=None,
            n_window_items=4):
    """A DriveResult carrying ONLY the structured signals the block-kind tree
    reads (seed step's mem_live_in.{symbolized,n_window_items}, symbolic_forwards,
    emitted_F). Everything else is a placeholder — the tree never reads it."""
    mem_live_in = {"n_window_items": n_window_items,
                   "symbolized": sorted(symbolized or []),
                   "decided_back": [], "unpinned": []}
    per_step = ({"step": "seed_entry_state", "symbolic_regs": [],
                 "mem_live_in": mem_live_in},)
    return DriveResult(
        closed=False, mode="encoded", parity=None, emitted_F=emitted_F,
        backing_ok=True, address_closure={}, mem_backing={}, per_step=per_step,
        entry_keys=(), view_path=None, checkpoints={},
        symbolic_forwards=symbolic_forwards)


def _diag(verdict, *, idx=0, pc=0x1000):
    return StagingDiagnosis(
        window=(0, 4), window_is_idx=True, verdict=verdict,
        blind_loads=(BlindLoad(idx=idx, pc=pc, ea_regs=("x1",), ea_symbolic=True),)
        if verdict == VERDICT_SYMBOLIC_ADDRESS else ())


_WIN = (0, 4)


# =========================================================================== #
# §需求1 — the 5 classes, each surfaced from its own structural signal.
# =========================================================================== #

def test_block_opaque_staging_symbolic_ea_no_forward():
    # EA symbolic + forwarding rescued nothing → opaque staging (F0 shape).
    res = _result(symbolic_forwards=0)
    bk, detail = _classify_block_kind(
        res, items=[], window=_WIN, window_is_idx=True, inputs=("carrier",),
        staging_diag=_diag(VERDICT_SYMBOLIC_ADDRESS))
    assert bk == BLOCK_OPAQUE_STAGING
    assert detail["ea_symbolic"] and detail["symbolic_forwards"] == 0


def test_block_window_boundary_mismatch_zero_items():
    # window matched 0 trace items → boundary mismatch.
    res = _result(n_window_items=0)
    bk, detail = _classify_block_kind(
        res, items=[], window=_WIN, window_is_idx=True, inputs=("carrier",),
        staging_diag=_diag(VERDICT_KNOWN_ADDR))
    assert bk == BLOCK_WINDOW_BOUNDARY_MISMATCH
    assert detail["n_window_items"] == 0


def test_block_window_boundary_mismatch_symbol_outside_window():
    # the symbolized byte is loaded OUTSIDE the window band → boundary mismatch.
    trace = [_ins(9, 0x9000, "ldr w0, [x1]", reads={"x1": 0x8000},
                  mem=[_mem_r(0x8000, 4, 7)])]               # idx 9 is OUTSIDE (0,4)
    res = _result(symbolized=[0x8000], n_window_items=4)
    bk, detail = _classify_block_kind(
        res, items=trace, window=_WIN, window_is_idx=True, inputs=("carrier",),
        staging_diag=_diag(VERDICT_KNOWN_ADDR))
    assert bk == BLOCK_WINDOW_BOUNDARY_MISMATCH
    assert "0x8000" in detail["symbolized_outside_window"]


def test_block_symbol_not_on_output_path_breaks_before_exit():
    # symbol loaded in-window, EA concrete, but its value never reaches the exit
    # node's DFG → symbol off the output path (TC2 shape) + a symbol_trace.
    trace = [
        _ins(0, 0x1000, "ldr w0, [x1]", reads={"x1": 0x8000},
             writes={"w0": 7}, mem=[_mem_r(0x8000, 4, 7)]),   # the symbol load
        _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 7}, writes={"w2": 14}),
        # exit node computes from a DIFFERENT register (w9) — symbol not on its path
        _ins(2, 0x1008, "mov w3, w9", reads={"w9": 3}, writes={"w3": 3}),
    ]
    res = _result(symbolized=[0x8000], n_window_items=3)
    bk, detail = _classify_block_kind(
        res, items=trace, window=(0, 2), window_is_idx=True, inputs=("carrier",),
        staging_diag=_diag(VERDICT_KNOWN_ADDR))
    assert bk == BLOCK_SYMBOL_NOT_ON_OUTPUT_PATH
    st = detail["symbol_trace"]
    assert st["entered_emit_dfg"] is False
    assert st["last_seen_idx"] == 1               # last node still carrying the symbol
    assert st["constantized_at_pc"] == "0x1004"   # PC where the chain ended


def test_block_emit_picked_constant():
    # nothing symbolized, EA concrete, but symex emitted an F that references no
    # input (collapsed to a constant at emit) → emit_picked_constant.
    res = _result(emitted_F="def f(carrier):\n    return 7\n", symbolized=[])
    bk, detail = _classify_block_kind(
        res, items=[], window=_WIN, window_is_idx=True, inputs=("carrier",),
        staging_diag=_diag(VERDICT_KNOWN_ADDR))
    assert bk == BLOCK_EMIT_PICKED_CONSTANT
    assert detail["emitted_F"].endswith("return 7\n")


def test_block_undetermined_constant_tail_carries_excluded():
    # nothing symbolized, EA concrete, nothing emitted → degenerate tail. Still a
    # verdict, with the excluded set (A8④ never卡死).
    res = _result(emitted_F=None, symbolized=[])
    bk, detail = _classify_block_kind(
        res, items=[], window=_WIN, window_is_idx=True, inputs=("carrier",),
        staging_diag=_diag(VERDICT_KNOWN_ADDR))
    assert bk == BLOCK_UNDETERMINED_CONSTANT
    assert isinstance(detail["excluded"], list) and len(detail["excluded"]) == 4
    # each excluded entry names the class it ruled out
    joined = " ".join(detail["excluded"])
    for cls in ("opaque_staging", "window_boundary_mismatch",
                "symbol_not_on_output_path", "emit_picked_constant"):
        assert cls in joined


def test_block_kinds_are_mutually_exclusive_priority():
    # opaque_staging (EA symbolic, no forward) OUTRANKS a co-present boundary/emit
    # signal — the ordered tree returns exactly one, the highest-priority match.
    res = _result(emitted_F="def f(carrier):\n    return 7\n",
                  symbolic_forwards=0, n_window_items=0)
    bk, _ = _classify_block_kind(
        res, items=[], window=_WIN, window_is_idx=True, inputs=("carrier",),
        staging_diag=_diag(VERDICT_SYMBOLIC_ADDRESS))
    assert bk == BLOCK_OPAQUE_STAGING        # priority 1 wins over the rest


# =========================================================================== #
# §需求3 — symbol_trace: symbol reaches emit DFG vs breaks at a PC (two forms).
# =========================================================================== #

def test_symbol_trace_enters_emit_dfg():
    # the symbol flows straight to the exit node → entered_emit_dfg true, no break.
    trace = [
        _ins(0, 0x1000, "ldr w0, [x1]", reads={"x1": 0x8000},
             writes={"w0": 7}, mem=[_mem_r(0x8000, 4, 7)]),
        _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 7}, writes={"w2": 14}),
        _ins(2, 0x1008, "eor w3, w2, w2", reads={"w2": 14}, writes={"w3": 0}),
    ]
    st = _dfg_symbol_trace(trace, seed_idxs=[0], window=(0, 2), window_is_idx=True)
    assert st["entered_emit_dfg"] is True
    assert st["constantized_at_pc"] is None
    assert st["last_seen_idx"] == 2


def test_symbol_trace_breaks_at_pc():
    # the symbol dies at idx1; the exit node (idx2) reads an unrelated reg → break.
    trace = [
        _ins(0, 0x1000, "ldr w0, [x1]", reads={"x1": 0x8000},
             writes={"w0": 7}, mem=[_mem_r(0x8000, 4, 7)]),
        _ins(1, 0x1004, "add w2, w0, w0", reads={"w0": 7}, writes={"w2": 14}),
        _ins(2, 0x1008, "mov w3, w9", reads={"w9": 3}, writes={"w3": 3}),
    ]
    st = _dfg_symbol_trace(trace, seed_idxs=[0], window=(0, 2), window_is_idx=True)
    assert st["entered_emit_dfg"] is False
    assert st["last_seen_idx"] == 1
    assert st["constantized_at_pc"] == "0x1004"


# =========================================================================== #
# End-to-end through the Verifier — the two real anchors (F0 / TC2 shapes).
# =========================================================================== #

_BASE = CaseConfig(
    target="synthetic.so", input_hash="ab12", run_id="run-1",
    seed_hint_addr=0x100, sink_hint_addr=0x200, entry_pc=0x0FFF,
    window=(0, 2), window_kind="idx", reg_file=("x0", "x1", "x16"),
    inputs=("carrier",), parity_min=8, symbolic_regs=("x0",),
    concrete_backing=build_concrete_backing(reg_values={"x16": 0x9000}),
    task="recover_window")

from engine.cvd import Candidate
from engine.cvd_recovery import RECOVER_WINDOW

_REC = Candidate(RECOVER_WINDOW, 0, "dispatch_type_rep", "rep",
                 payload={"window": [0, 2], "window_kind": "idx"})
_BOTH = {"alias_vs_compute": "compute", "which_static": []}


def test_verifier_opaque_terminal_carries_block_kind_and_coverage():
    # A collapsed-F opaque terminal now carries an explicit block_kind + the
    # build's capability coverage stamp (this build has the token → coverage_ok).
    # register-backed window (x16 pinned via concrete_backing, no external mem
    # live-in) so the run reaches the opaque terminal instead of pausing for a
    # symbolize-vs-back judgment.
    backed = [
        _ins(0, 0x1000, "ldr w0, [x16]", reads={"x16": 0x9000}),
        _ins(1, 0x1004, "mul w0, w0, w1", reads={"w0": 0, "w1": 0}, writes={"w0": 0}),
    ]

    def runner(_ctx):
        return {"propagated": False, "expr_source": ""}

    v = RecoverWindowVerifier(
        base_config=_BASE, triton_runner=runner, decisions=_BOTH
    ).verify(_REC, CvdState(backed, b"\x00"))
    assert v.status is VStatus.TERMINAL and v.terminal_kind == "opaque_staging"
    assert "block_kind" in v.evidence
    assert v.evidence["coverage_ok"] is True
    assert "recovery_block_kind_v1" in v.evidence["capabilities"]
    assert "coverage_warn" not in v.evidence       # this build has every token
