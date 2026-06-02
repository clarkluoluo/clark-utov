"""capability_request.md §P1-3 — differential localization tests."""

from __future__ import annotations

from engine.localize import (
    CandidateHypothesis,
    DivergencePoint,
    LocalizeResult,
    find_first_divergence,
    localize_divergence,
)
from engine.types import Instruction, MemOp


def _ins(idx, pc, regs_read=None, regs_write=None, mem=()):
    return Instruction(
        idx=idx, pc=pc, bytes_=b"\x00\x00\x00\x00", mnemonic="op",
        regs_read=regs_read or {}, regs_write=regs_write or {},
        mem=tuple(mem),
    )


def test_identical_traces_have_no_divergence():
    a = [_ins(i, 0x100 + 4*i, {}, {"x0": i}) for i in range(5)]
    b = [_ins(i, 0x100 + 4*i, {}, {"x0": i}) for i in range(5)]
    assert find_first_divergence(a, b) is None
    r = localize_divergence(a, b)
    assert r.divergence is None
    assert r.candidates == ()


def test_register_write_divergence_caught_and_hypothesised():
    good = [
        _ins(0, 0x100, {}, {"x0": 1}),
        _ins(1, 0x104, {}, {"x1": 2}),
        _ins(2, 0x108, {}, {"x2": 3}),
    ]
    bad = [
        _ins(0, 0x100, {}, {"x0": 1}),
        _ins(1, 0x104, {}, {"x1": 999}),     # diverges here
        _ins(2, 0x108, {}, {"x2": 3}),
    ]
    r = localize_divergence(good, bad)
    assert r.divergence is not None
    assert r.divergence.kind == "regs_write"
    assert r.divergence.pc == 0x104
    # candidate hypothesis must name the divergent register
    assert r.candidates
    assert any("x1" in c.subject for c in r.candidates)
    assert any("0x2" in (c.payload.get("good") or "") and
               "0x3e7" in (c.payload.get("bad") or "")
               for c in r.candidates)


def test_pc_divergence_emits_cf_candidate():
    """Same step index, different PC — control-flow branched."""
    good = [_ins(0, 0x100, {}, {}), _ins(1, 0x110, {}, {})]
    bad  = [_ins(0, 0x100, {}, {}), _ins(1, 0x120, {}, {})]
    r = localize_divergence(good, bad)
    assert r.divergence is not None
    assert r.divergence.kind == "pc"
    assert r.candidates[0].kind == "control_flow_divergence"


def test_length_divergence_when_good_shorter():
    good = [_ins(0, 0x100, {}, {})]
    bad  = [_ins(0, 0x100, {}, {}), _ins(1, 0x104, {}, {"x0": 7})]
    r = localize_divergence(good, bad)
    assert r.divergence is not None
    assert r.divergence.kind == "length"
    assert r.divergence.good_idx == -1
    assert r.divergence.bad_idx == 1


def test_mem_divergence_caught():
    good = [_ins(0, 0x100, {}, {},
                 mem=[MemOp(rw="w", addr=0x1000, val=0x11, size=4)])]
    bad  = [_ins(0, 0x100, {}, {},
                 mem=[MemOp(rw="w", addr=0x1000, val=0x99, size=4)])]
    r = localize_divergence(good, bad)
    assert r.divergence is not None
    assert r.divergence.kind == "mem"
    assert r.candidates[0].kind == "divergent_mem_write"


def test_resync_finds_realignment_after_one_step():
    """After one divergent step, both traces should re-converge — the
    engine reports the resync point so a byte-graft can be tested."""
    good = [
        _ins(0, 0x100, {}, {"x0": 1}),
        _ins(1, 0x104, {}, {"x1": 2}),        # diverge here
        _ins(2, 0x108, {}, {"x2": 3}),
        _ins(3, 0x10c, {}, {"x3": 4}),
    ]
    bad = [
        _ins(0, 0x100, {}, {"x0": 1}),
        _ins(1, 0x104, {}, {"x1": 999}),      # diverge
        _ins(2, 0x108, {}, {"x2": 3}),        # back in sync
        _ins(3, 0x10c, {}, {"x3": 4}),
    ]
    r = localize_divergence(good, bad)
    assert r.resync_at is not None
    assert r.resync_at == (2, 2)


def test_no_resync_when_traces_drift():
    good = [_ins(i, 0x100 + 4*i, {}, {"x0": i}) for i in range(5)]
    bad  = [
        _ins(0, 0x100, {}, {"x0": 0}),
        _ins(1, 0x999, {}, {"x9": 1}),  # never matches good again
        _ins(2, 0x998, {}, {"x9": 2}),
    ]
    r = localize_divergence(good, bad, resync_look_ahead=4)
    assert r.divergence is not None
    assert r.resync_at is None
