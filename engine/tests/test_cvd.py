"""CVD driver — the candidate-verification loop. Synthetic traces only.

Covers the three MVP paths (CVD_DESIGN §8): auto-pop the next candidate after an
elimination, auto-widen when in-scope candidates are exhausted, terminal
classification on a dead end — plus the §11 surprise weighting, the budget, and
the auditable record log.
"""

from __future__ import annotations

from engine.cvd import (
    BASE_VALUE,
    Candidate,
    CvdBudget,
    CvdOutcome,
    _roi,
    run_cvd,
)
from engine.types import Instruction, MemOp, MemSnapshot


def _ins(idx, mnem, reads=None, mem=()):
    return Instruction(idx=idx, pc=0x70000 + idx * 4, bytes_=b"\x00\x00\x00\x00",
                       mnemonic=mnem, regs_read=dict(reads or {}), regs_write={}, mem=mem)


def _le(b: bytes) -> int:
    return int.from_bytes(b, "little")


EXPECTED = bytes([0x34, 0x15, 0x5f, 0xe9])
SCRATCH = bytes([0x00, 0x11, 0x22, 0x33])
SCRATCH2 = bytes([0xaa, 0xbb, 0xcc, 0xdd])
SCR = 0x1000          # scratch buffer
REAL = 0x2000         # real output buffer == expected
SNAP = 0x72b18        # snapshot-located output


def _events(res):
    return [e["event"] for e in res.log]


# --- MVP path 1: auto-pop the next candidate after an elimination ----------

def test_auto_pop_next_candidate_after_elimination():
    trace = [
        _ins(0, "str x8, [x10]", reads={"x8": _le(SCRATCH), "x10": SCR},
             mem=(MemOp("w", SCR, _le(SCRATCH), 4),)),       # scratch (popped first)
        _ins(1, "str x8, [x9]", reads={"x8": _le(EXPECTED), "x9": REAL},
             mem=(MemOp("w", REAL, _le(EXPECTED), 4),)),     # real output
    ]
    res = run_cvd(trace, EXPECTED)
    assert res.outcome is CvdOutcome.SUCCESS
    assert res.sink_base == REAL
    assert res.verdict == "CONTINUOUS_BUFFER"
    # the driver eliminated the scratch then confirmed the real sink, by itself
    assert "ELIMINATED" in _events(res)
    assert "CONFIRMED" in _events(res)
    assert "WIDEN" not in _events(res)            # no widen needed in-scope


# --- MVP path 2: auto-widen when in-scope candidates are exhausted ----------

def test_auto_widen_when_in_scope_exhausted():
    # the real buffer write is OUTSIDE the initial window; only the scratch is in
    # scope. After the scratch is eliminated the driver widens (drops the window)
    # and finds the real buffer on its own.
    trace = [
        _ins(0, "str x8, [x10]", reads={"x8": _le(SCRATCH), "x10": SCR},
             mem=(MemOp("w", SCR, _le(SCRATCH), 4),)),       # in window
        _ins(1, "str x8, [x9]", reads={"x8": _le(EXPECTED), "x9": REAL},
             mem=(MemOp("w", REAL, _le(EXPECTED), 4),)),     # outside window
    ]
    res = run_cvd(trace, EXPECTED, window=(0, 0))            # scope 0 = only idx 0
    assert "WIDEN" in _events(res)
    assert res.outcome is CvdOutcome.SUCCESS
    assert res.sink_base == REAL


# --- MVP path 3: terminal classification on a dead end ----------------------

def test_terminal_classification_on_dead_end():
    # only a scratch write, no real buffer, no snapshot, nothing to widen.
    trace = [
        _ins(0, "str x8, [x10]", reads={"x8": _le(SCRATCH), "x10": SCR},
             mem=(MemOp("w", SCR, _le(SCRATCH), 4),)),
    ]
    res = run_cvd(trace, EXPECTED)
    assert res.outcome is CvdOutcome.TERMINAL
    assert res.verdict == "OUTPUT_NOT_OBSERVABLE"
    assert res.capability_request                          # precise hand-off, not empty
    assert "TERMINAL" in _events(res)


# --- obs-source widen surfaces a snapshot sink, then classifies its terminal -

def test_widen_to_snapshot_then_opaque_or_needs_terminal():
    # nothing in writes matches; the output is in a snapshot. Scope-0 misses it;
    # widen adds the snapshot source -> sink confirmed -> #3 returns a terminal
    # (no traced producer for a snapshot-only sink).
    trace = [
        _ins(0, "str x8, [x10]", reads={"x8": _le(SCRATCH), "x10": SCR},
             mem=(MemOp("w", SCR, _le(SCRATCH), 4),)),
    ]
    snaps = [MemSnapshot(addr=SNAP, data=EXPECTED, label="output")]
    res = run_cvd(trace, EXPECTED, snapshots=snaps)
    assert "WIDEN" in _events(res)
    assert "CONFIRMED" in _events(res)                     # snapshot sink confirmed after widen
    assert res.outcome is CvdOutcome.TERMINAL
    assert res.verdict in ("NEEDS_OBSERVATION", "OPAQUE_CALLEE")
    assert res.capability_request


# --- §11 surprise weighting: a rare signal outscores a common one -----------

def test_surprise_spikes_rare_signal_over_common():
    history = {"write_cluster": 20}                        # write_cluster is common now
    common = Candidate("sink", SCR, "write_cluster", "", base_value=1.0)
    rare = Candidate("sink", REAL, "rare_diagnostic", "", base_value=1.0)
    assert _roi(rare, history, stall=0) > _roi(common, history, stall=0)


def test_stall_context_boosts_score():
    history = {"write_cluster": 5}
    c = Candidate("sink", SCR, "write_cluster", "", base_value=1.0)
    assert _roi(c, history, stall=4) > _roi(c, history, stall=0)


# --- §7 bounds: budget exhaustion emits the remaining frontier --------------

def test_budget_exhaustion_emits_remaining_frontier():
    trace = [
        _ins(0, "str x8, [x10]", mem=(MemOp("w", 0x1000, _le(SCRATCH), 4),)),
        _ins(1, "str x8, [x11]", mem=(MemOp("w", 0x3000, _le(SCRATCH2), 4),)),
        _ins(2, "str x8, [x12]", mem=(MemOp("w", 0x5000, _le(SCRATCH), 4),)),
    ]
    res = run_cvd(trace, EXPECTED, budget=CvdBudget(max_candidates=1, max_widen=0))
    assert res.outcome is CvdOutcome.BUDGET_EXHAUSTED
    assert res.checkpoint and res.checkpoint["frontier"]   # remaining frontier, not truncated
